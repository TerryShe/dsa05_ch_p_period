"""Reliability-first local rules and the centralized causal scheduler of scenario v2.0.

Local rules (docs/research/scenario-v2.md §8) decide per UE from that UE's own
observation, including its current own gains ``own_gain_w``, and its own RNG.
``random_max`` and ``sticky_max`` of v1.0 are reused unchanged (``policies``).

Common quantities of one UE, all local:

* interference I_k: last slot's public RSSI, with the UE's own received power
  removed on its own last subchannel (``baselines.estimate_link``);
* rate at max power: ``bits(P_max g_k / I_k)`` with interference, or
  ``bits(P_max g_k / noise)`` alone;
* slack: slots to the deadline minus ceil(remaining bits / rate), the number of
  slots the UE can still skip if the rate held.

``CentralEdfPolicy`` is the reference N_EDF (§7): it sees every UE's queue and
current gains at once, so it is not a local rule. ``ConfiguredGrantPolicy`` is
the semi-static reference N_CG (2026-10-09 supplement to §7): exclusive periodic
occasions allocated once from mean gains, traffic and fading statistics.
``ConfiguredGrantOptPolicy`` (CG-opt, 2026-10-10 supplement) keeps its knowledge
and transmission rule and optimizes the allocation (``cg_opt``).
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dsa_period.baselines import estimate_link
from dsa_period.channel import ShadowedJakesChannel
from dsa_period.env import Outcome, UEObservation
from dsa_period.physics import PhysicalConfig, PowerMenu, shannon_bits
from dsa_period.policies import POLICIES, HeuristicPolicy, Policy
from dsa_period.traffic import TrafficPattern
from dsa_period.urgency import SummarySettings, YieldPolicy, parse_wrapper


def _slack(obs: UEObservation, rate: float) -> float:
    if rate <= 0:
        return -math.inf
    return obs.slots_to_deadline - math.ceil(obs.remaining_bits / rate - 1e-9)


class _LocalRule:
    """Shared per-UE plumbing: one RNG per UE, OFF when there is nothing to send."""

    name = "local"

    def __init__(self, config: PhysicalConfig, menu: PowerMenu) -> None:
        self.config, self.menu = config, menu
        self._top = menu.num_levels - 1
        self._pmax = menu.power_w[-1]
        self._rngs: list[np.random.Generator] = []

    def reset(self, rngs: Sequence[np.random.Generator]) -> None:
        self._rngs = list(rngs)

    def _decide(self, ue: int, obs: UEObservation) -> tuple[int, int]:
        raise NotImplementedError

    def act(self, observations: Sequence[UEObservation]) -> NDArray[np.int64]:
        if len(observations) != len(self._rngs):
            raise ValueError("need one observation per UE; call reset first")
        actions = np.zeros((len(observations), 2), dtype=np.int64)
        for ue, obs in enumerate(observations):
            if obs.own_gain_w is None:
                raise ValueError(f"{self.name} needs own_gain_w (scenario v2.0)")
            if obs.remaining_bits > 0:
                actions[ue] = self._decide(ue, obs)
            else:
                self._idle(ue, obs)
        return actions

    def _idle(self, ue: int, obs: UEObservation) -> None:
        return None

    def _rates(self, obs: UEObservation) -> NDArray[np.float64]:
        """Max-power mutual information per subchannel under last slot's interference."""
        _, interference = estimate_link(obs, self.config, self.menu)
        return shannon_bits(self.config, self._pmax * np.asarray(obs.own_gain_w) / interference)


class PPersistentPolicy(_LocalRule):
    """Deadline-aware p-persistent access (v2.0 §8).

    Channel: the subchannel with the largest own gain; max power. With slack s
    computed from the interference-free rate on that subchannel, transmit with
    probability 1 if s <= 0 and min(1, a / (s + 1)) otherwise: about ``a``
    attempts are spread over the s + 1 slots the UE can still choose from.
    """

    def __init__(self, config: PhysicalConfig, menu: PowerMenu, *, a: float = 1.0) -> None:
        super().__init__(config, menu)
        if not a > 0:
            raise ValueError("a must be positive")
        self.a = float(a)
        self.name = f"p_persistent(a={a:g})"

    def _decide(self, ue: int, obs: UEObservation) -> tuple[int, int]:
        gains = np.asarray(obs.own_gain_w)
        channel = int(np.argmax(gains))
        rate = float(
            shannon_bits(self.config, self._pmax * gains[channel] / self.config.noise_power_w)
        )
        slack = _slack(obs, rate)
        probability = 1.0 if slack <= 0 else min(1.0, self.a / (slack + 1))
        # Always draw, so a UE's random stream does not depend on its slack history.
        if self._rngs[ue].random() < probability:
            return channel, self._top
        return 0, 0


class OpportunisticPolicy(_LocalRule):
    """Opportunistic threshold rule (v2.0 §8).

    On the subchannel with the largest estimated max-power rate r (own current
    gain over last slot's interference), transmit at max power only if r reaches
    ``theta`` times the pace remaining_bits / slots_to_deadline, or if the slack
    at rate r is <= 0. Deterministic.
    """

    def __init__(self, config: PhysicalConfig, menu: PowerMenu, *, theta: float = 2.0) -> None:
        super().__init__(config, menu)
        if not theta > 0:
            raise ValueError("theta must be positive")
        self.theta = float(theta)
        self.name = f"opportunistic(theta={theta:g})"

    def _decide(self, ue: int, obs: UEObservation) -> tuple[int, int]:
        rates = self._rates(obs)
        channel = int(np.argmax(rates))
        rate = float(rates[channel])
        pace = obs.remaining_bits / obs.slots_to_deadline
        if rate >= self.theta * pace or _slack(obs, rate) <= 0:
            return channel, self._top
        return 0, 0


class ReservationPolicy(_LocalRule):
    """Periodic reservation, inspired by PT-CD (v2.0 §8).

    A UE keeps a reservation (offset o, subchannel c). Every packet is sent at max
    power on c from o slots after its arrival until it decodes. After
    ``failures`` consecutive packets that missed their deadline, the UE draws a
    new o uniformly from {0, ..., floor(offset * P) - 1} (at least {0}) and a new
    c uniformly. The first reservation is drawn at the first packet.
    """

    def __init__(
        self, config: PhysicalConfig, menu: PowerMenu, *, offset: float = 0.5, failures: int = 1
    ) -> None:
        super().__init__(config, menu)
        if not 0 < offset <= 1 or int(failures) != failures or failures < 1:
            raise ValueError("offset must be in (0, 1] and failures a positive integer")
        self.offset, self.failures = float(offset), int(failures)
        self.name = f"reservation(offset={offset:g},failures={failures:g})"

    def reset(self, rngs: Sequence[np.random.Generator]) -> None:
        super().reset(rngs)
        n = len(self._rngs)
        self._slot: list[int | None] = [None] * n
        self._channel = [0] * n
        self._misses = [0] * n
        self._open = [False] * n  # a packet was released and has not decoded yet
        self._fresh = [False] * n

    def _draw(self, ue: int, period: int) -> None:
        rng = self._rngs[ue]
        self._slot[ue] = int(rng.integers(max(1, math.floor(self.offset * period))))
        self._channel[ue] = int(rng.integers(self.config.num_subchannels))

    def _track(self, ue: int, obs: UEObservation) -> None:
        """Settle the previous packet when a new one appears; count misses."""
        if obs.previous_outcome == Outcome.DECODED:
            self._open[ue] = False
            self._misses[ue] = 0
        released = obs.remaining_bits > 0 and obs.slots_to_deadline == obs.period_slots
        if not released:
            return
        if self._open[ue]:
            self._misses[ue] += 1
        self._open[ue] = True
        if self._slot[ue] is None or self._misses[ue] >= self.failures:
            self._draw(ue, obs.period_slots)
            self._misses[ue] = 0

    def _decide(self, ue: int, obs: UEObservation) -> tuple[int, int]:
        self._track(ue, obs)
        elapsed = obs.period_slots - obs.slots_to_deadline
        if elapsed >= self._slot[ue]:
            return self._channel[ue], self._top
        return 0, 0

    def _idle(self, ue: int, obs: UEObservation) -> None:
        self._track(ue, obs)


class PacedStickyCsiPolicy(_LocalRule):
    """paced_sticky of v1.0 with current own gains (v2.0 §8).

    Link score of subchannel k: own gain over last slot's interference. Keep the
    current subchannel while its score is within ``switch_db`` of the best;
    otherwise, with probability ``switch_probability``, move uniformly to one of
    the subchannels within 3 dB of the best. Power: the lowest level whose rate
    covers (1 + margin) * remaining_bits / slots_to_deadline, max power if none.
    """

    def __init__(
        self,
        config: PhysicalConfig,
        menu: PowerMenu,
        *,
        margin: float = 0.5,
        switch_db: float = 3.0,
        switch_probability: float = 0.5,
    ) -> None:
        super().__init__(config, menu)
        if margin < 0 or switch_db < 0 or not 0 < switch_probability <= 1:
            raise ValueError("margin, switch_db must be >= 0 and switch_probability in (0, 1]")
        self.margin, self.switch_db, self.switch_probability = margin, switch_db, switch_probability
        self.name = f"paced_sticky_csi(margin={margin:g})"
        self._power = np.asarray(menu.power_w[1:])

    def reset(self, rngs: Sequence[np.random.Generator]) -> None:
        super().reset(rngs)
        self._current: list[int | None] = [None] * len(self._rngs)

    def _decide(self, ue: int, obs: UEObservation) -> tuple[int, int]:
        rng = self._rngs[ue]
        _, interference = estimate_link(obs, self.config, self.menu)
        score = np.asarray(obs.own_gain_w) / interference
        best = float(score.max())
        channel = self._current[ue]
        poor = channel is None or score[channel] < best * 10 ** (-self.switch_db / 10)
        if poor and (channel is None or rng.random() < self.switch_probability):
            candidates = np.flatnonzero(score >= best * 10**-0.3)
            channel = int(candidates[rng.integers(candidates.size)])
        self._current[ue] = channel
        required = (1 + self.margin) * obs.remaining_bits / obs.slots_to_deadline
        bits = shannon_bits(self.config, self._power * score[channel])
        enough = np.flatnonzero(bits >= required)
        return channel, int(enough[0]) + 1 if enough.size else self._top


class CentralEdfPolicy:
    """Centralized causal scheduler N_EDF (v2.0 §7); not a local rule.

    Each slot the AP ranks UEs holding a packet by slack (slots to the deadline
    minus the slots still needed at the interference-free max-power rate of the
    UE's best current subchannel), then by slots to the deadline, then by UE
    index. In that order each UE gets its best free subchannel at max power;
    each subchannel carries at most one UE. With ``defer_hopeless`` UEs whose
    slack is negative go after all others (they are still served if subchannels
    remain). Deterministic; it knows current gains of all UEs, never the future.
    """

    def __init__(
        self, config: PhysicalConfig, menu: PowerMenu, *, defer_hopeless: bool = False
    ) -> None:
        self.config, self.menu = config, menu
        self.defer_hopeless = bool(defer_hopeless)
        self.name = "central_edf" + ("(defer)" if self.defer_hopeless else "")
        self._top = menu.num_levels - 1

    def reset(self, rngs: Sequence[np.random.Generator]) -> None:
        return None

    def act(self, observations: Sequence[UEObservation]) -> NDArray[np.int64]:
        n = len(observations)
        actions = np.zeros((n, 2), dtype=np.int64)
        waiting = [ue for ue, obs in enumerate(observations) if obs.remaining_bits > 0]
        if not waiting:
            return actions
        gains = np.array([observations[ue].own_gain_w for ue in waiting])
        rate = shannon_bits(
            self.config, self.menu.power_w[-1] * gains.max(axis=1) / self.config.noise_power_w
        )
        keys = []
        for row, ue in enumerate(waiting):
            obs = observations[ue]
            slack = _slack(obs, float(rate[row]))
            hopeless = self.defer_hopeless and slack < 0
            keys.append((hopeless, slack, obs.slots_to_deadline, ue, row))
        free = set(range(self.config.num_subchannels))
        for *_, ue, row in sorted(keys):
            if not free:
                break
            order = np.argsort(-gains[row], kind="stable")
            channel = next(int(k) for k in order if int(k) in free)
            free.remove(channel)
            actions[ue] = (channel, self._top)
        return actions


# Configured-grant reference N_CG
# (docs/experiment-designs/configured-grant-baseline_20261009.md).
CG_BANK_SEED = 20_261_009
CG_BANK_SAMPLES = 20_000
_BANK_CHUNK = 500


@lru_cache(maxsize=4)  # banks A and B of CG-opt, two window lengths
def fading_bank(
    num_subchannels: int,
    doppler_hz: float,
    sinusoids: int,
    slot_duration_s: float,
    slots: int,
    samples: int = CG_BANK_SAMPLES,
    seed: int = CG_BANK_SEED,
) -> NDArray[np.float64]:
    """(samples, slots, K) small-scale power |h|^2 of independent fading realizations.

    The scenario's own sum-of-sinusoids generator with unit mean gain, one
    realization per sample and slots [0, slots) of each (design §3.5). The seed
    base differs from every protocol v2 seed base; cached per process, read-only.
    """
    seeds = np.random.SeedSequence(seed).spawn(samples)
    bank = np.empty((samples, slots, num_subchannels))
    for start in range(0, samples, _BANK_CHUNK):
        part = seeds[start : start + _BANK_CHUNK]
        channel = ShadowedJakesChannel(
            np.ones(len(part)),
            num_subchannels,
            doppler_hz=doppler_hz,
            slot_duration_s=slot_duration_s,
            ue_seeds=part,
            sinusoids=sinusoids,
        )
        h = channel.small_scale(np.arange(slots))
        bank[start : start + len(part)] = np.transpose(h.real**2 + h.imag**2, (1, 0, 2))
    bank.flags.writeable = False
    return bank


@dataclass(frozen=True)
class GrantAllocation:
    """Exclusive periodic occasions of every UE and the AP's predictions.

    ``occasions[i]`` lists UE i's (slot residue r mod P_i, subchannel k) in the
    order they were granted; ``outage[i]`` is the predicted probability that a
    packet misses with them, ``need[i]`` the packet size over the expected
    bits per slot; ``order`` is the UE granted at every step and ``blocked``
    counts UEs closed by placement while enough free resource units remained.
    """

    occasions: tuple[tuple[tuple[int, int], ...], ...]
    outage: tuple[float, ...]
    need: tuple[float, ...]
    order: tuple[int, ...]
    hyperperiod: int
    assigned_share: float
    blocked: int


def place_occasion(
    ue: int,
    owner: NDArray[np.int64],
    occasions: Sequence[tuple[int, int]],
    period: int,
    phase: int,
    column: int,
    single_channel: bool = False,
) -> tuple[int, int] | None:
    """Where UE ``ue`` gets its next occasion (design §3.4), or None if nowhere.

    ``owner`` is the (K, hyperperiod) grid of resource units (-1 free). A
    candidate (r, k) needs a residue r unused by the UE and every unit
    (r + nP, k) free. Preference: the subchannel with the fewest of the UE's
    occasions; a column (k, r mod ``column``) already partly used by another UE;
    the largest minimum window distance to the UE's occasions on that
    subchannel; the smallest (k, r). CG-single-channel keeps a UE on the
    subchannel of its first occasion, which goes to the subchannel with the most
    free units (phase-0 amendment: the shared first criterion would put every
    UE on subchannel 0).
    """
    num_subchannels, hyperperiod = owner.shape
    used = {r for r, _ in occasions}
    per_channel = np.bincount([k for _, k in occasions], minlength=num_subchannels)
    channels = range(num_subchannels)
    if single_channel and occasions:
        channels = [occasions[0][1]]
    elif single_channel:
        per_channel = -np.count_nonzero(owner == -1, axis=1)
    best, best_key = None, None
    for k in channels:
        positions = [(r - phase) % period for r, kk in occasions if kk == k]
        for r in range(period):
            if r in used or np.any(owner[k, r::period] != -1):
                continue
            cells = owner[k, r % column :: column]
            shared = bool(np.any((cells != -1) & (cells != ue)))
            p = (r - phase) % period
            spread = min(abs(p - q) for q in positions) if positions else math.inf
            key = (per_channel[k], not shared, -spread, k, r)
            if best_key is None or key < best_key:
                best, best_key = (r, k), key
    return best


def allocate_grants(
    config: PhysicalConfig,
    snr: NDArray[np.float64],
    packet_bits: NDArray[np.float64],
    period: NDArray[np.int64],
    phase: NDArray[np.int64],
    bank: NDArray[np.float64],
    single_channel: bool = False,
) -> GrantAllocation:
    """Greedy max-min allocation of exclusive periodic occasions (design §3.3).

    ``snr`` is each UE's mean SNR at max power alone; ``bank`` holds (samples,
    window slots, K) fading power. Every step grants one occasion to the open UE
    with the largest predicted outage (ties: fewer occasions per unit of need,
    then the lower index); a UE closes when it holds P_i occasions or none fits.
    """
    n = len(snr)
    num_subchannels = bank.shape[2]
    period = [int(p) for p in period]
    if max(period) > bank.shape[1] or num_subchannels != config.num_subchannels:
        raise ValueError("the fading bank must cover the longest window and every subchannel")
    hyperperiod = math.lcm(*period)
    column = math.gcd(*period)
    owner = np.full((num_subchannels, hyperperiod), -1, dtype=np.int64)
    expected = np.array([shannon_bits(config, s * bank[:, 0, :]).mean() for s in snr])
    need = np.asarray(packet_bits, dtype=np.float64) / expected
    occasions: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    accumulated = np.zeros((n, bank.shape[0]))
    outage = np.ones(n)
    open_ = np.ones(n, dtype=bool)
    order, blocked = [], 0
    while open_.any():
        ue = min(
            np.flatnonzero(open_),
            key=lambda i: (-outage[i], len(occasions[i]) / need[i], i),
        )
        spot = place_occasion(
            int(ue), owner, occasions[ue], period[ue], int(phase[ue]), column, single_channel
        )
        if spot is None:
            open_[ue] = False
            blocked += int(np.count_nonzero(owner == -1) >= hyperperiod // period[ue])
            continue
        r, k = spot
        owner[k, r :: period[ue]] = ue
        occasions[ue].append((r, k))
        order.append(int(ue))
        position = (r - int(phase[ue])) % period[ue]
        accumulated[ue] += shannon_bits(config, snr[ue] * bank[:, position, k])
        outage[ue] = float(np.mean(accumulated[ue] < packet_bits[ue]))
        if len(occasions[ue]) == period[ue]:
            open_[ue] = False
    return GrantAllocation(
        occasions=tuple(tuple(o) for o in occasions),
        outage=tuple(float(q) for q in outage),
        need=tuple(float(x) for x in need),
        order=tuple(order),
        hyperperiod=hyperperiod,
        assigned_share=float(np.mean(owner != -1)),
        blocked=blocked,
    )


class ConfiguredGrantPolicy:
    """Configured-grant reference N_CG (TSCAI + configured grant); not a local rule.

    ``configure`` gives the AP what TSCAI and long-term measurement give it:
    every UE's mean gain (path loss and shadowing), period, packet size and
    arrival phase, and the fading statistics. It never sees current gains,
    queues or outcomes. It allocates exclusive periodic occasions once per
    episode (``allocate_grants``); a UE then transmits at max power in its
    occasions while it holds bits. Deterministic.
    """

    def __init__(
        self, config: PhysicalConfig, menu: PowerMenu, *, single_channel: bool = False
    ) -> None:
        self.config, self.menu = config, menu
        self.single_channel = bool(single_channel)
        self.name = "configured_grant" + ("(single_channel)" if self.single_channel else "")
        self._top = menu.num_levels - 1
        self.allocation: GrantAllocation | None = None

    def configure(
        self, mean_gain_linear: NDArray[np.float64], traffic: TrafficPattern, fading: object
    ) -> None:
        period = np.asarray(traffic.period_slots, dtype=np.int64)
        self._period, self._phase = period, np.asarray(traffic.phase_slots, dtype=np.int64)
        bank = fading_bank(
            self.config.num_subchannels,
            float(fading.doppler_hz),
            int(fading.sinusoids),
            float(self.config.slot_duration_s),
            int(period.max()),
            CG_BANK_SAMPLES,
            CG_BANK_SEED,
        )
        snr = self.menu.power_w[-1] * np.asarray(mean_gain_linear) / self.config.noise_power_w
        self._snr = snr
        self.allocation = allocate_grants(
            self.config,
            snr,
            np.asarray(traffic.packet_bits, dtype=np.float64),
            period,
            self._phase,
            bank,
            self.single_channel,
        )
        self._grant = [dict(o) for o in self.allocation.occasions]

    def reset(self, rngs: Sequence[np.random.Generator]) -> None:
        if self.allocation is None:
            raise ValueError("configure the grants before reset")
        if len(rngs) != len(self._grant):
            raise ValueError("configured for a different number of UEs")
        self._t = 0
        self._offered = np.zeros(len(self._grant), dtype=np.int64)
        self._sent = np.zeros(len(self._grant), dtype=np.int64)

    def act(self, observations: Sequence[UEObservation]) -> NDArray[np.int64]:
        if len(observations) != len(self._grant):
            raise ValueError("need one observation per configured UE")
        actions = np.zeros((len(observations), 2), dtype=np.int64)
        for ue, obs in enumerate(observations):
            period = int(self._period[ue])
            if (self._t + obs.slots_to_deadline - self._phase[ue]) % period:
                raise ValueError("observations are out of step with the configured traffic")
            channel = self._grant[ue].get(self._t % period)
            if channel is None:
                continue
            self._offered[ue] += 1
            if obs.remaining_bits > 0:
                actions[ue] = (channel, self._top)
                self._sent[ue] += 1
        self._t += 1
        return actions

    def episode_diagnostics(self) -> dict:
        allocation = self.allocation
        return {
            "occasions": [len(o) for o in allocation.occasions],
            "grants": [[list(g) for g in o] for o in allocation.occasions],
            "predicted_outage": list(allocation.outage),
            "need": list(allocation.need),
            "mean_snr_db": (10 * np.log10(self._snr)).tolist(),
            "assigned_share": allocation.assigned_share,
            "blocked": allocation.blocked,
            "offered": self._offered.tolist(),
            "sent": self._sent.tolist(),
        }


class ConfiguredGrantOptPolicy(ConfiguredGrantPolicy):
    """Optimized static configured grant CG-opt (configured-grant-optimized_20261010.md).

    The knowledge, configuration form and transmission rule of
    ``ConfiguredGrantPolicy``; ``configure`` replaces the greedy by
    ``cg_opt.optimize_grants`` on the optimization bank A (seed 20,261,010) under
    the frozen budget. ``allocation.outage`` is the prediction on bank A; the
    diagnostics add the prediction on the CG bank B (out of sample) and the
    solver record. Deterministic unless a solve hits the time limit (recorded).
    """

    def __init__(self, config: PhysicalConfig, menu: PowerMenu) -> None:
        super().__init__(config, menu)
        self.name = "configured_grant_opt"

    def configure(
        self, mean_gain_linear: NDArray[np.float64], traffic: TrafficPattern, fading: object
    ) -> None:
        from dsa_period import cg_opt

        period = np.asarray(traffic.period_slots, dtype=np.int64)
        self._period, self._phase = period, np.asarray(traffic.phase_slots, dtype=np.int64)
        packet_bits = np.asarray(traffic.packet_bits, dtype=np.float64)
        banks = [
            fading_bank(
                self.config.num_subchannels,
                float(fading.doppler_hz),
                int(fading.sinusoids),
                float(self.config.slot_duration_s),
                int(period.max()),
                samples,
                seed,
            )
            for samples, seed in (
                (cg_opt.CG_OPT_BANK_SAMPLES, cg_opt.CG_OPT_BANK_SEED),
                (CG_BANK_SAMPLES, CG_BANK_SEED),
            )
        ]
        snr = self.menu.power_w[-1] * np.asarray(mean_gain_linear) / self.config.noise_power_w
        self._snr = snr
        result = cg_opt.optimize_grants(
            self.config, snr, packet_bits, period, self._phase, banks[0]
        )
        self.allocation = result.allocation
        self._record = result.record
        self._outage_b = cg_opt.predicted_outage(
            cg_opt.deployment(self.config, snr, packet_bits, period, self._phase, banks[1]),
            self.allocation.occasions,
        )
        self._grant = [dict(o) for o in self.allocation.occasions]

    def episode_diagnostics(self) -> dict:
        return super().episode_diagnostics() | {
            "predicted_outage_bank_b": self._outage_b,
            "solver": self._record,
        }


def _params(raw: str, allowed: dict[str, type]) -> dict:
    params = {}
    for item in filter(None, raw.split(",")):
        key, _, value = item.partition("=")
        if key not in allowed:
            raise ValueError(f"unknown parameter {key!r}; choose from {tuple(allowed)}")
        params[key] = allowed[key](float(value))
    return params


RULES = {
    "p_persistent": (PPersistentPolicy, {"a": float}),
    "opportunistic": (OpportunisticPolicy, {"theta": float}),
    "reservation": (ReservationPolicy, {"offset": float, "failures": int}),
    "paced_sticky_csi": (
        PacedStickyCsiPolicy,
        {"margin": float, "switch_db": float, "switch_probability": float},
    ),
    "central_edf": (CentralEdfPolicy, {"defer_hopeless": bool}),
    "configured_grant": (ConfiguredGrantPolicy, {"single_channel": bool}),
    "configured_grant_opt": (ConfiguredGrantOptPolicy, {}),
}


CALIBRATION_PATH = (
    Path(__file__).parents[2] / "docs" / "results" / "v2-calibration_20261001" / "calibration.json"
)


@lru_cache(maxsize=4)
def _calibration(path: str) -> dict:
    return json.loads(Path(path).read_text())


def calibrated_spec(family: str, num_ues: int, path: Path = CALIBRATION_PATH) -> str:
    """The frozen per-N parameter of a rule family (protocol v2 §5)."""
    per_n = _calibration(str(path))["families"][family]["per_n"]
    if str(num_ues) not in per_n:
        raise ValueError(f"no calibration of {family} for N={num_ues}")
    return per_n[str(num_ues)]["spec"]


def best_local_spec(num_ues: int, path: Path = CALIBRATION_PATH) -> str:
    """Best calibrated local rule at N (EXP-DSA05-006 §2), fixed by the calibration.

    Highest validation F at eps = 0.01, then at 0.05, then the lower mean
    worst-UE violation rate, then the family order of ``CALIBRATION_TABLE``.
    """
    families = _calibration(str(path))["families"]

    def key(position: int) -> tuple:
        row = families[tuple(CALIBRATION_TABLE)[position]]["per_n"][str(num_ues)]
        share = row["feasible_share"]
        return (-share["0.01"], -share["0.05"], row["worst_ue_violation_rate_mean"], position)

    best = min(range(len(CALIBRATION_TABLE)), key=key)
    return calibrated_spec(tuple(CALIBRATION_TABLE)[best], num_ues, path)


def make_policy_v2(spec: str, config: PhysicalConfig, menu: PowerMenu) -> Policy:
    """A scenario v2.0 policy for ``config.num_ues`` UEs.

    ``random_max``, ``sticky_max``, ``name:key=value,...`` of ``RULES``,
    ``calibrated:<family>`` (frozen per-N parameters), ``best_local`` (the best
    calibrated rule at that N) or ``marl:<checkpoint path>`` (EXP-DSA05-006; a
    summary checkpoint of EXP-DSA05-011 loads as a ``SummaryPolicy``). The
    EXP-DSA05-011 wrappers ``yield[theta=T]@SPEC`` and ``summary[theta=T,hide=H]@marl:PATH``
    and the EXP-DSA05-012 rules ``yield[theta=T,content=survival]@SPEC`` (M+S) and
    ``boost[]@SPEC`` (R17-like) are described in ``urgency.parse_wrapper``.
    """
    wrapped = parse_wrapper(spec)
    if wrapped is not None:
        wrapper, options, inner = wrapped
        if wrapper == "yield":
            base = make_policy_v2(inner, config, menu)
            settings = SummarySettings(
                content=options.get("content", "urgency"), theta_db=options["theta_db"]
            )
            return YieldPolicy(base, config, menu, settings)
        if wrapper == "boost":
            from dsa_period.survival import BoostPolicy

            return BoostPolicy(make_policy_v2(inner, config, menu), config, menu)
        from dsa_period.marl.trainer_v2 import load_capacity_policy

        return load_capacity_policy(Path(inner.partition(":")[2]), config, menu, **options)
    name, _, raw = spec.partition(":")
    if name == "calibrated":
        return make_policy_v2(calibrated_spec(raw, config.num_ues), config, menu)
    if name == "best_local":
        if raw:
            raise ValueError("best_local takes no parameters")
        return make_policy_v2(best_local_spec(config.num_ues), config, menu)
    if name == "marl":
        from dsa_period.marl.trainer_v2 import load_capacity_policy

        return load_capacity_policy(Path(raw), config, menu)
    if name in ("random_max", "sticky_max"):
        if raw:
            raise ValueError(f"{name} takes no parameters")
        assert name in POLICIES
        return HeuristicPolicy(name, config, menu)
    if name not in RULES:
        raise ValueError(f"unknown policy {spec!r}")
    cls, allowed = RULES[name]
    return cls(config, menu, **_params(raw, allowed))


# Per-N calibration of the local rules on the validation split (protocol v2 §5),
# pre-registered on 2026-10-01. Order matters: it breaks the last tie. The
# amendment of 2026-10-01 (after the first validation run selected the ends of
# the opportunistic and paced_sticky_csi ranges) appends values, keeping the
# original order in front; margin = inf always uses max power.
CALIBRATION_TABLE: dict[str, tuple[str, ...]] = {
    "random_max": ("random_max",),
    "sticky_max": ("sticky_max",),
    "p_persistent": tuple(f"p_persistent:a={a:g}" for a in (2, 4, 8, 16, 32, 64)),
    "opportunistic": tuple(f"opportunistic:theta={t:g}" for t in (1, 1.5, 2, 3, 4, 0.5, 0.75)),
    "reservation": tuple(
        f"reservation:offset={o:g},failures={f}" for o in (0.25, 0.5) for f in (1, 2, 4)
    ),
    "paced_sticky_csi": tuple(
        f"paced_sticky_csi:margin={m:g}" for m in (0.5, 1, 2, 4, 8, 16, 32, float("inf"))
    ),
}
