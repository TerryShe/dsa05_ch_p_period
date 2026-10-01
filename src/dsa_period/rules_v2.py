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
current gains at once, so it is not a local rule.
"""

import json
import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dsa_period.baselines import estimate_link
from dsa_period.env import Outcome, UEObservation
from dsa_period.physics import PhysicalConfig, PowerMenu, shannon_bits
from dsa_period.policies import POLICIES, HeuristicPolicy, Policy


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
    calibrated rule at that N) or ``marl:<checkpoint path>`` (EXP-DSA05-006).
    """
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
