"""Local actor features, deadline settlement and per-UE dual variables.

Everything is computed per UE from that UE's own ``UEObservation`` stream: its
gain estimate (own feedback and public RSSI), the per-subchannel interference it
infers, the deadline settlements it witnesses and its dual variable mu_i. No
topology, UE identifier, clock or other UE's private state enters the actor.

Action index 0 is OFF; index 1 + k * L + (l - 1) is (subchannel k, power ID l)
with L active power levels, matching the actor's channel-major logits.

Optional periodic memory (EXP-DSA05-002): for each lag l in ``lags`` the token of
subchannel k also carries the interference the UE inferred on k in the slot l
slots before the one being decided, and whether the UE itself transmitted on k
then. The UE keeps these in a local ring buffer of its own past observations;
the context carries one flag per lag telling whether that much history exists.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dsa_period.env import Outcome, UEObservation
from dsa_period.physics import LN2, PhysicalConfig, PowerMenu

TOKEN_DIM_EXTRA = 3  # RSSI, interference, own-transmission flag; plus 2 * L level entries
CONTEXT_DIM = 13
DB_SCALE = 30.0  # dB above the noise floor per unit feature
GAIN_DB_OFFSET, GAIN_DB_SCALE = 120.0, 40.0
COVERAGE_CLIP = 4.0  # log2(bits / required) is clipped to [-4, 4] and divided by 4
MAX_PERIOD_SLOTS = 40.0
VARIANTS = ("fixed", "global", "local")


def num_actions(config: PhysicalConfig, menu: PowerMenu) -> int:
    return 1 + config.num_subchannels * (menu.num_levels - 1)


def token_dim(menu: PowerMenu, lags: tuple[int, ...] = ()) -> int:
    return TOKEN_DIM_EXTRA + 2 * (menu.num_levels - 1) + 2 * len(lags)


def context_dim(lags: tuple[int, ...] = ()) -> int:
    return CONTEXT_DIM + len(lags)


def validate_lags(lags) -> tuple[int, ...]:
    values = tuple(int(lag) for lag in lags)
    if any(lag < 1 for lag in values) or len(set(values)) != len(values):
        raise ValueError("periodic lags must be distinct positive integers")
    return values


def decode_actions(indices: NDArray, menu: PowerMenu) -> NDArray[np.int64]:
    """Map action indices to (subchannel, power ID) rows; 0 is OFF = (0, 0)."""
    index = np.asarray(indices, dtype=np.int64)
    levels = menu.num_levels - 1
    active = index > 0
    actions = np.zeros((index.size, 2), dtype=np.int64)
    actions[active, 0] = (index[active] - 1) // levels
    actions[active, 1] = (index[active] - 1) % levels + 1
    return actions


def encode_actions(actions: NDArray, menu: PowerMenu) -> NDArray[np.int64]:
    rows = np.asarray(actions, dtype=np.int64)
    levels = menu.num_levels - 1
    return np.where(rows[:, 1] > 0, 1 + rows[:, 0] * levels + rows[:, 1] - 1, 0)


@dataclass(frozen=True)
class ObsArrays:
    """One environment's observations as per-UE arrays (row i is UE i's own view)."""

    remaining: NDArray[np.float64]
    packet: NDArray[np.float64]
    period: NDArray[np.float64]
    to_deadline: NDArray[np.float64]
    prev_channel: NDArray[np.int64]
    prev_power: NDArray[np.int64]
    prev_outcome: NDArray[np.int64]
    prev_service: NDArray[np.float64]
    rssi: NDArray[np.float64]  # (K,) public, identical for every UE


def stack(observations: Sequence[UEObservation]) -> ObsArrays:
    return ObsArrays(
        remaining=np.array([o.remaining_bits for o in observations]),
        packet=np.array([o.packet_bits for o in observations]),
        period=np.array([o.period_slots for o in observations], dtype=np.float64),
        to_deadline=np.array([o.slots_to_deadline for o in observations], dtype=np.float64),
        prev_channel=np.array([o.previous_action[0] for o in observations], dtype=np.int64),
        prev_power=np.array([o.previous_action[1] for o in observations], dtype=np.int64),
        prev_outcome=np.array([int(o.previous_outcome) for o in observations], dtype=np.int64),
        prev_service=np.array([o.previous_service_bits for o in observations]),
        rssi=np.array(observations[0].previous_rssi_w, dtype=np.float64),
    )


@dataclass(frozen=True)
class DualSettings:
    """How the penalty weight mu_i of a violation is set (EXP-DSA05-001 variants A/B/C).

    fixed: mu_i = weight. global: one mu for all UEs, set by the trainer between
    iterations and frozen at execution. local: each UE updates its own mu_i at
    each of its deadline settlements, mu_i <- clip(mu_i + eta (c_i - epsilon), 0,
    mu_max), and the actor observes mu_i / mu_max.
    """

    variant: str
    epsilon: float
    weight: float = 0.0
    eta: float = 0.0
    mu_max: float = 1.0

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if not 0 < self.epsilon < 1 or self.mu_max <= 0 or self.eta < 0 or self.weight < 0:
            raise ValueError("need 0 < epsilon < 1, mu_max > 0, eta >= 0, weight >= 0")

    @property
    def mu_scale(self) -> float:
        """Normalizer of mu in critic inputs."""
        return self.weight if self.variant == "fixed" and self.weight > 0 else self.mu_max


@dataclass(frozen=True)
class Settlement:
    settled: NDArray[np.bool_]
    violated: NDArray[np.bool_]


class LocalMemory:
    """Per-UE memory of one environment: gain estimates, deadline tracker, duals.

    Call ``observe`` on every new observation (including the first after reset),
    then read ``mu`` (the multiplier that was in force for the slot just settled),
    then ``update_duals`` and ``features``.
    """

    def __init__(
        self,
        config: PhysicalConfig,
        menu: PowerMenu,
        num_ues: int,
        duals: DualSettings,
        lags: tuple[int, ...] = (),
    ) -> None:
        self.config, self.menu, self.num_ues, self.duals = config, menu, num_ues, duals
        self.lags = validate_lags(lags)
        self._power = np.asarray(menu.power_w)
        self._noise_db = 10.0 * np.log10(config.noise_power_w)
        self.reset(np.zeros(num_ues))

    def reset(self, mu: NDArray[np.float64]) -> None:
        mu = np.asarray(mu, dtype=np.float64)
        if mu.shape != (self.num_ues,) or np.any(mu < 0) or np.any(mu > self.duals.mu_max):
            raise ValueError("initial mu must be a per-UE vector in [0, mu_max]")
        self.mu = mu.copy()
        self.gain = np.full(self.num_ues, np.nan)
        self.interference = np.zeros((self.num_ues, self.config.num_subchannels))
        self._released = np.zeros(self.num_ues, dtype=np.bool_)
        self._prev_remaining = np.zeros(self.num_ues)
        # Ring buffer of per-slot local history: slot s lives at row s % depth.
        depth = max(self.lags, default=1)
        shape = (depth, self.num_ues, self.config.num_subchannels)
        self._history_interference = np.zeros(shape)
        self._history_tx = np.zeros(shape)
        self._observations = 0  # observe() calls since reset; the next slot's index

    def observe(self, obs: ObsArrays) -> Settlement:
        """Update gain/interference estimates and detect this boundary's settlements.

        A release happened iff slots_to_deadline == period. Every release after
        a UE's first one settles its previous packet: violated iff it still had
        bits before the last slot and that slot did not decode it.
        """
        transmitted = (obs.prev_power > 0) & (
            (obs.prev_outcome == Outcome.TX) | (obs.prev_outcome == Outcome.DECODED)
        )
        interference = np.broadcast_to(obs.rssi, self.interference.shape).copy()
        if transmitted.any():
            ues = np.flatnonzero(transmitted)
            channels = obs.prev_channel[ues]
            sinr = np.expm1(obs.prev_service[ues] / self.config.channel_uses_per_slot * LN2)
            own = obs.rssi[channels] * sinr / (1.0 + sinr)
            interference[ues, channels] = obs.rssi[channels] / (1.0 + sinr)
            self.gain[ues] = own / self._power[obs.prev_power[ues]]
        self.interference = interference
        if self.lags and self._observations > 0:
            # This observation describes slot (calls - 1).
            row = (self._observations - 1) % self._history_tx.shape[0]
            self._history_interference[row] = interference
            self._history_tx[row] = 0.0
            ues = np.flatnonzero(obs.prev_power > 0)
            self._history_tx[row, ues, obs.prev_channel[ues]] = 1.0
        self._observations += 1
        boundary = obs.to_deadline == obs.period
        settled = boundary & self._released
        violated = settled & (self._prev_remaining > 0) & (obs.prev_outcome != Outcome.DECODED)
        self._released |= boundary
        self._prev_remaining = obs.remaining.copy()
        return Settlement(settled, violated)

    def update_duals(self, settlement: Settlement) -> None:
        if self.duals.variant != "local" or not settlement.settled.any():
            return
        step = self.duals.eta * (settlement.violated.astype(float) - self.duals.epsilon)
        self.mu = np.where(
            settlement.settled, np.clip(self.mu + step, 0.0, self.duals.mu_max), self.mu
        )

    def features(self, obs: ObsArrays) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Per-UE subchannel tokens (N, K, token_dim) and context (N, CONTEXT_DIM)."""
        n, k = self.num_ues, self.config.num_subchannels
        levels = self.menu.num_levels - 1
        has_packet = obs.remaining > 0
        known = ~np.isnan(self.gain)
        tokens = np.zeros((n, k, token_dim(self.menu)))
        tokens[:, :, 0] = (10.0 * np.log10(obs.rssi) - self._noise_db) / DB_SCALE
        tokens[:, :, 1] = (10.0 * np.log10(self.interference) - self._noise_db) / DB_SCALE
        sent = obs.prev_power > 0
        ues = np.flatnonzero(sent)
        tokens[ues, obs.prev_channel[ues], 2] = 1.0
        tokens[ues, obs.prev_channel[ues], 2 + obs.prev_power[ues]] = 1.0
        ready = known & has_packet
        if ready.any():
            required = obs.remaining[ready] / obs.to_deadline[ready]
            snr = (
                self._power[1:][None, None, :]
                * self.gain[ready][:, None, None]
                / self.interference[ready][:, :, None]
            )
            bits = self.config.channel_uses_per_slot * np.log1p(snr) / LN2
            ratio = np.log2(np.maximum(bits, 1e-300) / required[:, None, None])
            tokens[ready, :, 3 + levels :] = np.clip(ratio, -COVERAGE_CLIP, COVERAGE_CLIP) / 4.0
        context = np.zeros((n, CONTEXT_DIM))
        context[:, 0] = has_packet
        context[:, 1] = obs.remaining / obs.packet
        context[:, 2] = obs.to_deadline / obs.period
        context[:, 3] = obs.period / MAX_PERIOD_SLOTS
        context[np.arange(n), 4 + obs.prev_outcome] = 1.0
        context[:, 8] = np.minimum(obs.prev_service / obs.packet, 4.0)
        context[:, 9] = known
        gain_db = 10.0 * np.log10(np.where(known, self.gain, 1.0))
        context[:, 10] = np.where(known, (gain_db + GAIN_DB_OFFSET) / GAIN_DB_SCALE, 0.0)
        per_slot = np.where(has_packet, obs.remaining / obs.to_deadline, 0.0)
        context[:, 11] = np.log2(1.0 + per_slot / self.config.channel_uses_per_slot) / 4.0
        if self.duals.variant == "local":
            context[:, 12] = self.mu / self.duals.mu_max
        if self.lags:
            tokens, context = self._periodic_features(tokens, context)
        return tokens.astype(np.float32), context.astype(np.float32)

    def _periodic_features(self, tokens, context):
        """Append lagged interference / own-transmission per subchannel and validity flags."""
        levels = self.menu.num_levels - 1
        start = TOKEN_DIM_EXTRA + 2 * levels
        wide = np.zeros((*tokens.shape[:2], start + 2 * len(self.lags)))
        wide[:, :, :start] = tokens
        flags = np.zeros((self.num_ues, len(self.lags)))
        decided = self._observations - 1  # index of the slot about to be decided
        depth = self._history_tx.shape[0]
        for j, lag in enumerate(self.lags):
            slot = decided - lag
            if slot < 0:
                continue
            row = slot % depth
            interference = self._history_interference[row]
            wide[:, :, start + 2 * j] = (10.0 * np.log10(interference) - self._noise_db) / DB_SCALE
            wide[:, :, start + 2 * j + 1] = self._history_tx[row]
            flags[:, j] = 1.0
        return wide, np.concatenate((context, flags), axis=1)


def legal_mask(obs: ObsArrays, actions: int) -> NDArray[np.bool_]:
    """UEs without a packet may only choose OFF; others may choose anything."""
    mask = np.ones((obs.remaining.size, actions), dtype=np.bool_)
    mask[obs.remaining <= 0, 1:] = False
    return mask


def critic_state(
    tokens: NDArray,
    context: NDArray,
    true_gain: NDArray[np.float64],
    mu: NDArray[np.float64],
    mu_scale: float,
) -> NDArray[np.float32]:
    """Centralized training state: every UE's local features, true gain and mu."""
    gain = (10.0 * np.log10(true_gain) + GAIN_DB_OFFSET) / GAIN_DB_SCALE
    per_ue = np.concatenate(
        (tokens.reshape(tokens.shape[0], -1), context, gain[:, None], (mu / mu_scale)[:, None]),
        axis=1,
    )
    return per_ue.reshape(-1).astype(np.float32)


def critic_dim(config: PhysicalConfig, menu: PowerMenu, lags: tuple[int, ...] = ()) -> int:
    return config.num_ues * (config.num_subchannels * token_dim(menu, lags) + context_dim(lags) + 2)
