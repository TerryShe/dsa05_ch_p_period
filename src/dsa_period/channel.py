"""Channel models that supply per-slot (N, K) UE-to-AP power gains.

The kernel only consumes ``gains(slot)``. The first scenario version uses the
static, frequency-flat path-loss channel below. A fading model (block Rayleigh,
Gauss-Markov frequency-selective, ...) must implement the same protocol and own
its random stream, seeded independently of topology and traffic.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from dsa_period.topology import Topology


class ChannelModel(Protocol):
    num_ues: int
    num_subchannels: int

    def reset(self) -> None:
        """Restart the channel process for a new episode on the same instance."""

    def gains(self, slot: int) -> NDArray[np.float64]:
        """Return the (N, K) linear gains in force during ``slot``."""


class StaticChannel:
    """Time-invariant path loss; every subchannel sees the UE's own distance gain."""

    def __init__(self, topology: Topology, num_subchannels: int) -> None:
        if not isinstance(topology, Topology):
            raise ValueError("topology must be a Topology")
        if isinstance(num_subchannels, bool) or int(num_subchannels) != num_subchannels:
            raise ValueError("num_subchannels must be an integer")
        if num_subchannels < 1:
            raise ValueError("num_subchannels must be positive")
        self.topology = topology
        self.num_ues = topology.num_ues
        self.num_subchannels = int(num_subchannels)
        gains = np.repeat(topology.ue_ap_gain_linear[:, None], self.num_subchannels, axis=1)
        self._gains = np.frombuffer(gains.tobytes(), dtype=np.float64).reshape(gains.shape)

    def reset(self) -> None:
        return None

    def gains(self, slot: int) -> NDArray[np.float64]:
        if slot < 0:
            raise ValueError("slot must be nonnegative")
        return self._gains


class ShadowedJakesChannel:
    """Static mean gain (path loss and shadowing) times per-subchannel Jakes fading.

    Scenario v2.0 (docs/research/scenario-v2.md §2.2). Each (UE i, subchannel k)
    is an independent sum-of-sinusoids process, held constant within a slot and
    evaluated at the slot start,

        h(t) = M^-1/2 sum_n exp(j[2 pi fD (t0 + t tau) cos a_n + phi_n]),
        a_n = (2 pi n + theta) / M,

    with theta, phi_n ~ U[0, 2 pi) and a random start t0, so the ensemble
    autocorrelation is J0(2 pi fD dt). UE i draws its parameters from its own
    seed only, so a UE's fading does not depend on how many other UEs exist.
    One object is one fading realization: ``reset`` replays it, and a new
    realization needs a new channel with new seeds.
    """

    BLOCK_SLOTS = 256

    def __init__(
        self,
        mean_gain_linear: NDArray[np.float64],
        num_subchannels: int,
        *,
        doppler_hz: float,
        slot_duration_s: float,
        ue_seeds: list[np.random.SeedSequence],
        sinusoids: int = 32,
        start_time_max_s: float = 1000.0,
    ) -> None:
        mean = np.asarray(mean_gain_linear, dtype=np.float64)
        if mean.ndim != 1 or mean.size < 1 or np.any(~np.isfinite(mean)) or np.any(mean <= 0):
            raise ValueError("mean_gain_linear must be a positive finite vector")
        if len(ue_seeds) != mean.size:
            raise ValueError("need one seed per UE")
        if num_subchannels < 1 or sinusoids < 1:
            raise ValueError("num_subchannels and sinusoids must be positive")
        if doppler_hz < 0 or slot_duration_s <= 0 or start_time_max_s < 0:
            raise ValueError("doppler_hz, start_time_max_s >= 0 and slot_duration_s > 0 required")
        self.num_ues, self.num_subchannels = mean.size, int(num_subchannels)
        self.mean_gain_linear = mean
        self.slot_duration_s = float(slot_duration_s)
        shape = (self.num_ues, self.num_subchannels)
        start = np.empty(shape)
        frequency = np.empty((*shape, sinusoids))
        phase = np.empty((*shape, sinusoids))
        for ue, seed in enumerate(ue_seeds):
            rng = np.random.Generator(np.random.PCG64(seed))
            theta = rng.uniform(0.0, 2 * np.pi, self.num_subchannels)
            phase[ue] = rng.uniform(0.0, 2 * np.pi, (self.num_subchannels, sinusoids))
            start[ue] = rng.uniform(0.0, start_time_max_s, self.num_subchannels)
            angle = (2 * np.pi * np.arange(sinusoids)[None, :] + theta[:, None]) / sinusoids
            frequency[ue] = 2 * np.pi * doppler_hz * np.cos(angle)
        # Fold the start time into the phase: arg = w (t0 + t tau) + phi.
        self._frequency = frequency
        self._phase = phase + frequency * start[..., None]
        self._scale = 1.0 / np.sqrt(sinusoids)
        self._blocks: dict[int, NDArray[np.float64]] = {}

    def reset(self) -> None:
        return None

    def small_scale(self, slots: NDArray[np.int64]) -> NDArray[np.complex128]:
        """(len(slots), N, K) complex fading h at the start of the given slots."""
        time = np.asarray(slots, dtype=np.float64) * self.slot_duration_s
        arg = self._frequency[None] * time[:, None, None, None] + self._phase[None]
        return np.exp(1j * arg).sum(axis=3) * self._scale

    def _block(self, index: int) -> NDArray[np.float64]:
        block = self._blocks.get(index)
        if block is None:
            slots = np.arange(index * self.BLOCK_SLOTS, (index + 1) * self.BLOCK_SLOTS)
            h = self.small_scale(slots)
            power = h.real**2 + h.imag**2
            # |h|^2 = 0 has probability zero; keep gains strictly positive anyway.
            gains = self.mean_gain_linear[None, :, None] * np.maximum(power, np.finfo(float).tiny)
            block = np.frombuffer(gains.tobytes(), dtype=np.float64).reshape(gains.shape)
            self._blocks[index] = block
        return block

    def gains(self, slot: int) -> NDArray[np.float64]:
        if slot < 0:
            raise ValueError("slot must be nonnegative")
        return self._block(slot // self.BLOCK_SLOTS)[slot % self.BLOCK_SLOTS]

    def gain_block(self, start: int, stop: int) -> NDArray[np.float64]:
        """(stop - start, N, K) gains of slots [start, stop)."""
        if not 0 <= start <= stop:
            raise ValueError("require 0 <= start <= stop")
        if start == stop:
            return np.empty((0, self.num_ues, self.num_subchannels))
        first, last = start // self.BLOCK_SLOTS, (stop - 1) // self.BLOCK_SLOTS
        stacked = np.concatenate([self._block(b) for b in range(first, last + 1)])
        offset = first * self.BLOCK_SLOTS
        return stacked[start - offset : stop - offset]

    def power_gain(self, start: int, stop: int) -> NDArray[np.float64]:
        """(stop - start, N, K) small-scale power |h|^2 alone (for statistics tests)."""
        return self.gain_block(start, stop) / self.mean_gain_linear[None, :, None]
