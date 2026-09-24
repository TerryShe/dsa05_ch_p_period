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
