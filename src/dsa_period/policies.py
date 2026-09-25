"""Local policies behind one stateful interface for protocol evaluation.

A policy is reset once per episode with one private RNG per UE and then asked
for joint actions slot by slot. Each UE's decision may use only its own
observation, its own RNG and its own memory; no policy receives the topology,
the instance index or a clock. MARL actors implement the same interface.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from dsa_period.baselines import POLICIES, act, estimate_link
from dsa_period.env import UEObservation
from dsa_period.physics import PhysicalConfig, PowerMenu, shannon_bits


class Policy(Protocol):
    name: str

    def reset(self, rngs: Sequence[np.random.Generator]) -> None:
        """Start an episode with one private generator per UE."""

    def act(self, observations: Sequence[UEObservation]) -> NDArray[np.int64]:
        """Return (N, 2) actions of (subchannel, power ID) for the coming slot."""


class HeuristicPolicy:
    """The memoryless local heuristics of ``baselines``."""

    def __init__(self, name: str, config: PhysicalConfig, menu: PowerMenu) -> None:
        if name not in POLICIES:
            raise ValueError(f"unknown heuristic {name!r}")
        self.name, self.config, self.menu = name, config, menu
        self._rngs: list[np.random.Generator] = []

    def reset(self, rngs: Sequence[np.random.Generator]) -> None:
        self._rngs = list(rngs)

    def act(self, observations: Sequence[UEObservation]) -> NDArray[np.int64]:
        return act(self.name, observations, self.config, self.menu, self._rngs)


class PacedStickyPolicy:
    """Strong local baseline: remembered own gain, sticky subchannel, paced power.

    Per UE, from its own observation only:

    * own path gain: estimated from its own transmissions (the channel is static)
      and kept across packets and idle periods; until known, probe at max power;
    * interference per subchannel: last slot's public RSSI with the UE's own
      received power removed on its own subchannel (``baselines.estimate_link``);
    * subchannel: keep it while its interference is within ``switch_db`` of the
      least-interfered subchannel; otherwise, with probability ``switch_probability``,
      move uniformly to one of the subchannels within 3 dB of the least;
    * power: the lowest level whose mutual information on that subchannel covers
      (1 + margin) * remaining_bits / slots_to_deadline; max power if none does.
    """

    def __init__(
        self,
        config: PhysicalConfig,
        menu: PowerMenu,
        *,
        margin: float = 0.0,
        switch_db: float = 3.0,
        switch_probability: float = 0.5,
    ) -> None:
        if margin < 0 or switch_db < 0 or not 0 < switch_probability <= 1:
            raise ValueError("margin, switch_db must be >= 0 and switch_probability in (0, 1]")
        self.config, self.menu = config, menu
        self.margin, self.switch_db, self.switch_probability = margin, switch_db, switch_probability
        self.name = f"paced_sticky(margin={margin:g})"
        self._power = np.asarray(menu.power_w[1:])
        self._rngs: list[np.random.Generator] = []

    def reset(self, rngs: Sequence[np.random.Generator]) -> None:
        self._rngs = list(rngs)
        self._gain: list[float | None] = [None] * len(self._rngs)
        self._channel: list[int | None] = [None] * len(self._rngs)

    def _act_one(self, ue: int, obs: UEObservation) -> tuple[int, int]:
        rng = self._rngs[ue]
        gain, interference = estimate_link(obs, self.config, self.menu)
        if gain is not None:
            self._gain[ue] = gain
        top = self.menu.num_levels - 1
        channel = self._channel[ue]
        least = float(interference.min())
        crowded = channel is None or interference[channel] > least * 10 ** (self.switch_db / 10)
        if crowded and (channel is None or rng.random() < self.switch_probability):
            candidates = np.flatnonzero(interference <= least * 10**0.3)
            channel = int(candidates[rng.integers(candidates.size)])
        self._channel[ue] = channel
        if self._gain[ue] is None:
            return channel, top
        required = (1 + self.margin) * obs.remaining_bits / obs.slots_to_deadline
        bits = shannon_bits(self.config, self._power * self._gain[ue] / interference[channel])
        enough = np.flatnonzero(bits >= required)
        return channel, int(enough[0]) + 1 if enough.size else top

    def act(self, observations: Sequence[UEObservation]) -> NDArray[np.int64]:
        if len(observations) != len(self._rngs):
            raise ValueError("need one observation per UE; call reset first")
        actions = np.zeros((len(observations), 2), dtype=np.int64)
        for ue, obs in enumerate(observations):
            if obs.remaining_bits > 0:
                actions[ue] = self._act_one(ue, obs)
            else:
                # Idle UEs keep learning from public feedback but draw no randomness.
                gain, _ = estimate_link(obs, self.config, self.menu)
                if gain is not None:
                    self._gain[ue] = gain
        return actions


def make_policy(spec: str, config: PhysicalConfig, menu: PowerMenu) -> Policy:
    """Build a policy from ``name`` or ``paced_sticky:margin=0.2,switch_db=3``."""
    name, _, raw = spec.partition(":")
    if name in POLICIES:
        if raw:
            raise ValueError(f"{name} takes no parameters")
        return HeuristicPolicy(name, config, menu)
    if name == "paced_sticky":
        params = {}
        for item in filter(None, raw.split(",")):
            key, _, value = item.partition("=")
            if key not in {"margin", "switch_db", "switch_probability"}:
                raise ValueError(f"unknown paced_sticky parameter {key!r}")
            params[key] = float(value)
        return PacedStickyPolicy(config, menu, **params)
    raise ValueError(f"unknown policy spec {spec!r}")
