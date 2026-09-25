"""A trained actor behind the protocol ``Policy`` interface (stochastic execution).

Each UE runs the shared actor on its own features and samples its action with
its own protocol RNG: one uniform draw per UE with a packet, inverted through
the float64 CDF of its masked action distribution. UEs without a packet choose
OFF and draw nothing. The per-UE dual mu_i (variant C) starts at ``mu0`` and
evolves only with that UE's own deadline settlements.
"""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from dsa_period.env import UEObservation
from dsa_period.marl.features import (
    DualSettings,
    LocalMemory,
    decode_actions,
    legal_mask,
    stack,
)
from dsa_period.marl.networks import ChannelSetActor
from dsa_period.physics import PhysicalConfig, PowerMenu


def sample_from_logits(
    logits: NDArray, legal: NDArray[np.bool_], uniforms: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Inverse-CDF sampling in float64; rows are UEs, one uniform per row."""
    masked = np.where(legal, logits.astype(np.float64), -np.inf)
    shifted = np.exp(masked - masked.max(axis=1, keepdims=True))
    cdf = np.cumsum(shifted, axis=1)
    index = np.array(
        [np.searchsorted(row, u * row[-1], side="right") for row, u in zip(cdf, uniforms)],
        dtype=np.int64,
    )
    return np.minimum(index, logits.shape[1] - 1)


class MarlPolicy:
    def __init__(
        self,
        actor: ChannelSetActor,
        config: PhysicalConfig,
        menu: PowerMenu,
        duals: DualSettings,
        mu0: float = 0.0,
        name: str = "marl",
    ) -> None:
        self.actor, self.config, self.menu, self.duals = actor, config, menu, duals
        self.mu0, self.name = mu0, name
        self._memory = LocalMemory(config, menu, config.num_ues, duals)

    def reset(self, rngs: Sequence[np.random.Generator]) -> None:
        if len(rngs) != self.config.num_ues:
            raise ValueError("need one generator per UE")
        self._rngs = list(rngs)
        initial = self.mu0 if self.duals.variant == "local" else 0.0
        self._memory.reset(np.full(self.config.num_ues, initial))
        self._hidden = self.actor.initial_hidden(self.config.num_ues)

    def act(self, observations: Sequence[UEObservation]) -> NDArray[np.int64]:
        obs = stack(observations)
        self._memory.update_duals(self._memory.observe(obs))
        tokens, context = self._memory.features(obs)
        with torch.no_grad():
            logits, self._hidden = self.actor.step(
                torch.from_numpy(tokens), torch.from_numpy(context), self._hidden
            )
        legal = legal_mask(obs, self.actor.num_actions)
        active = obs.remaining > 0
        index = np.zeros(self.config.num_ues, dtype=np.int64)
        if active.any():
            uniforms = np.array([self._rngs[ue].random() for ue in np.flatnonzero(active)])
            index[active] = sample_from_logits(logits.numpy()[active], legal[active], uniforms)
        return decode_actions(index, self.menu)

    @property
    def mu(self) -> NDArray[np.float64]:
        return self._memory.mu.copy()
