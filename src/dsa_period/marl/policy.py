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
from dsa_period.marl.networks import ChannelSetActor, MessageActor, split_logits
from dsa_period.physics import PhysicalConfig, PowerMenu
from dsa_period.urgency import (
    HIDING,
    SummaryLink,
    SummarySettings,
    urgent,
    visible,
    with_summary,
)


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
        lags: tuple[int, ...] = (),
    ) -> None:
        self.actor, self.config, self.menu, self.duals = actor, config, menu, duals
        self.mu0, self.name = mu0, name
        self._memory = LocalMemory(config, menu, config.num_ues, duals, lags)

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


class SummaryPolicy(MarlPolicy):
    """A trained actor that also reads the urgency state summary (EXP-DSA05-011).

    Each slot the AP side (``SummaryLink``) turns the previous slot's uplink bits
    into the K-bit broadcast; every UE's actor sees it (after the evaluation-time
    ``hide`` mask) and its own previous bit. Actions are sampled as in
    ``MarlPolicy``. With a message head, every transmitting UE then draws a second
    uniform from its own generator for its message bit.
    """

    def __init__(
        self,
        actor: ChannelSetActor,
        config: PhysicalConfig,
        menu: PowerMenu,
        duals: DualSettings,
        settings: SummarySettings,
        mu0: float = 0.0,
        name: str = "marl",
        hide: str = "none",
    ) -> None:
        super().__init__(actor, config, menu, duals, mu0, name)
        if hide not in HIDING:
            raise ValueError(f"hide must be one of {HIDING}")
        if settings.message != isinstance(actor, MessageActor):
            raise ValueError("the message content needs a MessageActor and only it")
        self.settings, self.hide = settings, hide
        self.link = SummaryLink(settings, config, menu, config.num_ues)

    def reset(self, rngs: Sequence[np.random.Generator]) -> None:
        super().reset(rngs)
        self.link.reset()

    def act(self, observations: Sequence[UEObservation]) -> NDArray[np.int64]:
        obs = stack(observations)
        self._memory.update_duals(self._memory.observe(obs))
        tokens, context = self._memory.features(obs)
        urgent_now = urgent(self.config, self.menu, obs, self.settings.urgent_slack)
        seen = visible(self.link.broadcast(obs), self.hide, urgent_now)
        tokens, context = with_summary(tokens, context, seen, self.link.own_previous)
        with torch.no_grad():
            logits, self._hidden = self.actor.step(
                torch.from_numpy(tokens), torch.from_numpy(context), self._hidden
            )
        action_logits, message_logits = split_logits(logits, self.actor.num_actions)
        legal = legal_mask(obs, self.actor.num_actions)
        active = obs.remaining > 0
        index = np.zeros(self.config.num_ues, dtype=np.int64)
        if active.any():
            uniforms = np.array([self._rngs[ue].random() for ue in np.flatnonzero(active)])
            index[active] = sample_from_logits(
                action_logits.numpy()[active], legal[active], uniforms
            )
        transmitting = index > 0
        messages = None
        if message_logits is not None:
            messages = np.zeros(self.config.num_ues, dtype=np.int64)
            if transmitting.any():
                ues = np.flatnonzero(transmitting)
                uniforms = np.array([self._rngs[ue].random() for ue in ues])
                both = np.ones((ues.size, MessageActor.MESSAGES), dtype=np.bool_)
                messages[ues] = sample_from_logits(message_logits.numpy()[ues], both, uniforms)
        self.link.commit(self.link.uplink(obs, transmitting, messages), transmitting)
        return decode_actions(index, self.menu)
