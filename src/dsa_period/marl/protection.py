"""Learned survival protection on a frozen base policy (EXP-DSA05-012 design §11, SP).

The base policy M (an L0 capacity MAPPO checkpoint) keeps every normal
decision. A UE with a packet hands its decision to the protection actor iff

* (a) it is in the survival state (its last settled packet missed), or
* (b) it is not, M's proposed action transmits on subchannel k and k is marked
  in the survival broadcast, ignoring the mark of the subchannel the UE itself
  used in the previous slot while attaching a bit: exactly the UEs the M+S rule
  (``urgency.YieldPolicy`` with the survival content) forces OFF.

The protection actor is the SC-B actor (L0 features, the survival state and the
K-bit survival broadcast). It starts as a copy of M: the weights of the inputs
M lacks are zero (``expand_actor_state``), and the critic likewise
(``expand_critic_state``). Training (``trainer_v2``) updates the actor on gated
decisions only; ``ProtectionPolicy`` executes the same split at evaluation.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from dsa_period.env import UEObservation
from dsa_period.marl.features import DualSettings, LocalMemory, decode_actions, legal_mask, stack
from dsa_period.marl.networks import ChannelSetActor
from dsa_period.marl.policy import sample_from_logits
from dsa_period.physics import PhysicalConfig, PowerMenu
from dsa_period.survival import SurvivalSettings, SurvivalTracker
from dsa_period.urgency import SummaryLink, SummarySettings, with_summary


@dataclass(frozen=True)
class ProtectionSettings:
    """The frozen base checkpoint (an L0 capacity MAPPO run) and its SHA-256."""

    base: str
    base_sha256: str | None = None


def protection_gate(
    active: NDArray[np.bool_],
    state: NDArray[np.bool_],
    proposed: NDArray[np.int64],
    marked: NDArray[np.bool_],
    own: NDArray[np.bool_],
    prev_channel: NDArray[np.int64],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Gates (a) and (b) per UE; ``proposed`` holds M's (subchannel, power ID) rows.

    ``own`` is the UE's own previous uplink bit on a transmission of the last slot.
    """
    by_state = np.asarray(state, dtype=np.bool_) & active
    channel = proposed[:, 0]
    sending = active & ~by_state & (proposed[:, 1] > 0)
    by_yield = sending & marked[channel] & ~(own & (prev_channel == channel))
    return by_state, by_yield


def _column_map(base: torch.Tensor, target_cols: int, index: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(base.shape[0], target_cols, dtype=base.dtype)
    out[:, index] = base
    return out


def expand_actor_state(
    base: dict, base_dims: tuple[int, int], target_dims: tuple[int, int], hidden: int
) -> dict:
    """M's actor weights for the protection actor: new input weights are zero.

    ``*_dims`` are (token dim, context dim). The base features are the leading
    entries of the target's token and context vectors (the survival state and the
    summary are appended), so every base input keeps its weight column.
    """
    (t0, c0), (t1, c1) = base_dims, target_dims
    ctx = torch.arange(c0)
    index = {
        "encoder.0.weight": (t1 + c1, torch.cat((torch.arange(t0), t1 + ctx))),
        "eta.0.weight": (hidden + c1, torch.cat((torch.arange(hidden), hidden + ctx))),
        "psi.0.weight": (2 * hidden + c1, torch.cat((torch.arange(2 * hidden), 2 * hidden + ctx))),
        "omega.0.weight": (hidden + c1, torch.cat((torch.arange(hidden), hidden + ctx))),
    }
    out = {}
    for name, value in base.items():
        out[name] = _column_map(value, *index[name]) if name in index else value.clone()
    return out


def expand_critic_state(
    base: dict, base_dims: tuple[int, int], target_dims: tuple[int, int], num_channels: int
) -> dict:
    """M's set-critic weights for the protection critic (per-UE input: K tokens, context, mu)."""
    (t0, c0), (t1, c1) = base_dims, target_dims
    tokens = torch.cat([k * t1 + torch.arange(t0) for k in range(num_channels)])
    start = num_channels * t1
    index = torch.cat((tokens, start + torch.arange(c0), torch.tensor([start + c1])))
    out = {name: value.clone() for name, value in base.items()}
    out["phi.0.weight"] = _column_map(base["phi.0.weight"], start + c1 + 1, index)
    return out


class ProtectionPolicy:
    """SP at evaluation: the base policy acts, gated UEs re-decide with the protection actor.

    Each slot the base policy samples first (one uniform per UE with a packet from
    the UE's own generator, as ``MarlPolicy``); each gated UE then draws a second
    uniform for the protection actor. The survival link, tracker and features are
    those of the SC-B actor (``SummaryPolicy`` with the survival content).
    ``episode_diagnostics`` counts the gated decisions of the episode.
    """

    def __init__(
        self,
        base,
        actor: ChannelSetActor,
        config: PhysicalConfig,
        menu: PowerMenu,
        duals: DualSettings,
        settings: SummarySettings,
        survival: SurvivalSettings,
        name: str = "sp",
    ) -> None:
        if settings.content != "survival" or not survival.features:
            raise ValueError("the protection actor reads the survival state and broadcast")
        self.base, self.actor, self.config, self.menu = base, actor, config, menu
        self.settings, self.survival, self.name = settings, survival, name
        self._memory = LocalMemory(config, menu, config.num_ues, duals)
        self.tracker = SurvivalTracker(config.num_ues, survival.tolerance)
        self.link = SummaryLink(settings, config, menu, config.num_ues)

    def reset(self, rngs: Sequence[np.random.Generator]) -> None:
        if len(rngs) != self.config.num_ues:
            raise ValueError("need one generator per UE")
        self.base.reset(rngs)
        self._rngs = list(rngs)
        self._memory.reset(np.zeros(self.config.num_ues))
        self._hidden = self.actor.initial_hidden(self.config.num_ues)
        self.tracker.reset()
        self.link.reset()
        self.counts = dict.fromkeys(
            ("decisions", "gated_state", "gated_yield", "changed_state", "changed_yield",
             "off_yield"), 0,
        )  # fmt: skip

    def features(self, obs) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray]:
        """Observe the boundary; the protection actor's tokens and context, and the broadcast."""
        settlement = self._memory.observe(obs)
        self._memory.update_duals(settlement)
        tokens, context = self._memory.features(obs)
        self.tracker.update(settlement, obs)
        context = np.concatenate((context, self.tracker.features()), axis=1)
        marked = self.link.broadcast(obs)
        seen = np.broadcast_to(marked, (self.config.num_ues, marked.size)).copy()
        tokens, context = with_summary(tokens, context, seen, self.link.own_previous)
        return tokens, context, marked

    def act(self, observations: Sequence[UEObservation]) -> NDArray[np.int64]:
        obs = stack(observations)
        proposed = np.array(self.base.act(observations), dtype=np.int64, copy=True)
        tokens, context, marked = self.features(obs)
        with torch.no_grad():
            logits, self._hidden = self.actor.step(
                torch.from_numpy(tokens), torch.from_numpy(context), self._hidden
            )
        active = obs.remaining > 0
        state = self.tracker.in_state & active
        own = self.link.own_previous & (obs.prev_power > 0)
        by_state, by_yield = protection_gate(active, state, proposed, marked, own, obs.prev_channel)
        gated = by_state | by_yield
        self.last_gate = (by_state, by_yield)
        actions = proposed.copy()
        if gated.any():
            legal = legal_mask(obs, self.actor.num_actions)
            actions[gated] = self.respond(np.flatnonzero(gated), logits.numpy(), legal, proposed)
        transmitting = actions[:, 1] > 0
        self.link.commit(self.link.uplink(obs, transmitting, state=state), transmitting)
        changed = (actions != proposed).any(axis=1)
        counts = self.counts
        counts["decisions"] += int(active.sum())
        counts["gated_state"] += int(by_state.sum())
        counts["gated_yield"] += int(by_yield.sum())
        counts["changed_state"] += int((changed & by_state).sum())
        counts["changed_yield"] += int((changed & by_yield).sum())
        counts["off_yield"] += int((by_yield & ~transmitting).sum())
        return actions

    def respond(
        self, ues: NDArray[np.int64], logits: NDArray, legal: NDArray[np.bool_], proposed
    ) -> NDArray[np.int64]:
        """(subchannel, power ID) rows of the gated ``ues``, sampled from the protection actor."""
        uniforms = np.array([self._rngs[ue].random() for ue in ues])
        index = sample_from_logits(logits[ues], legal[ues], uniforms)
        return decode_actions(index, self.menu)

    def episode_diagnostics(self) -> dict:
        return dict(self.counts)
