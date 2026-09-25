"""Recurrent MAPPO update with a shared actor and a centralized team critic.

Rollouts keep whole ordered environment sequences. The team advantage (discounted
GAE, bootstrapped at technical truncation, since the task is continuing) is
shared by every UE; UEs without a packet have a single legal action and are
excluded from the policy loss and entropy.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from dsa_period.marl.networks import CentralCritic, ChannelSetActor, masked_log_probs


def discounted_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    truncated: Tensor,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """GAE over [T, E]. ``next_values[t]`` is V(s_{t+1}) evaluated before any reset.

    There is no terminal state: truncation keeps the bootstrap but cuts the trace.
    """
    advantages = torch.empty_like(rewards)
    trace = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        trace = delta + gamma * lam * (~truncated[t]).to(rewards.dtype) * trace
        advantages[t] = trace
    return advantages, advantages + values


@dataclass
class Rollout:
    """Actor fields are [T, E, N, ...]; transition fields are [T, E]."""

    tokens: np.ndarray
    context: np.ndarray
    legal: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    active: np.ndarray
    state: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    rewards: np.ndarray
    truncated: np.ndarray
    starts: np.ndarray
    initial_hidden: Tensor  # [E * N, K, h] before obs[0]


@dataclass(frozen=True)
class PPOSettings:
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_envs: int = 4


class RecurrentMAPPO:
    def __init__(
        self, actor: ChannelSetActor, critic: CentralCritic, settings: PPOSettings, seed: int
    ) -> None:
        self.actor, self.critic, self.settings = actor, critic, settings
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=settings.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=settings.critic_lr, eps=1e-5
        )
        self.shuffle = torch.Generator().manual_seed(seed)

    def update(self, rollout: Rollout, entropy_coef: float) -> dict[str, float]:
        s = self.settings
        steps, envs, ues = rollout.actions.shape
        rewards = torch.from_numpy(rollout.rewards)
        values = torch.from_numpy(rollout.values)
        advantages, returns = discounted_gae(
            rewards,
            values,
            torch.from_numpy(rollout.next_values),
            torch.from_numpy(rollout.truncated),
            s.gamma,
            s.gae_lambda,
        )
        active = torch.from_numpy(rollout.active)
        per_ue = advantages.unsqueeze(-1).expand(steps, envs, ues)
        chosen = per_ue[active]
        if chosen.numel() > 1:
            per_ue = (per_ue - chosen.mean()) / (chosen.std(unbiased=False) + 1e-8)
        tokens = torch.from_numpy(rollout.tokens)
        context = torch.from_numpy(rollout.context)
        legal = torch.from_numpy(rollout.legal)
        actions = torch.from_numpy(rollout.actions)
        old_log_probs = torch.from_numpy(rollout.log_probs)
        state = torch.from_numpy(rollout.state)
        starts = torch.from_numpy(rollout.starts)
        hidden = rollout.initial_hidden.reshape(envs, ues, *rollout.initial_hidden.shape[1:])
        totals: dict[str, float] = {}
        count = 0
        replay_error = None
        for _ in range(s.ppo_epochs):
            order = torch.randperm(envs, generator=self.shuffle)
            for start in range(0, envs, s.minibatch_envs):
                idx = order[start : start + s.minibatch_envs]
                m = idx.numel()

                def flat(x: Tensor) -> Tensor:
                    return x[:, idx].reshape(steps, m * ues, *x.shape[3:])

                logits, _ = self.actor.sequence(
                    flat(tokens),
                    flat(context),
                    hidden[idx].reshape(m * ues, *hidden.shape[2:]),
                    starts[:, idx].unsqueeze(-1).expand(steps, m, ues).reshape(steps, m * ues),
                )
                mask = flat(legal)
                log_probs_all = masked_log_probs(logits, mask)
                log_probs = log_probs_all.gather(-1, flat(actions).unsqueeze(-1)).squeeze(-1)
                probs = log_probs_all.exp()
                entropy = -(probs * log_probs_all.masked_fill(~mask, 0.0)).sum(-1)
                act = flat(active)
                old = flat(old_log_probs)
                if replay_error is None:
                    replay_error = (
                        float((log_probs - old).detach()[act].abs().max()) if act.any() else 0.0
                    )
                metrics = {"value_loss": 0.0}
                if act.any():
                    log_ratio = (log_probs - old)[act]
                    ratio = log_ratio.exp()
                    adv = flat(per_ue)[act]
                    surrogate = torch.minimum(
                        ratio * adv, ratio.clamp(1 - s.clip, 1 + s.clip) * adv
                    )
                    actor_loss = -surrogate.mean()
                    mean_entropy = entropy[act].mean()
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    (actor_loss - entropy_coef * mean_entropy).backward()
                    actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), s.max_grad_norm)
                    self.actor_optimizer.step()
                    with torch.no_grad():
                        metrics.update(
                            actor_loss=float(actor_loss),
                            entropy=float(mean_entropy),
                            approx_kl=float(((ratio - 1) - log_ratio).mean()),
                            clip_fraction=float(((ratio - 1).abs() > s.clip).float().mean()),
                            actor_grad_norm=float(actor_norm),
                        )
                value = self.critic(state[:, idx])
                value_loss = 0.5 * (value - returns[:, idx]).square().mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), s.max_grad_norm)
                self.critic_optimizer.step()
                metrics.update(
                    value_loss=float(value_loss.detach()), critic_grad_norm=float(critic_norm)
                )
                for key, val in metrics.items():
                    totals[key] = totals.get(key, 0.0) + val
                count += 1
        result = {key: val / count for key, val in totals.items()}
        result.update(
            replay_log_prob_error=replay_error or 0.0,
            advantage_mean=float(advantages.mean()),
            return_mean=float(returns.mean()),
            entropy_coef=entropy_coef,
        )
        return result

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "shuffle": self.shuffle.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.shuffle.set_state(state["shuffle"])
