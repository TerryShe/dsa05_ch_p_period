"""EXP-DSA05-006 training: reliability-only constrained recurrent MAPPO on scenario v2.0.

Each parallel environment keeps one number of UEs for the whole run
(``env_ues``). Every episode draws a fresh instance from the training pool of
protocol v2 and fresh fading, runs ``episode_slots`` slots from slot 0 and is
truncated (bootstrapped). The UEs of all environments form one batch for the
shared actor; the set critic pools the UEs of each environment. Team reward of
environment e in slot t (no energy term, design §3.3):

    r_e(t) = -(1 / N_e) sum_i mu_i(t) c_i(t),

with c_i(t) = 1 iff UE i's packet violated its deadline at the end of slot t,
mu_i = weight (variant A, "fixed") or UE i's own dual (variant C, "local").
Checkpoints are written every ``checkpoint_every_slots``; validation runs
afterwards in parallel (design §5), not inside the training loop.
"""

import json
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from dsa_period import splits, splits_v2
from dsa_period.config import Scenario
from dsa_period.marl.features import (
    DualSettings,
    LocalMemory,
    context_dim,
    critic_ue_dim,
    critic_ue_features,
    decode_actions,
    legal_mask,
    num_actions,
    stack,
    token_dim,
)
from dsa_period.marl.networks import ChannelSetActor, SetCritic, masked_log_probs
from dsa_period.marl.policy import MarlPolicy
from dsa_period.marl.ppo import PPOSettings, discounted_gae
from dsa_period.physics import PhysicalConfig, PowerMenu

CHECKPOINT_KIND = "capacity-mappo"
CHECKPOINT_SCHEMA = 1
VARIANTS = ("fixed", "local")


@dataclass(frozen=True)
class CapacityTrainingSettings:
    budget_slots: int = 5_000_000
    episode_slots: int = 600
    env_ues: tuple[int, ...] = (4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 16, 16, 20, 20)
    rollout_slots: int = 128
    entropy_start: float = 0.01
    entropy_decay_fraction: float = 0.5
    checkpoint_every_slots: int = 1_000_000
    torch_threads: int = 1
    hidden: int = 64
    critic_hidden: int = 256

    def __post_init__(self) -> None:
        values = tuple(int(n) for n in self.env_ues)
        if not values or any(n < 1 for n in values):
            raise ValueError("env_ues must list at least one positive number of UEs")
        object.__setattr__(self, "env_ues", values)


@dataclass(frozen=True)
class CapacityConstraint:
    """Variant A ("fixed", mu_i = weight) or C ("local", per-UE duals; mu0 at execution)."""

    variant: str
    weight: float = 1.0
    eta: float = 0.0
    mu_max: float = 1.0
    mu0: float = 0.0

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.variant == "local" and not 0 <= self.mu0 <= self.mu_max:
            raise ValueError("mu0 must lie in [0, mu_max]")

    def duals(self, epsilon: float) -> DualSettings:
        return DualSettings(self.variant, epsilon, self.weight, self.eta, self.mu_max)


@dataclass
class RaggedRollout:
    """UE fields are [T, U, ...] over the concatenated UEs; env fields are [T, E]."""

    tokens: np.ndarray
    context: np.ndarray
    legal: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    active: np.ndarray
    critic_ue: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    rewards: np.ndarray
    truncated: np.ndarray
    starts: np.ndarray
    initial_hidden: Tensor  # [U, K, h] before obs[0]


class RaggedMAPPO:
    """Recurrent MAPPO over environments with different numbers of UEs.

    Same update as ``ppo.RecurrentMAPPO``: team GAE per environment shared by
    its UEs, advantages normalized over UEs with a packet, clipped surrogate,
    minibatches of whole environments; the critic is the set critic.
    """

    def __init__(
        self,
        actor: ChannelSetActor,
        critic: SetCritic,
        settings: PPOSettings,
        seed: int,
        env_of_ue: np.ndarray,
    ) -> None:
        self.actor, self.critic, self.settings = actor, critic, settings
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=settings.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=settings.critic_lr, eps=1e-5
        )
        self.shuffle = torch.Generator().manual_seed(seed)
        self.env_of_ue = torch.as_tensor(env_of_ue, dtype=torch.long)
        self.num_envs = int(env_of_ue.max()) + 1

    def values(self, critic_ue: Tensor, envs: Tensor) -> Tensor:
        """critic_ue [T, U_sel, D] of the UEs of ``envs`` (sorted) -> values [T, len(envs)]."""
        steps, ues, _ = critic_ue.shape
        local = torch.searchsorted(envs, self.env_of_ue[torch.isin(self.env_of_ue, envs)])
        segment = (torch.arange(steps).unsqueeze(-1) * envs.numel() + local).reshape(-1)
        flat = self.critic(critic_ue.reshape(steps * ues, -1), segment, steps * envs.numel())
        return flat.reshape(steps, envs.numel())

    def update(self, rollout: RaggedRollout, entropy_coef: float) -> dict[str, float]:
        s = self.settings
        advantages, returns = discounted_gae(
            torch.from_numpy(rollout.rewards),
            torch.from_numpy(rollout.values),
            torch.from_numpy(rollout.next_values),
            torch.from_numpy(rollout.truncated),
            s.gamma,
            s.gae_lambda,
        )
        active = torch.from_numpy(rollout.active)
        per_ue = advantages[:, self.env_of_ue]
        chosen = per_ue[active]
        if chosen.numel() > 1:
            per_ue = (per_ue - chosen.mean()) / (chosen.std(unbiased=False) + 1e-8)
        tokens = torch.from_numpy(rollout.tokens)
        context = torch.from_numpy(rollout.context)
        legal = torch.from_numpy(rollout.legal)
        actions = torch.from_numpy(rollout.actions)
        old_log_probs = torch.from_numpy(rollout.log_probs)
        critic_ue = torch.from_numpy(rollout.critic_ue)
        starts = torch.from_numpy(rollout.starts)[:, self.env_of_ue]
        totals: dict[str, float] = {}
        count = 0
        replay_error = None
        for _ in range(s.ppo_epochs):
            order = torch.randperm(self.num_envs, generator=self.shuffle)
            for start in range(0, self.num_envs, s.minibatch_envs):
                envs = order[start : start + s.minibatch_envs].sort().values
                cols = torch.isin(self.env_of_ue, envs)
                logits, _ = self.actor.sequence(
                    tokens[:, cols], context[:, cols], rollout.initial_hidden[cols], starts[:, cols]
                )
                mask = legal[:, cols]
                log_probs_all = masked_log_probs(logits, mask)
                log_probs = log_probs_all.gather(-1, actions[:, cols].unsqueeze(-1)).squeeze(-1)
                probs = log_probs_all.exp()
                entropy = -(probs * log_probs_all.masked_fill(~mask, 0.0)).sum(-1)
                act = active[:, cols]
                old = old_log_probs[:, cols]
                if replay_error is None:
                    replay_error = (
                        float((log_probs - old).detach()[act].abs().max()) if act.any() else 0.0
                    )
                metrics = {"value_loss": 0.0}
                if act.any():
                    log_ratio = (log_probs - old)[act]
                    ratio = log_ratio.exp()
                    adv = per_ue[:, cols][act]
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
                value = self.values(critic_ue[:, cols], envs)
                value_loss = 0.5 * (value - returns[:, envs]).square().mean()
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


def build_actor(config: PhysicalConfig, menu: PowerMenu, hidden: int) -> ChannelSetActor:
    return ChannelSetActor(
        token_dim(menu, own_csi=True),
        context_dim(),
        config.num_subchannels,
        menu.num_levels - 1,
        hidden,
    )


class CapacityTrainer:
    def __init__(
        self,
        scenario: Scenario,
        settings: CapacityTrainingSettings,
        ppo_settings: PPOSettings,
        constraint: CapacityConstraint,
        seed: int,
    ) -> None:
        if not scenario.is_v2:
            raise ValueError("EXP-DSA05-006 trains on scenario v2.0")
        torch.set_num_threads(settings.torch_threads)
        self.scenario, self.settings, self.constraint, self.seed = (
            scenario,
            settings,
            constraint,
            seed,
        )
        self.config, self.menu = scenario.physical, scenario.menu
        self.duals = constraint.duals(scenario.objective["max_violation_rate"])
        self.scenarios = [scenario.with_num_ues(n) for n in settings.env_ues]
        self.env_ues = np.array(settings.env_ues)
        self.offsets = np.concatenate([[0], np.cumsum(self.env_ues)])
        self.env_of_ue = np.repeat(np.arange(self.env_ues.size), self.env_ues)
        torch.manual_seed(seed)
        actor = build_actor(self.config, self.menu, settings.hidden)
        critic = SetCritic(critic_ue_dim(self.config, self.menu), settings.critic_hidden)
        self.ppo = RaggedMAPPO(actor, critic, ppo_settings, seed, self.env_of_ue)
        self.rng = np.random.default_rng([seed, 0x5EED])
        self.sampler = torch.Generator().manual_seed(seed + 1)
        self.actions = num_actions(self.config, self.menu)
        ues, k = int(self.offsets[-1]), self.config.num_subchannels
        self.memories = [
            LocalMemory(sc.physical, self.menu, sc.physical.num_ues, self.duals)
            for sc in self.scenarios
        ]
        self.envs = [None] * self.env_ues.size
        self.obs = [None] * self.env_ues.size
        self.tokens = np.zeros((ues, k, token_dim(self.menu, own_csi=True)), np.float32)
        self.context = np.zeros((ues, context_dim()), np.float32)
        self.legal = np.zeros((ues, self.actions), np.bool_)
        self.critic_ue = np.zeros((ues, critic_ue_dim(self.config, self.menu)), np.float32)
        self.starts = np.zeros(self.env_ues.size, np.bool_)
        self.hidden = actor.initial_hidden(ues)
        self.instances_drawn: list[int] = []
        for env in range(self.env_ues.size):
            self._reset_env(env)

    # ----------------------------------------------------------------- helpers
    def _rows(self, env: int) -> slice:
        return slice(int(self.offsets[env]), int(self.offsets[env + 1]))

    def _penalty(self, env: int) -> np.ndarray:
        if self.duals.variant == "fixed":
            return np.full(int(self.env_ues[env]), self.constraint.weight)
        return self.memories[env].mu.copy()

    def _refresh(self, env: int) -> None:
        rows, memory, obs = self._rows(env), self.memories[env], self.obs[env]
        self.tokens[rows], self.context[rows] = memory.features(obs)
        self.legal[rows] = legal_mask(obs, self.actions)
        self.critic_ue[rows] = critic_ue_features(
            self.tokens[rows], self.context[rows], self._penalty(env), self.duals.mu_scale
        )

    def _reset_env(self, env: int) -> None:
        index = splits.sample_train_index(self.rng)
        self.instances_drawn.append(index)
        scenario = self.scenarios[env]
        n = scenario.physical.num_ues
        topology, traffic = scenario.instance(index)
        seeds = splits_v2.training_fading_seeds(self.rng, n)
        self.envs[env] = scenario.make_env(topology, traffic, seeds, self.settings.episode_slots)
        mu = self.rng.uniform(0.0, self.duals.mu_max, n) if self.duals.variant == "local" else 0
        memory = self.memories[env]
        memory.reset(np.broadcast_to(np.asarray(mu, float), (n,)))
        self.obs[env] = stack(self.envs[env].reset())
        memory.update_duals(memory.observe(self.obs[env]))
        self._refresh(env)
        self.hidden[self._rows(env)] = 0.0
        self.starts[env] = True

    def _values(self) -> np.ndarray:
        with torch.no_grad():
            return (
                self.ppo.critic(
                    torch.from_numpy(self.critic_ue),
                    torch.from_numpy(self.env_of_ue),
                    self.env_ues.size,
                )
                .numpy()
                .astype(np.float32)
            )

    # ------------------------------------------------------------------ rollout
    def collect(self) -> tuple[RaggedRollout, dict]:
        steps, envs = self.settings.rollout_slots, self.env_ues.size
        ues = int(self.offsets[-1])
        buf = {
            "tokens": np.zeros((steps, *self.tokens.shape), np.float32),
            "context": np.zeros((steps, *self.context.shape), np.float32),
            "legal": np.zeros((steps, *self.legal.shape), np.bool_),
            "actions": np.zeros((steps, ues), np.int64),
            "log_probs": np.zeros((steps, ues), np.float32),
            "active": np.zeros((steps, ues), np.bool_),
            "critic_ue": np.zeros((steps, *self.critic_ue.shape), np.float32),
            "values": np.zeros((steps, envs), np.float32),
            "next_values": np.zeros((steps, envs), np.float32),
            "rewards": np.zeros((steps, envs), np.float32),
            "truncated": np.zeros((steps, envs), np.bool_),
            "starts": np.zeros((steps, envs), np.bool_),
        }
        initial_hidden = self.hidden.clone()
        settled = np.zeros(envs, np.int64)
        violations = np.zeros(envs, np.int64)
        tx = active_decisions = 0
        actor = self.ppo.actor
        for t in range(steps):
            buf["tokens"][t], buf["context"][t], buf["legal"][t] = (
                self.tokens,
                self.context,
                self.legal,
            )
            buf["critic_ue"][t], buf["starts"][t] = self.critic_ue, self.starts
            buf["active"][t] = self.legal[:, 1:].any(axis=-1)
            with torch.no_grad():
                logits, self.hidden = actor.step(
                    torch.from_numpy(self.tokens), torch.from_numpy(self.context), self.hidden
                )
                log_probs = masked_log_probs(logits, torch.from_numpy(self.legal))
                chosen = torch.multinomial(log_probs.exp(), 1, generator=self.sampler).squeeze(-1)
                buf["log_probs"][t] = log_probs.gather(-1, chosen[:, None]).squeeze(-1).numpy()
            buf["values"][t] = self._values()
            chosen = chosen.numpy()
            buf["actions"][t] = chosen
            self.starts[:] = False
            for env in range(envs):
                rows = self._rows(env)
                step = self.envs[env].step(decode_actions(chosen[rows], self.menu))
                memory = self.memories[env]
                self.obs[env] = stack(step.observations)
                settlement = memory.observe(self.obs[env])
                if not (
                    np.array_equal(settlement.violated, step.violations.astype(bool))
                    and np.array_equal(settlement.settled, (step.violations + step.on_time) > 0)
                ):
                    raise AssertionError("local settlement tracker disagrees with the environment")
                penalty = self._penalty(env)
                cost = float(penalty @ settlement.violated)
                buf["rewards"][t, env] = -cost / self.env_ues[env]
                memory.update_duals(settlement)
                self._refresh(env)
                buf["truncated"][t, env] = step.truncated
                settled[env] += int(settlement.settled.sum())
                violations[env] += int(settlement.violated.sum())
                tx += int(step.physical.active.sum())
                active_decisions += int(buf["active"][t, rows].sum())
            buf["next_values"][t] = self._values()
            for env in np.flatnonzero(buf["truncated"][t]):
                self._reset_env(int(env))
        rollout = RaggedRollout(**buf, initial_hidden=initial_hidden)
        by_n = {}
        for n in np.unique(self.env_ues):
            group = self.env_ues == n
            by_n[str(int(n))] = float(violations[group].sum() / max(settled[group].sum(), 1))
        stats = {
            "iteration_slots": steps * envs,
            "violation_rate": float(violations.sum() / max(settled.sum(), 1)),
            "violation_rate_by_n": by_n,
            "tx_fraction_of_active": tx / max(active_decisions, 1),
            "occupancy": tx / (steps * envs * self.config.num_subchannels),
            "mean_reward": float(buf["rewards"].mean()),
            "mu_mean": float(np.mean([self._penalty(e).mean() for e in range(envs)])),
        }
        return rollout, stats

    # -------------------------------------------------------------- checkpoints
    def checkpoint(self, slots: int, iteration: int) -> dict:
        return {
            "kind": CHECKPOINT_KIND,
            "schema": CHECKPOINT_SCHEMA,
            "slots": slots,
            "iteration": iteration,
            "seed": self.seed,
            "settings": asdict(self.settings),
            "constraint": asdict(self.constraint),
            "duals": asdict(self.duals),
            "model": {
                "token_dim": token_dim(self.menu, own_csi=True),
                "context_dim": context_dim(),
                "num_channels": self.config.num_subchannels,
                "num_levels": self.menu.num_levels - 1,
                "hidden": self.settings.hidden,
                "critic_ue_dim": critic_ue_dim(self.config, self.menu),
                "critic_hidden": self.settings.critic_hidden,
            },
            "ppo": self.ppo.state_dict(),
            "rng": {
                "numpy": self.rng.bit_generator.state,
                "sampler": self.sampler.get_state(),
                "torch": torch.get_rng_state(),
            },
        }

    def run(self, output: Path, log=print) -> dict:
        s = self.settings
        (output / "checkpoints").mkdir(parents=True)
        slots_per_iteration = s.rollout_slots * self.env_ues.size
        slots, iteration = 0, 0
        next_checkpoint = s.checkpoint_every_slots
        saved = []
        started = time.perf_counter()
        with open(output / "training.jsonl", "w") as training_log:
            while slots < s.budget_slots:
                tick = time.perf_counter()
                rollout, stats = self.collect()
                collect_s = time.perf_counter() - tick
                slots += slots_per_iteration
                iteration += 1
                progress = slots / (s.entropy_decay_fraction * s.budget_slots)
                coef = s.entropy_start * max(0.0, 1.0 - progress)
                tick = time.perf_counter()
                metrics = self.ppo.update(rollout, coef)
                update_s = time.perf_counter() - tick
                line = {
                    "iteration": iteration,
                    "slots": slots,
                    **stats,
                    **metrics,
                    "collect_s": collect_s,
                    "update_s": update_s,
                    "slots_per_s": slots_per_iteration / (collect_s + update_s),
                }
                training_log.write(json.dumps(line) + "\n")
                if iteration % 50 == 0:
                    training_log.flush()
                    log(
                        f"[{slots:>9d}] reward {stats['mean_reward']:.5f} "
                        f"viol {stats['violation_rate']:.4f} occ {stats['occupancy']:.2f} "
                        f"ent {metrics.get('entropy', 0):.3f} mu {stats['mu_mean']:.2f} "
                        f"{line['slots_per_s']:.0f} slot/s"
                    )
                if slots >= next_checkpoint or slots >= s.budget_slots:
                    name = f"step-{slots:09d}"
                    torch.save(
                        self.checkpoint(slots, iteration), output / "checkpoints" / f"{name}.pt"
                    )
                    saved.append(name)
                    log(f"[checkpoint {name}]")
                    while next_checkpoint <= slots:
                        next_checkpoint += s.checkpoint_every_slots
        result = {
            "slots": slots,
            "iterations": iteration,
            "wall_s": time.perf_counter() - started,
            "checkpoints": saved,
            "training_instances_drawn": len(self.instances_drawn),
        }
        (output / "result.json").write_text(json.dumps(result, indent=2))
        return result


@lru_cache(maxsize=8)
def _load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_capacity_policy(
    path: Path, config: PhysicalConfig, menu: PowerMenu, mu0: float | None = None
) -> MarlPolicy:
    """A protocol policy for any N from an EXP-DSA05-006 checkpoint (cached per process)."""
    checkpoint = _load_checkpoint(str(path))
    if checkpoint.get("kind") != CHECKPOINT_KIND or checkpoint["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError(f"{path} is not an EXP-DSA05-006 checkpoint")
    model = checkpoint["model"]
    actor = ChannelSetActor(
        model["token_dim"],
        model["context_dim"],
        model["num_channels"],
        model["num_levels"],
        model["hidden"],
    )
    actor.load_state_dict(checkpoint["ppo"]["actor"])
    actor.eval()
    duals = DualSettings(**checkpoint["duals"])
    constraint = CapacityConstraint(**checkpoint["constraint"])
    start = constraint.mu0 if mu0 is None else mu0
    return MarlPolicy(actor, config, menu, duals, start, name=f"marl:{path}")
