"""EXP-DSA05-001 training loop: constrained recurrent MAPPO under protocol v1.

Each training episode draws a fresh instance from the protocol's training pool
and runs ``episode_slots`` slots from slot 0, then is truncated (bootstrapped).
Team reward per slot: r_t = -(sum_i E_i(t) / E0 + sum_i mu_i(t) c_i(t)) / N with
E0 = tau * p_max and c_i(t) = 1 iff UE i's packet violated its deadline at the
end of slot t. Periodic validation follows protocol v1 exactly; every validated
actor is checkpointed and the protocol rule selects among them at the end.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dsa_period import splits
from dsa_period.channel import StaticChannel
from dsa_period.config import Scenario
from dsa_period.env import PeriodicUplinkEnv
from dsa_period.evaluation import aggregate, evaluate_policy, load_references, select
from dsa_period.marl.features import (
    CONTEXT_DIM,
    DualSettings,
    LocalMemory,
    critic_dim,
    critic_state,
    decode_actions,
    legal_mask,
    num_actions,
    stack,
    token_dim,
)
from dsa_period.marl.networks import CentralCritic, ChannelSetActor, masked_log_probs
from dsa_period.marl.policy import MarlPolicy
from dsa_period.marl.ppo import PPOSettings, RecurrentMAPPO, Rollout

CHECKPOINT_SCHEMA = 1


@dataclass(frozen=True)
class TrainingSettings:
    budget_slots: int = 2_000_000
    episode_slots: int = 600
    num_envs: int = 16
    rollout_slots: int = 128
    entropy_start: float = 0.01
    entropy_decay_fraction: float = 0.5
    validation_every_slots: int = 250_000
    torch_threads: int = 1
    hidden: int = 64
    critic_hidden: int = 256


@dataclass(frozen=True)
class ConstraintSettings:
    """Variant parameters: fixed uses weight; global eta/mu_init/mu_max; local eta/mu_max/mu0."""

    variant: str
    weight: float = 0.0
    eta: float = 0.0
    mu_max: float = 1.0
    mu_init: float = 0.0
    mu0: float = 0.0

    def duals(self, epsilon: float) -> DualSettings:
        return DualSettings(self.variant, epsilon, self.weight, self.eta, self.mu_max)


def build_models(scenario: Scenario, settings: TrainingSettings) -> tuple:
    config, menu = scenario.physical, scenario.menu
    actor = ChannelSetActor(
        token_dim(menu), CONTEXT_DIM, config.num_subchannels, menu.num_levels - 1, settings.hidden
    )
    critic = CentralCritic(critic_dim(config, menu), settings.critic_hidden)
    return actor, critic


class Trainer:
    def __init__(
        self,
        scenario: Scenario,
        settings: TrainingSettings,
        ppo_settings: PPOSettings,
        constraint: ConstraintSettings,
        seed: int,
        validation_indices: list[int] | None = None,
    ) -> None:
        torch.set_num_threads(settings.torch_threads)
        self.scenario, self.settings, self.constraint, self.seed = (
            scenario,
            settings,
            constraint,
            seed,
        )
        self.config, self.menu = scenario.physical, scenario.menu
        self.duals = constraint.duals(scenario.objective["max_violation_rate"])
        torch.manual_seed(seed)
        actor, critic = build_models(scenario, settings)
        self.ppo = RecurrentMAPPO(actor, critic, ppo_settings, seed)
        self.rng = np.random.default_rng([seed, 0x5EED])
        self.sampler = torch.Generator().manual_seed(seed + 1)
        self.mu_global = constraint.mu_init
        self.energy_unit = self.config.slot_duration_s * self.menu.power_w[-1]
        self.actions = num_actions(self.config, self.menu)
        self.validation_indices = (
            list(splits.indices("validation")) if validation_indices is None else validation_indices
        )
        self.references = load_references("validation", self.config.num_ues)
        n, e = self.config.num_ues, settings.num_envs
        self.memories = [LocalMemory(self.config, self.menu, n, self.duals) for _ in range(e)]
        self.envs: list[PeriodicUplinkEnv] = [None] * e  # type: ignore[list-item]
        self.true_gain = np.zeros((e, n))
        self.obs = [None] * e
        self.tokens = np.zeros(
            (e, n, self.config.num_subchannels, token_dim(self.menu)), np.float32
        )
        self.context = np.zeros((e, n, CONTEXT_DIM), np.float32)
        self.legal = np.zeros((e, n, self.actions), np.bool_)
        self.state = np.zeros((e, critic_dim(self.config, self.menu)), np.float32)
        self.starts = np.zeros(e, np.bool_)
        self.hidden = actor.initial_hidden(e * n)
        self.instances_drawn: list[int] = []
        for env in range(e):
            self._reset_env(env)

    # ----------------------------------------------------------------- helpers
    def _penalty(self, env: int) -> np.ndarray:
        n = self.config.num_ues
        if self.duals.variant == "fixed":
            return np.full(n, self.constraint.weight)
        if self.duals.variant == "global":
            return np.full(n, self.mu_global)
        return self.memories[env].mu.copy()

    def _refresh(self, env: int) -> None:
        memory, obs = self.memories[env], self.obs[env]
        self.tokens[env], self.context[env] = memory.features(obs)
        self.legal[env] = legal_mask(obs, self.actions)
        self.state[env] = critic_state(
            self.tokens[env],
            self.context[env],
            self.true_gain[env],
            self._penalty(env),
            self.duals.mu_scale,
        )

    def _reset_env(self, env: int) -> None:
        index = splits.sample_train_index(self.rng)
        self.instances_drawn.append(index)
        topology, traffic = self.scenario.instance(index)
        self.envs[env] = PeriodicUplinkEnv(
            self.config,
            self.menu,
            StaticChannel(topology, self.config.num_subchannels),
            traffic,
            self.settings.episode_slots,
        )
        self.true_gain[env] = topology.ue_ap_gain_linear
        n = self.config.num_ues
        mu = self.rng.uniform(0.0, self.duals.mu_max, n) if self.duals.variant == "local" else 0
        memory = self.memories[env]
        memory.reset(np.broadcast_to(np.asarray(mu, float), (n,)))
        self.obs[env] = stack(self.envs[env].reset())
        memory.update_duals(memory.observe(self.obs[env]))
        self._refresh(env)
        self.hidden[env * n : (env + 1) * n] = 0.0
        self.starts[env] = True

    # ------------------------------------------------------------------ rollout
    def collect(self) -> tuple[Rollout, dict]:
        steps, envs, n = self.settings.rollout_slots, self.settings.num_envs, self.config.num_ues
        k, dt = self.config.num_subchannels, token_dim(self.menu)
        buf = {
            "tokens": np.zeros((steps, envs, n, k, dt), np.float32),
            "context": np.zeros((steps, envs, n, CONTEXT_DIM), np.float32),
            "legal": np.zeros((steps, envs, n, self.actions), np.bool_),
            "actions": np.zeros((steps, envs, n), np.int64),
            "log_probs": np.zeros((steps, envs, n), np.float32),
            "active": np.zeros((steps, envs, n), np.bool_),
            "state": np.zeros((steps, envs, self.state.shape[1]), np.float32),
            "values": np.zeros((steps, envs), np.float32),
            "next_values": np.zeros((steps, envs), np.float32),
            "rewards": np.zeros((steps, envs), np.float32),
            "truncated": np.zeros((steps, envs), np.bool_),
            "starts": np.zeros((steps, envs), np.bool_),
        }
        initial_hidden = self.hidden.clone()
        stats = {"energy_j": 0.0, "settled": 0, "violations": 0, "tx": 0, "active_decisions": 0}
        levels = np.zeros(self.menu.num_levels, dtype=np.int64)
        actor, critic = self.ppo.actor, self.ppo.critic
        for t in range(steps):
            buf["tokens"][t], buf["context"][t], buf["legal"][t] = (
                self.tokens,
                self.context,
                self.legal,
            )
            buf["state"][t], buf["starts"][t] = self.state, self.starts
            buf["active"][t] = self.legal[:, :, 1:].any(axis=-1)
            with torch.no_grad():
                logits, self.hidden = actor.step(
                    torch.from_numpy(self.tokens.reshape(envs * n, k, dt)),
                    torch.from_numpy(self.context.reshape(envs * n, CONTEXT_DIM)),
                    self.hidden,
                )
                log_probs = masked_log_probs(
                    logits, torch.from_numpy(self.legal.reshape(envs * n, -1))
                )
                chosen = torch.multinomial(log_probs.exp(), 1, generator=self.sampler).squeeze(-1)
                buf["log_probs"][t] = (
                    log_probs.gather(-1, chosen[:, None]).squeeze(-1).numpy().reshape(envs, n)
                )
                buf["values"][t] = critic(torch.from_numpy(self.state)).numpy()
            chosen = chosen.numpy().reshape(envs, n)
            buf["actions"][t] = chosen
            self.starts[:] = False
            for env in range(envs):
                step = self.envs[env].step(decode_actions(chosen[env], self.menu))
                memory = self.memories[env]
                self.obs[env] = stack(step.observations)
                settlement = memory.observe(self.obs[env])
                if not (
                    np.array_equal(settlement.violated, step.violations.astype(bool))
                    and np.array_equal(settlement.settled, (step.violations + step.on_time) > 0)
                ):
                    raise AssertionError("local settlement tracker disagrees with the environment")
                penalty = self._penalty(env)
                energy = float(step.physical.tx_energy_j.sum())
                cost = float(penalty @ settlement.violated)
                buf["rewards"][t, env] = -(energy / self.energy_unit + cost) / n
                memory.update_duals(settlement)
                self._refresh(env)
                buf["truncated"][t, env] = step.truncated
                stats["energy_j"] += energy
                stats["settled"] += int(settlement.settled.sum())
                stats["violations"] += int(settlement.violated.sum())
                stats["tx"] += int(step.physical.active.sum())
                stats["active_decisions"] += int(buf["active"][t, env].sum())
                levels += np.bincount(step.physical.actions[:, 1], minlength=self.menu.num_levels)
            with torch.no_grad():
                buf["next_values"][t] = critic(torch.from_numpy(self.state)).numpy()
            for env in np.flatnonzero(buf["truncated"][t]):
                self._reset_env(int(env))
        rollout = Rollout(**buf, initial_hidden=initial_hidden)
        slots = steps * envs
        stats.update(
            slots=slots,
            mw_per_ue=1e3 * stats["energy_j"] / (slots * n * self.config.slot_duration_s),
            violation_rate=stats["violations"] / max(stats["settled"], 1),
            tx_fraction_of_active=stats["tx"] / max(stats["active_decisions"], 1),
            power_level_share=(levels[1:] / max(levels[1:].sum(), 1)).tolist(),
            mean_reward=float(buf["rewards"].mean()),
            mu_mean=float(np.mean([self._penalty(e).mean() for e in range(envs)])),
        )
        return rollout, stats

    # --------------------------------------------------------------- validation
    def policy(self) -> MarlPolicy:
        return MarlPolicy(self.ppo.actor, self.config, self.menu, self.duals, self.constraint.mu0)

    def validate(self) -> tuple[dict, list[dict]]:
        eval_scenario = self.scenario
        episodes = evaluate_policy(
            eval_scenario, self.policy, "validation", self.validation_indices, self.references
        )
        return aggregate({str(self.seed): episodes}), episodes

    def checkpoint(self, slots: int, iteration: int) -> dict:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "slots": slots,
            "iteration": iteration,
            "seed": self.seed,
            "settings": asdict(self.settings),
            "constraint": asdict(self.constraint),
            "duals": asdict(self.duals),
            "mu_global": self.mu_global,
            "model": {
                "token_dim": token_dim(self.menu),
                "context_dim": CONTEXT_DIM,
                "num_channels": self.config.num_subchannels,
                "num_levels": self.menu.num_levels - 1,
                "hidden": self.settings.hidden,
                "critic_dim": critic_dim(self.config, self.menu),
                "critic_hidden": self.settings.critic_hidden,
            },
            "ppo": self.ppo.state_dict(),
            "rng": {
                "numpy": self.rng.bit_generator.state,
                "sampler": self.sampler.get_state(),
                "torch": torch.get_rng_state(),
            },
        }

    # -------------------------------------------------------------------- run
    def run(self, output: Path, log=print) -> dict:
        s = self.settings
        (output / "checkpoints").mkdir(parents=True)
        (output / "validation").mkdir()
        slots_per_iteration = s.rollout_slots * s.num_envs
        summaries: dict[str, dict] = {}
        slots, iteration, next_validation = 0, 0, 0
        started = time.perf_counter()
        with open(output / "training.jsonl", "w") as training_log:
            while True:
                if slots >= next_validation or slots >= s.budget_slots:
                    name = f"step-{slots:09d}"
                    tick = time.perf_counter()
                    summary, episodes = self.validate()
                    summaries[name] = summary
                    torch.save(
                        self.checkpoint(slots, iteration), output / "checkpoints" / f"{name}.pt"
                    )
                    record = {
                        "checkpoint": name,
                        "slots": slots,
                        "summary": summary,
                        "episodes": episodes,
                    }
                    (output / "validation" / f"{name}.json").write_text(json.dumps(record))
                    m = summary["metrics"]
                    log(
                        f"[validate {name}] F {m['feasible']['mean']:.3f} "
                        f"E/UB {m['e_over_ub']['mean']:.3f} worst-UE OCR "
                        f"{m['worst_ue_ocr']['mean']:.4f} mW {m['mw_per_ue']['mean']:.2f} "
                        f"({time.perf_counter() - tick:.0f}s)"
                    )
                    while next_validation <= slots:
                        next_validation += s.validation_every_slots
                if slots >= s.budget_slots:
                    break
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
                if self.duals.variant == "global":
                    rate = stats["violations"] / max(stats["settled"], 1)
                    self.mu_global = float(
                        np.clip(
                            self.mu_global + self.constraint.eta * (rate - self.duals.epsilon),
                            0.0,
                            self.duals.mu_max,
                        )
                    )
                line = {
                    "iteration": iteration,
                    "slots": slots,
                    **stats,
                    **metrics,
                    "mu_global": self.mu_global,
                    "collect_s": collect_s,
                    "update_s": update_s,
                    "slots_per_s": slots_per_iteration / (collect_s + update_s),
                }
                training_log.write(json.dumps(line) + "\n")
                if iteration % 50 == 0:
                    training_log.flush()
                    log(
                        f"[{slots:>9d}] reward {stats['mean_reward']:.4f} mW {stats['mw_per_ue']:.2f} "
                        f"viol {stats['violation_rate']:.4f} ent {metrics.get('entropy', 0):.3f} "
                        f"mu {stats['mu_mean']:.2f} {line['slots_per_s']:.0f} slot/s"
                    )
        names = sorted(summaries)
        result = {
            "slots": slots,
            "iterations": iteration,
            "wall_s": time.perf_counter() - started,
            "selected": select(summaries),
            "final": names[-1],
            "candidates": {
                name: {
                    "feasible": summaries[name]["metrics"]["feasible"]["mean"],
                    "e_over_ub": summaries[name]["metrics"]["e_over_ub"]["mean"],
                }
                for name in names
            },
            "training_instances_drawn": len(self.instances_drawn),
        }
        (output / "result.json").write_text(json.dumps(result, indent=2))
        return result


def load_policy_factory(path: Path, scenario: Scenario):
    """Rebuild a checkpoint's actor and return a factory of protocol policies."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported checkpoint schema")
    model = checkpoint["model"]
    actor = ChannelSetActor(
        model["token_dim"],
        model["context_dim"],
        model["num_channels"],
        model["num_levels"],
        model["hidden"],
    )
    actor.load_state_dict(checkpoint["ppo"]["actor"])
    duals = DualSettings(**checkpoint["duals"])
    mu0 = checkpoint["constraint"]["mu0"]

    def make() -> MarlPolicy:
        return MarlPolicy(actor, scenario.physical, scenario.menu, duals, mu0)

    return make, checkpoint
