"""EXP-DSA05-005 training loop: XR-SBPC-style SAC-Discrete under protocol v1.

Episodes, training instances and validation follow EXP-DSA05-001: each of
``num_envs`` environments draws a fresh training instance, runs ``episode_slots``
slots and is truncated (its last transition bootstraps like any other). Every
environment step stores one transition per UE, idle UEs included; once the buffer
is full, every step of the environments performs one gradient update. Rewards:
v1 / v1s per UE (``xr_sbpc``); v2 is EXP-DSA05-001's team reward
-(sum_i E_i / E0 + w sum_i c_i) / N, given to every UE. Validation checkpoints hold
the policy network; the protocol rule selects among them at the end, and
``result.json`` has the same fields as a MAPPO run's (mu0 is always 0).
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
    DualSettings,
    LocalMemory,
    context_dim,
    decode_actions,
    legal_mask,
    num_actions,
    stack,
    token_dim,
)
from dsa_period.marl.sac import (
    PrioritizedReplay,
    SacAgent,
    SacPolicy,
    SacSettings,
    flat_features,
    masked_policy,
    policy_checkpoint,
)
from dsa_period.marl.xr_sbpc import RewardSettings, xr_rewards


@dataclass(frozen=True)
class SacTrainingSettings:
    budget_slots: int = 20_000_000
    episode_slots: int = 600
    num_envs: int = 16
    validation_every_slots: int = 1_000_000
    torch_threads: int = 1
    log_every_steps: int = 1000


class SacTrainer:
    def __init__(
        self,
        scenario: Scenario,
        settings: SacTrainingSettings,
        sac: SacSettings,
        reward: RewardSettings,
        seed: int,
        validation_indices: list[int] | None = None,
    ) -> None:
        torch.set_num_threads(settings.torch_threads)
        self.scenario, self.settings, self.sac, self.reward_settings, self.seed = (
            scenario,
            settings,
            sac,
            reward,
            seed,
        )
        self.config, self.menu = scenario.physical, scenario.menu
        self.epsilon = scenario.objective["max_violation_rate"]
        self.duals = DualSettings("fixed", epsilon=self.epsilon)
        self.actions = num_actions(self.config, self.menu)
        self.obs_dim = self.config.num_subchannels * token_dim(self.menu) + context_dim()
        self.agent = SacAgent(self.obs_dim, self.actions, sac, seed)
        self.buffer = PrioritizedReplay(
            sac.buffer_size, self.obs_dim, self.actions, sac.per_alpha, sac.per_eps, [seed, 0xB0F]
        )
        self.rng = np.random.default_rng([seed, 0x5EED])
        self.sampler = torch.Generator().manual_seed(seed + 1)
        self.energy_unit = self.config.slot_duration_s * self.menu.power_w[-1]
        self.validation_indices = (
            list(splits.indices("validation")) if validation_indices is None else validation_indices
        )
        self.references = load_references("validation", self.config.num_ues)
        n, e = self.config.num_ues, settings.num_envs
        self.memories = [LocalMemory(self.config, self.menu, n, self.duals) for _ in range(e)]
        self.envs: list[PeriodicUplinkEnv] = [None] * e  # type: ignore[list-item]
        self.obs = [None] * e
        self.features = np.zeros((e, n, self.obs_dim), np.float32)
        self.legal = np.zeros((e, n, self.actions), np.bool_)
        self.instances_drawn: list[int] = []
        for env in range(e):
            self._reset_env(env)

    # ----------------------------------------------------------------- helpers
    def _refresh(self, env: int) -> None:
        memory, obs = self.memories[env], self.obs[env]
        self.features[env] = flat_features(memory, obs)
        self.legal[env] = legal_mask(obs, self.actions)

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
        memory = self.memories[env]
        memory.reset(np.zeros(self.config.num_ues))
        self.obs[env] = stack(self.envs[env].reset())
        memory.update_duals(memory.observe(self.obs[env]))
        self._refresh(env)

    def rewards(self, before, step) -> np.ndarray:
        """Per-UE reward of one slot from the state at its start and the env step."""
        r = self.reward_settings
        if r.kind == "v2":
            energy = float(step.physical.tx_energy_j.sum())
            team = -(energy / self.energy_unit + r.weight * float(step.violations.sum()))
            return np.full(self.config.num_ues, team / self.config.num_ues)
        return xr_rewards(
            r,
            before.remaining,
            before.to_deadline,
            step.physical.service_bits,
            step.completed,
            step.violations.astype(bool),
        )

    # -------------------------------------------------------------------- step
    def step(self) -> dict:
        """Advance every environment one slot and store the N transitions of each."""
        envs, n = self.settings.num_envs, self.config.num_ues
        with torch.no_grad():
            logits = self.agent.policy(torch.from_numpy(self.features.reshape(envs * n, -1)))
            probs, _ = masked_policy(logits, torch.from_numpy(self.legal.reshape(envs * n, -1)))
            chosen = torch.multinomial(probs, 1, generator=self.sampler).squeeze(-1)
        chosen = chosen.numpy().reshape(envs, n)
        stats = dict(
            reward=0.0, energy_j=0.0, settled=0, violations=0, tx=0, active=0, completions=0
        )
        levels = np.zeros(self.menu.num_levels, dtype=np.int64)
        for env in range(envs):
            before = self.obs[env]
            features, legal = self.features[env].copy(), self.legal[env].copy()
            step = self.envs[env].step(decode_actions(chosen[env], self.menu))
            memory = self.memories[env]
            self.obs[env] = stack(step.observations)
            settlement = memory.observe(self.obs[env])
            if not (
                np.array_equal(settlement.violated, step.violations.astype(bool))
                and np.array_equal(settlement.settled, (step.violations + step.on_time) > 0)
            ):
                raise AssertionError("local settlement tracker disagrees with the environment")
            memory.update_duals(settlement)
            reward = self.rewards(before, step)
            self._refresh(env)
            self.buffer.add(
                features, chosen[env], reward, self.features[env], legal, self.legal[env]
            )
            stats["reward"] += float(reward.sum())
            stats["energy_j"] += float(step.physical.tx_energy_j.sum())
            stats["settled"] += int(settlement.settled.sum())
            stats["violations"] += int(settlement.violated.sum())
            stats["tx"] += int(step.physical.active.sum())
            stats["active"] += int((before.remaining > 0).sum())
            stats["completions"] += int(step.completed.sum())
            levels += np.bincount(step.physical.actions[:, 1], minlength=self.menu.num_levels)
            if step.truncated:
                self._reset_env(env)
        stats["levels"] = levels
        return stats

    # --------------------------------------------------------------- validation
    def policy(self) -> SacPolicy:
        return SacPolicy(self.agent.policy, self.config, self.menu, self.epsilon)

    def validate(self) -> tuple[dict, list[dict]]:
        episodes = evaluate_policy(
            self.scenario, self.policy, "validation", self.validation_indices, self.references
        )
        return aggregate({str(self.seed): episodes}), episodes

    def checkpoint(self, slots: int, updates: int) -> dict:
        return policy_checkpoint(
            self.agent,
            slots=slots,
            updates=updates,
            seed=self.seed,
            settings=asdict(self.settings),
            reward=asdict(self.reward_settings),
        )

    # -------------------------------------------------------------------- run
    def run(self, output: Path, log=print) -> dict:
        s = self.settings
        (output / "checkpoints").mkdir(parents=True)
        (output / "validation").mkdir()
        summaries: dict[str, dict] = {}
        slots, updates, steps, next_validation = 0, 0, 0, 0
        window = self._empty_window()
        started = time.perf_counter()
        with open(output / "training.jsonl", "w") as training_log:
            while True:
                if slots >= next_validation or slots >= s.budget_slots:
                    name = f"step-{slots:09d}"
                    tick = time.perf_counter()
                    summary, episodes = self.validate()
                    torch.save(
                        self.checkpoint(slots, updates), output / "checkpoints" / f"{name}.pt"
                    )
                    summaries[name] = summary
                    record = {
                        "checkpoint": name,
                        "slots": slots,
                        "by_mu0": [{"mu0": 0.0, "summary": summary, "episodes": episodes}],
                    }
                    (output / "validation" / f"{name}.json").write_text(json.dumps(record))
                    m = summary["metrics"]
                    log(
                        f"[validate {name}] F {m['feasible']['mean']:.3f} "
                        f"E/UB {m['e_over_ub']['mean']:.3f} "
                        f"worst-UE OCR {m['worst_ue_ocr']['mean']:.4f} "
                        f"mW {m['mw_per_ue']['mean']:.2f} ({time.perf_counter() - tick:.0f}s)"
                    )
                    while next_validation <= slots:
                        next_validation += s.validation_every_slots
                if slots >= s.budget_slots:
                    break
                tick = time.perf_counter()
                stats = self.step()
                slots += s.num_envs
                steps += 1
                window["step_s"] += time.perf_counter() - tick
                if self.buffer.size >= self.buffer.capacity:
                    tick = time.perf_counter()
                    beta = self.sac.per_beta_start + (1.0 - self.sac.per_beta_start) * min(
                        1.0, slots / s.budget_slots
                    )
                    metrics = self.agent.update(self.buffer, beta)
                    updates += 1
                    window["update_s"] += time.perf_counter() - tick
                    for key, value in metrics.items():
                        window["metrics"].setdefault(key, []).append(value)
                for key in ("reward", "energy_j", "settled", "violations", "tx", "active"):
                    window[key] += stats[key]
                window["completions"] += stats["completions"]
                window["levels"] += stats["levels"]
                window["steps"] += 1
                if steps % s.log_every_steps == 0:
                    line = self._log_line(window, slots, updates)
                    training_log.write(json.dumps(line) + "\n")
                    training_log.flush()
                    log(
                        f"[{slots:>9d}] reward {line['mean_reward']:.3f} mW {line['mw_per_ue']:.2f} "
                        f"viol {line['violation_rate']:.4f} alpha {line.get('alpha', float('nan')):.4f} "
                        f"ent {line.get('entropy', float('nan')):.3f} {line['slots_per_s']:.0f} slot/s"
                    )
                    window = self._empty_window()
        selected = select(summaries)
        final_checkpoint = f"step-{slots:09d}"
        result = {
            "slots": slots,
            "updates": updates,
            "wall_s": time.perf_counter() - started,
            "selected": selected,
            "selected_checkpoint": selected,
            "selected_mu0": 0.0,
            "final_checkpoint": final_checkpoint,
            "final": final_checkpoint,
            "candidates": {
                name: {
                    "feasible": summaries[name]["metrics"]["feasible"]["mean"],
                    "e_over_ub": summaries[name]["metrics"]["e_over_ub"]["mean"],
                }
                for name in sorted(summaries)
            },
            "training_instances_drawn": len(self.instances_drawn),
        }
        (output / "result.json").write_text(json.dumps(result, indent=2))
        return result

    def _empty_window(self) -> dict:
        return {
            "steps": 0,
            "reward": 0.0,
            "energy_j": 0.0,
            "settled": 0,
            "violations": 0,
            "tx": 0,
            "active": 0,
            "completions": 0,
            "levels": np.zeros(self.menu.num_levels, dtype=np.int64),
            "metrics": {},
            "step_s": 0.0,
            "update_s": 0.0,
        }

    def _log_line(self, window: dict, slots: int, updates: int) -> dict:
        n, envs = self.config.num_ues, self.settings.num_envs
        env_slots = window["steps"] * envs
        levels = window["levels"]
        return {
            "slots": slots,
            "updates": updates,
            "mean_reward": window["reward"] / max(env_slots * n, 1),
            "mw_per_ue": 1e3
            * window["energy_j"]
            / (max(env_slots, 1) * n * self.config.slot_duration_s),
            "violation_rate": window["violations"] / max(window["settled"], 1),
            "tx_fraction_of_active": window["tx"] / max(window["active"], 1),
            "completions": window["completions"],
            "power_level_share": (levels[1:] / max(levels[1:].sum(), 1)).tolist(),
            **{key: float(np.mean(values)) for key, values in window["metrics"].items()},
            "step_s": window["step_s"],
            "update_s": window["update_s"],
            "slots_per_s": env_slots / max(window["step_s"] + window["update_s"], 1e-9),
        }
