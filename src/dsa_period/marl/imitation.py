"""EXP-DSA05-010: DAgger imitation of the central scheduler N_EDF by the local actor.

Design: docs/experiment-designs/010-edf-imitation-diagnostic_20261008.md, §2.

The student is the unchanged EXP-006 actor (CSG GRU on the local features of
``LocalMemory``, own current CSI, 21 actions) and sees exactly what it sees at
execution (``MarlPolicy``). The teacher is ``CentralEdfPolicy`` on every UE's
current observation; it only labels. All environments of a run have the same N
(protocol v2 §4, 2026-10-04 supplement). Iteration 1 is executed by the teacher,
later iterations by the student (sampling, like at execution); every visited
state gets the teacher's label. After each iteration the student is trained on
everything collected so far: cross-entropy over UE-slots with a packet, no
reweighting, truncated BPTT over 128-slot segments with the hidden state carried
(detached) from the start of each episode. Features are stored as float16.
"""

import json
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dsa_period import splits, splits_v2
from dsa_period.config import Scenario
from dsa_period.marl.features import (
    LocalMemory,
    context_dim,
    critic_ue_dim,
    decode_actions,
    encode_actions,
    legal_mask,
    num_actions,
    stack,
    token_dim,
)
from dsa_period.marl.networks import masked_log_probs
from dsa_period.marl.trainer_v2 import CapacityConstraint, build_actor
from dsa_period.rules_v2 import CentralEdfPolicy

CHECKPOINT_KIND = "capacity-imitation"
CHECKPOINT_SCHEMA = 1


@dataclass(frozen=True)
class ImitationSettings:
    iterations: int = 6
    episodes_per_env: int = 30
    episode_slots: int = 600
    num_envs: int = 14
    segment_slots: int = 128
    epochs: int = 4
    batch_episodes: int = 16
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    hidden: int = 64
    torch_threads: int = 1


def teacher_labels(teacher: CentralEdfPolicy, observations, menu) -> np.ndarray:
    """Student action indices of the teacher's joint action (OFF = 0)."""
    return encode_actions(teacher.act(observations), menu)


class Episodes:
    """Aggregated dataset: one entry per collected episode of N UEs."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, tokens, context, legal, labels, active) -> None:
        self.items.append(
            {
                "tokens": np.asarray(tokens, np.float16),
                "context": np.asarray(context, np.float16),
                "legal": np.asarray(legal, np.bool_),
                "labels": np.asarray(labels, np.int64),
                "active": np.asarray(active, np.bool_),
            }
        )

    def batch(self, indices) -> dict[str, torch.Tensor]:
        """Episodes side by side on the UE axis: [T, E * N, ...]."""
        out = {}
        for key in ("tokens", "context", "legal", "labels", "active"):
            joined = np.concatenate([self.items[i][key] for i in indices], axis=1)
            tensor = torch.from_numpy(joined)
            out[key] = tensor.float() if key in ("tokens", "context") else tensor
        return out


def segment_loss(actor, batch: dict, start: int, stop: int, hidden):
    """Cross-entropy of one segment over UE-slots with a packet, and the next hidden."""
    steps = stop - start
    starts = torch.zeros(steps, hidden.shape[0], dtype=torch.bool)
    logits, hidden = actor.sequence(
        batch["tokens"][start:stop], batch["context"][start:stop], hidden, starts
    )
    log_probs = masked_log_probs(logits, batch["legal"][start:stop])
    chosen = log_probs.gather(-1, batch["labels"][start:stop].unsqueeze(-1)).squeeze(-1)
    active = batch["active"][start:stop]
    if not active.any():
        return None, hidden
    return -chosen[active].mean(), hidden


class ImitationTrainer:
    def __init__(self, scenario: Scenario, settings: ImitationSettings, num_ues: int, seed: int):
        if not scenario.is_v2:
            raise ValueError("EXP-DSA05-010 trains on scenario v2.0")
        torch.set_num_threads(settings.torch_threads)
        self.settings, self.seed, self.num_ues = settings, seed, num_ues
        self.scenario = scenario.with_num_ues(num_ues)
        self.config, self.menu = self.scenario.physical, self.scenario.menu
        self.constraint = CapacityConstraint("fixed", weight=1.0)
        self.duals = self.constraint.duals(self.scenario.objective["max_violation_rate"])
        torch.manual_seed(seed)
        self.actor = build_actor(self.config, self.menu, settings.hidden)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=settings.lr, eps=1e-5)
        self.teacher = CentralEdfPolicy(self.config, self.menu)
        self.rng = np.random.default_rng([seed, 0x1A1])
        self.sampler = torch.Generator().manual_seed(seed + 1)
        self.shuffle = torch.Generator().manual_seed(seed + 2)
        self.actions = num_actions(self.config, self.menu)
        self.data = Episodes()
        self.instances_drawn: list[int] = []

    # ---------------------------------------------------------------- collect
    def _new_episode(self):
        index = splits.sample_train_index(self.rng)
        self.instances_drawn.append(index)
        topology, traffic = self.scenario.instance(index)
        seeds = splits_v2.training_fading_seeds(self.rng, self.num_ues)
        env = self.scenario.make_env(topology, traffic, seeds, self.settings.episode_slots)
        memory = LocalMemory(self.config, self.menu, self.num_ues, self.duals)
        memory.reset(np.zeros(self.num_ues))
        raw = env.reset()
        obs = stack(raw)
        memory.update_duals(memory.observe(obs))
        return env, memory, raw, obs

    def collect(self, by_teacher: bool) -> dict:
        """Run num_envs environments for episodes_per_env episodes; label every state."""
        s, n = self.settings, self.num_ues
        stats = {"active": 0, "agree": 0, "tx": 0, "collided": 0, "settled": 0, "violated": 0}
        for _ in range(s.episodes_per_env):
            episodes = [self._new_episode() for _ in range(s.num_envs)]
            ues = s.num_envs * n
            hidden = self.actor.initial_hidden(ues)
            buf = {
                "tokens": np.zeros((s.episode_slots, ues, self.config.num_subchannels,
                                    token_dim(self.menu, own_csi=True)), np.float32),
                "context": np.zeros((s.episode_slots, ues, context_dim()), np.float32),
                "legal": np.zeros((s.episode_slots, ues, self.actions), np.bool_),
                "labels": np.zeros((s.episode_slots, ues), np.int64),
                "active": np.zeros((s.episode_slots, ues), np.bool_),
            }  # fmt: skip
            for t in range(s.episode_slots):
                features = [memory.features(obs) for _, memory, _, obs in episodes]
                tokens = np.concatenate([f[0] for f in features])
                context = np.concatenate([f[1] for f in features])
                legal = np.concatenate([legal_mask(obs, self.actions) for *_, obs in episodes])
                labels = np.concatenate(
                    [teacher_labels(self.teacher, raw, self.menu) for _, _, raw, _ in episodes]
                )
                active = legal[:, 1:].any(axis=-1)
                with torch.no_grad():
                    logits, hidden = self.actor.step(
                        torch.from_numpy(tokens), torch.from_numpy(context), hidden
                    )
                    probs = masked_log_probs(logits, torch.from_numpy(legal)).exp()
                    sampled = torch.multinomial(probs, 1, generator=self.sampler).squeeze(-1)
                executed = labels if by_teacher else sampled.numpy()
                buf["tokens"][t], buf["context"][t], buf["legal"][t] = tokens, context, legal
                buf["labels"][t], buf["active"][t] = labels, active
                stats["active"] += int(active.sum())
                stats["agree"] += int((sampled.numpy()[active] == labels[active]).sum())
                for e, (env, memory, _, _) in enumerate(episodes):
                    rows = slice(e * n, (e + 1) * n)
                    actions = decode_actions(executed[rows], self.menu)
                    step = env.step(actions)
                    transmitting = actions[:, 1] > 0
                    channels = actions[transmitting, 0]
                    stats["tx"] += int(transmitting.sum())
                    stats["collided"] += int(
                        sum(c for c in np.bincount(channels, minlength=4) if c > 1)
                    )
                    obs = stack(step.observations)
                    settlement = memory.observe(obs)
                    memory.update_duals(settlement)
                    stats["settled"] += int(settlement.settled.sum())
                    stats["violated"] += int(settlement.violated.sum())
                    episodes[e] = (env, memory, step.observations, obs)
            for e in range(s.num_envs):
                rows = slice(e * n, (e + 1) * n)
                self.data.add(
                    *(buf[k][:, rows] for k in ("tokens", "context", "legal", "labels", "active"))
                )
        return {
            "executed_by": "teacher" if by_teacher else "student",
            "student_agreement": stats["agree"] / max(stats["active"], 1),
            "collision_share": stats["collided"] / max(stats["tx"], 1),
            "violation_rate": stats["violated"] / max(stats["settled"], 1),
        }

    # ------------------------------------------------------------------- train
    def train(self) -> dict:
        """epochs over the aggregated data; truncated BPTT per segment, hidden carried."""
        s = self.settings
        losses = []
        for _ in range(s.epochs):
            order = torch.randperm(len(self.data.items), generator=self.shuffle).tolist()
            total, count = 0.0, 0
            for b in range(0, len(order), s.batch_episodes):
                batch = self.data.batch(order[b : b + s.batch_episodes])
                hidden = self.actor.initial_hidden(batch["labels"].shape[1])
                for start in range(0, s.episode_slots, s.segment_slots):
                    stop = min(start + s.segment_slots, s.episode_slots)
                    loss, hidden = segment_loss(self.actor, batch, start, stop, hidden)
                    hidden = hidden.detach()
                    if loss is None:
                        continue
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), s.max_grad_norm)
                    self.optimizer.step()
                    total, count = total + float(loss.detach()), count + 1
            losses.append(total / max(count, 1))
        return {"loss_by_epoch": losses, "episodes": len(self.data.items)}

    # -------------------------------------------------------------- checkpoint
    def checkpoint(self, slots: int, iteration: int) -> dict:
        return {
            "kind": CHECKPOINT_KIND,
            "schema": CHECKPOINT_SCHEMA,
            "slots": slots,
            "iteration": iteration,
            "seed": self.seed,
            "num_ues": self.num_ues,
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
            },
            "actor": self.actor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def run(self, output: Path, log=print) -> dict:
        s = self.settings
        (output / "checkpoints").mkdir(parents=True)
        slots, started = 0, time.perf_counter()
        per_iteration = s.num_envs * s.episodes_per_env * s.episode_slots
        with open(output / "training.jsonl", "w") as handle:
            for iteration in range(1, s.iterations + 1):
                tick = time.perf_counter()
                collected = self.collect(by_teacher=iteration == 1)
                collect_s = time.perf_counter() - tick
                tick = time.perf_counter()
                trained = self.train()
                slots += per_iteration
                line = {
                    "iteration": iteration,
                    "slots": slots,
                    **collected,
                    **trained,
                    "collect_s": collect_s,
                    "train_s": time.perf_counter() - tick,
                }
                handle.write(json.dumps(line) + "\n")
                handle.flush()
                name = f"step-{slots:09d}"
                torch.save(self.checkpoint(slots, iteration), output / "checkpoints" / f"{name}.pt")
                log(
                    f"[iteration {iteration}] {collected['executed_by']} agreement "
                    f"{collected['student_agreement']:.3f} viol {collected['violation_rate']:.4f} "
                    f"loss {trained['loss_by_epoch'][-1]:.4f} ({line['collect_s']:.0f}+{line['train_s']:.0f} s)"
                )
        result = {
            "slots": slots,
            "iterations": s.iterations,
            "wall_s": time.perf_counter() - started,
            "episodes": len(self.data.items),
            "distinct_training_instances": len(set(self.instances_drawn)),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2))
        return result
