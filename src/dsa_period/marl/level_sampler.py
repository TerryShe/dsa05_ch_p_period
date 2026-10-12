"""Training-deployment samplers of EXP-DSA05-009 (design §5).

A level is a training-pool instance index at a fixed number of UEs N: its
topology, shadowing and traffic phases. Every visit draws fresh fading and
execution randomness, so a replay revisits the deployment, not a trajectory.
Each N group (the environments with that N) has its own sampler.

* ``uniform``: the trainer's own draw (``splits.sample_train_index``), so the
  EXP-006/008 path is unchanged bit for bit.
* ``stratified`` (S1): with probability ``hard_share`` a level from the hard
  stratum, hardness below ``tau[N]`` (rejection sampling); otherwise uniform.
* ``plr`` (P1, P2): prioritized level replay (Jiang et al., ICML 2021, eqs. 1-4)
  with a rolling buffer of at most ``buffer_size`` scored levels per group.
  With probability ``replay_probability`` (once ``min_fill`` levels are scored)
  a level is replayed from P = (1 - rho) P_S + rho P_C, with rank prioritization
  h = 1 / rank (ties share their mean rank) at temperature ``beta`` and the
  staleness distribution P_C; otherwise a new level is drawn uniformly. A full
  buffer keeps the highest scores: a new level replaces the lowest-scored one
  only if it scores higher. The fixed replay probability, buffer size and
  minimum fill are design choices for an unbounded pool, not the paper's.

The hardness label (``deployment_hardness``) is training-side and policy free:
the smallest large-scale service margin c_i / r_i over the UEs, c_i the
interference-free Shannon bits per slot at max power on the mean gain (path
loss and shadowing, no fading) and r_i the packet bits per period slot.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from dsa_period import splits
from dsa_period.physics import shannon_bits

SAMPLERS = ("uniform", "stratified", "plr")
SCORES = ("gae", "avoidable")


@dataclass(frozen=True)
class SamplerSettings:
    kind: str = "uniform"
    hard_share: float = 0.5
    tau: dict = field(default_factory=dict)  # str(N) -> hardness threshold
    max_rejections: int = 100_000
    score: str = "gae"
    replay_probability: float = 0.5
    buffer_size: int = 500
    min_fill: int = 50
    beta: float = 0.1
    rho: float = 0.1

    def __post_init__(self) -> None:
        if self.kind not in SAMPLERS:
            raise ValueError(f"sampler kind must be one of {SAMPLERS}")
        if self.score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}")
        if not (0 <= self.hard_share <= 1 and 0 <= self.replay_probability <= 1):
            raise ValueError("probabilities must lie in [0, 1]")
        if not (0 <= self.rho <= 1 and self.beta > 0 and 1 <= self.min_fill <= self.buffer_size):
            raise ValueError("invalid PLR settings")


def deployment_hardness(scenario, topology, traffic) -> float:
    """Smallest large-scale service margin over the UEs (lower is harder)."""
    top = scenario.menu.power_w[-1]
    bits = shannon_bits(
        scenario.physical, top * topology.ue_ap_gain_linear / scenario.physical.noise_power_w
    )
    need = traffic.packet_bits / traffic.period_slots
    return float(np.min(bits / need))


def rank_weights(scores: np.ndarray, beta: float) -> np.ndarray:
    """P_S of eq. (3) with h = 1 / rank (descending scores, ties at their mean rank)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(scores.size)
    ranks[order] = np.arange(1, scores.size + 1)
    for value in np.unique(scores):  # ties share their mean rank
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    weights = (1.0 / ranks) ** (1.0 / beta)
    return weights / weights.sum()


def staleness_weights(stamps: np.ndarray, clock: int) -> np.ndarray:
    """P_C of eq. (4): proportional to clock - C_i (uniform when all are fresh)."""
    stale = np.maximum(clock - stamps.astype(float), 0.0)  # stamps never exceed the clock
    total = stale.sum()
    return stale / total if total > 0 else np.full(stamps.size, 1.0 / stamps.size)


class LevelSampler:
    """Deployment sampler of one N group."""

    def __init__(self, settings: SamplerSettings, num_ues: int, seed: int, hardness) -> None:
        self.settings, self.num_ues = settings, num_ues
        self.rng = np.random.default_rng([seed, 0x9E7, num_ues])
        self.hardness = hardness  # index -> hardness at this N (stratified only)
        self.clock = 0  # episodes started in this group (PLR's c)
        self.levels: list[int] = []
        self.scores: list[float] = []
        self.stamps: list[int] = []
        self.counts = {"new": 0, "replay": 0, "hard": 0, "rejections": 0}
        self.visits: dict[int, int] = {}

    def next_level(self, draw_uniform: Callable[[], int]) -> tuple[int, bool, int]:
        """(instance index, replayed?, timestamp) of the next episode of this group."""
        s = self.settings
        self.clock += 1
        replayed = False
        if s.kind == "stratified" and self.rng.random() < s.hard_share:
            tau = s.tau[str(self.num_ues)]
            for _ in range(s.max_rejections):
                index = splits.sample_train_index(self.rng)
                self.counts["rejections"] += 1
                if self.hardness(index) < tau:
                    self.counts["hard"] += 1
                    break
            else:
                raise RuntimeError("no hard deployment found; check tau")
        elif (
            s.kind == "plr"
            and len(self.levels) >= s.min_fill
            and self.rng.random() < s.replay_probability
        ):
            prob = (1 - s.rho) * rank_weights(np.array(self.scores), s.beta)
            prob += s.rho * staleness_weights(np.array(self.stamps), self.clock)
            index = self.levels[int(self.rng.choice(len(self.levels), p=prob / prob.sum()))]
            replayed = True
            self.counts["replay"] += 1
        else:
            index = draw_uniform()
            self.counts["new"] += 1
        self.visits[index] = self.visits.get(index, 0) + 1
        return index, replayed, self.clock

    def record(self, index: int, score: float, stamp: int) -> None:
        """Score of a finished episode on ``index`` (PLR only), sampled at ``stamp``."""
        if self.settings.kind != "plr":
            return
        if index in self.levels:
            k = self.levels.index(index)
            self.scores[k], self.stamps[k] = score, stamp
            return
        if len(self.levels) < self.settings.buffer_size:
            self.levels.append(index)
            self.scores.append(score)
            self.stamps.append(stamp)
            return
        k = int(np.argmin(self.scores))
        if score > self.scores[k]:
            self.levels[k], self.scores[k], self.stamps[k] = index, score, stamp

    def diagnostics(self) -> dict:
        out = {**self.counts, "buffer": len(self.levels)}
        if self.scores:
            out["score_quantiles"] = np.quantile(self.scores, [0.0, 0.5, 0.9, 1.0]).tolist()
            out["score_zero_share"] = float(np.mean(np.array(self.scores) == 0))
        if self.visits:
            index = max(self.visits, key=self.visits.get)
            out["most_visited"] = [int(index), self.visits[index]]
            out["distinct_levels"] = len(self.visits)
        return out

    def state_dict(self) -> dict:
        return {
            "rng": self.rng.bit_generator.state,
            "clock": self.clock,
            "levels": list(self.levels),
            "scores": list(self.scores),
            "stamps": list(self.stamps),
            "counts": dict(self.counts),
            "visits": dict(self.visits),
        }

    def load_state_dict(self, state: dict) -> None:
        """Continue from ``state_dict`` (a resumed training run, EXP-DSA05-011B)."""
        self.rng.bit_generator.state = state["rng"]
        self.clock = int(state["clock"])
        self.levels, self.scores = list(state["levels"]), list(state["scores"])
        self.stamps = list(state["stamps"])
        self.counts = dict(state["counts"])
        self.visits = {int(k): int(v) for k, v in state["visits"].items()}
