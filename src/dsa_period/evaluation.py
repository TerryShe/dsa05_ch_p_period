"""Evaluation under MARL protocol v1 (docs/research/marl-protocol-v1.md).

``run_episode`` plays one protocol episode with the frozen execution seeds and
scores it; ``aggregate`` and ``paired_difference`` produce the protocol's point
estimates and two-level (training seed, then instance) bootstrap intervals.
Energies are normalized per instance by the archived central references.
"""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dsa_period import splits
from dsa_period.config import Scenario
from dsa_period.metrics import window_summary
from dsa_period.policies import Policy

REFERENCE_DIR = Path(__file__).parents[2] / "docs" / "results" / "protocol-v1-references_20260924"
FEASIBLE_EPISODE_RATE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_926
METRICS = ("feasible", "e_over_ub", "e_over_lb", "ocr", "worst_ue_ocr", "mw_per_ue")


def load_references(split: str, num_ues: int, directory: Path = REFERENCE_DIR) -> dict:
    """Per-instance strict window energies {instance: (lower_j, upper_j)} of a split."""
    record = json.loads((directory / f"{split}.json").read_text())
    runs = [run for run in record["runs"] if run["num_ues"] == num_ues]
    if len(runs) != 1:
        raise ValueError(f"no unique {split} reference for N={num_ues}")
    table = {row["instance"]: (row["lower_j"], row["upper_j"]) for row in runs[0]["per_instance"]}
    if sorted(table) != list(splits.indices(split)):
        raise ValueError(f"{split} references do not cover exactly the frozen split")
    return table


def run_episode(
    scenario: Scenario,
    policy: Policy,
    split: str,
    instance: int,
    repeat: int,
    references: Mapping[int, tuple[float, float]] | None = None,
) -> dict:
    """One protocol episode: frozen instance, frozen per-UE execution RNGs, full horizon."""
    num_ues = scenario.physical.num_ues
    rngs = splits.execution_generators(split, num_ues, instance, repeat)
    topology, traffic = scenario.instance(instance)
    env = scenario.make_env(topology, traffic)
    policy.reset(rngs)
    observations = env.reset()
    while not env.truncated:
        observations = env.step(policy.act(observations)).observations
    summary = window_summary(
        env.job_records(),
        env.energy_by_slot(),
        warmup_slots=scenario.warmup_slots,
        horizon_slots=scenario.horizon_slots,
        slot_duration_s=scenario.physical.slot_duration_s,
        num_ues=num_ues,
    )
    cap = scenario.objective["max_violation_rate"]
    rows = summary["per_ue"]
    # Integer test avoids float noise: violations <= floor(cap * packets).
    allowed = [math.floor(cap * row["packets"] + 1e-9) for row in rows]
    violations = [row["packets"] - row["on_time"] for row in rows]
    result = {
        "instance": instance,
        "repeat": repeat,
        "feasible": all(v <= a for v, a in zip(violations, allowed, strict=True)),
        "per_ue_violation_rate": [
            v / row["packets"] for v, row in zip(violations, rows, strict=True)
        ],
        "ocr": summary["ocr"],
        "worst_ue_ocr": summary["worst_ue_ocr"],
        "packet_energy_j": summary["packet_energy_j"],
        "mw_per_ue": 1e3 * summary["average_power_per_ue_w"],
        "per_ue_mw": [1e3 * row["average_power_w"] for row in rows],
        "distance_m": topology.ue_ap_distances_m.tolist(),
        "e_over_lb": None,
        "e_over_ub": None,
    }
    if references is not None:
        lower, upper = references[instance]
        result["e_over_lb"] = summary["packet_energy_j"] / lower
        result["e_over_ub"] = summary["packet_energy_j"] / upper
    return result


def evaluate_policy(
    scenario: Scenario,
    make: Callable[[], Policy],
    split: str,
    instance_indices: Sequence[int] | None = None,
    references: Mapping[int, tuple[float, float]] | None = None,
) -> list[dict]:
    """All protocol episodes of one policy (one training seed) on a split."""
    chosen = splits.indices(split) if instance_indices is None else instance_indices
    repeats = splits.EVALUATION_REPEATS.get(split, 1)
    return [
        run_episode(scenario, make(), split, instance, repeat, references)
        for instance in chosen
        for repeat in range(repeats)
    ]


def _matrix(episodes_by_seed: Mapping[str, Sequence[dict]], metric: str) -> tuple[list, NDArray]:
    """(instances, [seed, instance] array) with repeats averaged within each cell."""
    seeds = sorted(episodes_by_seed)
    instances = sorted({e["instance"] for e in episodes_by_seed[seeds[0]]})
    values = np.empty((len(seeds), len(instances)))
    for s, seed in enumerate(seeds):
        cells: dict[int, list[float]] = {}
        for episode in episodes_by_seed[seed]:
            cells.setdefault(episode["instance"], []).append(float(episode[metric]))
        if sorted(cells) != instances:
            raise ValueError("every training seed must cover the same instances")
        values[s] = [np.mean(cells[i]) for i in instances]
    return instances, values


def _bootstrap(values: NDArray, other: NDArray | None, resamples: int, seed: int) -> NDArray:
    """Two-level bootstrap of mean(values) [- mean(other)], instances resampled jointly."""
    rng = np.random.default_rng(seed)
    out = np.empty(resamples)
    chunk = 500
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        inst = rng.integers(values.shape[1], size=(size, values.shape[1]))
        seeds = rng.integers(values.shape[0], size=(size, values.shape[0]))
        stat = values[seeds[:, :, None], inst[:, None, :]].mean(axis=(1, 2))
        if other is not None:
            other_seeds = rng.integers(other.shape[0], size=(size, other.shape[0]))
            stat = stat - other[other_seeds[:, :, None], inst[:, None, :]].mean(axis=(1, 2))
        out[start : start + size] = stat
    return out


def _interval(samples: NDArray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def aggregate(
    episodes_by_seed: Mapping[str, Sequence[dict]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Protocol summary of one method: means, per-seed means and 95% intervals."""
    first = next(iter(episodes_by_seed.values()))
    metrics = [m for m in METRICS if first[0][m] is not None]
    result: dict = {"training_seeds": sorted(episodes_by_seed), "metrics": {}}
    for metric in metrics:
        instances, values = _matrix(episodes_by_seed, metric)
        result["metrics"][metric] = {
            "mean": float(values.mean()),
            "per_seed": dict(zip(sorted(episodes_by_seed), values.mean(axis=1).tolist())),
            "ci95": _interval(_bootstrap(values, None, resamples, seed)),
        }
    result["instances"] = len(instances)
    result["episodes"] = sum(len(v) for v in episodes_by_seed.values())
    violation = np.concatenate(
        [e["per_ue_violation_rate"] for v in episodes_by_seed.values() for e in v]
    )
    result["per_ue_violation_rate"] = {
        "p50": float(np.percentile(violation, 50)),
        "p95": float(np.percentile(violation, 95)),
        "max": float(violation.max()),
    }
    result["satisfies_constraint"] = result["metrics"]["feasible"]["mean"] >= FEASIBLE_EPISODE_RATE
    return result


def paired_difference(
    a: Mapping[str, Sequence[dict]],
    b: Mapping[str, Sequence[dict]],
    metric: str,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """mean(a) - mean(b) on the same instances with a two-level paired bootstrap."""
    instances_a, values_a = _matrix(a, metric)
    instances_b, values_b = _matrix(b, metric)
    if instances_a != instances_b:
        raise ValueError("paired comparison needs identical instances")
    samples = _bootstrap(values_a, values_b, resamples, seed)
    return {"mean": float(values_a.mean() - values_b.mean()), "ci95": _interval(samples)}


def select(summaries: Mapping[str, dict]) -> str:
    """Protocol selection rule over validation summaries (checkpoints or settings).

    Among candidates with feasibility rate >= 0.95, the lowest mean E/UB wins;
    if none qualifies, the highest feasibility rate, then the lowest E/UB.
    """
    if not summaries:
        raise ValueError("nothing to select from")

    def key(name: str) -> tuple:
        metrics = summaries[name]["metrics"]
        feasible = metrics["feasible"]["mean"]
        energy = metrics["e_over_ub"]["mean"]
        if feasible >= FEASIBLE_EPISODE_RATE:
            return (0, energy, name)
        return (1, -feasible, energy, name)

    return min(summaries, key=key)
