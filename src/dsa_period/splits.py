"""Frozen instance splits and evaluation seeds of MARL protocol v1.

Instance i of a scenario uses topology seed ``spatial.topology_seed + i`` and
phase seed ``traffic.phase_seed + i`` (see ``config.Scenario.instance``). The
splits are disjoint half-open index ranges shared by every N variant, so their
seeds never overlap. Training may only draw from ``train``; model selection may
only read ``validation``; ``test`` is read once a model is frozen. ``reserve``
stays untouched for a later confirmation set. ``dev`` holds every instance
already used for calibration, references and debugging (0-99 among them).
"""

import hashlib
import json
from collections.abc import Iterable

import numpy as np

PROTOCOL_ID = "marl-protocol-v1"

SPLITS: dict[str, tuple[int, int]] = {
    "dev": (0, 10_000),
    "validation": (10_000, 10_064),
    "test": (20_000, 20_200),
    "reserve": (30_000, 40_000),
    "train": (1_000_000, 2_000_000),
}

# Stochastic executions per evaluated instance.
EVALUATION_REPEATS = {"validation": 1, "test": 2}

# Execution seeds are shared by every method (common random numbers), so method
# comparisons on the same (split, N, instance, repeat) are paired.
EXECUTION_SEED_BASE = 20_260_925
SPLIT_CODES = {"dev": 0, "validation": 1, "test": 2, "reserve": 3, "train": 4}


def indices(split: str) -> range:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {tuple(SPLITS)}")
    start, stop = SPLITS[split]
    return range(start, stop)


def split_of(index: int) -> str | None:
    for name, (start, stop) in SPLITS.items():
        if start <= index < stop:
            return name
    return None


def require(split: str, index: int) -> int:
    """Return ``index`` if it belongs to ``split``; refuse it otherwise."""
    if split_of(index) != split:
        raise ValueError(f"instance {index} is not in the {split} split")
    return index


def sample_train_index(rng: np.random.Generator) -> int:
    """Draw one training instance index uniformly from the training pool."""
    start, stop = SPLITS["train"]
    return int(rng.integers(start, stop))


def execution_generators(
    split: str, num_ues: int, instance: int, repeat: int
) -> list[np.random.Generator]:
    """Per-UE policy RNGs of one evaluation episode, identical for every method."""
    require(split, instance)
    if not 0 <= repeat < EVALUATION_REPEATS.get(split, 1):
        raise ValueError(f"repeat {repeat} is outside the {split} evaluation contract")
    sequence = np.random.SeedSequence(
        [EXECUTION_SEED_BASE, SPLIT_CODES[split], num_ues, instance, repeat]
    )
    return [np.random.Generator(np.random.PCG64(child)) for child in sequence.spawn(num_ues)]


def instance_digest(scenario, instance_indices: Iterable[int]) -> str:
    """SHA-256 over the generated arrays of the given instances, in order.

    Uses the topology array hashes and the traffic arrays only, so the digest
    changes exactly when a generated instance changes, not with metadata.
    """
    digest = hashlib.sha256()
    for index in instance_indices:
        topology, traffic = scenario.instance(index)
        record = traffic.to_record()
        payload = {
            "index": index,
            "topology": topology.to_record()["array_sha256"],
            "traffic": {
                key: record[key]
                for key in ("class_index", "period_slots", "packet_bits", "phase_slots")
            },
        }
        digest.update(json.dumps(payload, sort_keys=True).encode())
    return digest.hexdigest()
