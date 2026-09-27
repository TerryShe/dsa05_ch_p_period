"""Frozen instance splits and evaluation seeds of MARL protocol v1.

Instance i of a scenario uses topology seed ``spatial.topology_seed + i`` and
phase seed ``traffic.phase_seed + i`` (see ``config.Scenario.instance``). The
splits are disjoint half-open index ranges shared by every N variant, so their
seeds never overlap. Training may only draw from ``train``; model selection may
only read ``validation``; ``test`` is read once a model is frozen. ``reserve``
stays untouched for a later confirmation set. ``dev`` holds every instance
already used for calibration, references and debugging (0-99 among them).

Confirmation sets (protocol section 7) are held-out slices of ``reserve``, taken
once the test split has been read. They are evaluated like the test split, draw
their execution seeds with the reserve pool's code, and leave the frozen splits
and repeats above untouched (``split_of`` still reports them as "reserve").
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

# confirm1: EXP-DSA05-005 (docs/experiment-designs/005-xr-sbpc-baseline_20260926.md).
CONFIRMATION_SETS: dict[str, tuple[int, int]] = {"confirm1": (30_000, 30_200)}
CONFIRMATION_REPEATS = 2
HELD_OUT = ("test", *CONFIRMATION_SETS)


def _bounds(split: str) -> tuple[int, int]:
    if split in SPLITS:
        return SPLITS[split]
    if split in CONFIRMATION_SETS:
        return CONFIRMATION_SETS[split]
    raise ValueError(f"unknown split {split!r}; choose from {(*SPLITS, *CONFIRMATION_SETS)}")


def indices(split: str) -> range:
    start, stop = _bounds(split)
    return range(start, stop)


def evaluation_repeats(split: str) -> int:
    """Stochastic executions per evaluated instance of a split or confirmation set."""
    if split in CONFIRMATION_SETS:
        return CONFIRMATION_REPEATS
    return EVALUATION_REPEATS.get(split, 1)


def seed_code(split: str) -> int:
    """Execution-seed code: a confirmation set uses the code of the reserve pool."""
    return SPLIT_CODES["reserve"] if split in CONFIRMATION_SETS else SPLIT_CODES[split]


def split_of(index: int) -> str | None:
    for name, (start, stop) in SPLITS.items():
        if start <= index < stop:
            return name
    return None


def require(split: str, index: int) -> int:
    """Return ``index`` if it belongs to ``split``; refuse it otherwise."""
    start, stop = _bounds(split)
    if not start <= index < stop:
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
    if not 0 <= repeat < evaluation_repeats(split):
        raise ValueError(f"repeat {repeat} is outside the {split} evaluation contract")
    sequence = np.random.SeedSequence(
        [EXECUTION_SEED_BASE, seed_code(split), num_ues, instance, repeat]
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
