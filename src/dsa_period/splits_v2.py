"""Instance splits, execution seeds and fading seeds of MARL protocol v2 (scenario v2.0).

Protocol v2 keeps the index ranges of protocol v1 (``splits.SPLITS``). Scenario
v2.0 uses new topology and phase seed bases, so every v2 instance differs from
every v1 instance, and the v1 test and confirmation sets are never reused.

Each evaluation episode (split, instance, repeat) has one execution seed and one
fading seed per UE, identical for every method (common random numbers). They are
spawned per UE from a sequence that does not contain N: UE i gets the same seeds
whatever the number of UEs, so together with the nested topology and traffic
(``topology.generate_shadowed_topology``, ``traffic.build_round_robin_pattern``)
the deployment with n UEs is the first n UEs of every larger deployment.

Frozen on 2026-10-01 with scenario v2.0 (docs/research/marl-protocol-v2.md); the
manifest marl-protocol-v2.json and tests/test_frozen_scenario_v2.py guard it.
"""

import hashlib
import json

import numpy as np

from dsa_period import splits

PROTOCOL_ID = "marl-protocol-v2"
SPLITS = splits.SPLITS
SPLIT_CODES = splits.SPLIT_CODES
EVALUATION_REPEATS = {"dev": 1, "validation": 1, "test": 1}
EXECUTION_SEED_BASE = 20_260_930
FADING_SEED_BASE = 20_261_001


def indices(split: str) -> range:
    return splits.indices(split)


def require(split: str, index: int) -> int:
    return splits.require(split, index)


def _ue_sequences(
    base: int, split: str, instance: int, repeat: int, num_ues: int
) -> list[np.random.SeedSequence]:
    if split not in EVALUATION_REPEATS:
        raise ValueError(f"protocol v2 evaluates only {tuple(EVALUATION_REPEATS)}")
    require(split, instance)
    if not 0 <= repeat < EVALUATION_REPEATS[split]:
        raise ValueError(f"repeat {repeat} is outside the {split} evaluation contract")
    if num_ues < 1:
        raise ValueError("num_ues must be positive")
    parent = np.random.SeedSequence([base, SPLIT_CODES[split], instance, repeat])
    return parent.spawn(num_ues)


def execution_generators(
    split: str, num_ues: int, instance: int, repeat: int
) -> list[np.random.Generator]:
    """Per-UE policy RNGs of one evaluation episode, identical for every method."""
    return [
        np.random.Generator(np.random.PCG64(seq))
        for seq in _ue_sequences(EXECUTION_SEED_BASE, split, instance, repeat, num_ues)
    ]


def fading_seeds(
    split: str, num_ues: int, instance: int, repeat: int
) -> list[np.random.SeedSequence]:
    """Per-UE fading seeds of one evaluation episode, shared by methods and references."""
    return _ue_sequences(FADING_SEED_BASE, split, instance, repeat, num_ues)


def training_fading_seeds(rng: np.random.Generator, num_ues: int) -> list[np.random.SeedSequence]:
    """Fresh per-UE fading seeds for one training episode, drawn from the trainer's RNG."""
    entropy = [int(value) for value in rng.integers(0, 2**32, size=4, dtype=np.uint64)]
    return np.random.SeedSequence(entropy).spawn(num_ues)


def instance_digest(scenario, instance_indices, split: str) -> str:
    """SHA-256 over the generated instances and their fading, in order.

    Covers the topology arrays (positions, path gain, shadowing), the traffic
    arrays and every UE's complex fading at slots 0-7 of the episode, so the
    digest changes whenever a generated episode changes.
    """
    digest = hashlib.sha256()
    n = scenario.physical.num_ues
    for index in instance_indices:
        topology, traffic = scenario.instance(index)
        record = traffic.to_record()
        channel = scenario.make_channel(topology, fading_seeds(split, n, index, 0))
        fading = np.asarray(channel.small_scale(np.arange(8)), dtype="<c16")
        payload = {
            "index": index,
            "topology": topology.path.to_record()["array_sha256"],
            "shadowing": topology.to_record()["shadowing_sha256"],
            "traffic": {
                key: record[key]
                for key in ("class_index", "period_slots", "packet_bits", "phase_slots")
            },
            "fading": hashlib.sha256(fading.tobytes()).hexdigest(),
        }
        digest.update(json.dumps(payload, sort_keys=True).encode())
    return digest.hexdigest()
