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

Sets registered after the freeze leave the frozen splits, codes and repeats
untouched (EXP-DSA05-008, docs/research/exp008-registration.md and
exp008-sets.json):

* screening sets: fixed slices of ``dev`` that select checkpoints. They are
  ordinary dev episodes (code 0), read only through ``screening_instances``;
  ``check_dev_range`` refuses any other dev use that overlaps them.
* confirmation sets: slices of ``reserve``, read once after a model is frozen.
  Their seeds use the reserve code 3 and the same per-UE, N-free scheme.
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

# Checkpoint screening sets: EXP-DSA05-008 (design §4.1) on dev 2000-2255 at N = 6, 8;
# EXP-DSA05-009 (design §3) on dev 3000-3255 at every training N (an evaluation may
# cover a subset of the registered grid: a per-N model is evaluated at its own N).
SCREENING_SETS: dict[str, dict] = {
    "exp008-screen256": {"range": (2_000, 2_256), "grid": (6, 8)},
    "exp009-screen256": {"range": (3_000, 3_256), "grid": (2, 4, 6, 8, 10, 12, 16, 20)},
    # EXP-DSA05-010 (design §3.1): the diagnostic evaluation, dev 4000-4511 at N = 6, 8, 10.
    "exp010-eval512": {"range": (4_000, 4_512), "grid": (6, 8, 10)},
    # EXP-DSA05-011 (design §4.2): the evaluation of the urgency summary, dev 5000-5511.
    "exp011-eval512": {"range": (5_000, 5_512), "grid": (6, 8, 10, 12, 16)},
}
# Confirmation sets, one execution each: v2-confirm1 (EXP-008 §7.1, reserve 30000-30499)
# and v2-confirm2 (EXP-009 §3, reserve 30500-30999).
CONFIRMATION_SETS: dict[str, tuple[int, int]] = {
    "v2-confirm1": (30_000, 30_500),
    "v2-confirm2": (30_500, 31_000),
}
CONFIRMATION_REPEATS = 1
HELD_OUT = ("test", *CONFIRMATION_SETS)


def _bounds(split: str) -> tuple[int, int]:
    if split in CONFIRMATION_SETS:
        return CONFIRMATION_SETS[split]
    if split in SPLITS:
        return SPLITS[split]
    raise ValueError(f"unknown protocol-v2 split {split!r}")


def indices(split: str) -> range:
    start, stop = _bounds(split)
    return range(start, stop)


def require(split: str, index: int) -> int:
    start, stop = _bounds(split)
    if not start <= index < stop:
        raise ValueError(f"instance {index} is not in the {split} split")
    return index


def evaluation_repeats(split: str) -> int:
    """Executions per evaluated instance of a frozen split or a confirmation set."""
    if split in CONFIRMATION_SETS:
        return CONFIRMATION_REPEATS
    if split in EVALUATION_REPEATS:
        return EVALUATION_REPEATS[split]
    raise ValueError(f"protocol v2 evaluates only {(*EVALUATION_REPEATS, *CONFIRMATION_SETS)}")


def screening_instances(name: str) -> list[int]:
    """The dev instances of a registered screening set, in order."""
    start, stop = SCREENING_SETS[name]["range"]
    return [require("dev", i) for i in range(start, stop)]


def check_dev_range(instances) -> None:
    """Refuse a plain dev evaluation that touches a registered screening set."""
    for name, entry in SCREENING_SETS.items():
        start, stop = entry["range"]
        hit = [i for i in instances if start <= i < stop]
        if hit:
            raise ValueError(
                f"dev instances {hit[0]}..{hit[-1]} belong to the screening set {name}; "
                "read it only through its registered entry point"
            )


def _ue_sequences(
    base: int, split: str, instance: int, repeat: int, num_ues: int
) -> list[np.random.SeedSequence]:
    repeats = evaluation_repeats(split)
    require(split, instance)
    if not 0 <= repeat < repeats:
        raise ValueError(f"repeat {repeat} is outside the {split} evaluation contract")
    if num_ues < 1:
        raise ValueError("num_ues must be positive")
    code = SPLIT_CODES["reserve"] if split in CONFIRMATION_SETS else SPLIT_CODES[split]
    parent = np.random.SeedSequence([base, code, instance, repeat])
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
