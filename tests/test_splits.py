"""Guard the frozen MARL protocol v1 splits, instance content and execution seeds."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from dsa_period import splits
from dsa_period.config import load_scenario

ROOT = Path(__file__).parents[1]
MANIFEST = json.loads((ROOT / "docs" / "research" / "marl-protocol-v1.json").read_text())
SCENARIO_FREEZE = json.loads((ROOT / "docs" / "research" / "scenario-v1-freeze.json").read_text())


def test_manifest_matches_the_split_module():
    assert MANIFEST["protocol_id"] == splits.PROTOCOL_ID and MANIFEST["status"] == "frozen"
    assert {k: (v["start"], v["stop"]) for k, v in MANIFEST["splits"].items()} == splits.SPLITS
    assert MANIFEST["evaluation_repeats"] == splits.EVALUATION_REPEATS
    assert MANIFEST["execution_seeds"]["base"] == splits.EXECUTION_SEED_BASE
    assert MANIFEST["execution_seeds"]["split_codes"] == splits.SPLIT_CODES


def test_splits_are_disjoint_and_the_used_instances_are_in_dev():
    for (a, (s1, e1)), (b, (s2, e2)) in combinations(splits.SPLITS.items(), 2):
        assert e1 <= s2 or e2 <= s1, f"{a} overlaps {b}"
    assert all(splits.split_of(i) == "dev" for i in range(100)), "reports used 0-99"
    assert splits.split_of(9_999_999) is None


def test_manifest_covers_every_frozen_scenario_variant():
    frozen = {(v["num_ues"], v["config_sha256"]) for v in SCENARIO_FREEZE["variants"]}
    assert {(v["num_ues"], v["config_sha256"]) for v in MANIFEST["variants"]} == frozen


@pytest.mark.parametrize("variant", MANIFEST["variants"], ids=lambda v: f"n{v['num_ues']}")
def test_validation_test_and_train_probe_instances_are_unchanged(variant):
    scenario = load_scenario(ROOT / variant["config_path"])
    assert (
        splits.instance_digest(scenario, splits.indices("validation"))
        == (variant["validation_digest"])
    )
    assert splits.instance_digest(scenario, splits.indices("test")) == variant["test_digest"]
    probe = MANIFEST["train_probe_indices"]
    assert all(splits.split_of(i) == "train" for i in probe)
    assert splits.instance_digest(scenario, probe) == variant["train_probe_digest"]


def test_require_refuses_indices_from_other_splits():
    assert splits.require("test", 20_000) == 20_000
    for split, index in (("test", 10_000), ("validation", 20_000), ("train", 5)):
        with pytest.raises(ValueError):
            splits.require(split, index)


def test_training_draws_stay_in_the_training_pool():
    rng = np.random.default_rng(0)
    draws = [splits.sample_train_index(rng) for _ in range(1000)]
    assert all(splits.split_of(i) == "train" for i in draws)
    assert len(set(draws)) > 990


def test_execution_seeds_are_reproducible_paired_and_distinct():
    def first_draws(*key):
        return [g.integers(1 << 30) for g in splits.execution_generators(*key)]

    assert first_draws("test", 8, 20_000, 0) == first_draws("test", 8, 20_000, 0)
    keys = [("test", 8, 20_000, 1), ("test", 8, 20_001, 0), ("test", 12, 20_000, 0)]
    base = first_draws("test", 8, 20_000, 0)
    assert all(first_draws(*key)[:8] != base for key in keys)
    assert len(set(base)) == 8, "UEs get distinct streams"
    with pytest.raises(ValueError):
        splits.execution_generators("test", 8, 20_000, 2)
    with pytest.raises(ValueError):
        splits.execution_generators("validation", 8, 20_000, 0)
