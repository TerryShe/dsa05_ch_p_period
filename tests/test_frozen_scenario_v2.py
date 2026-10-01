"""Guard the frozen scenario v2.0 and MARL protocol v2 (2026-10-01).

Changing any frozen value requires a new scenario or protocol version in new
files, not an edit of configs/scenario-v2.toml or of the generating code.
"""

import hashlib
import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

from dsa_period import splits_v2
from dsa_period.capacity import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, EPSILONS
from dsa_period.config import load_scenario
from dsa_period.rules_v2 import CALIBRATION_TABLE

ROOT = Path(__file__).parents[1]
SCENARIO = json.loads((ROOT / "docs" / "research" / "scenario-v2-freeze.json").read_text())
PROTOCOL = json.loads((ROOT / "docs" / "research" / "marl-protocol-v2.json").read_text())
CONFIG = ROOT / SCENARIO["config_path"]
EVERY_N = sorted(int(n) for n in PROTOCOL["digests"])


def _freeze_module():
    spec = importlib.util.spec_from_file_location("freeze", ROOT / "scripts/freeze_scenario_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scenario_manifest_is_frozen_and_matches_the_config_bytes():
    assert SCENARIO["status"] == "frozen" and SCENARIO["scenario_version"] == "2.0"
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == SCENARIO["config_sha256"]
    assert tomllib.loads(CONFIG.read_text()) == SCENARIO["parameters"]


def test_frozen_config_loads_with_the_recorded_derived_values():
    derived = SCENARIO["derived"]
    scenario = load_scenario(CONFIG, 12)
    assert 1 + scenario.physical.num_subchannels * (scenario.menu.num_levels - 1) == 21
    assert derived["num_joint_actions"] == 21
    assert scenario.horizon_slots - scenario.warmup_slots == derived["evaluation_slots"] == 20_000
    assert derived["packets_per_ue"] == {"fast": 2000, "medium": 1000, "slow": 500}
    assert derived["allowance"]["0.01"] == {"fast": 20, "medium": 10, "slow": 5}
    _, traffic = scenario.instance(0)
    assert traffic.offered_bits_per_slot() == pytest.approx(12 * 614.4)


def test_protocol_manifest_matches_the_code():
    assert PROTOCOL["status"] == "frozen" and PROTOCOL["protocol_id"] == splits_v2.PROTOCOL_ID
    assert {k: (v["start"], v["stop"]) for k, v in PROTOCOL["splits"].items()} == splits_v2.SPLITS
    assert PROTOCOL["evaluation_repeats"] == splits_v2.EVALUATION_REPEATS
    assert PROTOCOL["split_codes"] == splits_v2.SPLIT_CODES
    assert PROTOCOL["execution_seeds"]["base"] == splits_v2.EXECUTION_SEED_BASE
    assert PROTOCOL["fading_seeds"]["base"] == splits_v2.FADING_SEED_BASE
    assert PROTOCOL["epsilons"] == list(EPSILONS)
    assert PROTOCOL["bootstrap"] == {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED}
    assert PROTOCOL["calibration_table"] == {k: list(v) for k, v in CALIBRATION_TABLE.items()}
    capacity = load_scenario(CONFIG, 2).capacity
    assert PROTOCOL["grid"] == capacity["grid"] == [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
    assert PROTOCOL["reference_extra"] == capacity["reference_extra"] == [36, 40]
    assert EVERY_N == sorted({*capacity["grid"], *capacity["reference_extra"]})


@pytest.mark.parametrize("num_ues", EVERY_N)
def test_validation_test_and_train_probe_instances_are_unchanged(num_ues):
    scenario = load_scenario(CONFIG, num_ues)
    recorded = PROTOCOL["digests"][str(num_ues)]
    for split in ("validation", "test"):
        assert (
            splits_v2.instance_digest(scenario, splits_v2.indices(split), split)
            == (recorded[split])
        ), split
    assert _freeze_module().train_probe_digest(scenario) == recorded["train_probe"]


def test_the_calibration_amendment_only_appends_to_the_preregistered_table():
    original = {
        "opportunistic": [f"opportunistic:theta={t:g}" for t in (1, 1.5, 2, 3, 4)],
        "paced_sticky_csi": [f"paced_sticky_csi:margin={m:g}" for m in (0.5, 1, 2, 4, 8)],
    }
    for family, specs in original.items():
        assert list(CALIBRATION_TABLE[family][: len(specs)]) == specs
    assert [a["date"] for a in PROTOCOL["amendments"]] == ["2026-10-01"]
