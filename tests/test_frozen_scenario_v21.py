"""Guard the frozen scenario v2.1 (2026-10-14).

Changing any frozen value requires a new scenario version in new files, not an
edit of configs/scenario-v2.1.toml or of the generating code.
"""

import hashlib
import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

from dsa_period.config import load_scenario

ROOT = Path(__file__).parents[1]
MANIFEST = json.loads((ROOT / "docs" / "research" / "scenario-v2.1-freeze.json").read_text())
PREVIOUS = json.loads((ROOT / "docs" / "research" / "scenario-v2-freeze.json").read_text())
CONFIG = ROOT / MANIFEST["config_path"]
EVERY_N = sorted(int(n) for n in MANIFEST["digests"])


def _freeze_module():
    spec = importlib.util.spec_from_file_location(
        "freeze_v21", ROOT / "scripts/freeze_scenario_v21.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_is_frozen_and_matches_the_config_bytes():
    assert MANIFEST["status"] == "frozen" and MANIFEST["scenario_version"] == "2.1"
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == MANIFEST["config_sha256"]
    assert tomllib.loads(CONFIG.read_text()) == MANIFEST["parameters"]


def test_previous_version_is_the_frozen_v20():
    assert MANIFEST["previous"] == {
        "scenario_version": "2.0",
        "config_path": PREVIOUS["config_path"],
        "config_sha256": PREVIOUS["config_sha256"],
    }
    old = tomllib.loads((ROOT / PREVIOUS["config_path"]).read_text())
    new = MANIFEST["parameters"]
    assert new["spatial"] == {**old["spatial"], "min_distance_m": 10.0}
    assert {k: v for k, v in new.items() if k not in ("experiment", "spatial")} == {
        k: v for k, v in old.items() if k not in ("experiment", "spatial")
    }


def test_frozen_config_loads_with_the_recorded_derived_values():
    derived = MANIFEST["derived"]
    scenario = load_scenario(CONFIG, 12)
    assert scenario.min_distance_m == derived["min_distance_m"] == 10.0
    assert {k: v for k, v in derived.items() if k != "min_distance_m"} == PREVIOUS["derived"]
    assert EVERY_N == [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40]


@pytest.mark.parametrize("num_ues", EVERY_N)
def test_dev_and_train_probe_instances_are_unchanged(num_ues):
    assert _freeze_module().digests(num_ues) == MANIFEST["digests"][str(num_ues)]
