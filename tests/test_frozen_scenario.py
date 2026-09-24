"""Guard the frozen v1.0 working points (N=4/6/8/12) against accidental edits.

Changing any frozen value requires a new scenario version in new config files
and a new freeze manifest, not an edit of the scenario-v1-n*.toml files.
"""

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from dsa_period.config import load_scenario

ROOT = Path(__file__).parents[1]
MANIFEST = json.loads((ROOT / "docs" / "research" / "scenario-v1-freeze.json").read_text())
SHARED = MANIFEST["shared_parameters"]
VARIANTS = {variant["num_ues"]: variant for variant in MANIFEST["variants"]}


def test_manifest_freezes_the_four_ue_counts():
    assert MANIFEST["status"] == "frozen" and MANIFEST["scenario_version"] == "1.0"
    assert sorted(VARIANTS) == [4, 6, 8, 12]
    assert sorted(p.name for p in (ROOT / "configs").glob("scenario-v1-*.toml")) == sorted(
        Path(v["config_path"]).name for v in VARIANTS.values()
    )


@pytest.mark.parametrize("num_ues", [4, 6, 8, 12])
def test_frozen_config_bytes_match_the_manifest(num_ues):
    path = ROOT / VARIANTS[num_ues]["config_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == VARIANTS[num_ues]["config_sha256"]


@pytest.mark.parametrize("num_ues", [4, 6, 8, 12])
def test_variants_differ_only_in_ue_count_and_class_split(num_ues):
    variant = VARIANTS[num_ues]
    raw = tomllib.loads((ROOT / variant["config_path"]).read_text())
    assert raw["experiment"]["scenario_version"] == "1.0"
    assert raw["physical"] == {**SHARED["physical"], "num_ues": num_ues}
    for section in ("spatial", "power", "run", "objective"):
        assert raw[section] == SHARED[section], section
    assert raw["traffic"]["phase_seed"] == SHARED["traffic"]["phase_seed"]
    assert raw["traffic"]["classes"] == [
        {**shared, "num_ues": variant["class_ue_counts"][shared["name"]]}
        for shared in SHARED["traffic"]["classes"]
    ]


@pytest.mark.parametrize("num_ues", [4, 6, 8, 12])
def test_frozen_variants_load_with_the_recorded_derived_values(num_ues):
    variant = VARIANTS[num_ues]
    derived = MANIFEST["shared_derived"]
    scenario = load_scenario(ROOT / variant["config_path"])
    _, traffic = scenario.instance(0)
    actions = 1 + scenario.physical.num_subchannels * (scenario.menu.num_levels - 1)
    assert actions == derived["num_joint_actions"] == 21
    assert traffic.offered_bits_per_slot() == pytest.approx(variant["offered_bits_per_slot"])
    assert traffic.offered_bits_per_slot() == pytest.approx(num_ues * 614.4)
    assert scenario.horizon_slots - scenario.warmup_slots == derived["evaluation_slots"]
