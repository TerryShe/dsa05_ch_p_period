"""Scenario v2.1: scenario v2.0 plus a 10 m minimum UE-AP distance (scenario-v2.1.md)."""

import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ks_2samp

from dsa_period.config import load_scenario
from dsa_period.topology import _SHADOWED_SAMPLING, generate_shadowed_topology

ROOT = Path(__file__).parents[1]
V20 = ROOT / "configs" / "scenario-v2.toml"
V21 = ROOT / "configs" / "scenario-v2.1.toml"
N = 20
INSTANCES = range(300)


@pytest.fixture(scope="module")
def pairs():
    """(v2.0, v2.1) deployments of instances 0-299 at N = 20."""
    old, new = load_scenario(V20, N), load_scenario(V21, N)
    return [(old.instance(i), new.instance(i)) for i in INSTANCES]


def test_the_config_equals_v20_apart_from_version_and_min_distance():
    old = tomllib.loads(V20.read_text())
    new = tomllib.loads(V21.read_text())
    assert new["experiment"] == {
        **old["experiment"],
        "id": "scenario-v2.1",
        "scenario_version": "2.1",
    }
    assert new["spatial"] == {**old["spatial"], "min_distance_m": 10.0}
    for section in old.keys() - {"experiment", "spatial"}:
        assert new[section] == old[section]
    assert new.keys() == old.keys()


def test_v20_has_no_min_distance_and_unchanged_topology_records():
    scenario = load_scenario(V20, 8)
    assert scenario.min_distance_m == 0.0
    topology, _ = scenario.instance(0)
    assert topology.min_distance_m == 0.0
    assert topology.to_record()["sampling_method"] == _SHADOWED_SAMPLING
    assert "min_distance_m" not in topology.to_record()["path"]["spatial"]


def test_every_ue_is_at_least_10_m_from_the_ap(pairs):
    for _, (topology, _) in pairs:
        assert topology.ue_ap_distances_m.min() >= 10.0
        assert topology.min_distance_m == 10.0
        assert topology.to_record()["sampling_method"].endswith("below 10 m")


def test_deployments_are_nested_across_n():
    big = load_scenario(V21, N)
    for i in range(40):
        topology, traffic = big.instance(i)
        for n in (2, 6, 8, 12):
            small, small_traffic = big.with_num_ues(n).instance(i)
            assert big.with_num_ues(n).min_distance_m == 10.0
            np.testing.assert_array_equal(small.ue_positions_m, topology.ue_positions_m[:n])
            np.testing.assert_array_equal(small.shadowing_db, topology.shadowing_db[:n])
            np.testing.assert_array_equal(small_traffic.phase_slots, traffic.phase_slots[:n])


def test_v20_ues_before_the_first_near_ue_are_unchanged(pairs):
    unchanged = 0
    for (old, old_traffic), (new, new_traffic) in pairs:
        near = np.flatnonzero(old.ue_ap_distances_m < 10.0)
        first = near[0] if len(near) else N
        np.testing.assert_array_equal(new.ue_positions_m[:first], old.ue_positions_m[:first])
        np.testing.assert_array_equal(new.shadowing_db[:first], old.shadowing_db[:first])
        np.testing.assert_array_equal(new.redraws[:first], old.redraws[:first])
        np.testing.assert_array_equal(new_traffic.phase_slots, old_traffic.phase_slots)
        if first < N:
            assert not np.array_equal(new.ue_positions_m[first], old.ue_positions_m[first])
        else:
            unchanged += 1
    # 1 - 0.99^20 = 18% of deployments change at N = 20 (scenario-v2.1.md §4)
    assert 0.70 < unchanged / len(pairs) < 0.92


def test_accepted_ues_follow_v20_conditioned_on_the_min_distance(pairs):
    old = np.concatenate([o.ue_ap_distances_m for (o, _), _ in pairs])
    new = np.concatenate([n.ue_ap_distances_m for _, (n, _) in pairs])
    old_shadow = np.concatenate([o.shadowing_db for (o, _), _ in pairs])
    new_shadow = np.concatenate([n.shadowing_db for _, (n, _) in pairs])
    keep = old >= 10.0
    assert ks_2samp(new, old[keep]).pvalue > 0.01
    assert ks_2samp(new_shadow, old_shadow[keep]).pvalue > 0.01
    # uniform by area on the 10-100 m annulus, before the coverage rule thins the edge
    inner = np.mean(new < 50.0)
    assert inner == pytest.approx((50**2 - 10**2) / (100**2 - 10**2), abs=0.03)


def test_min_distance_is_validated():
    v21 = load_scenario(V21, 4)
    with pytest.raises(ValueError, match="min_distance_m"):
        replace(v21, min_distance_m=-1.0)
    with pytest.raises(ValueError, match="min_distance_m"):
        replace(v21, min_distance_m=100.0)
    v1 = load_scenario(ROOT / "configs" / "scenario-v1-n8.toml")
    with pytest.raises(ValueError, match="v2 scenario"):
        replace(v1, min_distance_m=10.0)
    _, traffic = v21.instance(0)
    with pytest.raises(ValueError, match="min_distance_m"):
        generate_shadowed_topology(
            v21.spatial, v21.shadowing, 4, v21.covered(traffic), min_distance_m=100.0
        )
