"""Scenario v2.0: fading channel, shadowing with coverage, any N, own CSI, protocol v2 seeds."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import j0

from dsa_period import splits, splits_v2
from dsa_period.channel import ShadowedJakesChannel
from dsa_period.config import load_scenario
from dsa_period.physics import compute_step
from dsa_period.traffic import round_robin_counts

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "scenario-v2.toml"
FREEZE_V1 = json.loads((ROOT / "docs" / "research" / "scenario-v1-freeze.json").read_text())


@pytest.fixture(scope="module")
def v2():
    return load_scenario(CONFIG, 8)


def test_sos_fading_has_unit_power_finite_m_variance_and_jakes_autocorrelation():
    ues, fd, sinusoids = 4000, 10.0, 32
    channel = ShadowedJakesChannel(
        np.ones(ues),
        4,
        doppler_hz=fd,
        slot_duration_s=1e-3,
        ue_seeds=np.random.SeedSequence(7).spawn(ues),
        sinusoids=sinusoids,
    )
    lags = np.array([0, 10, 20, 40])
    h = channel.small_scale(lags)  # (lags, UEs, K): 16,000 independent processes
    power = np.abs(h[0]) ** 2
    assert power.mean() == pytest.approx(1.0, abs=0.02)
    assert power.var() == pytest.approx(1 - 1 / sinusoids, abs=0.05)
    for position, lag in enumerate(lags[1:], 1):
        correlation = np.mean(h[0] * np.conj(h[position])).real
        assert correlation == pytest.approx(j0(2 * np.pi * fd * lag * 1e-3), abs=0.02)


def test_gains_are_mean_gain_times_fading_power_at_the_slot_start():
    mean = np.array([1e-9, 2e-10, 5e-11])
    channel = ShadowedJakesChannel(
        mean, 4, doppler_hz=10.0, slot_duration_s=1e-3, ue_seeds=np.random.SeedSequence(1).spawn(3)
    )
    h = channel.small_scale(np.array([10, 300]))
    assert channel.gains(10) == pytest.approx(mean[:, None] * np.abs(h[0]) ** 2)
    assert channel.gains(300) == pytest.approx(mean[:, None] * np.abs(h[1]) ** 2)
    assert np.array_equal(channel.gain_block(250, 310)[50], channel.gains(300))
    channel.reset()
    assert np.array_equal(channel.gains(300), channel.gain_block(300, 301)[0])


def test_round_robin_counts_match_the_frozen_v1_variants():
    for variant in FREEZE_V1["variants"]:
        counts = variant["class_ue_counts"]
        assert round_robin_counts(3, variant["num_ues"]) == [
            counts["fast"],
            counts["medium"],
            counts["slow"],
        ]


def test_v2_loads_any_n_and_v1_files_refuse_an_n():
    scenario = load_scenario(CONFIG, 13)
    assert scenario.physical.num_ues == 13 and scenario.is_v2
    assert [c.num_ues for c in scenario.traffic_classes] == [5, 4, 4]
    assert scenario.horizon_slots == 20_200 and scenario.warmup_slots == 200
    assert scenario.objective["max_violation_rate"] == 0.01
    with pytest.raises(ValueError):
        load_scenario(CONFIG)
    with pytest.raises(ValueError):
        load_scenario(ROOT / "configs" / "scenario-v1-n8.toml", 8)


def test_evaluation_packet_counts_give_integral_allowances(v2):
    _, traffic = v2.instance(0)
    due = traffic.packets_due(v2.warmup_slots, v2.horizon_slots)
    assert due.tolist() == [2000, 1000, 500, 2000, 1000, 500, 2000, 1000]
    for eps in (0.01, 0.05):
        assert np.allclose(eps * due, np.round(eps * due))


def test_deployments_topology_traffic_and_fading_are_nested_across_n(v2):
    small = v2.with_num_ues(4)
    t8, f8 = v2.instance(5)
    t4, f4 = small.instance(5)
    assert np.array_equal(t8.ue_positions_m[:4], t4.ue_positions_m)
    assert np.array_equal(t8.shadowing_db[:4], t4.shadowing_db)
    assert np.array_equal(f8.phase_slots[:4], f4.phase_slots)
    assert f8.class_index.tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    c8 = v2.make_channel(t8, splits_v2.fading_seeds("dev", 8, 5, 0))
    c4 = small.make_channel(t4, splits_v2.fading_seeds("dev", 4, 5, 0))
    assert np.array_equal(c8.gain_block(0, 300)[:, :4], c4.gain_block(0, 300))


def test_every_ue_meets_the_coverage_rule(v2):
    scenario = load_scenario(CONFIG, 32)
    for index in range(3):
        topology, traffic = scenario.instance(index)
        rule = scenario.covered(traffic)
        assert all(rule(ue, g) for ue, g in enumerate(topology.ue_ap_gain_linear))
        path = topology.path.ue_ap_gain_linear
        assert topology.ue_ap_gain_linear == pytest.approx(
            path * 10 ** (topology.shadowing_db / 10)
        )


def test_own_gain_is_the_gain_the_environment_uses_in_that_slot(v2):
    topology, traffic = v2.instance(1)
    env = v2.make_env(topology, traffic, splits_v2.fading_seeds("dev", 8, 1, 0), horizon_slots=60)
    observations = env.reset()
    rng = np.random.default_rng(0)
    for slot in range(60):
        gains = np.array([obs.own_gain_w for obs in observations])
        assert np.array_equal(gains, env.channel.gains(slot))
        actions = np.column_stack([rng.integers(4, size=8), rng.integers(6, size=8)])
        step = env.step(actions)
        expected = compute_step(v2.physical, v2.menu, gains, step.physical.actions)
        assert np.array_equal(step.physical.service_bits, expected.service_bits)
        observations = step.observations


def test_v1_observations_carry_no_own_gain():
    scenario = load_scenario(ROOT / "configs" / "scenario-v1-n8.toml")
    topology, traffic = scenario.instance(0)
    assert all(obs.own_gain_w is None for obs in scenario.make_env(topology, traffic).reset())
    with pytest.raises(ValueError):
        scenario.make_env(topology, traffic, splits_v2.fading_seeds("dev", 8, 0, 0))


def test_v2_environment_needs_fading_seeds(v2):
    topology, traffic = v2.instance(0)
    with pytest.raises(ValueError):
        v2.make_env(topology, traffic)


def test_protocol_v2_seeds_are_shared_nested_and_distinct_from_v1():
    a = splits_v2.execution_generators("validation", 8, 10_003, 0)
    b = splits_v2.execution_generators("validation", 4, 10_003, 0)
    assert [g.integers(2**32) for g in a[:4]] == [g.integers(2**32) for g in b]
    v1 = splits.execution_generators("validation", 8, 10_003, 0)
    fresh = splits_v2.execution_generators("validation", 8, 10_003, 0)
    assert [g.integers(2**32) for g in v1] != [g.integers(2**32) for g in fresh]
    f1 = splits_v2.fading_seeds("test", 8, 20_000, 0)
    f2 = splits_v2.fading_seeds("test", 8, 20_000, 0)
    assert [s.generate_state(2).tolist() for s in f1] == [s.generate_state(2).tolist() for s in f2]
    with pytest.raises(ValueError):
        splits_v2.fading_seeds("test", 8, 20_000, 1)  # one execution per test instance
    with pytest.raises(ValueError):
        splits_v2.fading_seeds("test", 8, 10_000, 0)  # not a test instance


def test_v2_instances_differ_from_v1_instances(v2):
    v1 = load_scenario(ROOT / "configs" / "scenario-v1-n8.toml")
    for index in (0, 20_000):
        assert not np.allclose(
            v1.instance(index)[0].ue_positions_m, v2.instance(index)[0].ue_positions_m
        )


def test_doppler_sensitivity_keeps_instances_and_changes_only_the_fading(v2):
    slow = v2.with_doppler(2.0)
    assert slow.fading.doppler_hz == 2.0 and slow.instance(3)[1].phase_slots.tolist() == (
        v2.instance(3)[1].phase_slots.tolist()
    )
    topology, _ = v2.instance(3)
    seeds = splits_v2.fading_seeds("dev", 8, 3, 0)
    fast_gain = v2.make_channel(topology, seeds).gains(100)
    slow_gain = slow.make_channel(topology, seeds).gains(100)
    assert not np.allclose(fast_gain, slow_gain, rtol=1e-3, atol=0.0)


def test_methods_and_references_see_bit_identical_fading(v2):
    topology, traffic = v2.instance(2)
    seeds = splits_v2.fading_seeds("validation", 8, 10_002, 0)
    env = v2.make_env(topology, traffic, seeds, horizon_slots=50)
    env.reset()
    reference = v2.make_channel(topology, splits_v2.fading_seeds("validation", 8, 10_002, 0))
    assert np.array_equal(env.channel.gain_block(0, 600), reference.gain_block(0, 600))
