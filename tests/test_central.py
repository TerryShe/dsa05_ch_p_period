import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from conftest import MENU, build_env, physical

from dsa_period.central import (
    _compositions,
    act_central,
    central_reference,
    continuous_lower_bound_j,
    optimal_packet_plan,
)
from dsa_period.config import load_scenario
from dsa_period.experiments import central_reference as runner
from dsa_period.metrics import window_summary

N4 = str(Path(__file__).parents[1] / "configs" / "scenario-v1-n4.toml")
N8 = str(Path(__file__).parents[1] / "configs" / "scenario-v1-n8.toml")


def test_compositions_enumerate_every_bounded_count_vector():
    rows = _compositions(3, 2)
    assert len(rows) == math.comb(5, 2)
    assert {tuple(r) for r in rows} == {(a, b) for a in range(4) for b in range(4) if a + b <= 3}


def brute_force(bits, energy, packet, max_slots):
    """Try every per-slot level sequence, including OFF (level 0)."""
    best = None
    for seq in itertools.product(range(len(bits) + 1), repeat=max_slots):
        chosen = [lvl - 1 for lvl in seq if lvl]
        if sum(bits[c] for c in chosen) >= packet:
            cost = sum(energy[c] for c in chosen)
            best = cost if best is None else min(best, cost)
    return best


@pytest.mark.parametrize("seed", range(12))
def test_packet_plan_matches_brute_force_over_all_slot_sequences(seed):
    rng = np.random.default_rng(seed)
    levels = int(rng.integers(1, 4))
    bits = np.sort(rng.uniform(50, 500, levels)) + np.arange(levels)
    energy = np.sort(rng.uniform(1, 10, levels)) + np.arange(levels)
    max_slots = int(rng.integers(1, 6))
    packet = float(rng.uniform(0.2, 1.0) * bits[-1] * max_slots)
    plan = optimal_packet_plan(bits, energy, packet, max_slots)
    expected = brute_force(bits, energy, packet, max_slots)
    if expected is None:
        assert plan is None
        return
    assert plan.energy_j == pytest.approx(expected)
    assert plan.slots <= max_slots
    assert plan.mutual_information_bits >= packet
    assert np.dot(plan.counts, bits) == pytest.approx(plan.mutual_information_bits)


def test_packet_plan_is_none_when_maximum_power_cannot_finish():
    assert optimal_packet_plan(np.array([1.0, 2.0]), np.array([1.0, 3.0]), 10.0, 4) is None


def test_plan_levels_list_highest_power_first():
    plan = optimal_packet_plan(np.array([100.0, 150.0]), np.array([1.0, 2.0]), 400.0, 3)
    assert plan.counts == (1, 2)
    assert plan.levels() == (2, 2, 1)


@pytest.mark.parametrize("distance", [10.0, 40.0, 70.0, 99.0])
def test_continuous_bound_never_exceeds_the_discrete_optimum(distance):
    config = physical(1)
    gain = 10 ** (-(40 + 40 * math.log10(distance)) / 10)
    power = np.asarray(MENU.power_w[1:])
    bits = 1000 * np.log2(1 + power * gain / config.noise_power_w)
    plan = optimal_packet_plan(bits, 0.001 * power, 3000.0, 10)
    bound = continuous_lower_bound_j(config, gain, 3000.0, 10, power[-1])
    if plan is not None:
        assert bound is not None and bound <= plan.energy_j


def test_reference_rejects_more_ues_than_subchannels():
    scenario = load_scenario(N8)
    topology, traffic = scenario.instance(0)
    with pytest.raises(NotImplementedError):
        central_reference(scenario.physical, scenario.menu, topology, traffic)


def test_orthogonal_witness_meets_every_deadline_at_the_planned_energy():
    scenario = load_scenario(N4)
    topology, traffic = scenario.instance(0)
    reference = central_reference(scenario.physical, scenario.menu, topology, traffic)
    env = scenario.make_env(topology, traffic)
    observations = env.reset()
    while not env.truncated:
        step = env.step(act_central(reference, observations))
        observations = step.observations
        channels = step.physical.actions[step.physical.active, 0]
        assert len(set(channels.tolist())) == len(channels), "subchannels must stay orthogonal"
    settled = [job for job in env.job_records() if job["success"] is not None]
    assert settled and all(job["success"] for job in settled)
    for job in settled:
        assert job["energy_j"] == pytest.approx(reference.plans[job["ue"]].energy_j, rel=1e-12)
        assert job["tx_slots"] == reference.plans[job["ue"]].slots


def test_packets_due_matches_the_window_packet_set():
    env = build_env(
        [[20.0, 0.0], [0.0, 60.0]], [3, 7], [1.0, 1.0], [2, 5], num_subchannels=2, horizon=30
    )
    while not env.truncated:
        env.step([[0, 1], [1, 1]])
    summary = window_summary(
        env.job_records(),
        env.energy_by_slot(),
        warmup_slots=7,
        horizon_slots=30,
        slot_duration_s=0.001,
        num_ues=2,
    )
    due = env.traffic.packets_due(7, 30)
    assert due.tolist() == [row["packets"] for row in summary["per_ue"]] == [8, 3]


def test_window_energy_skips_floor_cap_packets_per_ue():
    scenario = load_scenario(N4)
    topology, traffic = scenario.instance(1)
    reference = central_reference(scenario.physical, scenario.menu, topology, traffic)
    energy = reference.window_energy_j(np.array([20, 19, 40, 1]), 0.05)
    assert energy["skipped_packets"] == [1, 0, 2, 0]
    per_packet = np.array([plan.energy_j for plan in reference.plans])
    assert energy["capped_j"] == pytest.approx(np.array([19, 19, 38, 1]) @ per_packet)


def test_runner_writes_a_verified_record_and_refuses_to_overwrite(tmp_path):
    output = tmp_path / "central"
    runner.main(["--config", N4, "--instances", "2", "--output", str(output)])
    record = json.loads((output / "summary.json").read_text())
    assert len(record["runs"]) == 2
    for run in record["runs"]:
        assert all(run["witness"]["checks"].values())
        assert run["witness"]["summary"]["ocr"] == 1.0
    with pytest.raises(SystemExit):
        runner.main(["--config", N4, "--instances", "1", "--output", str(output)])
