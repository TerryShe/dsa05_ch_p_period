import json
from pathlib import Path

import numpy as np
import pytest
from conftest import MENU, SPATIAL, TOP, physical

from dsa_period.central import solo_reference
from dsa_period.central_timeshare import (
    _assign_slots,
    act_cycle,
    cyclic_windows,
    solve_orthogonal_cycle,
)
from dsa_period.channel import StaticChannel
from dsa_period.config import load_scenario
from dsa_period.env import PeriodicUplinkEnv
from dsa_period.experiments import central_bounds
from dsa_period.topology import topology_from_positions
from dsa_period.traffic import TrafficPattern

CONFIGS = Path(__file__).parents[1] / "configs"


def pattern(periods, bits, phases):
    n = len(periods)
    return TrafficPattern(
        ("c",), np.zeros(n, int), np.array(periods), np.array(bits, float), np.array(phases)
    )


def test_cyclic_windows_partition_the_hyperperiod_for_every_ue():
    cycle, windows = cyclic_windows(pattern([10, 20, 40], [1.0] * 3, [7, 13, 39]))
    assert cycle == 40
    for ue in range(3):
        slots = np.concatenate([w.slots for w in windows if w.ue == ue])
        assert sorted(slots.tolist()) == list(range(40))
    first = [w for w in windows if w.ue == 2][0]
    assert first.slots[:2].tolist() == [39, 0], "windows wrap around the cycle in time order"


def test_slot_assignment_respects_windows_capacity_and_demand():
    _, windows = cyclic_windows(pattern([4, 4, 4], [1.0] * 3, [0, 1, 2]))
    demand = np.array([2, 1, 3])
    assigned = _assign_slots(4, 2, windows, demand)
    assert assigned.sum(axis=1).max() <= 2
    assert assigned.sum(axis=0).tolist() == demand.tolist()
    with pytest.raises(RuntimeError):
        _assign_slots(4, 1, windows, np.array([4, 4, 4]))


@pytest.mark.parametrize(
    "num_ues, instance",
    # N=4 instances 10 and 83 stopped ~0.15% / 0.05% short of optimal when the MILP
    # was posed in joules (HiGHS's absolute gap); they guard the unit scaling.
    [(4, 0), (4, 10), (4, 83), (6, 0)],
)
def test_upper_bound_equals_the_solo_optimum_when_capacity_is_slack(num_ues, instance):
    scenario = load_scenario(CONFIGS / f"scenario-v1-n{num_ues}.toml")
    topology, traffic = scenario.instance(instance)
    lower = solo_reference(scenario.physical, scenario.menu, topology, traffic)
    schedule = solve_orthogonal_cycle(scenario.physical, scenario.menu, topology, traffic)
    per_cycle = sum(
        plan.energy_j * (schedule.cycle_slots // int(traffic.period_slots[ue]))
        for ue, plan in enumerate(lower.plans)
    )
    assert schedule.cycle_energy_j == pytest.approx(per_cycle, rel=1e-9)


def two_ues_one_channel():
    """Both UEs need 4 slots at the lowest power alone but must share one channel."""
    config = physical(2, 1)
    positions = [[60.0, 0.0], [0.0, 60.0]]
    topology = topology_from_positions(SPATIAL, positions)
    gain = topology.ue_ap_gain_linear[0]
    low = 1000 * np.log2(1 + MENU.power_w[1] * gain / config.noise_power_w)
    traffic = pattern([4, 4], [3.5 * low, 3.5 * low], [0, 0])
    return config, topology, traffic


def test_capacity_forces_higher_power_and_the_replay_meets_every_deadline():
    config, topology, traffic = two_ues_one_channel()
    lower = solo_reference(config, MENU, topology, traffic)
    schedule = solve_orthogonal_cycle(config, MENU, topology, traffic)
    assert (schedule.power_ids > 0).sum(axis=1).max() <= 1
    assert schedule.cycle_energy_j > sum(plan.energy_j for plan in lower.plans)
    env = PeriodicUplinkEnv(config, MENU, StaticChannel(topology, 1), traffic, 40)
    env.reset()
    while not env.truncated:
        step = env.step(act_cycle(schedule, env.elapsed_steps))
        assert step.physical.active.sum() <= 1
    settled = [job for job in env.job_records() if job["success"] is not None]
    assert settled and all(job["success"] for job in settled)


def test_orthogonal_bound_is_none_when_capacity_cannot_carry_the_load():
    config, topology, traffic = two_ues_one_channel()
    gain = topology.ue_ap_gain_linear[0]
    top = 1000 * np.log2(1 + MENU.power_w[TOP] * gain / config.noise_power_w)
    heavy = pattern([4, 4], [2.5 * top, 2.5 * top], [0, 0])
    assert solve_orthogonal_cycle(config, MENU, topology, heavy) is None


def test_bounds_runner_brackets_and_verifies_a_loaded_instance(tmp_path):
    output = tmp_path / "bounds"
    config = str(CONFIGS / "scenario-v1-n12.toml")
    central_bounds.main(["--config", config, "--instances", "1", "--output", str(output)])
    run = json.loads((output / "summary.json").read_text())["runs"][0]
    assert all(run["upper"]["checks"].values())
    assert run["upper_over_lower"] >= 1.0
    assert run["upper"]["slot_utilization"] <= 1.0
    with pytest.raises(SystemExit):
        central_bounds.main(["--config", config, "--instances", "1", "--output", str(output)])
