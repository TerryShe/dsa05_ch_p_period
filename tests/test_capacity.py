"""Scenario v2.0 capacity evaluation, local rules, the central scheduler and the LP bounds."""

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from dsa_period import splits_v2
from dsa_period.capacity import (
    allowance,
    capacity,
    evaluated_packets,
    run_capacity_episode,
    select_per_n,
    summarize,
    undeliverable,
)
from dsa_period.capacity_bounds import feasible_at, minimum_slots, solve_bound
from dsa_period.config import Scenario, load_scenario
from dsa_period.env import Outcome, UEObservation
from dsa_period.rules_v2 import (
    CentralEdfPolicy,
    OpportunisticPolicy,
    PPersistentPolicy,
    ReservationPolicy,
    make_policy_v2,
)
from dsa_period.traffic import TrafficPattern

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "scenario-v2.toml"


@pytest.fixture(scope="module")
def v2():
    return load_scenario(CONFIG, 4)


def obs(remaining, to_deadline, gains, *, period=10, packet=6000.0, rssi=None, **kw):
    return UEObservation(
        remaining_bits=remaining,
        packet_bits=packet,
        period_slots=period,
        slots_to_deadline=to_deadline,
        previous_action=kw.get("previous_action", (0, 0)),
        previous_outcome=kw.get("previous_outcome", Outcome.NO_TX),
        previous_service_bits=kw.get("previous_service_bits", 0.0),
        previous_rssi_w=np.asarray(rssi if rssi is not None else [1e-13] * len(gains)),
        own_gain_w=np.asarray(gains, dtype=float),
    )


# ------------------------------------------------------------------ capacity rule
def test_capacity_takes_the_grid_point_before_the_first_drop():
    grid = [4, 8, 12, 16]
    assert capacity(grid, [1.0, 0.96, 0.9, 1.0]) == {"n_star": 8, "at_least": False}
    assert capacity(grid, [0.94, 1.0, 1.0, 1.0]) == {"n_star": 0, "at_least": False}
    assert capacity(grid, [1.0, 1.0, 0.95, 0.95]) == {"n_star": 16, "at_least": True}
    with pytest.raises(ValueError):
        capacity([8, 4], [1.0, 1.0])


def test_allowance_is_the_integral_floor():
    assert [allowance(0.01, m) for m in (2000, 1000, 500, 45, 90)] == [20, 10, 5, 0, 0]
    assert [allowance(0.05, m) for m in (2000, 1000, 500)] == [100, 50, 25]


def test_summary_reports_f_and_capacity_per_eps():
    def episode(instance, ok):
        return {
            "instance": instance,
            "feasible": {"0.01": ok, "0.05": True},
            "infeasible_for_all": {"0.01": False, "0.05": False},
        }

    results = {
        4: {"-": [episode(i, True) for i in range(20)]},
        8: {"-": [episode(i, i != 0) for i in range(20)]},  # F = 0.95
        12: {"-": [episode(i, i > 1) for i in range(20)]},  # F = 0.90
    }
    summary = summarize(results, resamples=200)
    assert summary["per_eps"]["0.01"]["capacity"] == {"n_star": 8, "at_least": False}
    assert summary["per_eps"]["0.05"]["capacity"] == {"n_star": 12, "at_least": True}
    assert sum(summary["per_eps"]["0.01"]["capacity_bootstrap"].values()) == pytest.approx(1.0)


# ------------------------------------------------------------- undeliverable
def test_undeliverable_packets_are_found_from_the_best_alone_rate():
    traffic = TrafficPattern(("c",), [0, 0], [4, 4], [10.0, 10.0], [0, 2])
    packets = evaluated_packets(traffic, warmup=4, horizon=12)
    assert packets["deadline"].tolist() == [8, 12, 6, 10]  # UE 0, then UE 1
    best = np.full((12, 2), 3.0)
    best[4:8, 0] = 2.0  # UE 0's packet (4, 8] gets 8 < 10 bits
    assert undeliverable(packets, best).tolist() == [True, False, False, False]


def test_an_episode_counts_undeliverable_packets_as_violations(v2):
    topology, traffic = v2.instance(0)
    big = TrafficPattern(
        traffic.class_names,
        traffic.class_index,
        traffic.period_slots,
        np.full(4, 1e7),  # no UE can deliver 1e7 bits within one period
        traffic.phase_slots,
    )

    class Fixed(Scenario):
        def instance(self, index):
            return topology, big

    far = Fixed(**{f.name: getattr(v2, f.name) for f in fields(v2)})
    policy = make_policy_v2("central_edf", far.physical, far.menu)
    result = run_capacity_episode(far, policy, "dev", 0)
    assert result["undeliverable"] == result["packets"] == result["violations"]
    assert result["infeasible_for_all"] == {"0.01": True, "0.05": True}
    assert result["feasible"] == {"0.01": False, "0.05": False}


# ------------------------------------------------------------------ local rules
def test_central_edf_serves_the_smallest_slack_on_its_best_free_subchannel(v2):
    policy = CentralEdfPolicy(v2.physical, v2.menu)
    g = 1e-11
    observations = [
        obs(6000.0, 9, [g, 2 * g, g, g]),  # large slack
        obs(6000.0, 1, [g, 3 * g, g, g]),  # smallest slack: gets subchannel 1
        obs(6000.0, 2, [g, 4 * g, 2 * g, g]),  # next: 1 is taken, gets 2
        obs(0.0, 5, [g, g, g, g]),  # nothing to send
        obs(6000.0, 3, [5 * g, g, g, g]),
        obs(6000.0, 4, [g, g, g, 6 * g]),
    ]
    actions = policy.act(observations)
    top = v2.menu.num_levels - 1
    assert actions[1].tolist() == [1, top] and actions[2].tolist() == [2, top]
    assert actions[4].tolist() == [0, top] and actions[5].tolist() == [3, top]
    assert actions[0].tolist() == [0, 0] and actions[3].tolist() == [0, 0]  # 4 subchannels used


def test_p_persistent_always_sends_without_slack(v2):
    policy = PPersistentPolicy(v2.physical, v2.menu, a=0.01)
    policy.reset([np.random.default_rng(i) for i in range(1)])
    urgent = obs(6000.0, 1, [1e-12, 1e-12, 3e-12, 1e-12])
    assert policy.act([urgent])[0].tolist() == [2, v2.menu.num_levels - 1]


def test_opportunistic_waits_for_a_good_channel_unless_urgent(v2):
    policy = OpportunisticPolicy(v2.physical, v2.menu, theta=1e6)
    policy.reset([np.random.default_rng(0)])
    relaxed = obs(600.0, 9, [1e-9] * 4)
    assert policy.act([relaxed])[0].tolist() == [0, 0]
    urgent = obs(6000.0, 1, [1e-15] * 4)
    assert policy.act([urgent])[0, 1] == v2.menu.num_levels - 1


def test_reservation_keeps_offset_and_subchannel_until_consecutive_misses(v2):
    policy = ReservationPolicy(v2.physical, v2.menu, offset=0.5, failures=2)
    policy.reset([np.random.default_rng(3)])
    g = [1e-11] * 4
    top = v2.menu.num_levels - 1
    policy.act([obs(6000.0, 10, g)])  # release: draws the reservation
    first = (policy._slot[0], policy._channel[0])
    assert 0 <= first[0] < 5
    # Sends on the reserved subchannel from the reserved offset on.
    assert policy.act([obs(6000.0, 10 - first[0], g)])[0].tolist() == [first[1], top]
    # Decoded before the next release: reservation kept, no miss.
    policy.act([obs(0.0, 4, g, previous_outcome=Outcome.DECODED)])
    policy.act([obs(6000.0, 10, g)])
    assert (policy._slot[0], policy._channel[0]) == first and policy._misses[0] == 0
    # One miss (next release without decoding): kept, one miss counted.
    policy.act([obs(6000.0, 10, g)])
    assert (policy._slot[0], policy._channel[0]) == first and policy._misses[0] == 1
    # A second consecutive miss: redrawn, counter reset.
    state = policy._rngs[0].bit_generator.state
    policy.act([obs(6000.0, 10, g)])
    assert policy._misses[0] == 0 and policy._rngs[0].bit_generator.state != state


def test_every_policy_spec_builds(v2):
    for spec in (
        "random_max",
        "sticky_max",
        "p_persistent:a=4",
        "opportunistic:theta=2",
        "reservation:offset=0.25,failures=2",
        "paced_sticky_csi:margin=0.5,switch_db=3",
        "central_edf:defer_hopeless=1",
    ):
        make_policy_v2(spec, v2.physical, v2.menu)
    with pytest.raises(ValueError):
        make_policy_v2("p_persistent:b=1", v2.physical, v2.menu)


# ------------------------------------------------------------------- LP bounds
def _toy(v2):
    """Two UEs, one subchannel, period 2, 1000 bits per slot alone, 1500-bit packets."""
    physical = replace(v2.physical, num_ues=2, num_subchannels=1)
    classes = (replace(v2.traffic_classes[0], num_ues=1), replace(v2.traffic_classes[1], num_ues=1))
    scenario = replace(
        v2, physical=physical, traffic_classes=classes, warmup_slots=40, horizon_slots=46
    )
    traffic = TrafficPattern(("c",), [0, 0], [2, 2], [1500.0, 1500.0], [0, 0])
    # SNR 1 at max power: tau * B * log2(2) = 1000 bits per slot.
    gain = physical.noise_power_w / v2.menu.power_w[-1]
    return scenario, traffic, np.full((46, 2, 1), gain)


def test_lp_bounds_match_a_hand_calculation(v2):
    scenario, traffic, gains = _toy(v2)
    # Orthogonal: a 1500-bit packet needs 2 whole slots of 1000 bits, and a
    # 2-slot window on one subchannel holds 2 slots, so only one of the two
    # packets of each window fits: each UE misses half of its packets.
    orth = solve_bound(scenario, traffic, gains, "orth")
    assert orth["s_star"] == pytest.approx(0.5, abs=1e-6)
    # With one subchannel the merged relaxation is the same LP.
    merged = solve_bound(scenario, traffic, gains, "orth_agg")
    assert merged["s_star"] == pytest.approx(0.5, abs=1e-6)
    # Sum capacity 1000 log2(3) = 1585 bits per slot covers 3000 per window.
    mac = solve_bound(scenario, traffic, gains, "mac")
    assert mac["s_star"] == pytest.approx(0.0, abs=1e-6)
    assert feasible_at(orth["s_star"], 0.51) and not feasible_at(orth["s_star"], 0.45)


def test_minimum_slots_takes_the_best_slots_first():
    best = np.array([3.0, 1.0, 2.0, 5.0, 5.0, 1.0, 1.0])
    rows = np.array([0, 0, 0, 1, 1, 2, 2])
    needed = minimum_slots(best, rows, np.array([4.5, 5.0, 3.0]), np.array([3, 2, 2]))
    assert needed.tolist() == [2.0, 1.0, np.inf]


def test_lp_bounds_are_at_most_what_the_central_scheduler_achieves(v2):
    base = load_scenario(CONFIG, 12)
    scenario = replace(
        base,
        traffic_classes=tuple(
            replace(c, packet_bits=4 * c.packet_bits) for c in base.traffic_classes
        ),
        horizon_slots=1240,
        warmup_slots=40,
    )
    worst = []
    for instance in range(2):
        episode = run_capacity_episode(
            scenario,
            make_policy_v2("central_edf", scenario.physical, scenario.menu),
            "dev",
            instance,
        )
        topology, traffic = scenario.instance(instance)
        channel = scenario.make_channel(topology, splits_v2.fading_seeds("dev", 12, instance, 0))
        gains = channel.gain_block(0, scenario.horizon_slots)
        orth = solve_bound(scenario, traffic, gains, "orth")["s_star"]
        mac = solve_bound(scenario, traffic, gains, "mac")["s_star"]
        merged = solve_bound(scenario, traffic, gains, "orth_agg")["s_star"]
        assert orth <= episode["worst_violation_rate"] + 1e-6
        assert mac <= orth + 1e-6
        assert merged <= orth + 1e-6
        worst.append(episode["worst_violation_rate"])
    assert max(worst) > 0, "the test needs a scheduler that misses packets"


def test_per_n_selection_follows_the_tie_breaks():
    def summary(rows):
        return {
            "per_eps": {
                key: {
                    "rows": [
                        {
                            "num_ues": n,
                            "feasible_share": shares[i],
                            "worst_ue_violation_rate_mean": worst,
                        }
                        for n, shares, worst in rows
                    ]
                }
                for i, key in enumerate(("0.01", "0.05"))
            }
        }

    summaries = {
        "a": summary([(2, (1.0, 1.0), 0.002), (4, (0.9, 0.95), 0.01), (6, (0.0, 0.2), 0.3)]),
        "b": summary([(2, (1.0, 1.0), 0.001), (4, (0.9, 1.0), 0.02), (6, (0.0, 0.2), 0.3)]),
        "c": summary([(2, (0.98, 1.0), 0.0), (4, (0.8, 1.0), 0.0), (6, (0.0, 0.1), 0.0)]),
    }
    chosen = select_per_n(summaries, ["a", "b", "c"])
    assert chosen[2]["spec"] == "b"  # equal F at both eps: lower worst-UE rate
    assert chosen[4]["spec"] == "b"  # equal F at 0.01: higher F at 0.05
    assert chosen[6]["spec"] == "a"  # all equal: table order
    assert chosen[4]["feasible_share"] == {"0.01": 0.9, "0.05": 1.0}


def test_calibration_merges_runs_on_identical_instances_only():
    from dsa_period.experiments.calibrate_rules_v2 import merge

    def run(specs, instances=(10_000, 10_001)):
        return {
            "protocol_id": "marl-protocol-v2",
            "split": "validation",
            "instances": list(instances),
            "grid": [2, 4],
            "config": "configs/scenario-v2.toml",
            "formal": True,
            "provenance": {"commit": "x", "dirty": False},
            "summaries": {spec: {} for spec in specs},
        }

    merged = merge([run(["a", "b"]), run(["c"])])
    assert sorted(merged["summaries"]) == ["a", "b", "c"] and merged["formal"]
    with pytest.raises(ValueError):
        merge([run(["a"]), run(["a"])])
    with pytest.raises(ValueError):
        merge([run(["a"]), run(["c"], instances=(10_000,))])
