import json
from pathlib import Path

import numpy as np
import pytest
from conftest import MENU, TOP, build_env, solo_bits

from dsa_period import splits
from dsa_period.central import solo_reference
from dsa_period.central_timeshare import act_cycle, solve_orthogonal_cycle
from dsa_period.config import load_scenario
from dsa_period.evaluation import (
    _bootstrap,
    aggregate,
    load_references,
    paired_difference,
    run_episode,
    select,
)
from dsa_period.experiments import evaluate_policies
from dsa_period.policies import PacedStickyPolicy, make_policy

CONFIGS = Path(__file__).parents[1] / "configs"


def scenario(n):
    return load_scenario(CONFIGS / f"scenario-v1-n{n}.toml")


class CyclePolicy:
    """Centralized witness (uses a clock), only to check the evaluation plumbing."""

    name = "cycle"

    def __init__(self, schedule):
        self.schedule = schedule

    def reset(self, rngs):
        self.slot = 0

    def act(self, observations):
        actions = act_cycle(self.schedule, self.slot)
        self.slot += 1
        return actions


def test_references_cover_the_frozen_splits():
    for split in ("validation", "test"):
        table = load_references(split, 4)
        assert sorted(table) == list(splits.indices(split))
        assert all(upper == pytest.approx(lower, rel=1e-9) for lower, upper in table.values())
    table = load_references("test", 12)
    assert all(upper > lower for lower, upper in table.values())


def test_the_upper_bound_schedule_scores_exactly_one_through_the_harness():
    sc = scenario(8)
    topology, traffic = sc.instance(10_000)
    schedule = solve_orthogonal_cycle(sc.physical, sc.menu, topology, traffic)
    refs = load_references("validation", 8)
    episode = run_episode(sc, CyclePolicy(schedule), "validation", 10_000, 0, refs)
    assert episode["feasible"] and episode["ocr"] == 1.0
    assert episode["e_over_ub"] == pytest.approx(1.0, rel=1e-9)
    assert episode["e_over_lb"] >= 1.0


def test_episodes_are_reproducible_and_repeats_use_different_seeds():
    sc = scenario(8)
    make = lambda: make_policy("random_max", sc.physical, sc.menu)  # noqa: E731
    first = run_episode(sc, make(), "test", 20_000, 0)
    assert run_episode(sc, make(), "test", 20_000, 0) == first
    assert run_episode(sc, make(), "test", 20_000, 1)["packet_energy_j"] != first["packet_energy_j"]
    rates = first["per_ue_violation_rate"]
    assert first["feasible"] == all(rate <= 0.05 + 1e-12 for rate in rates)


def test_paced_sticky_reaches_the_exact_optimum_on_an_uncontended_instance():
    sc = scenario(4)
    topology, traffic = sc.instance(0)
    lower = solo_reference(sc.physical, sc.menu, topology, traffic).window_energy_j(
        traffic.packets_due(sc.warmup_slots, sc.horizon_slots), 0.05
    )["strict_j"]
    episode = run_episode(sc, PacedStickyPolicy(sc.physical, sc.menu), "dev", 0, 0)
    assert episode["ocr"] == 1.0
    assert episode["packet_energy_j"] == pytest.approx(lower, rel=1e-9)


def test_paced_sticky_keeps_its_gain_estimate_across_idle_periods():
    s = solo_bits([30.0, 0.0])
    env = build_env([[30.0, 0.0]], [6], [1.5 * s], [0], num_subchannels=2)
    policy = PacedStickyPolicy(env.config, MENU)
    policy.reset([np.random.default_rng(0)])
    obs = env.reset()
    first = policy.act(obs)
    assert first[0, 1] == TOP, "probes at max power while its gain is unknown"
    powers = []
    while not env.truncated:
        actions = policy.act(obs)
        powers.append(int(actions[0, 1]))
        obs = env.step(actions).observations
    later = [p for p in powers[6:] if p]
    assert later and max(later) < TOP, "later packets never re-probe at max power"


def test_policy_specs_are_parsed_and_validated():
    sc = scenario(4)
    assert make_policy("paced_sticky:margin=0.25", sc.physical, sc.menu).margin == 0.25
    for spec in ("oracle", "random_max:margin=1", "paced_sticky:speed=2"):
        with pytest.raises(ValueError):
            make_policy(spec, sc.physical, sc.menu)


def episodes(values, instances=range(4)):
    return [
        {
            "instance": i,
            "feasible": v,
            "e_over_ub": v,
            "ocr": v,
            "worst_ue_ocr": v,
            "mw_per_ue": v,
            "e_over_lb": None,
            "per_ue_violation_rate": [0.0],
        }
        for i, v in zip(instances, values, strict=True)
    ]


def test_bootstrap_intervals_behave_on_known_inputs():
    constant = aggregate({"a": episodes([1.0] * 4)}, resamples=200)
    assert constant["metrics"]["e_over_ub"]["ci95"] == [1.0, 1.0]
    assert constant["satisfies_constraint"]
    two_seeds = _bootstrap(np.array([[0.0] * 4, [1.0] * 4]), None, 2000, 0)
    assert two_seeds.mean() == pytest.approx(0.5, abs=0.05) and two_seeds.std() > 0.1
    same = {"a": episodes([0.2, 0.4, 0.6, 0.8])}
    assert paired_difference(same, same, "e_over_ub", resamples=200) == {
        "mean": 0.0,
        "ci95": [0.0, 0.0],
    }


def test_selection_rule_prefers_feasible_then_energy_then_feasibility():
    def summary(feasible, energy):
        return {"metrics": {"feasible": {"mean": feasible}, "e_over_ub": {"mean": energy}}}

    assert select({"a": summary(0.96, 3.0), "b": summary(0.99, 2.0), "c": summary(0.5, 1.0)}) == "b"
    assert select({"a": summary(0.90, 3.0), "b": summary(0.80, 1.0)}) == "a"


def test_runner_guards_the_test_split_and_runs_dev(tmp_path):
    config = str(CONFIGS / "scenario-v1-n4.toml")
    with pytest.raises(SystemExit):
        evaluate_policies.main(["--config", config, "--split", "test", "--policies", "random_max"])
    with pytest.raises(SystemExit):
        evaluate_policies.main(
            [
                "--config",
                config,
                "--split",
                "validation",
                "--instances",
                "3",
                "--policies",
                "random_max",
            ]
        )
    output = tmp_path / "dev"
    evaluate_policies.main(
        [
            "--config",
            config,
            "--split",
            "dev",
            "--first-instance",
            "0",
            "--instances",
            "2",
            "--policies",
            "random_max",
            "paced_sticky",
            "--output",
            str(output),
        ]
    )
    record = json.loads((output / "summary.json").read_text())
    assert record["instances"] == [0, 1] and record["selected_by_protocol_rule"] is None
    assert len(record["results"]["paced_sticky"]["episodes"]) == 2
