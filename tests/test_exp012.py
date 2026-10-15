"""EXP-DSA05-012 phase 0: survival state, outage metrics, M+S and R17-like rules,
the survival content, the per-slot outage cost and the branch episodes (design §2-§4)."""

import copy
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from dsa_period import splits_v2
from dsa_period.capacity import run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.env import Outcome
from dsa_period.marl.features import Settlement, stack
from dsa_period.marl.ppo import PPOSettings
from dsa_period.marl.trainer_v2 import (
    BranchSettings,
    CapacityConstraint,
    CapacityTrainer,
    CapacityTrainingSettings,
    load_capacity_policy,
    model_dims,
)
from dsa_period.rules_v2 import make_policy_v2
from dsa_period.survival import (
    STATE_DIM,
    BoostPolicy,
    SurvivalSettings,
    SurvivalTracker,
    service_record,
    settlement_memory,
)
from dsa_period.urgency import SummarySettings, YieldPolicy, parse_wrapper

ROOT = Path(__file__).parents[1]
V21 = ROOT / "configs" / "scenario-v2.1.toml"


def _obs(n: int, decoded=()):
    outcome = np.full(n, int(Outcome.NO_TX))
    outcome[list(decoded)] = int(Outcome.DECODED)

    class Obs:
        prev_outcome = outcome

    return Obs()


def _settle(n: int, missed=()):
    violated = np.zeros(n, dtype=np.bool_)
    violated[list(missed)] = True
    return Settlement(violated.copy(), violated)


def test_tracker_enters_on_a_miss_leaves_on_a_decode_and_marks_outages():
    tracker = SurvivalTracker(3, tolerance=1)
    tracker.update(_settle(3, [0, 1]), _obs(3))
    assert tracker.consecutive.tolist() == [1, 1, 0] and not tracker.outage.any()
    assert tracker.in_state.tolist() == [True, True, False]
    tracker.update(_settle(3), _obs(3, decoded=[1]))  # UE 1 decodes: state left at once
    assert tracker.consecutive.tolist() == [1, 0, 0]
    tracker.update(_settle(3, [0]), _obs(3))  # second consecutive miss: outage (s = 1)
    assert tracker.consecutive[0] == 2 and tracker.outage.tolist() == [True, False, False]
    tracker.update(_settle(3, [0]), _obs(3))  # extends the outage
    assert tracker.outage[0] and tracker.consecutive[0] == 3
    features = tracker.features()
    assert features.shape == (3, STATE_DIM)
    assert features[0].tolist() == [0, 0, 1, 0] and features[1].tolist() == [1, 0, 0, 1]
    tracker.update(_settle(3), _obs(3, decoded=[0]))
    assert not tracker.outage.any() and tracker.consecutive[0] == 0
    two = SurvivalTracker(1, tolerance=2)
    for _ in range(2):
        two.update(_settle(1, [0]), _obs(1))
    assert not two.outage[0]
    two.update(_settle(1, [0]), _obs(1))
    assert two.outage[0]


def _job(ue, deadline, completion=None, success=None):
    return {"ue": ue, "deadline_slot": deadline, "completion_slot": completion, "success": success}


def test_service_record_times_outages_from_the_deadline_to_the_actual_delivery():
    period = 10
    jobs = [
        _job(0, 210, 205, True),
        _job(0, 220, None, False),  # first miss: survival state, still available
        _job(0, 230, None, False),  # second miss: outage from 230
        _job(0, 240, None, False),  # extends it
        _job(0, 250, 243, True),  # delivered at 243: outage 230 -> 243 = 13 slots
        _job(0, 260, None, False),
        _job(0, 270, 262, True),  # a single miss: no outage
        _job(0, 280, None, False),
        _job(0, 290, None, False),  # outage from 290, never recovers before the horizon
        _job(0, 300, None, None),  # pending at the horizon
    ]
    record = service_record(jobs, 1, warmup=200, horizon=295, tolerance=1)
    assert record == {"outages": [2], "unavailable_slots": [13 + 5]}
    two = service_record(jobs, 1, warmup=200, horizon=295, tolerance=2)
    assert two == {"outages": [1], "unavailable_slots": [3]}  # 240 -> 243
    early = [_job(0, 190, None, False), _job(0, 200, None, False), _job(0, 210, 204, True)]
    clipped = service_record(early, 1, warmup=200, horizon=400, tolerance=1)
    assert clipped == {"outages": [0], "unavailable_slots": [4]}  # starts at 200, not counted
    assert period == 10


def test_episode_outage_slots_match_the_tracker_slot_by_slot():
    scenario = load_scenario(V21, 16)
    topology, traffic = scenario.instance(3)
    horizon, warmup = 3000, 200
    env = scenario.make_env(
        topology, traffic, splits_v2.fading_seeds("dev", 16, 3, 0), horizon_slots=horizon
    )
    policy = make_policy_v2("random_max", scenario.physical, scenario.menu)
    policy.reset(splits_v2.execution_generators("dev", 16, 3, 0))
    memory, tracker = settlement_memory(scenario.physical, scenario.menu), SurvivalTracker(16, 1)
    observations = env.reset()
    down = np.zeros(16, dtype=np.int64)
    for t in range(horizon):
        obs = stack(observations)
        tracker.update(memory.observe(obs), obs)
        if t >= warmup:
            down += tracker.outage
        observations = env.step(policy.act(observations)).observations
    record = service_record(env.job_records(), 16, warmup, horizon, 1)
    assert record["unavailable_slots"] == down.tolist()
    assert sum(record["unavailable_slots"]) > 0 and sum(record["outages"]) > 0


def test_capacity_episodes_record_missed_packets_and_survival():
    scenario = load_scenario(V21, 12)
    episode = run_capacity_episode(
        scenario, make_policy_v2("random_max", scenario.physical, scenario.menu), "dev", 64
    )
    assert [len(m) for m in episode["missed_packets"]] == episode["violations"]
    assert set(episode["survival"]) == {"1", "2"} and episode["survival_window"] == 20_000
    for record in episode["survival"].values():
        assert len(record["outages"]) == len(record["unavailable_slots"]) == 12
    one, two = episode["survival"]["1"], episode["survival"]["2"]
    assert all(a >= b for a, b in zip(one["unavailable_slots"], two["unavailable_slots"]))


def test_wrapper_specs_for_m_plus_s_and_the_boost_rule():
    assert parse_wrapper("yield[theta=-10,content=survival]@best_local") == (
        "yield",
        {"theta_db": -10.0, "content": "survival"},
        "best_local",
    )
    assert parse_wrapper("boost[]@best_local") == ("boost", {}, "best_local")
    with pytest.raises(ValueError):
        parse_wrapper("yield[theta=-10,content=message]@best_local")
    scenario = load_scenario(V21, 8)
    ms = make_policy_v2("yield[theta=-10,content=survival]@random_max", scenario.physical,
                        scenario.menu)  # fmt: skip
    assert isinstance(ms, YieldPolicy) and ms.settings.content == "survival"
    assert ms.name == "yield[theta=-10,content=survival]@random_max"
    old = make_policy_v2("yield[theta=-10]@random_max", scenario.physical, scenario.menu)
    assert old.name == "yield[theta=-10]@random_max" and old.tracker is None
    boost = make_policy_v2("boost[]@random_max", scenario.physical, scenario.menu)
    assert isinstance(boost, BoostPolicy)


def test_m_plus_s_sends_only_survival_bits_and_exempts_those_ues():
    scenario = load_scenario(V21, 12)
    policy = make_policy_v2("yield[theta=all,content=survival]@random_max", scenario.physical,
                            scenario.menu)  # fmt: skip
    env = scenario.make_env(*scenario.instance(5), splits_v2.fading_seeds("dev", 12, 5, 0),
                            horizon_slots=800)  # fmt: skip
    policy.reset(splits_v2.execution_generators("dev", 12, 5, 0))
    observations = env.reset()
    sent_by_state = []
    for _ in range(800):
        actions = policy.act(observations)
        state = policy.tracker.in_state & (stack(observations).remaining > 0)
        sent_by_state.append((policy.link.sent, state, actions[:, 1] > 0))
        observations = env.step(actions).observations
    for sent, state, transmitting in sent_by_state:
        assert not (sent & ~state).any() and np.array_equal(sent, state & transmitting)
    assert any(sent.any() for sent, _, _ in sent_by_state)


def test_boost_sends_ues_in_the_survival_state_on_their_best_subchannel_at_max_power():
    scenario = load_scenario(V21, 16)
    policy = BoostPolicy(make_policy_v2("random_max", scenario.physical, scenario.menu),
                         scenario.physical, scenario.menu)  # fmt: skip
    env = scenario.make_env(*scenario.instance(7), splits_v2.fading_seeds("dev", 16, 7, 0),
                            horizon_slots=800)  # fmt: skip
    policy.reset(splits_v2.execution_generators("dev", 16, 7, 0))
    observations = env.reset()
    checked = 0
    for _ in range(800):
        actions = policy.act(observations)
        obs = stack(observations)
        for ue in np.flatnonzero(policy.tracker.in_state & (obs.remaining > 0)):
            assert actions[ue, 0] == np.argmax(obs.own_gain[ue])
            assert actions[ue, 1] == scenario.menu.num_levels - 1
            checked += 1
        observations = env.step(actions).observations
    assert checked > 0 and policy.boosted.sum() == checked


def _trainer(branching: BranchSettings | None, survival=None, information=None, seed=3):
    settings = CapacityTrainingSettings(
        budget_slots=10**9, env_ues=(16, 20), rollout_slots=40, episode_slots=40, hidden=16
    )
    return CapacityTrainer(
        load_scenario(V21, 20), settings, PPOSettings(), CapacityConstraint("fixed"), seed,
        information=information, survival=survival or SurvivalSettings(), branching=branching,
    )  # fmt: skip


@pytest.fixture(scope="module")
def branched():
    trainer = _trainer(
        BranchSettings(fraction=1.0, periods=1, buffer=8),
        information=SummarySettings("survival"),
    )
    runs = []
    for _ in range(4):
        rollout, stats = trainer.collect()
        runs.append((rollout, stats, trainer.train_step(rollout, 0.01)))
    return trainer, runs


def test_branch_episodes_start_from_snapshots_and_replay_exactly(branched):
    trainer, runs = branched
    assert trainer.branch_counts["snapshots"] > 0 and trainer.branch_counts["branches"] > 0
    restored = [r.restart_hidden[r.starts[:, trainer.env_of_ue]].abs().sum() for r, _, _ in runs]
    assert any(float(x) > 0 for x in restored)  # some starts resume a non-zero memory
    for rollout, stats, metrics in runs:
        assert metrics["replay_log_prob_error"] < 1e-5
        assert {"outage_ue_slots", "entries", "state_decisions"} <= set(stats["survival"])
        dims = model_dims(trainer.config, trainer.menu, trainer.information, trainer.survival)
        assert rollout.context.shape[-1] == dims[1] and rollout.critic_ue.shape[-1] == dims[2]


def test_a_branch_resumes_the_snapshot_state_bit_for_bit():
    trainer = _trainer(BranchSettings(fraction=1.0, periods=1, buffer=8))
    trainer.collect()
    n = int(trainer.env_ues[0])
    assert trainer.snapshots.get(n), "the random policy should miss at N = 16 within 40 slots"
    snapshot = trainer.snapshots[n][0]
    trainer._branch_env(0, snapshot)
    rows = trainer._rows(0)
    assert trainer.envs[0].elapsed_steps == snapshot["env"].elapsed_steps
    assert np.array_equal(trainer.trackers[0].consecutive, snapshot["tracker"].consecutive)
    assert torch.equal(trainer.hidden[rows], snapshot["hidden"])
    assert trainer.branch_end[0] == snapshot["episode_slot"] + snapshot["length"]
    actions = np.zeros((n, 2), dtype=np.int64)
    a = copy.deepcopy(trainer.envs[0])
    b = copy.deepcopy(snapshot["env"])
    for _ in range(20):
        sa, sb = a.step(actions), b.step(actions)
        assert np.array_equal(sa.violations, sb.violations)


def test_outage_slots_cost_lambda_d_each():
    plain = _trainer(None, SurvivalSettings(outage_slot_weight=0.0))
    weighted = _trainer(None, SurvivalSettings(outage_slot_weight=1.0))
    (r0, s0), (r1, s1) = plain.collect(), weighted.collect()
    assert np.array_equal(r0.actions, r1.actions)  # same seeds, same first rollout
    extra = (r0.rewards - r1.rewards) * plain.env_ues[None, :]
    assert extra.sum() == pytest.approx(s1["survival"]["outage_ue_slots"])
    assert s1["survival"]["outage_ue_slots"] == s0["survival"]["outage_ue_slots"]


def test_survival_settings_are_checked_and_saved(tmp_path):
    with pytest.raises(ValueError):
        CapacityTrainer(
            load_scenario(V21, 4), CapacityTrainingSettings(env_ues=(4,)), PPOSettings(),
            CapacityConstraint("fixed"), 1, information=SummarySettings("survival"),
        )  # fmt: skip
    with pytest.raises(ValueError):
        CapacityTrainer(
            load_scenario(V21, 4), CapacityTrainingSettings(env_ues=(4,)), PPOSettings(),
            CapacityConstraint("fixed"), 1, branching=BranchSettings(),
        )  # fmt: skip
    trainer = _trainer(BranchSettings(), information=SummarySettings("survival"))
    checkpoint = trainer.checkpoint(1, 1)
    assert checkpoint["survival"] == {"tolerance": 1, "features": True, "outage_slot_weight": 1.0}
    assert checkpoint["branching"]["fraction"] == 0.25
    path = tmp_path / "step-000000001.pt"
    torch.save(checkpoint, path)
    scenario = load_scenario(V21, 8)
    policy = load_capacity_policy(path, scenario.physical, scenario.menu)
    assert policy.survival.features and policy.settings.content == "survival"
    policy.reset(splits_v2.execution_generators("dev", 8, 0, 0))
    env = scenario.make_env(*scenario.instance(0), splits_v2.fading_seeds("dev", 8, 0, 0),
                            horizon_slots=100)  # fmt: skip
    observations = env.reset()
    for _ in range(100):
        observations = env.step(policy.act(observations)).observations


def test_pilot_configs_equal_the_m_recipe_apart_from_the_exp012_sections():
    base = tomllib.loads((ROOT / "configs/exp009-step2-s1.toml").read_text())
    extra = {"pilot-m": set(), "pilot-ms": {"survival"}, "pilot-scb": {"survival", "information"},
             "pilot-sc": {"survival", "information", "branching"},
             "pilot1b-scb-ld01": {"survival", "information"},
             "pilot1b-scb-ld03": {"survival", "information"}}  # fmt: skip
    weight = {"pilot1b-scb-ld01": 0.1, "pilot1b-scb-ld03": 0.3}  # stage 1b (design §10.5)
    for key, sections in extra.items():
        raw = tomllib.loads((ROOT / f"configs/exp012-{key}.toml").read_text())
        assert set(raw) == set(base) | sections
        for section in set(base) - {"experiment"}:
            assert raw[section] == base[section], (key, section)
        assert raw["experiment"]["scenario"] == "configs/scenario-v2.1.toml"
        assert raw["experiment"]["seeds"] == [101, 202]
        if "survival" in raw:
            assert raw["survival"]["tolerance"] == 1
            assert raw["survival"]["outage_slot_weight"] == weight.get(key, 1.0)
            assert raw["survival"]["features"] == (key != "pilot-ms")
        if "branching" in raw:
            assert raw["branching"] == {"fraction": 0.25, "periods": 3, "buffer": 64,
                                        "snapshot_probability": 1.0}  # fmt: skip


def test_exp012_sets_are_registered_on_scenario_v21():
    manifest = json.loads((ROOT / "docs/research/exp012-sets.json").read_text())
    assert manifest["scenario_config"] == "configs/scenario-v2.1.toml"
    assert set(manifest["screening"]) == {"exp012-pilot256", "exp012-eval512"}
    pilot = manifest["screening"]["exp012-pilot256"]
    assert (pilot["start"], pilot["stop"], pilot["grid"]) == (1400, 1656, [8, 10, 12])
    for name in ("exp012-pilot256", "exp012-eval512"):
        start, stop = splits_v2.SCREENING_SETS[name]["range"]
        for other, entry in splits_v2.SCREENING_SETS.items():
            if other != name:
                a, b = entry["range"]
                assert stop <= a or b <= start, (name, other)
    n = 8
    digest = splits_v2.instance_digest(load_scenario(V21, n), range(1400, 1656), "dev")
    assert manifest["screening"]["exp012-pilot256"]["digests"][str(n)] == digest


def _analysis():
    import importlib.util

    spec = importlib.util.spec_from_file_location("a12", ROOT / "scripts/analyze_exp012.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_per_deployment_quantities_and_the_gate_a_rule():
    a = _analysis()
    episode = {
        "survival": {"1": {"outages": [0, 2], "unavailable_slots": [0, 400]},
                     "2": {"outages": [0, 1], "unavailable_slots": [0, 100]}},
        "survival_window": 20_000, "missed_packets": [[], [3, 4, 9]], "packets": [100, 50],
        "worst_violation_rate": 0.06, "violation_rate": [0.0, 0.06],
    }  # fmt: skip
    row = a.per_deployment(episode, 1)
    assert row["worst_unavailability"] == 0.02 and row["worst_outages"] == 2
    assert row["mean_unavailability"] == 0.01 and row["any_outage"] == 1.0
    assert (row["pairs"], row["after_miss"]) == (1.0, 3.0)
    assert (row["misses"], row["isolated_misses"], row["misses_in_runs"]) == (3.0, 1.0, 2.0)
    assert a.per_deployment(episode, 2)["worst_unavailability"] == 0.005

    def methods(q, p2_sc=0.010, p2_ms=0.010):
        return {
            name: {"10": {"worst_unavailability": value, "w": p2_sc if name == "SC" else p2_ms}}
            for name, value in q.items()
        }

    base = {"SC": 0.008, "SC-B": 0.009, "M-S": 0.012, "M": 0.020, "M+S": 0.012, "M+Y": 0.011,
            "R17-like": 0.015}  # fmt: skip
    assert a.gate_a(methods(base))["decision"] == "continue"
    assert a.gate_a(methods(base))["best_rule"] == "M+Y"
    assert a.gate_a(methods({**base, "M-S": 0.009}))["decision"] == "adjust"  # reward alone
    assert a.gate_a(methods(base, p2_sc=0.012))["decision"] == "adjust"  # P2 > 1.10 M+S
    assert a.gate_a(methods({**base, "SC": 0.021}))["decision"] == "stop"


def test_stage_1b_carries_forward_the_eligible_lambda_with_the_lowest_q_u():
    a = _analysis()

    def methods(q, p2):
        return {name: {"10": {"worst_unavailability": value, "w": p2.get(name, 0.010)}}
                for name, value in q.items()}  # fmt: skip

    q = {"SC": 0.008, "SC-B": 0.007, "M-S": 0.03, "M": 0.020, "M+S": 0.012, "M+Y": 0.011,
         "R17-like": 0.015, "SC-B[0.1]": 0.006, "SC-B[0.3]": 0.008}  # fmt: skip
    selection = a.select_1b(methods(q, {}))
    assert selection["best_rule"] == "M+Y" and selection["chosen"] == "SC-B[0.1]"
    # the lower Q_U fails the P2 guard: the other candidate is carried forward
    selection = a.select_1b(methods(q, {"SC-B[0.1]": 0.0111}))
    assert selection["eligible"] == ["SC-B[0.3]"] and selection["chosen"] == "SC-B[0.3]"
    # neither is 20% below the best rule: none is chosen
    selection = a.select_1b(methods({**q, "SC-B[0.1]": 0.0089, "SC-B[0.3]": 0.0095}, {}))
    assert selection["chosen"] is None and selection["decision"] == "none eligible"
