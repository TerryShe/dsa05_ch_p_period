"""EXP-DSA05-011: urgency state summary, yield rule, summary training, selection and analysis."""

import dataclasses
import importlib.util
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from dsa_period import splits_v2
from dsa_period.capacity import run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.experiments import evaluate_capacity
from dsa_period.experiments import select_exp011 as sel
from dsa_period.experiments.train_capacity_mappo import information_from, settings_from
from dsa_period.marl.features import ObsArrays, stack
from dsa_period.marl.networks import ChannelSetActor, MessageActor, split_logits
from dsa_period.marl.policy import SummaryPolicy
from dsa_period.marl.ppo import PPOSettings
from dsa_period.marl.trainer_v2 import (
    CapacityConstraint,
    CapacityTrainer,
    CapacityTrainingSettings,
    _load_checkpoint,
    joint_with_message,
    model_dims,
)
from dsa_period.physics import shannon_bits
from dsa_period.rules_v2 import make_policy_v2
from dsa_period.urgency import (
    SummaryLink,
    SummarySettings,
    YieldPolicy,
    parse_wrapper,
    slack,
    urgent,
    visible,
    with_summary,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "scenario-v2.toml"


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _module("probe", "scripts/probe_broadcast.py")
analysis = _module("analyze_exp011", "scripts/analyze_exp011.py")
driver = _module("run_exp011", "scripts/run_exp011.py")


def short(n: int, horizon: int = 300):
    scenario = load_scenario(CONFIG, n)
    return dataclasses.replace(scenario, horizon_slots=horizon, warmup_slots=40)


def _obs(**fields) -> ObsArrays:
    n, k = 3, 4
    base = dict(
        remaining=np.zeros(n),
        packet=np.full(n, 1000.0),
        period=np.full(n, 10.0),
        to_deadline=np.full(n, 10.0),
        prev_channel=np.zeros(n, np.int64),
        prev_power=np.zeros(n, np.int64),
        prev_outcome=np.zeros(n, np.int64),
        prev_service=np.zeros(n),
        rssi=np.ones(k),
        own_gain=np.full((n, k), 1e-9),
    )
    base.update(fields)
    return ObsArrays(**base)


# ------------------------------------------------------------------ summary link
def test_slack_matches_the_probe():
    scenario = short(8)
    topology, traffic = scenario.instance(1000)
    env = scenario.make_env(topology, traffic, splits_v2.fading_seeds("dev", 8, 1000, 0))
    policy = make_policy_v2("best_local", scenario.physical, scenario.menu)
    policy.reset(splits_v2.execution_generators("dev", 8, 1000, 0))
    observations = env.reset()
    for _ in range(60):
        ours = slack(scenario.physical, scenario.menu, stack(observations))
        np.testing.assert_array_equal(ours, probe.slack(scenario, observations))
        observations = env.step(policy.act(observations)).observations


def test_link_reads_by_threshold_ors_per_subchannel_and_delays_one_slot():
    scenario = load_scenario(CONFIG, 3)
    config, menu = scenario.physical, scenario.menu
    link = SummaryLink(SummarySettings("occupancy", -10.0), config, menu, 3)
    threshold = float(shannon_bits(config, 0.1))
    assert link.threshold_bits == pytest.approx(threshold)
    sending = np.array([True, True, True])
    link.commit(np.ones(3, bool), sending)
    obs = _obs(
        prev_channel=np.array([1, 1, 3]),
        prev_power=np.array([5, 5, 5]),
        prev_service=np.array([threshold - 1.0, threshold, 0.0]),
    )
    np.testing.assert_array_equal(link.broadcast(obs), [False, True, False, False])
    np.testing.assert_array_equal(link.read, [False, True, False])
    assert (link.bits_sent, link.bits_read) == (3, 1)
    every = SummaryLink(SummarySettings("occupancy", None), config, menu, 3)
    every.commit(np.ones(3, bool), sending)
    np.testing.assert_array_equal(every.broadcast(obs), [False, True, False, True])
    link.commit(np.zeros(3, bool), np.zeros(3, bool))
    assert not link.broadcast(_obs()).any()  # bits live for exactly one slot


def test_link_refuses_a_bit_of_a_ue_that_did_not_transmit():
    scenario = load_scenario(CONFIG, 3)
    link = SummaryLink(SummarySettings(), scenario.physical, scenario.menu, 3)
    link.sent = np.array([True, False, False])
    with pytest.raises(AssertionError):
        link.broadcast(_obs())


def test_uplink_contents():
    scenario = load_scenario(CONFIG, 3)
    config, menu = scenario.physical, scenario.menu
    obs = _obs(
        remaining=np.array([500.0, 500.0, 0.0]),
        to_deadline=np.array([1.0, 30.0, 5.0]),
        own_gain=np.array([[1e-12] * 4, [1e-6] * 4, [1e-6] * 4]),
    )
    sending = np.array([True, True, False])
    hot = urgent(config, menu, obs)
    assert hot.tolist() == [True, False, False]
    for content, expected in (("urgency", [1, 0, 0]), ("occupancy", [1, 1, 0])):
        link = SummaryLink(SummarySettings(content), config, menu, 3)
        assert link.uplink(obs, sending).astype(int).tolist() == expected
    link = SummaryLink(SummarySettings("message"), config, menu, 3)
    assert link.uplink(obs, sending, np.array([0, 1, 1])).tolist() == [False, True, False]
    with pytest.raises(ValueError):
        link.uplink(obs, sending)
    for content, own in (("urgency", [True, True, False]), ("occupancy", [False] * 3)):
        link = SummaryLink(SummarySettings(content), config, menu, 3)
        link.commit(np.ones(3, bool), sending)
        assert link.own_previous.tolist() == own  # occupancy has no own bit (design §2.5)


def test_hiding_masks_act_on_the_named_ues_only():
    marked = np.array([True, False, True, False])
    hot = np.array([True, False, False])
    assert visible(marked, "none", hot).sum() == 6
    assert visible(marked, "all", hot).sum() == 0
    np.testing.assert_array_equal(visible(marked, "nonurgent", hot).any(axis=1), [1, 0, 0])
    np.testing.assert_array_equal(visible(marked, "urgent", hot).any(axis=1), [0, 1, 1])
    with pytest.raises(ValueError):
        visible(marked, "some", hot)


def test_summary_features_keep_the_l0_features_bit_for_bit():
    tokens = np.random.default_rng(0).normal(size=(3, 4, 15)).astype(np.float32)
    context = np.random.default_rng(1).normal(size=(3, 13)).astype(np.float32)
    seen = np.zeros((3, 4), bool)
    wide, longer = with_summary(tokens, context, seen, np.zeros(3, bool))
    np.testing.assert_array_equal(wide[:, :, :15], tokens)
    np.testing.assert_array_equal(longer[:, :13], context)
    assert not wide[:, :, 15].any() and not longer[:, 13].any()


@pytest.mark.parametrize(("mode", "theta"), [("B3", None), ("B3s", 0.0)])
def test_yield_rule_reproduces_the_probe(mode, theta):
    # Dev 1003 at N = 10 (probe slice): the probe defers 22 (B3) and 2 (B3s) times and
    # loses flags under B3s, so every branch of the rule and of the reading is exercised.
    scenario = short(10, horizon=600)
    inner = make_policy_v2("best_local", scenario.physical, scenario.menu)
    reference = probe.run(scenario, probe.Broadcast(inner, scenario, mode), 1003)
    assert reference["silenced"] > 0
    wrapped = YieldPolicy(
        make_policy_v2("best_local", scenario.physical, scenario.menu),
        scenario.physical,
        scenario.menu,
        SummarySettings(theta_db=theta),
    )
    ours = run_capacity_episode(scenario, wrapped, "dev", 1003)
    assert ours["violations"] == reference["violations"]
    assert int(wrapped.deferred.sum()) == reference["silenced"]
    assert wrapped.link.bits_sent == reference["flags_sent"]
    assert wrapped.link.bits_read == reference["flags_heard"]


def test_wrapper_specs():
    assert parse_wrapper("best_local") is None
    assert parse_wrapper("yield[theta=all]@best_local") == (
        "yield",
        {"theta_db": None},
        "best_local",
    )
    for bad in (
        "yield[]@best_local",
        "yield[hide=all]@best_local",
        "summary[theta=-5]@best_local",
        "summary[hide=some]@marl:x.pt",
        "yield[theta=-5]",
    ):
        with pytest.raises(ValueError):
            parse_wrapper(bad)
    scenario = load_scenario(CONFIG, 4)
    policy = make_policy_v2("yield[theta=-10]@best_local", scenario.physical, scenario.menu)
    assert isinstance(policy, YieldPolicy) and policy.settings.theta_db == -10.0


def test_evaluation_workers_pin_one_thread_for_wrapped_marl_specs(monkeypatch):
    """A forked worker left on the parent's torch thread pool deadlocks (EXP-011 run)."""
    calls = []
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    monkeypatch.setattr(evaluate_capacity, "make_policy_v2", lambda *args: None)
    monkeypatch.setattr(evaluate_capacity, "run_capacity_episode", lambda *args: {})
    for spec in (
        "marl:x.pt",
        "yield[theta=-10]@marl:x.pt",
        "summary[theta=0,hide=all]@marl:x.pt",
        "yield[theta=-10]@best_local",
    ):
        evaluate_capacity._episode((str(CONFIG), spec, 4, "dev", 0))
    assert calls == [1, 1, 1]


# --------------------------------------------------------------- actor and PPO
def test_message_actor_scores_actions_like_the_base_actor():
    torch.manual_seed(3)
    base = ChannelSetActor(16, 14, 4, 5, 32)
    torch.manual_seed(3)
    talker = MessageActor(16, 14, 4, 5, 32)
    tokens, context = torch.randn(5, 4, 16), torch.randn(5, 14)
    logits, hidden = base.step(tokens, context, base.initial_hidden(5))
    both, hidden2 = talker.step(tokens, context, talker.initial_hidden(5))
    actions, words = split_logits(both, talker.num_actions)
    torch.testing.assert_close(actions, logits)
    torch.testing.assert_close(hidden, hidden2)
    assert words.shape == (5, 2) and split_logits(logits, base.num_actions)[1] is None


def test_the_message_counts_only_when_the_ue_transmits():
    log_probs, entropy = torch.tensor([-1.0, -2.0]), torch.tensor([0.5, 0.5])
    probs = torch.tensor([[0.25, 0.75], [0.25, 0.75]])
    words = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # uniform: log 1/2, entropy log 2
    joint, total = joint_with_message(
        log_probs, entropy, probs, words, torch.tensor([0, 3]), torch.tensor([1, 1])
    )
    log2 = float(np.log(2))
    torch.testing.assert_close(joint, torch.tensor([-1.0, -2.0 - log2]))
    torch.testing.assert_close(total, torch.tensor([0.5 + 0.75 * log2] * 2))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Tiny summary trainings of every content, saved as checkpoints."""
    scenario = load_scenario(CONFIG, 8)
    settings = CapacityTrainingSettings(
        budget_slots=10**9, env_ues=(4, 6), rollout_slots=16, episode_slots=60
    )
    out = {}
    for content in ("urgency", "occupancy", "message"):
        trainer = CapacityTrainer(
            scenario, settings, PPOSettings(), CapacityConstraint("fixed"), 5,
            information=SummarySettings(content),
        )  # fmt: skip
        for _ in range(2):
            rollout, stats = trainer.collect()
            metrics = trainer.train_step(rollout, 0.01)
        path = tmp_path_factory.mktemp(content) / "step-000000001.pt"
        torch.save(trainer.checkpoint(1, 1), path)
        out[content] = (path, rollout, stats, metrics, trainer)
    return out


def test_summary_training_replays_its_joint_log_probabilities(trained):
    for content, (_, rollout, stats, metrics, trainer) in trained.items():
        assert metrics["replay_log_prob_error"] < 1e-5, content
        assert (rollout.messages is not None) == (content == "message")
        assert set(stats["summary"]) == {"bits_sent", "bits_read", "marked_share"}
        dims = model_dims(trainer.config, trainer.menu, trainer.information)
        assert rollout.tokens.shape[-1] == dims[0] and rollout.context.shape[-1] == dims[1]
        assert rollout.critic_ue.shape[-1] == dims[2]
        own_bit = rollout.context[..., -1]
        if content == "occupancy":
            assert not own_bit.any()  # no uplink bit (design §2.5)
        if content == "message":
            assert own_bit.any()  # random messages of transmitting UEs
    assert (
        trained["occupancy"][2]["summary"]["bits_sent"]
        >= trained["urgency"][2]["summary"]["bits_sent"]
    )


def test_summary_checkpoints_load_at_any_n_with_wrappers(trained):
    for content, (path, *_rest) in trained.items():
        _load_checkpoint.cache_clear()
        for n in (4, 12):
            scenario = short(n, horizon=120)
            policy = make_policy_v2(f"marl:{path}", scenario.physical, scenario.menu)
            assert isinstance(policy, SummaryPolicy) and policy.settings.content == content
            run_capacity_episode(scenario, policy, "dev", 1)
        wrapped = make_policy_v2(
            f"summary[theta=all,hide=urgent]@marl:{path}", scenario.physical, scenario.menu
        )
        assert wrapped.settings.theta_db is None and wrapped.hide == "urgent"


def test_a_summary_wrapper_refuses_an_l0_checkpoint(tmp_path):
    scenario = load_scenario(CONFIG, 4)
    settings = CapacityTrainingSettings(budget_slots=10**9, env_ues=(4,), rollout_slots=8)
    trainer = CapacityTrainer(scenario, settings, PPOSettings(), CapacityConstraint("fixed"), 1)
    path = tmp_path / "step-000000001.pt"
    torch.save(trainer.checkpoint(1, 1), path)
    with pytest.raises(ValueError):
        make_policy_v2(f"summary[hide=all]@marl:{path}", scenario.physical, scenario.menu)


# ------------------------------------------------------------- configs and sets
def test_configs_are_s1_plus_the_summary():
    s1 = tomllib.loads((ROOT / "configs" / "exp009-step2-s1.toml").read_text())
    expected = {
        "u": ("U", "urgency", -10.0),
        "o": ("O", "occupancy", -10.0),
        "l": ("L", "message", -10.0),
        "u0": ("U0", "urgency", 0.0),
    }
    for key, (condition, content, theta) in expected.items():
        raw = tomllib.loads((ROOT / "configs" / f"exp011-{key}.toml").read_text())
        assert {k: v for k, v in raw.items() if k not in ("experiment", "information")} == {
            k: v for k, v in s1.items() if k != "experiment"
        }
        assert raw["experiment"]["condition"] == condition
        assert raw["experiment"]["seeds"] == [101, 202, 303]
        assert information_from(raw) == SummarySettings(content, theta, 3)
        assert settings_from(raw, "fixed")[0].env_ues == tuple(s1["training"]["env_ues"])
    assert information_from(s1) is None


def test_eval_slice_is_registered_and_guarded():
    register = _module("register_sets", "scripts/register_exp008_sets.py")
    manifest = json.loads((ROOT / "docs" / "research" / "exp011-sets.json").read_text())
    assert manifest == json.loads(json.dumps(register.build("EXP-DSA05-011")))
    assert splits_v2.screening_instances("exp011-eval512") == list(range(5000, 5512))
    assert splits_v2.SCREENING_SETS["exp011-eval512"]["grid"] == (6, 8, 10, 12, 16)
    with pytest.raises(ValueError):
        splits_v2.check_dev_range([5511])


# ------------------------------------------------------------ selection by P2
def test_selection_prefers_low_p2_then_high_f_then_later():
    rows = [
        {"P2": 0.002, "F01": 0.9, "slots": 1},
        {"P2": 0.001, "F01": 0.8, "slots": 2},
        {"P2": 0.001, "F01": 0.9, "slots": 3},
        {"P2": 0.001, "F01": 0.9, "slots": 4},
    ]
    assert sel.best(rows)["slots"] == 4


def test_select_reads_conditions_from_manifests(tmp_path):
    screen = {}
    for condition, seed in (("U", 101), ("S1", 202)):
        run = tmp_path / f"{condition}-{seed}"
        (run / "checkpoints").mkdir(parents=True)
        (run / "manifest.json").write_text(
            json.dumps({"experiment": {"id": "x", "condition": condition}, "seed": seed})
        )
        for slots, p2 in ((1_000_000, 0.003), (2_000_000, 0.002)):
            spec = f"marl:{run}/checkpoints/step-{slots:09d}.pt"
            screen[spec] = {n: {"P2": p2 * n, "F01": 0.9} for n in (6, 8, 10, 12, 16, 20)}
    result = sel.select(screen, {"S1": "M"}, {"U": sel.GRID, "M": sel.GRID})
    assert set(result) == {"U", "M"}
    assert result["M"]["202"]["8"]["slots"] == 2_000_000
    assert "20" not in result["U"]["101"]
    with pytest.raises(ValueError):
        sel.select(screen, {"S1": "M"}, {"U": (6, 8, 24), "M": sel.GRID})


# -------------------------------------------------------------- analysis plan
def test_decision_rules():
    assert analysis.main_decision({8: -0.1, 10: -0.1, 12: -0.1}) == "effective"
    assert analysis.main_decision({8: -0.1, 10: 0.1, 12: -0.1}) == "partly effective"
    assert analysis.main_decision({8: 0.0, 10: 0.1, 12: 0.2}) == "no evidence"
    rows = {t: {"ci95": [-1.0, -0.1]} for t in analysis.THETAS}
    rows["0"] = {"ci95": [-1.0, 0.1]}
    assert analysis.boundary(rows) == "-5"
    rows = {t: {"ci95": [-1.0, 0.1]} for t in analysis.THETAS}
    assert analysis.boundary(rows) is None
    assert analysis.capacity_on((6, 8, 10), [0.99, 0.94, 0.9]) == 6
    assert analysis.capacity_on((6, 8, 10), [0.9, 0.94, 0.9]) is None


def _fake_methods() -> dict:
    def layers(tag, grid):
        return [{str(n): f"marl:outputs/exp011/{tag}-{s}/checkpoints/step-{n}.pt" for n in grid}
                for s in range(3)]  # fmt: skip

    return {
        **{c: layers(c, driver.GRID) for c in ("U", "O", "L", "M")},
        "U0": layers("U0", driver.CORE),
        "M_exp009": layers("M", driver.GRID),
        "best_local": "best_local",
        "cg_opt": "configured_grant_opt",
        "edf": "central_edf",
    }


def test_driver_plan_has_every_group_and_unique_chunks():
    plan = driver.plan(_fake_methods())
    groups, grids = plan["groups"], plan["grids"]
    for name in ("U", "M", "M+Y", "O", "L", "best_local+Y", "CG-opt", "N_EDF", "M_exp009"):
        assert grids[name] == list(driver.GRID), name
    for name in ("U0", "U@all", "U@-15", "U@-5", "U@0", "M+Y@0"):
        assert grids[name] == list(driver.CORE), name
    assert groups["M+Y"][0]["8"].startswith("yield[theta=-10]@marl:")
    assert groups["U-hide-nonurgent"][1]["6"].startswith("summary[hide=nonurgent]@marl:")
    pairs = driver.pairs_of(plan)
    labels = [driver.label_of(spec, n) for spec, n in pairs]
    assert len(set(labels)) == len(labels)
    # 8 learned groups x 3 layers x 5 N, 4 single specs x 5 N, U0 and the 8 threshold
    # groups x 3 layers x 3 N; M_exp009 equals M here, so it adds no chunk.
    assert len(pairs) == 8 * 3 * 5 + 4 * 5 + 9 * 3 * 3
    for name in plan["groups"]:
        for n in grids[name]:
            for member in groups[name] if isinstance(groups[name], list) else [groups[name]]:
                spec = member if isinstance(member, str) else member[str(n)]
                if spec.startswith(("yield", "summary")):
                    assert parse_wrapper(spec) is not None


def test_the_analysis_runs_on_every_group_of_the_plan():
    plan = json.loads(json.dumps(driver.plan(_fake_methods())))
    rng = np.random.default_rng(0)
    instances = list(range(30))
    episodes: dict = {}
    for spec, n in driver.pairs_of(plan):
        shift = 0.0 if spec.startswith("marl:outputs/exp011/U-") else 0.002
        rows = []
        for i in instances:
            rates = rng.uniform(0, 0.01, n) + shift * (i % 3 == 0)
            rows.append(
                {
                    "instance": i,
                    "worst_violation_rate": float(rates.max()),
                    "violation_rate": rates.tolist(),
                    "feasible": {"0.01": bool(rates.max() <= 0.01)},
                    "infeasible_for_all": {"0.01": i == 7},
                }
            )
        episodes.setdefault(spec, {})[n] = rows
    result = analysis.analyze(episodes, plan, instances, 200)
    assert result["decision"] in ("effective", "partly effective", "no evidence")
    assert set(result["D_N"]) == set(driver.GRID)
    assert result["coordination"]["enters_final_confirmation"] in ("U", "M+Y")
    assert set(result["receive"]["U"][8]["by_theta"]) == set(analysis.THETAS)
    assert result["sensitivity"][8]["excluding_infeasible_for_all"]["deployments"] == 29
    assert set(result["mechanism"][10]) >= {"all", "nonurgent", "urgent", "H_M1", "H_M2"}
    json.dumps(result)
