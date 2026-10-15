"""EXP-DSA05-012 stage 0' (design §11.5): SP, learned protection on a frozen base M.

Checks: the expanded actor and critic reproduce M; the gate's yield part is the M+S
yield set; ungated UEs take M's proposal; the actions and old log probabilities of
ungated decisions do not enter the update (the direct loss mask; ungated slots still
shape the recurrent state); branch episodes restore both recurrent states and log
their ages and lengths; the evaluation path, fed the recorded training draws, follows
the training episode; the configs, the checkpoint guards and gate A2.
"""

import copy
import importlib.util
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from dsa_period import splits_v2
from dsa_period.config import load_scenario
from dsa_period.marl.features import decode_actions
from dsa_period.marl.networks import ChannelSetActor, SetCritic
from dsa_period.marl.ppo import PPOSettings
from dsa_period.marl.protection import (
    ProtectionPolicy,
    ProtectionSettings,
    expand_actor_state,
    expand_critic_state,
    protection_gate,
)
from dsa_period.marl.trainer_v2 import (
    BranchSettings,
    CapacityConstraint,
    CapacityTrainer,
    CapacityTrainingSettings,
    load_capacity_policy,
    model_dims,
)
from dsa_period.rules_v2 import make_policy_v2
from dsa_period.survival import SurvivalSettings
from dsa_period.urgency import SummarySettings, YieldPolicy

ROOT = Path(__file__).parents[1]
V21 = ROOT / "configs" / "scenario-v2.1.toml"
SURVIVAL = SummarySettings("survival")


def _settings(env_ues=(12,), slots=40):
    return CapacityTrainingSettings(
        budget_slots=10**9, env_ues=env_ues, rollout_slots=slots, episode_slots=slots,
        hidden=16, critic_hidden=32,
    )  # fmt: skip


@pytest.fixture(scope="module")
def base_path(tmp_path_factory):
    """A small L0 base run (one PPO update, so its weights are not the initial ones)."""
    trainer = CapacityTrainer(
        load_scenario(V21, 20), _settings((12, 16)), PPOSettings(), CapacityConstraint("fixed"), 1
    )
    trainer.train_step(trainer.collect()[0], 0.01)
    path = tmp_path_factory.mktemp("base") / "step-000000001.pt"
    torch.save(trainer.checkpoint(1, 1), path)
    return path


def _sp(base_path, env_ues=(12,), slots=40, branching=None, seed=5):
    return CapacityTrainer(
        load_scenario(V21, 20), _settings(env_ues, slots), PPOSettings(),
        CapacityConstraint("fixed"), seed, information=SURVIVAL, survival=SurvivalSettings(),
        branching=branching, protection=ProtectionSettings(str(base_path)),
    )  # fmt: skip


def test_expanded_actor_and_critic_reproduce_m():
    scenario = load_scenario(V21, 8)
    config, menu = scenario.physical, scenario.menu
    k, levels = config.num_subchannels, menu.num_levels - 1
    (t0, c0, d0), (t1, c1, d1) = (
        model_dims(config, menu, None, None),
        model_dims(config, menu, SURVIVAL, SurvivalSettings()),
    )
    assert (t1, c1) == (t0 + 1, c0 + 5)  # b_k per token; survival state and own bit
    torch.manual_seed(0)
    base, target = ChannelSetActor(t0, c0, k, levels, 16), ChannelSetActor(t1, c1, k, levels, 16)
    target.load_state_dict(expand_actor_state(base.state_dict(), (t0, c0), (t1, c1), 16))
    tokens, context = torch.randn(9, k, t0), torch.randn(9, c0)
    wide_tokens = torch.cat((tokens, torch.randn(9, k, 1)), dim=-1)
    wide_context = torch.cat((context, torch.randn(9, 5)), dim=-1)
    hidden = torch.randn(9, k, 16)
    for _ in range(3):
        a, ha = base.step(tokens, context, hidden)
        b, hb = target.step(wide_tokens, wide_context, hidden)
        assert torch.allclose(a, b, atol=1e-6) and torch.allclose(ha, hb, atol=1e-6)
        hidden = ha
    critic, wide = SetCritic(d0, 32), SetCritic(d1, 32)
    wide.load_state_dict(expand_critic_state(critic.state_dict(), (t0, c0), (t1, c1), k))
    per_ue = torch.randn(9, d0)
    flat = per_ue[:, : k * t0].reshape(9, k, t0)
    widened = torch.cat(
        (torch.cat((flat, torch.randn(9, k, 1)), -1).reshape(9, -1), per_ue[:, k * t0 : -1],
         torch.randn(9, 5), per_ue[:, -1:]), dim=1,
    )  # fmt: skip
    segment = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2])
    assert torch.allclose(critic(per_ue, segment, 3), wide(widened, segment, 3), atol=1e-6)


def test_the_trainer_starts_sp_from_m_and_keeps_m_frozen(base_path):
    trainer = _sp(base_path)
    base = torch.load(base_path, weights_only=False)
    t0, c0 = base["model"]["token_dim"], base["model"]["context_dim"]
    assert np.array_equal(trainer.tokens[:, :, :t0], trainer.base_tokens)
    assert np.array_equal(trainer.context[:, :c0], trainer.base_context)
    with torch.no_grad():
        a, _ = trainer.base_actor.step(
            torch.from_numpy(trainer.base_tokens), torch.from_numpy(trainer.base_context),
            trainer.base_actor.initial_hidden(12),
        )  # fmt: skip
        b, _ = trainer.ppo.actor.step(
            torch.from_numpy(trainer.tokens), torch.from_numpy(trainer.context),
            trainer.ppo.actor.initial_hidden(12),
        )  # fmt: skip
    assert torch.allclose(a, b, atol=1e-6)
    before = copy.deepcopy(trainer.base_actor.state_dict())
    for _ in range(3):
        trainer.train_step(trainer.collect()[0], 0.01)
    after = trainer.base_actor.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)


class RuleSP(ProtectionPolicy):
    """SP whose response is the M+S rule: yield-gated UEs stay OFF, state-gated act as M."""

    def respond(self, ues, logits, legal, proposed):
        out = proposed[ues].copy()
        out[~self.tracker.in_state[ues]] = 0
        return out


def _episode(n=12, instance=1_000_005, slots=800):
    scenario = load_scenario(V21, n)
    env = scenario.make_env(*scenario.instance(instance),
                            splits_v2.training_fading_seeds(np.random.default_rng(instance), n),
                            horizon_slots=slots)  # fmt: skip
    rngs = lambda: [np.random.default_rng([instance, ue]) for ue in range(n)]  # noqa: E731
    return scenario, env, rngs


def test_the_gate_case_by_case():
    active = np.array([True, True, True, True, True, False])
    state = np.array([True, False, False, False, False, False])
    proposed = np.array([[1, 2], [1, 2], [0, 1], [1, 0], [1, 3], [1, 2]])  # (subchannel, power)
    marked = np.array([False, True])
    own = np.array([False, False, False, False, True, False])
    prev_channel = np.array([0, 0, 0, 0, 1, 0])
    by_state, by_yield = protection_gate(active, state, proposed, marked, own, prev_channel)
    assert by_state.tolist() == [True, False, False, False, False, False]
    # UE 1 sends on a marked subchannel; UE 2's is unmarked; UE 3 stays OFF; UE 4's mark is
    # its own bit of the last slot on the same subchannel; UE 5 has no packet.
    assert by_yield.tolist() == [False, True, False, False, False, False]
    prev_channel[4] = 0  # its own bit was on another subchannel: the mark is someone else's
    assert protection_gate(active, state, proposed, marked, own, prev_channel)[1][4]


def test_the_yield_gate_is_the_m_plus_s_yield_set():
    scenario, env, rngs = _episode()
    config, menu = scenario.physical, scenario.menu
    tokens, context, _ = model_dims(config, menu, SURVIVAL, SurvivalSettings())
    actor = ChannelSetActor(tokens, context, config.num_subchannels, menu.num_levels - 1, 16)
    duals = CapacityConstraint("fixed").duals(0.01)
    sp = RuleSP(make_policy_v2("random_max", config, menu), actor, config, menu, duals,
                SURVIVAL, SurvivalSettings())  # fmt: skip
    ms = YieldPolicy(make_policy_v2("random_max", config, menu), config, menu, SURVIVAL)
    sp.reset(rngs())
    ms.reset(rngs())
    observations = env.reset()
    for _ in range(800):
        a, b = sp.act(observations), ms.act(observations)
        assert np.array_equal(a, b)
        assert np.array_equal(sp.link.sent, ms.link.sent)
        observations = env.step(a).observations
    assert sp.counts["gated_yield"] == ms.deferred.sum() > 0
    assert sp.counts["gated_state"] > 0


def test_ungated_ues_act_as_m_and_gated_ones_draw_once_more(base_path):
    scenario, env, rngs = _episode()
    trainer = _sp(base_path)
    policy = _load(trainer, base_path.parent / "sp.pt", scenario)
    proposals = []
    inner_act = policy.base.act

    def record(observations):
        out = inner_act(observations)
        proposals.append(np.array(out, copy=True))
        return out

    policy.base.act = record
    policy.reset(rngs())
    observations = env.reset()
    gated_any = False
    for _ in range(800):
        actions = policy.act(observations)
        by_state, by_yield = policy.last_gate
        gated = by_state | by_yield
        assert np.array_equal(actions[~gated], proposals[-1][~gated])
        gated_any |= bool(gated.any())
        observations = env.step(actions).observations
    assert gated_any


def _load(trainer, path, scenario):
    torch.save(trainer.checkpoint(1, 1), path)
    policy = load_capacity_policy(path, scenario.physical, scenario.menu)
    assert isinstance(policy, ProtectionPolicy)
    return policy


def test_ungated_actions_and_log_probs_do_not_enter_the_update(base_path):
    trainer = _sp(base_path, env_ues=(12, 16), slots=60, branching=BranchSettings(1.0, 1, 8))
    rollout = None
    for _ in range(6):  # until a rollout holds gated decisions
        rollout, _ = trainer.collect()
        if rollout.active.any():
            break
    assert rollout.active.any()
    packet = rollout.legal[..., 1:].any(-1)
    assert not (rollout.active & ~packet).any()
    assert (packet & ~rollout.active).any()
    noisy = copy.deepcopy(rollout)
    ungated = ~rollout.active
    noisy.actions[ungated] = 0  # OFF is always legal
    noisy.log_probs[ungated] = -3.0
    a, b = copy.deepcopy(trainer.ppo), copy.deepcopy(trainer.ppo)
    metrics = a.update(rollout, 0.01)
    b.update(noisy, 0.01)
    assert metrics["replay_log_prob_error"] < 1e-5
    sa, sb = a.actor.state_dict(), b.actor.state_dict()
    assert all(torch.equal(sa[name], sb[name]) for name in sa)
    assert not all(torch.equal(sa[name], trainer.ppo.actor.state_dict()[name]) for name in sa)


def test_branches_restore_both_recurrent_states(base_path):
    trainer = _sp(base_path, env_ues=(16, 20), slots=60, branching=BranchSettings(1.0, 1, 8))
    errors = []
    for _ in range(4):
        rollout, stats = trainer.collect()
        errors.append(trainer.train_step(rollout, 0.01)["replay_log_prob_error"])
    assert trainer.branch_counts["branches"] > 0 and max(errors) < 1e-5
    assert {"gated_state", "gated_yield", "gated_share_of_active"} <= set(stats["protection"])
    n = int(trainer.env_ues[0])
    snapshot = trainer.snapshots[n][0]
    trainer._branch_env(0, snapshot)
    rows = trainer._rows(0)
    hidden, tokens, context = snapshot["base"]
    assert torch.equal(trainer.base_hidden[rows], hidden) and torch.equal(
        trainer.hidden[rows], snapshot["hidden"]
    )
    assert np.array_equal(trainer.base_tokens[rows], tokens)
    assert np.array_equal(trainer.base_context[rows], context)
    assert np.array_equal(trainer.marked[0], snapshot["marked"])


def test_branch_logs_record_snapshot_ages_lengths_and_gated_exposure(base_path):
    trainer = _sp(base_path, env_ues=(16, 20), slots=60, branching=BranchSettings(1.0, 1, 8))
    logs = []
    for _ in range(5):
        rollout, stats = trainer.collect()
        trainer.train_step(rollout, 0.01)
        logs.append(stats)
    ages = [age for stats in logs for age in stats["branching"]["started_age"]]
    ended = [
        (length, nominal)
        for stats in logs
        for length, nominal in zip(
            stats["branching"]["ended_length"], stats["branching"]["ended_nominal"]
        )
    ]
    assert len(ages) == trainer.branch_counts["branches"] and min(ages) >= 0
    assert max(ages) > 0  # snapshots outlive PPO updates
    assert ended and all(0 < length <= nominal for length, nominal in ended)
    short = sum(stats["branching"]["ended_short"] for stats in logs)
    assert short == sum(length < nominal for length, nominal in ended)
    gated = sum(stats["protection"]["gated_state"] + stats["protection"]["gated_yield"]
                for stats in logs)  # fmt: skip
    inside = sum(stats["protection"]["gated_in_branches"] for stats in logs)
    assert 0 < inside <= gated


def test_the_evaluation_path_follows_a_training_episode_given_its_draws(base_path, tmp_path):
    trainer = _sp(base_path, env_ues=(12,), slots=40, seed=9)
    for _ in range(3):  # move the actor away from M first
        trainer.train_step(trainer.collect()[0], 0.01)
    assert trainer.episode_slot[0] == 0  # rollouts of 40 slots hold whole episodes
    env = copy.deepcopy(trainer.envs[0])  # the episode the next rollout plays
    rollout, _ = trainer.collect()
    scenario = load_scenario(V21, 12)
    policy = _load(trainer, tmp_path / "sp.pt", scenario)
    menu = scenario.menu
    script = {"t": 0}

    class ScriptedBase:
        def reset(self, rngs):
            pass

        def act(self, observations):
            return decode_actions(rollout.base_actions[script["t"]], menu)

    seen = {}

    class Scripted(ProtectionPolicy):
        def features(self, obs):
            tokens, context, marked = super().features(obs)
            seen["tokens"], seen["context"] = tokens, context
            return tokens, context, marked

        def respond(self, ues, logits, legal, proposed):
            return decode_actions(rollout.actions[script["t"]][ues], menu)

    policy.__class__ = Scripted
    policy.base = ScriptedBase()
    policy.reset([np.random.default_rng(ue) for ue in range(12)])
    observations = env.reset()
    for t in range(40):
        script["t"] = t
        actions = policy.act(observations)
        by_state, by_yield = policy.last_gate
        assert np.array_equal(by_state | by_yield, rollout.active[t])
        assert np.array_equal(seen["tokens"], rollout.tokens[t])
        assert np.array_equal(seen["context"], rollout.context[t])
        assert np.array_equal(actions, decode_actions(rollout.actions[t], menu))
        observations = env.step(actions).observations
    assert rollout.active.any()


def test_sp_guards(base_path, tmp_path):
    scenario = load_scenario(V21, 20)
    args = (scenario, _settings(), PPOSettings(), CapacityConstraint("fixed"), 1)
    with pytest.raises(ValueError, match="survival"):
        CapacityTrainer(*args, survival=SurvivalSettings(),
                        protection=ProtectionSettings(str(base_path)))  # fmt: skip
    with pytest.raises(ValueError, match="SHA-256"):
        CapacityTrainer(*args, information=SURVIVAL, survival=SurvivalSettings(),
                        protection=ProtectionSettings(str(base_path), "0" * 64))  # fmt: skip
    sc = CapacityTrainer(*args, information=SURVIVAL, survival=SurvivalSettings())
    torch.save(sc.checkpoint(1, 1), tmp_path / "sc.pt")
    with pytest.raises(ValueError, match="L0"):
        CapacityTrainer(*args, information=SURVIVAL, survival=SurvivalSettings(),
                        protection=ProtectionSettings(str(tmp_path / "sc.pt")))  # fmt: skip
    trainer = _sp(base_path)
    checkpoint = trainer.checkpoint(1, 1)
    assert checkpoint["protection"]["base"] == str(base_path)
    with pytest.raises(ValueError, match="resumed"):
        _sp(base_path).resume(checkpoint)


def test_pilot_2_configs_add_only_the_protection_base():
    bases = {
        "101": "outputs/exp012/pilot/m-seed101/checkpoints/step-005001472.pt",
        "202": "outputs/exp012/pilot/m-seed202/checkpoints/step-005001472.pt",
    }
    pairing = {"101": "101", "202": "202", "303": "101", "404": "202"}
    for key, source in (("sp", "sc"), ("spb", "scb")):
        raw = tomllib.loads((ROOT / f"configs/exp012-pilot1c-{key}.toml").read_text())
        old = tomllib.loads((ROOT / f"configs/exp012-pilot-{source}.toml").read_text())
        assert set(raw) == set(old) | {"protection"}
        for section in set(old) - {"experiment"}:
            assert raw[section] == old[section], (key, section)
        assert raw["experiment"]["seeds"] == [101, 202, 303, 404]
        assert raw["survival"]["outage_slot_weight"] == 1.0
        table = raw["protection"]
        assert table["base"] == {seed: bases[m] for seed, m in pairing.items()}
        digests = {table["base_sha256"][seed] for seed in ("101", "202")}
        assert len(digests) == 2
        assert all(table["base_sha256"][s] == table["base_sha256"][m] for s, m in pairing.items())


def test_the_pilot_2_slice_is_registered_apart_from_every_other_set():
    manifest = json.loads((ROOT / "docs/research/exp012-r4-sets.json").read_text())
    assert manifest["scenario_config"] == "configs/scenario-v2.1.toml"
    assert list(manifest["screening"]) == ["exp012-pilot2-256"]
    entry = manifest["screening"]["exp012-pilot2-256"]
    assert (entry["start"], entry["stop"], entry["grid"]) == (1656, 1912, [8, 10, 12])
    start, stop = splits_v2.SCREENING_SETS["exp012-pilot2-256"]["range"]
    for other, registered in splits_v2.SCREENING_SETS.items():
        if other != "exp012-pilot2-256":
            a, b = registered["range"]
            assert stop <= a or b <= start, other
    for a, b in manifest["usage_checked_on_registration"]["v2_dev_ranges_used"]:
        assert stop <= a or b <= start
    digest = splits_v2.instance_digest(load_scenario(V21, 10), range(1656, 1912), "dev")
    assert entry["digests"]["10"] == digest


@pytest.fixture
def analysis():
    spec = importlib.util.spec_from_file_location("a12", ROOT / "scripts/analyze_exp012.py")
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    return a


def gate_methods(q=None, p2=None, seeds=None, mean_rate=None):
    q = {"SP": 0.006, "SP-B": 0.007, "SP0": 0.010, "M": 0.020, "M+S": 0.012, "M+Y": 0.011,
         "R17-like": 0.015, "SC-B": 0.009, **(q or {})}  # fmt: skip
    p2, seeds, mean_rate = p2 or {}, seeds or {}, mean_rate or {}
    return {
        name: {"10": {"worst_unavailability": value, "w": p2.get(name, 0.010),
                      "mean_rate": mean_rate.get(name, 0.005),
                      "by_seed_w": seeds.get(name, [p2.get(name, 0.010)] * 2)}}
        for name, value in q.items()
    }  # fmt: skip


def test_gate_a2_rule(analysis):
    gate = analysis.gate_a2(gate_methods())
    assert gate["best_reference"] == "SP0" and gate["decision"] == "continue"
    assert gate["passing"] == ["SP", "SP-B"] and gate["carried"] == "SP"
    # SP0 is a reference like the rules: 20% below it is required
    gate = analysis.gate_a2(gate_methods(q={"SP0": 0.008}))
    assert gate["passing"] == ["SP"]
    # one SP seed fails the reliability bound: SP-B is carried
    gate = analysis.gate_a2(gate_methods(seeds={"SP": [0.009, 0.0126, 0.009, 0.009]}))
    assert gate["passing"] == ["SP-B"] and gate["carried"] == "SP-B"
    # the mean P2 bound
    gate = analysis.gate_a2(gate_methods(p2={"SP": 0.0111, "SP-B": 0.0111}))
    assert gate["decision"] == "adjust" and gate["carried"] is None
    # learning no better than M+S, or no better than the untrained SP0: stop
    assert analysis.gate_a2(gate_methods(q={"SP": 0.012, "SP-B": 0.013}))["decision"] == "stop"
    assert analysis.gate_a2(gate_methods(q={"SP0": 0.005}))["decision"] == "stop"


@pytest.mark.parametrize("variant_q", [0.0, 0.006])
def test_gate_a2_zero_best_reference_is_insufficient_evidence(analysis, variant_q):
    gate = analysis.gate_a2(gate_methods(q={"SP0": 0.0, "SP": variant_q, "SP-B": variant_q}))
    assert gate["decision"] == "insufficient_evidence"
    assert gate["carried"] is None and gate["passing"] == []
    assert "zero" in gate["reason"] and "relative improvement" in gate["reason"]
    for row in gate["variants"].values():
        assert row["ratio_to_best_reference"] is None and row["ratio_to_SP0"] is None
        assert row["checks"]["Q_U_vs_best_reference"] is False


@pytest.mark.parametrize("field", ["p2", "mean_rate"])
@pytest.mark.parametrize("variant_rate", [0.0, 0.001])
def test_gate_a2_zero_reliability_reference_cannot_establish_noninferiority(
    analysis, field, variant_rate
):
    rates = {"M+S": 0.0, "SP": variant_rate, "SP-B": variant_rate}
    gate = analysis.gate_a2(gate_methods(**{field: rates}))
    assert gate["decision"] == "insufficient_evidence" and gate["passing"] == []
    assert gate["carried"] is None
    metric = "P2" if field == "p2" else "mean_rate"
    assert metric in gate["reason"] and "noninferiority cannot be established" in gate["reason"]
    for row in gate["variants"].values():
        assert row[f"ratio_{metric}_to_M+S"] is None
        assert row["checks"][f"{metric}_vs_M+S"] is False
        if field == "p2":
            assert row["ratio_P2_by_seed"] is None
            assert row["checks"]["P2_every_seed"] is False
    # Missing reliability evidence also takes priority over the ordinary stop branch.
    stopped = analysis.gate_a2(gate_methods(q={"SP": 0.012, "SP-B": 0.013}, **{field: rates}))
    assert stopped["decision"] == "insufficient_evidence" and stopped["carried"] is None


def test_gate_a2_mean_rate_guard_and_exact_tie(analysis):
    gate = analysis.gate_a2(gate_methods(mean_rate={"SP": 0.00551}))
    assert gate["variants"]["SP"]["checks"]["P2_vs_M+S"] is True
    assert gate["variants"]["SP"]["checks"]["mean_rate_vs_M+S"] is False
    assert gate["passing"] == ["SP-B"] and gate["carried"] == "SP-B"
    gate = analysis.gate_a2(gate_methods(mean_rate={"SP": 0.00551, "SP-B": 0.00551}))
    assert gate["decision"] == "adjust" and gate["carried"] is None
    gate = analysis.gate_a2(gate_methods(q={"SP": 0.007}))
    assert gate["passing"] == ["SP", "SP-B"] and gate["carried"] == "SP-B"
    assert gate["tie_break"] == "SP-B"


def test_gate_a2_per_seed_bound_uses_overall_m_plus_s_mean(analysis):
    gate = analysis.gate_a2(
        gate_methods(seeds={"M+S": [0.009, 0.011], "SP": [0.0124, 0.009, 0.009, 0.009]})
    )
    assert gate["variants"]["SP"]["checks"]["P2_every_seed"] is True
    assert gate["variants"]["SP"]["ratio_P2_by_seed"][0] == pytest.approx(1.24)
    reference = gate["reliability_reference"]
    assert reference["P2"] == 0.010 and reference["mean_rate"] == 0.005
    assert reference["aggregation"] == "equal_weight_mean_over_both_base_policy_layers"
    assert reference["per_seed_P2_denominator"] == "overall_M+S_mean"


def test_pilot1c_cli_zero_ratios_render_na(analysis, monkeypatch, tmp_path, capsys):
    evaluation = tmp_path / "synthetic-evaluation"
    evaluation.mkdir()
    (evaluation / "summary.json").write_text(
        json.dumps(
            {"formal": True, "provenance": {"dirty": False}, "config": "configs/scenario-v2.1.toml"}
        )
    )
    plan, output = tmp_path / "plan.json", tmp_path / "analysis.json"
    plan.write_text("{}")
    methods = gate_methods(q={"SP0": 0.0}, p2={"M+S": 0.0}, mean_rate={"M+S": 0.0})

    def synthetic_pilot(*args, **kwargs):
        assert kwargs["conditional_on_fitted_policies"] is True
        return {"methods": {"s=1": methods}}

    monkeypatch.setattr(analysis, "pilot", synthetic_pilot)
    analysis.main(
        ["pilot1c", "--evaluations", str(evaluation), "--plan", str(plan), "--output", str(output)]
    )
    stdout = capsys.readouterr().out
    assert stdout.count("N/A") == 6
    assert "insufficient_evidence" in stdout
    assert json.loads(output.read_text())["gate_A2"]["carried"] is None


@pytest.mark.parametrize("conditional", [False, True])
def test_pilot_bootstrap_keeps_fitted_policy_layers_fixed_only_when_requested(
    analysis, monkeypatch, conditional
):
    # Layer variation must remain fixed, while paired deployment effects cancel.
    instances = [0, 1, 2]
    plan = {
        "grid": [10],
        "groups": {"SP": [{"10": f"SP-{i}"} for i in range(4)],
                   "M+S": [{"10": f"M+S-{i}"} for i in range(2)]},
        "base_seed": {"101": 101, "202": 202, "303": 101, "404": 202},
    }  # fmt: skip
    monkeypatch.setattr(analysis.base, "load", lambda *args, **kwargs: ({}, instances))
    monkeypatch.setattr(analysis, "DRAWS", 1000)

    def synthetic_matrix(episodes, layers, n, field, instances, s):
        values = np.arange(1, len(layers) + 1)[:, None] + np.array([0, 10, 20])[None, :]
        return values * (0.001 if field == "mean_rate" else 1.0)

    monkeypatch.setattr(analysis, "matrix", synthetic_matrix)
    compare = analysis.base.compare
    seen_shapes = []

    def record_compare(a, b, draws):
        seen_shapes.append((a.shape, b.shape))
        return compare(a, b, draws)

    monkeypatch.setattr(analysis.base, "compare", record_compare)
    result = analysis.pilot(
        [], plan, ("SP",), order=("SP", "M+S"), conditional_on_fitted_policies=conditional
    )
    a = synthetic_matrix({}, plan["groups"]["SP"], 10, "w", instances, 1)
    b = synthetic_matrix({}, plan["groups"]["M+S"], 10, "w", instances, 1)
    legacy = compare(a, b, analysis.DRAWS)
    row = result["methods"]["s=1"]["SP"]["10"]
    comparison = result["versus_SP"]["s=1"]["M+S"]["10"]
    assert comparison["w"]["difference"] == legacy["difference"]
    assert comparison["w"]["ratio"] == legacy["ratio"]
    assert legacy["ci95"][0] < legacy["ci95"][1]
    if conditional:
        assert all(shapes == ((1, 3), (1, 3)) for shapes in seen_shapes)
        assert comparison["w"]["ci95"] == [1.0, 1.0]
        assert comparison["mean_rate"]["ci95"] == pytest.approx([0.001, 0.001])
        assert row["by_seed_mean_rate"] == pytest.approx([0.011, 0.012, 0.013, 0.014])
        assert result["base_seed"] == plan["base_seed"]
        assert result["bootstrap"]["conditional_on_fitted_policies"] is True
        assert result["bootstrap"]["no_training_seed_uncertainty"] is True
        assert result["bootstrap"]["resampling_unit"] == "paired_deployment"
        assert result["bootstrap"]["fixed_policy_layers"] == {"SP": 4, "M+S": 2}
    else:
        assert all(shapes == ((4, 3), (2, 3)) for shapes in seen_shapes)
        assert comparison["w"] == legacy
        assert "mean_rate" not in comparison and "by_seed_mean_rate" not in row
        assert "bootstrap" not in result and "base_seed" not in result


M101 = ROOT / "outputs/exp012/pilot/m-seed101/checkpoints/step-005001472.pt"


@pytest.mark.skipif(not M101.exists(), reason="needs the pilot's M checkpoint")
def test_sp0_is_the_untrained_sp_checkpoint(tmp_path):
    from dsa_period.experiments.train_capacity_mappo import main

    output = tmp_path / "sp0"
    main(["--config", "configs/exp012-pilot1c-sp.toml", "--scenario", str(V21.relative_to(ROOT)),
          "--variant", "fixed", "--seed", "303", "--output", str(output),
          "--initial-checkpoint"])  # fmt: skip
    assert [p.name for p in (output / "checkpoints").iterdir()] == ["step-000000000.pt"]
    checkpoint = torch.load(output / "checkpoints/step-000000000.pt", weights_only=False)
    assert checkpoint["slots"] == 0 and checkpoint["protection"]["base"].endswith(
        "m-seed101/checkpoints/step-005001472.pt"
    )
    base = torch.load(M101, weights_only=False)
    expanded = expand_actor_state(
        base["ppo"]["actor"],
        (base["model"]["token_dim"], base["model"]["context_dim"]),
        (checkpoint["model"]["token_dim"], checkpoint["model"]["context_dim"]),
        base["model"]["hidden"],
    )
    actor = checkpoint["ppo"]["actor"]
    assert all(torch.equal(actor[name], expanded[name]) for name in actor)
    assert json.loads((output / "manifest.json").read_text())["initial_checkpoint_only"]
