"""Phase-0 acceptance of EXP-DSA05-001: features, duals, networks, PPO, trainer."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import MENU, TOP, build_env

from dsa_period.baselines import estimate_link
from dsa_period.config import load_scenario
from dsa_period.evaluation import load_references, run_episode
from dsa_period.experiments import train_constrained_mappo
from dsa_period.marl.features import (
    CONTEXT_DIM,
    DualSettings,
    LocalMemory,
    Settlement,
    critic_state,
    decode_actions,
    encode_actions,
    num_actions,
    stack,
    token_dim,
)
from dsa_period.marl.networks import ChannelSetActor
from dsa_period.marl.policy import MarlPolicy, sample_from_logits
from dsa_period.marl.ppo import PPOSettings, discounted_gae
from dsa_period.marl.trainer import (
    ConstraintSettings,
    Trainer,
    TrainingSettings,
    build_models,
    load_policy_factory,
)

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs"
EXPERIMENT = CONFIGS / "exp001-constrained-mappo.toml"
LOCAL = DualSettings("local", epsilon=0.05, eta=2.0, mu_max=10.0)


def scenario(n):
    return load_scenario(CONFIGS / f"scenario-v1-n{n}.toml")


def tiny_settings(**changes):
    base = dict(
        budget_slots=64,
        episode_slots=40,
        num_envs=2,
        rollout_slots=16,
        validation_every_slots=32,
        hidden=16,
        critic_hidden=32,
    )
    return TrainingSettings(**(base | changes))


def test_action_codec_is_channel_major_and_round_trips():
    sc = scenario(8)
    total = num_actions(sc.physical, sc.menu)
    assert total == 21
    rows = decode_actions(np.arange(total), sc.menu)
    assert rows[0].tolist() == [0, 0]
    assert rows[1].tolist() == [0, 1] and rows[5].tolist() == [0, 5] and rows[6].tolist() == [1, 1]
    assert encode_actions(rows, sc.menu).tolist() == list(range(total))


def test_local_settlements_match_the_environment_over_a_random_episode():
    sc = scenario(8)
    topology, traffic = sc.instance(3)
    env = sc.make_env(topology, traffic)
    memory = LocalMemory(sc.physical, sc.menu, 8, LOCAL)
    memory.observe(stack(env.reset()))
    rng = np.random.default_rng(0)
    counts = [0, 0]
    for _ in range(600):
        # Mostly OFF, otherwise the lowest power on a random subchannel: both outcomes occur.
        index = np.where(rng.random(8) < 0.8, 0, 1 + 5 * rng.integers(0, 4, size=8))
        step = env.step(decode_actions(index, sc.menu))
        settlement = memory.observe(stack(step.observations))
        assert np.array_equal(settlement.violated, step.violations.astype(bool))
        assert np.array_equal(settlement.settled, (step.violations + step.on_time) > 0)
        counts[0] += int(step.violations.sum())
        counts[1] += int(step.on_time.sum())
    assert counts[0] > 0 and counts[1] > 0, "the check must see both outcomes"


def test_gain_and_interference_estimates_are_local_and_remembered():
    env = build_env([[30.0, 0.0], [0.0, 60.0]], [4, 4], [1e9, 1e9], [0, 0], num_subchannels=2)
    memory = LocalMemory(env.config, MENU, 2, LOCAL)
    memory.observe(stack(env.reset()))
    step = env.step([[0, TOP], [0, 2]])
    memory.observe(stack(step.observations))
    true = env.channel.gains(0)[:, 0]
    assert memory.gain == pytest.approx(true, rel=1e-9)
    _, interference = estimate_link(step.observations[0], env.config, MENU)
    assert memory.interference[0] == pytest.approx(interference, rel=1e-12)
    step = env.step([[0, 0], [1, 1]])
    memory.observe(stack(step.observations))
    assert memory.gain[0] == pytest.approx(true[0], rel=1e-9), "kept while silent"


def test_local_duals_move_only_with_own_settlements_and_stay_clipped():
    memory = LocalMemory(scenario(4).physical, scenario(4).menu, 4, LOCAL)
    memory.reset(np.array([0.05, 5.0, 9.5, 3.0]))
    settled = np.array([True, True, True, False])
    violated = np.array([False, True, True, False])
    memory.update_duals(Settlement(settled, violated))
    assert memory.mu == pytest.approx([0.0, 5.0 + 2 * 0.95, 10.0, 3.0])
    fixed = LocalMemory(scenario(4).physical, scenario(4).menu, 4, DualSettings("fixed", 0.05, 3.0))
    fixed.update_duals(Settlement(settled, violated))
    assert fixed.mu.tolist() == [0.0] * 4


def test_actor_features_exclude_topology_and_expose_mu_only_for_local_duals():
    sc = scenario(8)
    topology, traffic = sc.instance(3)
    env = sc.make_env(topology, traffic)
    obs = stack(env.reset())
    features = {}
    for variant in ("fixed", "local"):
        memory = LocalMemory(sc.physical, sc.menu, 8, DualSettings(variant, 0.05, 2.0, 1.0, 10.0))
        memory.reset(np.full(8, 4.0) if variant == "local" else np.zeros(8))
        memory.observe(obs)
        features[variant] = memory.features(obs)
    tokens, context = features["local"]
    assert tokens.shape == (8, 4, token_dim(sc.menu)) and context.shape == (8, CONTEXT_DIM)
    assert context[:, 12] == pytest.approx(0.4)
    assert np.all(features["fixed"][1][:, 12] == 0)
    np.testing.assert_array_equal(features["fixed"][0], tokens)
    state = critic_state(tokens, context, topology.ue_ap_gain_linear, np.full(8, 4.0), 10.0)
    assert state.shape == (8 * (4 * token_dim(sc.menu) + CONTEXT_DIM + 2),)


def test_coverage_marks_levels_that_meet_the_pace():
    env = build_env([[40.0, 0.0]], [10], [30_000.0], [0], num_subchannels=2)
    memory = LocalMemory(env.config, MENU, 1, LOCAL)
    memory.observe(stack(env.reset()))
    obs = stack(env.step([[1, TOP]]).observations)
    memory.observe(obs)
    tokens, _ = memory.features(obs)
    levels = MENU.num_levels - 1
    coverage = tokens[0, 0, 3 + levels :]
    assert np.all(np.diff(coverage) >= 0), "more power never covers less"
    required = obs.remaining[0] / obs.to_deadline[0]
    bits = 1000 * np.log2(
        1 + np.asarray(MENU.power_w[1:]) * memory.gain[0] / memory.interference[0, 0]
    )
    assert np.array_equal(coverage >= 0, bits >= required)


def test_actor_sequence_equals_stepping_with_resets():
    torch.manual_seed(0)
    actor = ChannelSetActor(13, CONTEXT_DIM, 4, 5, 16)
    tokens, context = torch.randn(5, 3, 4, 13), torch.randn(5, 3, CONTEXT_DIM)
    starts = torch.zeros(5, 3, dtype=torch.bool)
    starts[2, 1] = True
    hidden = torch.randn(3, 4, 16)
    replay, _ = actor.sequence(tokens, context, hidden, starts)
    h = hidden
    for t in range(5):
        h = h * (~starts[t]).float()[:, None, None]
        logits, h = actor.step(tokens[t], context[t], h)
        torch.testing.assert_close(replay[t], logits)
    assert replay.shape == (5, 3, 21)


def test_discounted_gae_matches_a_manual_computation_with_truncation():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.4], [0.3]])
    next_values = torch.tensor([[0.4], [0.9], [0.2]])
    truncated = torch.tensor([[False], [True], [False]])
    g, lam = 0.9, 0.8
    d = [1 + g * 0.4 - 0.5, 2 + g * 0.9 - 0.4, 3 + g * 0.2 - 0.3]
    expected = [d[0] + g * lam * d[1], d[1], d[2]]
    advantages, returns = discounted_gae(rewards, values, next_values, truncated, g, lam)
    assert advantages[:, 0].tolist() == pytest.approx(expected)
    assert returns == pytest.approx(advantages + values)


def test_inverse_cdf_sampling_respects_the_mask():
    logits = np.array([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
    legal = np.array([[True, False, True], [True, True, True]])
    assert sample_from_logits(logits, legal, np.array([0.0, 0.0])).tolist() == [0, 0]
    assert sample_from_logits(logits, legal, np.array([0.99, 0.99])).tolist() == [2, 2]
    assert sample_from_logits(logits, legal, np.array([0.6, 0.5])).tolist() == [2, 1]


def test_marl_policy_is_reproducible_and_idle_ues_draw_nothing():
    sc = scenario(4)
    torch.manual_seed(0)
    actor, _ = build_models(sc, tiny_settings())
    make = lambda: MarlPolicy(actor, sc.physical, sc.menu, LOCAL, mu0=2.0)  # noqa: E731
    first = run_episode(sc, make(), "dev", 7, 0)
    assert run_episode(sc, make(), "dev", 7, 0) == first
    policy = make()
    rngs = [np.random.default_rng(i) for i in range(4)]
    policy.reset(rngs)
    topology, traffic = sc.instance(7)
    env = sc.make_env(topology, traffic)
    obs = env.reset()
    before = [r.bit_generator.state for r in rngs]
    actions = policy.act(obs)
    for ue, o in enumerate(obs):
        if o.remaining_bits == 0:
            assert actions[ue].tolist() == [0, 0]
            assert rngs[ue].bit_generator.state == before[ue]


def test_rollout_replay_and_dual_bookkeeping_in_the_trainer():
    sc = scenario(4)
    constraint = ConstraintSettings("local", eta=2.0, mu_max=10.0, mu0=2.0)
    trainer = Trainer(
        sc, tiny_settings(), PPOSettings(), constraint, seed=3, validation_indices=[10_000]
    )
    rollout, stats = trainer.collect()
    assert rollout.actions.shape == (16, 2, 4)
    assert np.all(rollout.legal[~rollout.active][:, 1:] == 0)
    metrics = trainer.ppo.update(rollout, entropy_coef=0.01)
    assert metrics["replay_log_prob_error"] == 0.0
    assert stats["settled"] >= stats["violations"] >= 0
    assert np.all(rollout.rewards <= 0)


def test_trainer_run_selects_checkpoints_that_reload_to_the_same_policy(tmp_path):
    sc = scenario(4)
    constraint = ConstraintSettings("global", eta=5.0, mu_max=50.0, mu_init=5.0)
    trainer = Trainer(
        sc, tiny_settings(), PPOSettings(), constraint, seed=5, validation_indices=[10_000]
    )
    result = trainer.run(tmp_path / "run", log=lambda *_: None)
    names = sorted(p.stem for p in (tmp_path / "run" / "checkpoints").glob("*.pt"))
    assert names == ["step-000000000", "step-000000032", "step-000000064"]
    assert result["final"] == names[-1] and result["selected"] in names
    lines = [json.loads(line) for line in (tmp_path / "run" / "training.jsonl").open()]
    mu = 5.0
    for line in lines:
        mu = float(np.clip(mu + 5.0 * (line["violation_rate"] - 0.05), 0.0, 50.0))
        assert line["mu_global"] == pytest.approx(mu)
    make, _ = load_policy_factory(tmp_path / "run" / "checkpoints" / f"{names[-1]}.pt", sc)
    recorded = json.loads((tmp_path / "run" / "validation" / f"{names[-1]}.json").read_text())
    replayed = run_episode(sc, make(), "validation", 10_000, 0, load_references("validation", 4))
    assert replayed == recorded["episodes"][0]


def test_cli_refuses_formal_overrides_and_existing_outputs(tmp_path):
    base = [
        "--config",
        str(EXPERIMENT),
        "--scenario",
        str(CONFIGS / "scenario-v1-n4.toml"),
        "--variant",
        "local",
        "--seed",
        "1",
    ]
    with pytest.raises(SystemExit):
        train_constrained_mappo.main(
            base + ["--output", str(tmp_path / "a"), "--formal", "--budget-slots", "64"]
        )
    (tmp_path / "b").mkdir()
    with pytest.raises(SystemExit):
        train_constrained_mappo.main(base + ["--output", str(tmp_path / "b")])
