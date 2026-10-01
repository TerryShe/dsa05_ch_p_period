"""EXP-DSA05-006: v2 actor features, set critic, ragged MAPPO, policy specs, selection."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dsa_period import splits, splits_v2
from dsa_period.capacity import run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.experiments.select_capacity_checkpoints import (
    SELECTION_GRID,
    choose_variant,
    local_bar,
    score,
)
from dsa_period.marl.features import (
    GAIN_DB_OFFSET,
    GAIN_DB_SCALE,
    DualSettings,
    LocalMemory,
    context_dim,
    critic_ue_dim,
    critic_ue_features,
    stack,
    token_dim,
)
from dsa_period.marl.networks import ChannelSetActor, SetCritic
from dsa_period.marl.ppo import PPOSettings
from dsa_period.marl.trainer_v2 import (
    CapacityConstraint,
    CapacityTrainer,
    CapacityTrainingSettings,
    load_capacity_policy,
)
from dsa_period.physics import shannon_bits
from dsa_period.rules_v2 import CALIBRATION_PATH, best_local_spec, calibrated_spec, make_policy_v2

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "scenario-v2.toml"
CALIBRATION = json.loads(CALIBRATION_PATH.read_text())


def v2_observations(n=6, instance=0, slots=12):
    scenario = load_scenario(CONFIG, n)
    topology, traffic = scenario.instance(instance)
    env = scenario.make_env(topology, traffic, splits_v2.fading_seeds("dev", n, instance, 0), 50)
    obs = env.reset()
    for _ in range(slots):
        obs = env.step(np.zeros((n, 2), dtype=np.int64)).observations  # all OFF: packets wait
    return scenario, obs


def test_v2_tokens_carry_the_current_own_gain_and_use_it_for_coverage():
    scenario, observations = v2_observations()
    duals = DualSettings("local", 0.01, 0.0, 2.0, 50.0)
    memory = LocalMemory(scenario.physical, scenario.menu, 6, duals)
    memory.reset(np.full(6, 5.0))
    obs = stack(observations)
    memory.observe(obs)
    tokens, context = memory.features(obs)
    assert tokens.shape == (6, 4, token_dim(scenario.menu, own_csi=True))
    assert context.shape == (6, context_dim())
    own = np.array([o.own_gain_w for o in observations])
    expected = (10 * np.log10(own) + GAIN_DB_OFFSET) / GAIN_DB_SCALE
    assert tokens[:, :, -1] == pytest.approx(expected.astype(np.float32), rel=1e-6)
    assert np.all(context[:, 9] == 1.0)
    levels = scenario.menu.num_levels - 1
    ue = int(np.flatnonzero(obs.remaining > 0)[0])
    required = obs.remaining[ue] / obs.to_deadline[ue]
    power = np.asarray(scenario.menu.power_w[1:])
    for k in range(4):
        bits = shannon_bits(scenario.physical, power * own[ue, k] / memory.interference[ue, k])
        ratio = np.clip(np.log2(bits / required), -4, 4) / 4
        assert tokens[ue, k, 3 + levels : 3 + 2 * levels] == pytest.approx(ratio, abs=1e-6)


def test_set_critic_is_permutation_invariant_and_takes_any_number_of_ues():
    torch.manual_seed(0)
    critic = SetCritic(9, 16)
    x = torch.randn(9, 9)
    segment = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = critic(x, segment, 3)
    perm = torch.tensor([1, 0, 4, 2, 3, 8, 5, 7, 6])
    assert torch.allclose(values, critic(x[perm], segment[perm], 3))
    assert torch.allclose(values[2:], critic(x[5:], torch.zeros(4, dtype=torch.long), 1))


def test_concatenated_actor_batch_equals_per_environment_calls():
    torch.manual_seed(1)
    actor = ChannelSetActor(14, 13, 4, 5, 16)
    tokens, context = torch.randn(7, 4, 14), torch.randn(7, 13)
    hidden = actor.initial_hidden(7)
    joint, _ = actor.step(tokens, context, hidden)
    first, _ = actor.step(tokens[:3], context[:3], hidden[:3])
    second, _ = actor.step(tokens[3:], context[3:], hidden[3:])
    assert torch.allclose(joint, torch.cat((first, second)), atol=1e-6)


@pytest.fixture(scope="module")
def tiny_trainer():
    scenario = load_scenario(CONFIG, 3)
    settings = CapacityTrainingSettings(
        budget_slots=60,
        episode_slots=20,
        env_ues=(2, 3),
        rollout_slots=10,
        hidden=8,
        critic_hidden=16,
    )
    trainer = CapacityTrainer(
        scenario,
        settings,
        PPOSettings(ppo_epochs=2, minibatch_envs=1),
        CapacityConstraint("local", eta=2.0, mu_max=50.0, mu0=5.0),
        seed=7,
    )
    return trainer


def test_ragged_rollout_rewards_replay_and_training_instances(tiny_trainer):
    rollout, stats = tiny_trainer.collect()
    assert rollout.actions.shape == (10, 5) and rollout.rewards.shape == (10, 2)
    assert rollout.critic_ue.shape[-1] == critic_ue_dim(tiny_trainer.config, tiny_trainer.menu)
    assert np.all(rollout.rewards <= 0)
    metrics = tiny_trainer.ppo.update(rollout, 0.01)
    assert metrics["replay_log_prob_error"] < 1e-5
    start, stop = splits.SPLITS["train"]
    assert all(start <= i < stop for i in tiny_trainer.instances_drawn)


def test_checkpoint_policy_runs_at_an_unseen_number_of_ues(tiny_trainer, tmp_path):
    path = tmp_path / "step.pt"
    torch.save(tiny_trainer.checkpoint(0, 0), path)
    scenario = load_scenario(CONFIG, 5)
    policy = load_capacity_policy(path, scenario.physical, scenario.menu)
    assert policy.mu0 == 5.0
    short = scenario.__class__(**{**scenario.__dict__, "horizon_slots": 300, "warmup_slots": 40})
    episode = run_capacity_episode(short, policy, "dev", 0)
    assert episode["num_ues"] == 5 and len(episode["violation_rate"]) == 5
    spec_policy = make_policy_v2(f"marl:{path}", scenario.physical, scenario.menu)
    assert spec_policy.name == f"marl:{path}"


def test_critic_inputs_hold_actor_features_and_the_dual():
    tokens = np.ones((2, 4, 3), np.float32)
    context = np.zeros((2, 13), np.float32)
    features = critic_ue_features(tokens, context, np.array([5.0, 10.0]), 50.0)
    assert features.shape == (2, 4 * 3 + 13 + 1)
    assert features[:, -1] == pytest.approx([0.1, 0.2])


def test_calibrated_and_best_local_specs_follow_the_frozen_calibration():
    scenario = load_scenario(CONFIG, 8)
    for family, result in CALIBRATION["families"].items():
        for n, row in result["per_n"].items():
            assert calibrated_spec(family, int(n)) == row["spec"]
    for n in SELECTION_GRID:
        best = best_local_spec(n)
        share = max(
            CALIBRATION["families"][f]["per_n"][str(n)]["feasible_share"]["0.01"]
            for f in CALIBRATION["families"]
        )
        family = next(f for f in CALIBRATION["families"] if calibrated_spec(f, n) == best)
        assert CALIBRATION["families"][family]["per_n"][str(n)]["feasible_share"]["0.01"] == share
    policy = make_policy_v2("calibrated:paced_sticky_csi", scenario.physical, scenario.menu)
    assert policy.name == "paced_sticky_csi(margin=32)"
    with pytest.raises(ValueError):
        calibrated_spec("p_persistent", 36)


def test_selection_score_bar_and_variant_rule():
    rows = [
        {"num_ues": n, "feasible_share": f}
        for n, f in zip(SELECTION_GRID, [1, 1, 0.9, 0.6, 0.3, 0.0])
    ]
    assert score({"per_eps": {"0.01": {"rows": rows}}}) == pytest.approx(3.8 / 6)
    assert 0 < local_bar(CALIBRATION) < 1
    runs = {
        "a1": {"variant": "fixed", "selected_score": 0.70},
        "a2": {"variant": "fixed", "selected_score": 0.70},
        "c1": {"variant": "local", "selected_score": 0.69},
        "c2": {"variant": "local", "selected_score": 0.69},
    }
    assert choose_variant(runs)["chosen"] == "local"  # A ahead by less than 0.02
    runs["a1"]["selected_score"] = runs["a2"]["selected_score"] = 0.72
    assert choose_variant(runs)["chosen"] == "fixed"
