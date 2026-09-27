"""Phase-0 acceptance of EXP-DSA05-005: XR-SBPC rewards, SAC-Discrete, trainer, C1, diagnostics."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dsa_period import splits
from dsa_period.config import load_scenario
from dsa_period.evaluation import load_references, run_episode
from dsa_period.experiments import evaluate_marl, train_sac_baseline
from dsa_period.marl.features import DualSettings, LocalMemory, decode_actions, stack
from dsa_period.marl.sac import (
    PrioritizedReplay,
    SacAgent,
    SacPolicy,
    SacSettings,
    actor_objective,
    flat_features,
    masked_policy,
    soft_targets,
)
from dsa_period.marl.sac_trainer import SacTrainer, SacTrainingSettings
from dsa_period.marl.trainer import load_policy_factory
from dsa_period.marl.xr_sbpc import PacketTracker, RewardSettings, packet_summary, xr_rewards

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs"
EXPERIMENT = CONFIGS / "exp005-xr-sbpc-baseline.toml"
V1, V1S, V2 = RewardSettings("v1"), RewardSettings("v1s"), RewardSettings("v2")


def scenario(n):
    return load_scenario(CONFIGS / f"scenario-v1-n{n}.toml")


def tiny_sac(**changes):
    base = dict(batch_size=8, buffer_size=64, hidden=(16, 8))
    return SacSettings(**(base | changes))


def tiny_training(**changes):
    base = dict(budget_slots=64, episode_slots=40, num_envs=2, validation_every_slots=32)
    return SacTrainingSettings(**(base | changes))


# ---------------------------------------------------------------- rewards
def test_v1_reward_follows_the_four_branches_of_xr_sbpc_eq7():
    # UE: idle, completes, expires, above target (R/phi = 2), below target, OFF with packet.
    remaining = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    to_deadline = np.array([5, 4, 1, 4, 4, 4])
    service = np.array([0.0, 150.0, 30.0, 50.0, 10.0, 0.0])
    completed = np.array([False, True, False, False, False, False])
    expired = np.array([False, False, True, False, False, False])
    reward = xr_rewards(V1, remaining, to_deadline, service, completed, expired)
    assert reward.tolist() == [0.0, 50.0, -100.0, 10.0, -10.0, -10.0]
    sparse = xr_rewards(V1S, remaining, to_deadline, service, completed, expired)
    assert sparse.tolist() == [0.0, 50.0, -100.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        xr_rewards(V1, remaining, to_deadline, service, ~completed, expired)  # idle UE completes
    with pytest.raises(ValueError):
        xr_rewards(V2, remaining, to_deadline, service, completed, expired)
    with pytest.raises(ValueError):
        RewardSettings("v3")


def test_rewards_follow_the_environment_over_a_random_episode():
    sc = scenario(8)
    trainer = SacTrainer(sc, tiny_training(), tiny_sac(), V1, seed=0, validation_indices=[10_000])
    topology, traffic = sc.instance(3)
    env = sc.make_env(topology, traffic)
    obs = stack(env.reset())
    rng = np.random.default_rng(1)
    events = {"completed": 0, "expired": 0}
    for _ in range(400):
        actions = np.where(rng.random(8) < 0.3, 1 + rng.integers(0, 20, 8), 0)
        step = env.step(decode_actions(actions, sc.menu))
        reward = trainer.rewards(obs, step)
        expired = step.violations.astype(bool)
        assert np.all(obs.to_deadline[expired] == 1), "expiry only in the last slot"
        assert np.all(reward[step.completed] == 50.0) and np.all(reward[expired] == -100.0)
        assert np.all(reward[obs.remaining <= 0] == 0.0)
        events["completed"] += int(step.completed.sum())
        events["expired"] += int(expired.sum())
        obs = stack(step.observations)
    assert events["completed"] > 0 and events["expired"] > 0


def test_v2_is_the_exp001_team_reward_for_every_ue():
    sc = scenario(4)
    trainer = SacTrainer(sc, tiny_training(), tiny_sac(), V2, seed=0, validation_indices=[10_000])
    env = sc.make_env(*sc.instance(0))
    obs = stack(env.reset())
    step = env.step(decode_actions(np.array([1, 7, 0, 0]), sc.menu))
    reward = trainer.rewards(obs, step)
    energy = step.physical.tx_energy_j.sum() / (sc.physical.slot_duration_s * sc.menu.power_w[-1])
    expected = -(energy + 10.0 * step.violations.sum()) / 4
    assert np.allclose(reward, expected) and len(set(reward.tolist())) == 1


# ------------------------------------------------------------- SAC pieces
def test_masked_policy_and_soft_targets_match_a_hand_computation():
    logits = torch.tensor([[0.0, float(np.log(3.0)), 5.0], [1.0, 2.0, 3.0]])
    legal = torch.tensor([[True, True, False], [True, False, False]])
    probs, log_probs = masked_policy(logits, legal)
    assert torch.allclose(probs, torch.tensor([[0.25, 0.75, 0.0], [1.0, 0.0, 0.0]]))
    assert log_probs[0, 2] == 0 and log_probs[1].abs().sum() == 0
    q1 = torch.tensor([[1.0, 2.0, 100.0], [4.0, 9.0, 9.0]])
    q2 = torch.tensor([[0.5, 3.0, 100.0], [6.0, 9.0, 9.0]])
    reward = torch.tensor([1.0, -1.0])
    y = soft_targets(reward, logits, legal, q1, q2, alpha=0.5, gamma=0.9)
    entropy0 = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    expected0 = 1.0 + 0.9 * (0.25 * 0.5 + 0.75 * 2.0 + 0.5 * entropy0)
    expected1 = -1.0 + 0.9 * 4.0  # idle next state: only OFF, entropy 0
    assert torch.allclose(y, torch.tensor([expected0, expected1], dtype=torch.float32))
    objective, entropy = actor_objective(logits, legal, q1, q2, alpha=0.5)
    assert torch.allclose(entropy, torch.tensor([entropy0, 0.0], dtype=torch.float32))
    assert objective[0].item() == pytest.approx(-0.5 * entropy0 - (0.25 * 0.5 + 0.75 * 2.0))


def test_prioritized_replay_is_circular_and_weights_follow_priorities():
    buffer = PrioritizedReplay(4, obs_dim=1, actions=2, alpha=1.0, eps=0.0, seed=0)
    legal = np.ones((3, 2), bool)
    buffer.add(np.arange(3.0)[:, None], np.zeros(3), np.arange(3.0), np.zeros((3, 1)), legal, legal)
    buffer.add(
        np.arange(3.0, 6.0)[:, None],
        np.ones(3),
        np.arange(3.0, 6.0),
        np.zeros((3, 1)),
        legal,
        legal,
    )
    assert buffer.size == 4 and buffer.cursor == 2
    assert buffer.reward.tolist() == [4.0, 5.0, 2.0, 3.0], "oldest rows overwritten in order"
    buffer.update(np.array([0, 1, 2, 3]), np.array([1.0, 1.0, 1.0, 5.0]))
    assert buffer.max_priority == 5.0
    index, weights = buffer.sample(20_000, beta=1.0)
    share = np.bincount(index, minlength=4) / index.size
    assert share == pytest.approx([0.125, 0.125, 0.125, 0.625], abs=0.01)
    assert weights.max() == pytest.approx(1.0)
    assert weights[index == 3].max() == pytest.approx(0.2), "IS weight = (N P)^-beta / max"
    _, flat = buffer.sample(100, beta=0.0)
    assert np.all(flat == 1.0)
    buffer.add(np.zeros((1, 1)), np.zeros(1), np.zeros(1), np.zeros((1, 1)), legal[:1], legal[:1])
    assert buffer.priority[2] == 5.0, "a new row enters at the current maximum priority"


def test_idle_transitions_train_q_but_not_the_actor_or_temperature():
    agent = SacAgent(obs_dim=3, actions=3, settings=tiny_sac(buffer_size=16), seed=0)
    buffer = PrioritizedReplay(16, 3, 3, alpha=0.6, eps=1e-6, seed=0)
    idle = np.tile([True, False, False], (16, 1))
    rng = np.random.default_rng(0)
    buffer.add(
        rng.normal(size=(16, 3)),
        np.zeros(16),
        rng.normal(size=16),
        rng.normal(size=(16, 3)),
        idle,
        idle,
    )
    policy = [p.clone() for p in agent.policy.parameters()]
    q1 = [p.clone() for p in agent.q1.parameters()]
    alpha = agent.alpha
    metrics = agent.update(buffer, beta=0.4)
    assert metrics["active_share"] == 0.0 and "actor_loss" not in metrics
    assert all(torch.equal(a, b) for a, b in zip(policy, agent.policy.parameters(), strict=True))
    assert agent.alpha == alpha
    assert not all(torch.equal(a, b) for a, b in zip(q1, agent.q1.parameters(), strict=True))
    # With a real choice in the batch, the actor and temperature move too.
    active = np.ones((16, 3), bool)
    buffer.add(
        rng.normal(size=(16, 3)),
        rng.integers(0, 3, 16),
        rng.normal(size=16),
        rng.normal(size=(16, 3)),
        active,
        active,
    )
    metrics = agent.update(buffer, beta=0.4)
    assert metrics["active_share"] == 1.0 and agent.alpha != alpha


def test_sac_features_are_exactly_the_actor_features_of_variant_a():
    sc = scenario(8)
    env = sc.make_env(*sc.instance(5))
    a_memory = LocalMemory(sc.physical, sc.menu, 8, DualSettings("fixed", 0.05, weight=10.0))
    policy = SacPolicy(torch.nn.Linear(65, 21), sc.physical, sc.menu, 0.05)
    policy.reset(splits.execution_generators("validation", 8, 10_000, 0))
    observations = env.reset()
    for _ in range(200):
        obs = stack(observations)
        a_memory.update_duals(a_memory.observe(obs))
        tokens, context = a_memory.features(obs)
        expected = np.concatenate((tokens.reshape(8, -1), context), axis=1)
        actions = policy.act(observations)
        assert np.array_equal(flat_features(policy._memory, obs), expected)
        observations = env.step(actions).observations


def test_sac_policy_samples_only_legal_actions_and_idle_ues_draw_nothing():
    sc = scenario(8)
    env = sc.make_env(*sc.instance(2))
    policy = SacPolicy(torch.nn.Linear(65, 21), sc.physical, sc.menu, 0.05)
    rngs = splits.execution_generators("validation", 8, 10_000, 0)
    policy.reset(rngs)
    observations = env.reset()
    for _ in range(50):
        idle = np.array([o.remaining_bits <= 0 for o in observations])
        states = [g.bit_generator.state for g in rngs]
        actions = policy.act(observations)
        assert np.all(actions[idle] == 0)
        for ue in np.flatnonzero(idle):
            assert rngs[ue].bit_generator.state == states[ue], "idle UEs draw nothing"
        observations = env.step(actions).observations


# ---------------------------------------------------------------- trainer
def test_trainer_stores_every_ue_and_updates_only_once_the_buffer_is_full():
    sc = scenario(4)
    trainer = SacTrainer(
        sc, tiny_training(), tiny_sac(buffer_size=40), V1, seed=0, validation_indices=[10_000]
    )
    trainer.step()
    assert trainer.buffer.size == 2 * 4, "one transition per UE and environment, idle UEs included"
    for _ in range(4):
        trainer.step()
    assert trainer.buffer.size == 40 and trainer.buffer.cursor == 0


def test_trainer_run_selects_checkpoints_that_reload_to_the_same_policy(tmp_path):
    sc = scenario(4)
    trainer = SacTrainer(
        sc, tiny_training(), tiny_sac(buffer_size=16), V1S, seed=5, validation_indices=[10_000]
    )
    result = trainer.run(tmp_path / "run", log=lambda *_: None)
    names = sorted(p.stem for p in (tmp_path / "run" / "checkpoints").glob("*.pt"))
    assert names == ["step-000000000", "step-000000032", "step-000000064"]
    assert result["final"] == names[-1] and result["selected"] in names
    assert result["updates"] == 64 // 2 - 16 // (2 * 4) + 1
    for name in names:
        make, checkpoint = load_policy_factory(tmp_path / "run" / "checkpoints" / f"{name}.pt", sc)
        assert checkpoint["kind"] == "sac" and checkpoint["reward"]["kind"] == "v1s"
        recorded = json.loads((tmp_path / "run" / "validation" / f"{name}.json").read_text())
        replayed = run_episode(
            sc, make(), "validation", 10_000, 0, load_references("validation", 4)
        )
        assert replayed == recorded["by_mu0"][0]["episodes"][0]
    with pytest.raises(ValueError):
        load_policy_factory(tmp_path / "run" / "checkpoints" / f"{names[-1]}.pt", sc, mu0=5.0)


def test_cli_refuses_formal_overrides_bad_profiles_and_existing_outputs(tmp_path):
    base = ["--config", str(EXPERIMENT), "--scenario", str(CONFIGS / "scenario-v1-n4.toml")]
    run = base + ["--profile", "v1-g099", "--seed", "1"]
    with pytest.raises(SystemExit):
        train_sac_baseline.main(
            run + ["--output", str(tmp_path / "a"), "--formal", "--budget-slots", "64"]
        )
    with pytest.raises(SystemExit):
        train_sac_baseline.main(
            base + ["--profile", "v9", "--seed", "1", "--output", str(tmp_path / "b")]
        )
    (tmp_path / "c").mkdir()
    with pytest.raises(SystemExit):
        train_sac_baseline.main(run + ["--output", str(tmp_path / "c")])


def test_config_profiles_cover_the_design_matrix():
    raw = __import__("tomllib").loads(EXPERIMENT.read_text())
    grid = {(p["reward"], p["gamma"]) for p in raw["profiles"].values()}
    assert grid == {(r, g) for r in ("v1", "v1s", "v2") for g in (0.5, 0.99)}
    training, sac, reward = train_sac_baseline.resolve(raw, "v2-g05")
    assert (sac.gamma, reward.kind, reward.weight) == (0.5, "v2", 10.0)
    assert (training.budget_slots, training.num_envs, training.episode_slots) == (
        20_000_000,
        16,
        600,
    )
    assert (sac.lr, sac.batch_size, sac.buffer_size, sac.tau, sac.target_entropy) == (
        1e-4,
        512,
        10_000,
        0.005,
        0.05,
    )


# ------------------------------------------------------ confirmation set C1
def test_confirmation_set_c1_is_a_reserve_slice_with_test_like_execution():
    c1 = splits.indices("confirm1")
    assert (c1.start, c1.stop) == (30_000, 30_200)
    assert all(splits.split_of(i) == "reserve" for i in (c1.start, c1.stop - 1))
    for name in ("dev", "validation", "test", "train"):
        start, stop = splits.SPLITS[name]
        assert c1.stop <= start or stop <= c1.start
    assert splits.evaluation_repeats("confirm1") == 2
    assert splits.EVALUATION_REPEATS == {"validation": 1, "test": 2}, "frozen v1 repeats unchanged"
    assert splits.require("confirm1", 30_199) == 30_199
    for index in (29_999, 30_200, 20_000):
        with pytest.raises(ValueError):
            splits.require("confirm1", index)
    with pytest.raises(ValueError):
        splits.execution_generators("confirm1", 8, 30_000, 2)
    generators = splits.execution_generators("confirm1", 8, 30_007, 1)
    sequence = np.random.SeedSequence([splits.EXECUTION_SEED_BASE, 3, 8, 30_007, 1])
    expected = [np.random.Generator(np.random.PCG64(c)) for c in sequence.spawn(8)]
    assert [g.integers(1 << 30) for g in generators] == [g.integers(1 << 30) for g in expected]


def test_evaluate_marl_guards_the_confirmation_set():
    n4 = str(CONFIGS / "scenario-v1-n4.toml")
    with pytest.raises(SystemExit):
        evaluate_marl.main(["--scenario", n4, "--baselines", "random_max", "--split", "confirm1"])
    with pytest.raises(SystemExit):
        evaluate_marl.main(
            ["--scenario", n4, "--baselines", "random_max", "--split", "confirm1", "--confirm-c1"]
        )


# --------------------------------------------------------- packet diagnostics
def test_packet_tracker_accumulates_shaping_and_ratios_per_packet():
    tracker = PacketTracker(period=np.array([4, 2]))
    # UE 0: 100 bits due in 4 slots; UE 1: 10 bits due in 2 slots.
    tracker.observe(0, [100, 10], [4, 2], [50, 4], [False, False], [False, False])
    tracker.observe(1, [50, 6], [3, 1], [0, 3], [False, False], [False, True])
    tracker.observe(2, [50, 0], [2, 1], [60, 0], [True, False], [False, False])
    jobs = [
        dict(ue=0, arrival_slot=0, deadline_slot=4, completion_slot=3, tx_slots=2, success=True),
        dict(
            ue=1, arrival_slot=0, deadline_slot=2, completion_slot=None, tx_slots=2, success=False
        ),
    ]
    collected = tracker.collect(jobs)
    assert collected["shaping"] == [pytest.approx(10.0 - 10.0), -10.0]
    assert collected["ratios"] == [pytest.approx(2.0), pytest.approx(0.8)]
    assert collected["completion_fraction"] == [0.75] and collected["expired"] == 1
    summary = packet_summary([collected, collected])
    assert summary["packets"] == 4 and summary["expired_share"] == 0.5
    assert summary["tx_slots_per_packet"] == 2.0
    assert summary["noncompleting_ratio_ge_1"] == 0.5
