"""EXP-DSA05-008 phase 0: training options, default-path identity, sets, selection, analysis."""

import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np

from dsa_period.config import load_scenario
from dsa_period.marl.ppo import PPOSettings
from dsa_period.marl.trainer_v2 import CapacityConstraint, CapacityTrainer, CapacityTrainingSettings

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "scenario-v2.toml"

# Keys that EXP-DSA05-006 already logged; the identity checks hash only these.
ORIGINAL_METRICS = (
    "value_loss",
    "actor_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "actor_grad_norm",
    "critic_grad_norm",
    "replay_log_prob_error",
    "advantage_mean",
    "return_mean",
    "entropy_coef",
)
ORIGINAL_STATS = (
    "iteration_slots",
    "violation_rate",
    "violation_rate_by_n",
    "tx_fraction_of_active",
    "occupancy",
    "mean_reward",
    "mu_mean",
)
ROLLOUT_FIELDS = ("actions", "log_probs", "rewards", "values", "next_values", "active", "truncated")


def fingerprint(trainer: CapacityTrainer, iterations: int, coef: float = 0.01) -> str:
    """SHA-256 over rollouts, the EXP-006 metrics and the final actor/critic parameters."""
    digest = hashlib.sha256()
    for _ in range(iterations):
        rollout, stats = trainer.collect()
        for name in ROLLOUT_FIELDS:
            digest.update(np.ascontiguousarray(getattr(rollout, name)).tobytes())
        metrics = trainer.ppo.update(rollout, coef)
        digest.update(json.dumps([metrics[k] for k in ORIGINAL_METRICS]).encode())
        digest.update(json.dumps({k: stats[k] for k in ORIGINAL_STATS}, sort_keys=True).encode())
    state = trainer.ppo.state_dict()
    for part in ("actor", "critic"):
        for key, value in state[part].items():
            digest.update(key.encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tiny_trainer(variant: str, seed: int = 7, **ppo) -> CapacityTrainer:
    scenario = load_scenario(CONFIG, 3)
    settings = CapacityTrainingSettings(
        budget_slots=60,
        episode_slots=20,
        env_ues=(2, 3),
        rollout_slots=10,
        hidden=8,
        critic_hidden=16,
    )
    constraint = (
        CapacityConstraint("local", eta=2.0, mu_max=50.0, mu0=5.0)
        if variant == "local"
        else CapacityConstraint("fixed", weight=1.0)
    )
    return CapacityTrainer(
        scenario,
        settings,
        PPOSettings(**{"ppo_epochs": 2, "minibatch_envs": 1, **ppo}),
        constraint,
        seed,
    )


def exp006_trainer(seed: int) -> CapacityTrainer:
    from dsa_period.experiments.train_capacity_mappo import settings_from

    raw = tomllib.loads((ROOT / "configs" / "exp006-capacity-mappo.toml").read_text())
    training, ppo, constraint, update = settings_from(raw, "fixed")
    return CapacityTrainer(load_scenario(CONFIG, 20), training, ppo, constraint, seed, update)


# ------------------------------------------------------------ default path identity
# Fingerprints recorded with the EXP-DSA05-006 trainer (commit c275070) before the
# EXP-008 options were added.
GOLDEN = {
    "fixed": "4409994da8ad7f5a968bf3951b9de1b91d7fc275b2a1fe20f543deadc598e80f",
    "local": "68bf2f41136de2ebb965c7122deab888c4decb391f9292f16e439a8049254a0f",
    "fixed_minibatch2": "8a64d3bf824c8b7a41f691cf97ba7607f7548624c8fdc2d5e5cd46873545e809",
}
# outputs/exp006-phase1/fixed-seed101/training.jsonl, iterations 1 and 2 (formal run).
PHASE1_SEED101 = [
    {
        "violation_rate": 0.2255278310940499,
        "mean_reward": -0.00897274911403656,
        "value_loss": 0.12076683327904902,
        "actor_loss": -0.06683216016244842,
        "approx_kl": 0.00019811844566675063,
    },
    {
        "violation_rate": 0.2147083685545224,
        "mean_reward": -0.00910295732319355,
        "value_loss": 0.01676462625619024,
        "actor_loss": -0.17874909657984972,
        "approx_kl": 6.885804107525928e-05,
    },
]


def test_default_update_is_the_exp006_update_bit_for_bit():
    assert fingerprint(tiny_trainer("fixed"), 3) == GOLDEN["fixed"]
    assert fingerprint(tiny_trainer("local"), 3) == GOLDEN["local"]
    assert fingerprint(tiny_trainer("fixed", minibatch_envs=2), 3) == GOLDEN["fixed_minibatch2"]


def test_exp006_config_reproduces_the_formal_phase1_run():
    trainer = exp006_trainer(101)
    s = trainer.settings
    slots = 0
    for expected in PHASE1_SEED101:
        rollout, stats = trainer.collect()
        slots += s.rollout_slots * len(s.env_ues)
        coef = s.entropy_start * max(0.0, 1.0 - slots / (s.entropy_decay_fraction * s.budget_slots))
        line = {**stats, **trainer.ppo.update(rollout, coef)}
        assert {k: line[k] for k in expected} == expected


# ------------------------------------------------------------------- F1-F5 options
import pytest  # noqa: E402
import torch  # noqa: E402

from dsa_period.marl.networks import ChannelSetActor, SetCritic  # noqa: E402
from dsa_period.marl.ppo import discounted_gae  # noqa: E402
from dsa_period.marl.trainer_v2 import RaggedMAPPO, UpdateSettings, ValueNorm  # noqa: E402


def bare_ppo(env_ues, options=None, **ppo) -> RaggedMAPPO:
    env_of_ue = np.repeat(np.arange(len(env_ues)), env_ues)
    torch.manual_seed(0)
    return RaggedMAPPO(
        ChannelSetActor(5, 3, 2, 2, 4),
        SetCritic(4, 8),
        PPOSettings(**ppo),
        0,
        env_of_ue,
        options or UpdateSettings(),
    )


def option_trainer(variant="fixed", env_ues=(2, 3), **kwargs) -> CapacityTrainer:
    ppo = {k: kwargs.pop(k) for k in list(kwargs) if k in PPOSettings.__dataclass_fields__}
    settings = CapacityTrainingSettings(
        budget_slots=60,
        episode_slots=20,
        env_ues=env_ues,
        rollout_slots=10,
        hidden=8,
        critic_hidden=16,
    )
    constraint = (
        CapacityConstraint("local", eta=2.0, mu_max=50.0, mu0=5.0)
        if variant == "local"
        else CapacityConstraint("fixed", weight=1.0)
    )
    return CapacityTrainer(
        load_scenario(CONFIG, max(env_ues)),
        settings,
        PPOSettings(**{"ppo_epochs": 2, "minibatch_envs": 1, **ppo}),
        constraint,
        7,
        UpdateSettings(**kwargs),
    )


def test_update_options_are_validated():
    with pytest.raises(ValueError):
        UpdateSettings(advantage_norm="per_env")
    with pytest.raises(ValueError):
        UpdateSettings(value_norm="per_n")


def test_f1_normalizes_each_n_group_and_handles_empty_constant_and_single_env_groups():
    # F4-like mix: N = 2 has one environment, N = 3 two, N = 4 one.
    ppo = bare_ppo((2, 3, 3, 4), UpdateSettings(advantage_norm="per_n"))
    torch.manual_seed(1)
    raw = torch.randn(6, 12)
    active = torch.rand(6, 12) > 0.3
    n_of_ue = ppo.n_of_ue
    active[:, n_of_ue == 4] = False  # no samples: left as is
    raw[:, n_of_ue == 2] = 0.7  # constant group: centered to 0
    out = ppo._normalize_per_n(raw, active)
    assert torch.isfinite(out).all()
    group3 = out[:, n_of_ue == 3][active[:, n_of_ue == 3]]
    assert float(group3.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(group3.std(unbiased=False)) == pytest.approx(1.0, abs=1e-5)
    assert torch.equal(out[:, n_of_ue == 4], raw[:, n_of_ue == 4])
    assert torch.all(out[:, n_of_ue == 2][active[:, n_of_ue == 2]] == 0)


def test_f1_rollout_groups_have_zero_mean_and_unit_spread():
    trainer = option_trainer(advantage_norm="per_n")
    rollout, _ = trainer.collect()
    prepared = trainer.ppo.advantages(rollout)
    for n in trainer.ppo.groups:
        cols = trainer.ppo.n_of_ue == n
        x = prepared["per_ue"][:, cols][prepared["active"][:, cols]]
        if x.numel() > 1 and float(prepared["raw"][:, cols][prepared["active"][:, cols]].std()) > 0:
            assert float(x.mean()) == pytest.approx(0.0, abs=1e-5)
            assert float(x.std(unbiased=False)) == pytest.approx(1.0, abs=1e-4)


def test_f2_weights_every_environment_with_samples_equally():
    ppo = bare_ppo((2, 3, 4), UpdateSettings(actor_loss_weighting="env"))
    samples = torch.tensor([1.0, 3.0, 10.0, 10.0, 10.0, 10.0])
    envs = torch.tensor([0, 0, 2, 2, 2, 2])  # environment 1 has no sample
    assert float(ppo._env_mean(samples, envs)) == pytest.approx((2.0 + 10.0) / 2)
    trainer = option_trainer(actor_loss_weighting="env")
    rollout, _ = trainer.collect()
    metrics = trainer.ppo.update(rollout, 0.01)
    assert np.isfinite(metrics["actor_loss"]) and np.isfinite(metrics["entropy"])


def test_f3_value_norm_roundtrip_moments_floor_and_restore():
    norm = ValueNorm()
    returns = torch.tensor([[0.01, -0.02], [0.03, 0.005]])  # variance far below 1e-2
    norm.update(returns)
    mean, var, floored = norm.moments()
    assert float(norm.debias) == pytest.approx(1 - ValueNorm.BETA)
    assert float(mean) == pytest.approx(float(returns.double().mean()))
    assert floored and float(var) == ValueNorm.VAR_FLOOR  # spread below the floor
    x = torch.randn(5, 3)
    assert torch.allclose(norm.denormalize(norm.normalize(x)), x, atol=1e-6)
    other = ValueNorm()
    other.load_state_dict(norm.state_dict())
    assert torch.equal(other.normalize(x), norm.normalize(x))


def test_f3_gae_in_reward_units_and_one_statistics_update_per_rollout():
    trainer = option_trainer(value_norm="global", minibatch_envs=1)
    norm = trainer.ppo.value_norm
    rollout, _ = trainer.collect()
    raw_values = norm.denormalize(torch.from_numpy(rollout.values))
    raw_next = norm.denormalize(torch.from_numpy(rollout.next_values))
    expected, returns = discounted_gae(
        torch.from_numpy(rollout.rewards),
        raw_values,
        raw_next,
        torch.from_numpy(rollout.truncated),
        0.99,
        0.95,
    )
    before = trainer.ppo.advantages(rollout, update_stats=False)
    assert torch.equal(before["advantages"], expected)  # GAE in reward units
    calls = []
    original = norm.update
    norm.update = lambda r: (calls.append(r.clone()), original(r))[1]
    metrics = trainer.ppo.update(rollout, 0.01)
    assert len(calls) == 1 and torch.equal(calls[0], returns)  # [T, E], not per UE
    assert metrics["critic_steps"] == 4  # 2 epochs x 2 minibatches, statistics moved once
    assert "value_norm" in metrics and metrics["by_n"]["2"]["value_rmse_normalized"] >= 0


def test_f3_checkpoint_restores_statistics_and_refuses_a_mismatch():
    trainer = option_trainer(value_norm="global")
    rollout, _ = trainer.collect()
    trainer.ppo.update(rollout, 0.01)
    state = trainer.checkpoint(10, 1)
    assert state["update"]["value_norm"] == "global" and "value_norm" in state["ppo"]
    fresh = option_trainer(value_norm="global")
    fresh.ppo.load_state_dict(state["ppo"])
    x = torch.randn(3)
    assert torch.equal(fresh.ppo.value_norm.normalize(x), trainer.ppo.value_norm.normalize(x))
    with pytest.raises(ValueError):
        option_trainer().ppo.load_state_dict(state["ppo"])


@pytest.mark.parametrize(
    ("env_ues", "minibatch", "per_epoch"),
    [((2, 3), 2, 1), ((2, 3), 1, 2), ((2, 2, 3, 3, 2), 2, 3)],  # last: 2 + 2 + 1
)
def test_f5_and_tail_minibatches_count_optimizer_steps(env_ues, minibatch, per_epoch):
    trainer = option_trainer(env_ues=env_ues, minibatch_envs=minibatch)
    rollout, _ = trainer.collect()
    metrics = trainer.ppo.update(rollout, 0.01)
    assert metrics["critic_steps"] == 2 * per_epoch
    assert metrics["actor_steps"] <= 2 * per_epoch
    assert metrics["replay_log_prob_error"] < 1e-5
    assert metrics["actor_samples"] == metrics["actor_steps"] * metrics["actor_batch_mean"]


def test_full_batch_covers_whole_trajectories():
    trainer = option_trainer(env_ues=(2, 3), minibatch_envs=2)
    rollout, _ = trainer.collect()
    metrics = trainer.ppo.update(rollout, 0.01)
    assert metrics["actor_samples"] == 2 * int(rollout.active.sum())  # every sample, each epoch


# ------------------------------------------------------------ training diagnostics
def test_per_n_diagnostics_mu_statistics_and_run_outputs(tmp_path):
    trainer = option_trainer("local", env_ues=(2, 3, 3))
    rollout, stats = trainer.collect()
    metrics = trainer.ppo.update(rollout, 0.01)
    by_n = metrics["by_n"]
    assert set(by_n) == {"2", "3"}
    assert by_n["3"]["potential_samples"] == 10 * 6
    assert sum(r["active_samples"] for r in by_n.values()) == int(rollout.active.sum())
    assert set(stats["settled_by_n"]) == {"2", "3"}
    mu = stats["mu_by_class_quarter"]
    assert np.sum(mu["count"]) == 10 * 8
    run = option_trainer(env_ues=(2, 3))
    result = run.run(tmp_path / "run", log=lambda *_: None)
    for key in (
        "peak_rss_mb",
        "distinct_training_instances",
        "episodes_by_n",
        "ue_samples_with_packet",
    ):
        assert key in result
    line = json.loads((tmp_path / "run" / "training.jsonl").read_text().splitlines()[0])
    assert {"by_n", "actor_steps", "critic_grad_clip_share", "violations_by_n"} <= set(line)


# ---------------------------------------------------------------------- configs
FACTORS = {
    "r0": {},
    "f1": {("update", "advantage_norm"): "per_n"},
    "f2": {("update", "actor_loss_weighting"): "env"},
    "f3": {("update", "value_norm"): "global"},
    "f4": {("training", "env_ues"): [4, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 16, 20]},
    "f5": {("ppo", "minibatch_envs"): 14},
}


def _flat(raw: dict) -> dict:
    return {
        (section, key): value
        for section, table in raw.items()
        if section not in ("experiment", "constraint")
        for key, value in table.items()
    }


def test_round1_configs_differ_from_exp006_only_in_their_factor():
    base = tomllib.loads((ROOT / "configs" / "exp006-capacity-mappo.toml").read_text())
    expected_base = _flat(base) | {
        ("update", "advantage_norm"): "global",
        ("update", "actor_loss_weighting"): "ue",
        ("update", "value_norm"): "none",
    }
    for name, factor in FACTORS.items():
        raw = tomllib.loads((ROOT / "configs" / f"exp008-round1-{name}.toml").read_text())
        assert raw["experiment"]["condition"] == name.upper()
        assert raw["experiment"]["seeds"] == [101, 202] and raw["experiment"]["round"] == 1
        assert raw["constraint"] == base["constraint"]
        assert _flat(raw) == expected_base | factor, name


def test_update_overrides_parse_strings():
    from dsa_period.experiments.train_capacity_mappo import parse_value, settings_from

    assert (
        parse_value("per_n") == "per_n" and parse_value("14") == 14 and parse_value("1e-3") == 1e-3
    )
    raw = tomllib.loads((ROOT / "configs" / "exp008-round1-f3.toml").read_text())
    assert settings_from(raw, "fixed")[3] == UpdateSettings(value_norm="global")


# ------------------------------------------------------------- screening and confirmation
from dsa_period import splits, splits_v2  # noqa: E402
from dsa_period.experiments import evaluate_capacity  # noqa: E402

SETS = json.loads((ROOT / "docs" / "research" / "exp008-sets.json").read_text())


def test_set_manifest_matches_the_code_and_the_generated_instances():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "register", ROOT / "scripts" / "register_exp008_sets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build() == SETS  # ranges, repeats, codes and every digest


def test_screening_set_is_reachable_only_through_its_entry_point():
    assert splits_v2.screening_instances("exp008-screen256") == list(range(2000, 2256))
    with pytest.raises(ValueError):
        splits_v2.check_dev_range(range(1990, 2001))
    splits_v2.check_dev_range(range(1000, 1400))  # step 0 range stays usable
    with pytest.raises(ValueError):
        evaluate_capacity.instances_for("dev", 2100, 10)
    assert evaluate_capacity.instances_for("dev", None, None, "exp008-screen256")[0] == 2000
    with pytest.raises(ValueError):
        evaluate_capacity.instances_for("validation", None, None, "exp008-screen256")


def test_confirmation_set_seeds_bounds_and_isolation():
    name = "v2-confirm1"
    assert list(splits_v2.indices(name)) == list(range(30_000, 30_500))
    assert splits_v2.evaluation_repeats(name) == 1
    with pytest.raises(ValueError):
        splits_v2.execution_generators(name, 4, 30_000, 1)  # one execution only
    with pytest.raises(ValueError):
        splits_v2.require(name, 30_500)
    start, stop = splits_v2.SPLITS["reserve"]
    assert start <= 30_000 and 30_500 <= stop
    for other in ("dev", "validation", "test", "train"):
        lo, hi = splits_v2.SPLITS[other]
        assert hi <= 30_000 or lo >= 30_500
    sequence = np.random.SeedSequence([splits_v2.EXECUTION_SEED_BASE, 3, 30_007, 0]).spawn(6)
    manual = [np.random.Generator(np.random.PCG64(s)).random() for s in sequence]
    assert [g.random() for g in splits_v2.execution_generators(name, 6, 30_007, 0)] == manual
    small = splits_v2.fading_seeds(name, 4, 30_007, 0)
    large = splits_v2.fading_seeds(name, 10, 30_007, 0)
    assert [s.entropy for s in small] == [s.entropy for s in large[:4]]
    assert [s.spawn_key for s in small] == [s.spawn_key for s in large[:4]]
    with pytest.raises(ValueError):
        splits_v2.execution_generators("confirm1", 4, 30_000, 0)  # the v1 set is not v2
    assert splits.CONFIRMATION_SETS == {"confirm1": (30_000, 30_200)}
    assert len(splits.execution_generators("confirm1", 4, 30_000, 1)) == 4


@pytest.mark.parametrize(
    "argv",
    [
        ["--split", "v2-confirm1"],
        ["--split", "v2-confirm1", "--confirm-c1"],
        ["--split", "dev", "--screen", "exp008-screen256"],
    ],
)
def test_held_out_sets_need_their_switches(argv):
    with pytest.raises(SystemExit):
        evaluate_capacity.main(["--config", str(CONFIG), "--policies", "best_local", *argv])


# ------------------------------------------------------- R0 comparison and diagnostics
def _load_script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_comparison_finds_a_single_changed_element(tmp_path):
    compare = _load_script("compare_capacity_runs")
    trainer = option_trainer()
    torch.save(trainer.checkpoint(10, 1), tmp_path / "a.pt")
    torch.save(trainer.checkpoint(10, 1), tmp_path / "b.pt")
    assert compare.compare_checkpoint(tmp_path / "a.pt", tmp_path / "b.pt") == []
    state = trainer.checkpoint(10, 1)
    key = next(iter(state["ppo"]["critic"]))
    state["ppo"]["critic"][key].view(-1)[0] += 1e-7
    torch.save(state, tmp_path / "c.pt")
    assert compare.compare_checkpoint(tmp_path / "a.pt", tmp_path / "c.pt") == [f"ppo/critic/{key}"]
    log = tmp_path / "ref.jsonl"
    log.write_text('{"iteration": 1, "x": 1.0, "collect_s": 3}\n{"iteration": 2, "x": 2.0}\n')
    same = tmp_path / "same.jsonl"
    same.write_text('{"iteration": 1, "x": 1.0, "collect_s": 9, "new": 0}\n')
    assert compare.compare_log(log, same) == (1, None)
    other = tmp_path / "other.jsonl"
    other.write_text('{"iteration": 1, "x": 1.5}\n')
    assert compare.compare_log(log, other) == (0, "iteration 1: x")


def test_gae_split_and_surrogate_parts_add_up():
    diagnose = _load_script("diagnose_exp008_training")
    torch.manual_seed(3)
    rewards, values, next_values = (torch.randn(12, 3) for _ in range(3))
    truncated = torch.zeros(12, 3, dtype=torch.bool)
    truncated[5, 1] = True
    gae, _ = discounted_gae(rewards, values, next_values, truncated, 0.99, 0.95)
    r, v = diagnose.decompose(rewards, values, next_values, truncated, 0.99, 0.95)
    assert torch.allclose(r + v, gae, atol=1e-6)
    for weighting in ("ue", "env"):
        trainer = option_trainer(env_ues=(2, 3, 3), actor_loss_weighting=weighting)
        rollout, _ = trainer.collect()
        prepared = trainer.ppo.advantages(rollout, update_stats=False)
        parts = diagnose.surrogate_parts(trainer, rollout, prepared["per_ue"], prepared["active"])
        logits, _ = trainer.ppo.actor.sequence(
            torch.from_numpy(rollout.tokens),
            torch.from_numpy(rollout.context),
            rollout.initial_hidden,
            torch.from_numpy(rollout.starts)[:, trainer.ppo.env_of_ue],
        )
        from dsa_period.marl.networks import masked_log_probs

        log_probs = masked_log_probs(logits, torch.from_numpy(rollout.legal))
        chosen = log_probs.gather(-1, torch.from_numpy(rollout.actions).unsqueeze(-1)).squeeze(-1)
        act = prepared["active"]
        terms = -(prepared["per_ue"] * chosen)[act]
        envs = trainer.ppo.env_of_ue.expand_as(act)[act]
        full = trainer.ppo._env_mean(terms, envs) if weighting == "env" else terms.mean()
        total = float(sum(parts.values()).detach())
        assert total == pytest.approx(float(full.detach()), abs=1e-6)  # float32 sums of O(1) terms


# ------------------------------------------------------------- selection rules
from dsa_period.experiments import select_exp008 as sel  # noqa: E402


def _cond(name, s_near, f6=None, end=None, val=None, seeds=(101, 202)):
    """Selection rows and validation shares of one condition (one value per seed)."""
    rows, validation = {}, {}
    for i, seed in enumerate(seeds):
        s = s_near[i]
        six = f6[i] if f6 else s + 0.05
        spec = f"marl:/runs/{name}-{seed}/checkpoints/step-005001472.pt"
        rows[f"/runs/{name}-{seed}"] = {
            "condition": name,
            "seed": seed,
            "selected": spec,
            "selected_S_near": s,
            "selected_F": {"F6": six, "F8": 2 * s - six},
            "end_slots": [4001536, 5001472],
            "end_S_near": [end if end is not None else s - 0.01] * 2,
        }
        validation[spec] = {n: 0.9 for n in sel.GUARD_GRID} | {2: 0.98} | (val or {})
    return rows, validation


def _table(*conds):
    selection, validation = {}, {}
    for rows, val in conds:
        selection.update(rows)
        validation.update(val)
    return sel.conditions(selection, validation)


def test_round1_rules_retain_only_a_clean_improvement():
    r0 = _cond("R0", (0.90, 0.90), end=0.89)
    table = _table(
        r0,
        _cond("F1", (0.915, 0.912)),  # +0.0135, both seeds up, guards fine
        _cond("F2", (0.942, 0.898)),  # +0.02 but seed 202 down
        _cond("F3", (0.915, 0.915), val={4: 0.875}),  # validation F(4) drops 0.025
        _cond("F4", (0.915, 0.915), end=0.885),  # end segment below R0
        _cond("F5", (0.915, 0.915), val={2: 0.94}),  # F(2) below 0.95
    )
    result = sel.round1(table)
    rules = {f: result["retention"][f]["rules"] for f in ("F2", "F3", "F4", "F5")}
    assert result["retained"] == ["F1"] and result["B1"] == "F1"
    assert not rules["F2"]["rule1_screen_improvement"]
    assert not rules["F3"]["rule2_validation_no_drop"]
    assert not rules["F4"]["rule3_end_segment"]
    assert not rules["F5"]["rule4_validation_F2"]


def test_round1_per_n_guard_blocks_trading_n6_for_n8():
    table = _table(
        _cond("R0", (0.90, 0.90), f6=(0.95, 0.95)),
        _cond("F1", (0.92, 0.92), f6=(0.93, 0.93)),  # F(6) -0.02, F(8) +0.06
    )
    check = sel.round1(table)["retention"]["F1"]
    assert check["rules"]["rule1_screen_improvement"]
    assert not check["rules"]["rule5_screen_per_n_no_drop"] and not check["retained"]


def test_b1_needs_comb_and_prefers_fewer_factors_within_the_tie_band():
    r0 = _cond("R0", (0.90, 0.90), end=0.89)
    f1 = _cond("F1", (0.915, 0.915))
    f2 = _cond("F2", (0.918, 0.918), val={10: 0.95})  # higher S_thr than F1
    first = sel.round1(_table(r0, f1, f2))
    assert first["comb_required"] == ["F1", "F2"] and first["B1"] is None
    comb = _cond("COMB", (0.920, 0.920))
    final = sel.round1(_table(r0, f1, f2, comb), ["F2", "F1"])
    assert final["comb"]["retained"] and set(final["candidates"]) == {"R0", "F1", "F2", "COMB"}
    assert final["B1"] == "F2"  # within 0.005 of COMB, one factor, higher S_thr than F1
    with pytest.raises(ValueError):
        sel.round1(_table(r0, f1, f2, comb), ["F1"])
    far = sel.round1(_table(r0, f1, f2, _cond("COMB", (0.93, 0.93))), ["F1", "F2"])
    assert far["B1"] == "COMB"


def test_b2_excludes_guard_failures_and_breaks_ties_by_simplicity():
    table = _table(
        _cond("F1", (0.920, 0.920)),
        _cond("A-2400", (0.930, 0.930), val={2: 0.94}),  # fails the F(2) guard
        _cond("C-600", (0.924, 0.924)),
        _cond("C-2400", (0.921, 0.921)),
    )
    result = sel.round2(table, "F1")
    assert not all(result["guards"]["A-2400"].values())
    assert result["candidates"] == ["A-600", "C-2400", "C-600"]
    assert result["B2"] == "A-600"  # 0.920 is within 0.005 of 0.924


def test_checkpoint_selection_prefers_later_ties_and_reports_the_end(tmp_path):
    run = tmp_path / "r0-seed101"
    (run / "checkpoints").mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": {"id": "x", "condition": "R0", "round": 1},
                "variant": "fixed",
                "seed": 101,
            }
        )
    )
    screen = {
        f"marl:{run}/checkpoints/step-00{k}000000.pt": {6: f6, 8: f8}
        for k, f6, f8 in ((1, 0.9, 0.8), (2, 0.95, 0.85), (3, 0.9, 0.9), (4, 0.85, 0.8))
    }
    row = sel.select(screen)[str(run)]
    assert row["selected_slots"] == 3_000_000  # ties with 2M at S_near 0.9: the later one
    assert row["end_slots"] == [3_000_000, 4_000_000]
    assert row["selected_F"] == {"F6": 0.9, "F8": 0.9}


def test_selection_reads_only_registered_screening_and_validation_outputs(tmp_path):
    def write(name, **record):
        path = tmp_path / name
        path.mkdir()
        base = {"protocol_id": splits_v2.PROTOCOL_ID, "formal": True, "summaries": {}}
        (path / "summary.json").write_text(json.dumps(base | record))
        return path

    screen = splits_v2.screening_instances("exp008-screen256")
    good = write("good", split="dev", screen="exp008-screen256", instances=screen, grid=[6, 8])
    assert sel.load_evaluations([good], "screen") == {}
    for name, record in {
        "plain_dev": {"split": "dev", "screen": None, "instances": screen, "grid": [6, 8]},
        "grid": {"split": "dev", "screen": "exp008-screen256", "instances": screen, "grid": [6]},
        "test": {"split": "test", "instances": list(range(20000, 20200)), "grid": [6, 8]},
    }.items():
        with pytest.raises(ValueError):
            sel.load_evaluations([write(name, **record)], "screen")
    val = write("val", split="validation", instances=list(range(10000, 10064)), grid=[2])
    assert sel.load_evaluations([val], "validation") == {}
    informal = write(
        "informal", split="validation", instances=list(range(10000, 10064)), formal=False
    )
    with pytest.raises(ValueError):
        sel.load_evaluations([informal], "validation")


# ------------------------------------------------------------ confirmation analysis
def _episodes(spec_f: dict[int, float], instances: int, seed: int) -> dict[int, list]:
    rng = np.random.default_rng(seed)
    out = {}
    for n, share in spec_f.items():
        feasible = np.zeros(instances, bool)
        feasible[rng.permutation(instances)[: round(share * instances)]] = True
        out[n] = [
            {
                "instance": i,
                "feasible": {"0.01": bool(f), "0.05": True},
                "infeasible_for_all": {"0.01": False, "0.05": False},
                "violation_rate": [0.0] * (n - 1) + [0.0 if f else 0.02],
                "worst_violation_rate": 0.0 if f else 0.02,
                "class_index": [j % 3 for j in range(n)],
                "distance_m": [90.0] * n,
                "undeliverable": [0] * n,
                "packets": [2000] * n,
            }
            for i, f in enumerate(feasible)
        ]
    return out


def test_confirmation_analysis_verdicts():
    analyze = _load_script("analyze_exp008_confirm")
    grid_f = {
        "M": {2: 1.0, 4: 1.0, 6: 1.0, 8: 0.975, 10: 0.9, 12: 0.7, 16: 0.3},
        "A": {2: 1.0, 4: 1.0, 6: 1.0, 8: 0.85, 10: 0.8, 12: 0.6, 16: 0.2},
        "L": {2: 1.0, 4: 0.975, 6: 0.9, 8: 0.6, 10: 0.5, 12: 0.3, 16: 0.1},
        "E": {2: 1.0, 4: 1.0, 6: 1.0, 8: 1.0, 10: 1.0, 12: 1.0, 16: 1.0, 20: 0.9},
    }
    episodes = {}
    for i in range(3):
        episodes[f"marl:m{i}"] = _episodes(grid_f["M"], 40, i)
        episodes[f"marl:a{i}"] = _episodes(grid_f["A"], 40, 10 + i)
    episodes["best_local"] = _episodes(grid_f["L"], 40, 20)
    episodes["central_edf"] = _episodes(grid_f["E"], 40, 30)
    methods = {
        "M": ["marl:m0", "marl:m1", "marl:m2"],
        "M_last": ["marl:m0", "marl:m1", "marl:m2"],  # same files: evaluated once
        "A_near": ["marl:a0", "marl:a1", "marl:a2"],
        "A_original": ["marl:a0", "marl:a1", "marl:a2"],
        "best_local": "best_local",
        "rules": ["best_local"],
        "edf": ["central_edf"],
    }
    result = analyze.analyze(episodes, methods, list(range(40)), 300)
    t = result["tests"]["0.01"]
    assert t["N_star"] == {
        "M": "8",
        "M_last": "8",
        "A_near": "6",
        "A_original": "6",
        "best_local": "4",
    }
    assert result["verdict_eps_0.01"] == {"H1": True, "H2": True, "H3": True, "all": True}
    assert t["H2"]["8"]["difference"] == pytest.approx(0.125)
    assert t["H2"]["6"]["ci95"] == [0.0, 0.0] and t["H2"]["6"]["holds"]
    assert t["joint_capacity_bootstrap"]["M-A_near"]["share_positive"] > 0.9
    assert t["gap_to_edf"]["M"] == 8 and t["gap_to_edf"]["best_local"] == 12
    assert t["scores"]["S_near"]["M-A_near"]["difference"] == pytest.approx(0.0625)
    worse = {**episodes, "marl:m1": _episodes({**grid_f["M"], 6: 0.9}, 40, 1)}
    t2 = analyze.analyze(worse, methods, list(range(40)), 300)["tests"]["0.01"]
    assert not t2["H2"]["6"]["holds"] and not t2["H2"]["holds"]  # N = 6 guard fails


ROUND2 = {
    "a2400": ("A-2400", "fixed", 2400),
    "c600": ("C-600", "local", 600),
    "c2400": ("C-2400", "local", 2400),
}


def test_round2_configs_differ_from_r0_only_in_episode_length_and_variant():
    base = tomllib.loads((ROOT / "configs" / "exp008-round1-r0.toml").read_text())
    for key, (name, variant, episode) in ROUND2.items():
        raw = tomllib.loads((ROOT / "configs" / f"exp008-round2-{key}.toml").read_text())
        experiment = raw["experiment"]
        assert (experiment["round"], experiment["condition"], experiment["variant"]) == (
            2,
            name,
            variant,
        )
        assert experiment["seeds"] == [101, 202] and experiment["b1"] == "R0"
        assert name in sel.ROUND2_ORDER
        assert raw["constraint"] == base["constraint"]
        assert _flat(raw) == _flat(base) | {("training", "episode_slots"): episode}, key
