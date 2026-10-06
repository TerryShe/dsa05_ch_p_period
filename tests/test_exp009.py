"""EXP-DSA05-009 phase 0: deployment samplers, per-N runs, sets, selection, analysis."""

import importlib.util
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from dsa_period import splits, splits_v2
from dsa_period.config import load_scenario
from dsa_period.experiments import evaluate_capacity
from dsa_period.experiments import select_exp009 as sel
from dsa_period.experiments.train_capacity_mappo import resolve_training, settings_from
from dsa_period.marl.level_sampler import (
    LevelSampler,
    SamplerSettings,
    deployment_hardness,
    rank_weights,
    staleness_weights,
)
from dsa_period.marl.ppo import PPOSettings
from dsa_period.marl.trainer_v2 import CapacityConstraint, CapacityTrainer, CapacityTrainingSettings

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "scenario-v2.toml"
TAU = tomllib.loads((ROOT / "configs" / "exp009-tau.toml").read_text())["tau"]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trainer(sampler=None, env_ues=(3, 3), episode_slots=20, rollout=10, seed=7) -> CapacityTrainer:
    settings = CapacityTrainingSettings(
        budget_slots=200,
        episode_slots=episode_slots,
        env_ues=env_ues,
        rollout_slots=rollout,
        hidden=8,
        critic_hidden=16,
    )
    return CapacityTrainer(
        load_scenario(CONFIG, max(env_ues)),
        settings,
        PPOSettings(ppo_epochs=1, minibatch_envs=1),
        CapacityConstraint("fixed", weight=1.0),
        seed,
        None,
        sampler,
    )


# ------------------------------------------------------------- PLR distribution
def test_rank_and_staleness_weights_follow_the_paper():
    scores = np.array([0.5, 0.1, 0.5, 0.0])
    weights = rank_weights(scores, beta=1.0)
    ranks = np.array([1.5, 3.0, 1.5, 4.0])  # ties share their mean rank
    assert weights == pytest.approx((1 / ranks) / (1 / ranks).sum())
    sharp = rank_weights(scores, beta=0.1)
    assert sharp == pytest.approx((1 / ranks) ** 10 / ((1 / ranks) ** 10).sum())
    assert staleness_weights(np.array([2, 5, 9]), 10) == pytest.approx(np.array([8, 5, 1]) / 14)
    assert staleness_weights(np.array([3, 3]), 3) == pytest.approx([0.5, 0.5])


def test_plr_buffer_min_fill_replacement_and_replay_mix():
    settings = SamplerSettings(kind="plr", buffer_size=3, min_fill=2, replay_probability=1.0)
    sampler = LevelSampler(settings, 6, 0, None)
    draws = iter(range(1_000_000, 1_000_100))
    index, replayed, stamp = sampler.next_level(lambda: next(draws))
    assert not replayed and stamp == 1  # fewer than min_fill scored levels: new level
    sampler.record(index, 0.3, stamp)
    index, replayed, stamp = sampler.next_level(lambda: next(draws))
    assert not replayed  # still below min_fill
    sampler.record(index, 0.2, stamp)
    sampler.record(1_000_040, 0.5, 2)  # scored levels fill the buffer
    assert sampler.scores == [0.3, 0.2, 0.5]
    sampler.record(1_000_050, 0.1, 2)  # full buffer, lower than every score: not kept
    assert 1_000_050 not in sampler.levels
    sampler.record(1_000_051, 0.25, 2)  # replaces the lowest (0.2)
    assert sampler.levels == [1_000_000, 1_000_051, 1_000_040]
    sampler.record(1_000_000, 0.9, 2)  # an existing level is updated in place
    assert sampler.scores[0] == 0.9 and sampler.stamps[0] == 2
    picks = [sampler.next_level(lambda: next(draws)) for _ in range(20)]
    assert all(replayed and level in sampler.levels for level, replayed, _ in picks)
    assert sampler.counts["replay"] == 20 and sampler.state_dict()["clock"] == 22


def test_stratified_sampler_hits_the_hard_stratum_at_the_designed_rate():
    scenario = load_scenario(CONFIG, 8)

    def hardness(index):
        return deployment_hardness(scenario, *scenario.instance(index))

    settings = SamplerSettings(kind="stratified", hard_share=0.5, tau=TAU)
    sampler = LevelSampler(settings, 8, 3, hardness)
    rng = np.random.default_rng(5)
    levels = [sampler.next_level(lambda: splits.sample_train_index(rng))[0] for _ in range(600)]
    hard = np.mean([hardness(i) < TAU["8"] for i in levels])
    assert hard == pytest.approx(0.5 + 0.5 * 0.1, abs=0.06)
    assert all(splits.split_of(i) == "train" for i in levels)


def test_hardness_thresholds_reproduce_from_the_train_pool():
    tau = load_script("compute_exp009_tau")
    module_tau = tau.thresholds(tau.sample_indices())
    assert module_tau == pytest.approx(TAU, rel=1e-12)


# ------------------------------------------------------------- trainer integration
def test_uniform_sampler_is_the_original_draw():
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from test_exp008 import GOLDEN, fingerprint, tiny_trainer

    explicit = tiny_trainer("fixed")
    assert explicit.sampler_settings == SamplerSettings() and not explicit.level_samplers
    assert fingerprint(explicit, 3) == GOLDEN["fixed"]
    assert "sampler" not in explicit.checkpoint(0, 0)


def test_replays_keep_the_deployment_and_draw_new_fading():
    settings = SamplerSettings(
        kind="plr", score="avoidable", buffer_size=5, min_fill=1, replay_probability=1.0
    )
    run = trainer(settings, env_ues=(3,), episode_slots=10, rollout=10)
    first_index = run.level[0][0]
    first_gain = run.envs[0].channel.gain_block(0, 1).copy()
    run.collect()  # the episode ends: scored, then the level is replayed
    assert run.level_samplers[3].levels == [first_index]
    assert run.level[0][0] == first_index
    assert not np.allclose(run.envs[0].channel.gain_block(0, 1), first_gain, rtol=1e-6, atol=0)
    state = run.checkpoint(10, 1)
    assert state["sampler"]["kind"] == "plr" and state["level_samplers"]["3"]["levels"]


def test_gae_score_is_the_episode_mean_absolute_advantage():
    settings = SamplerSettings(kind="plr", score="gae", buffer_size=10, min_fill=5)
    run = trainer(settings, env_ues=(3, 3), episode_slots=10, rollout=10)
    levels = [run.level[env][0] for env in range(2)]
    rollout, _ = run.collect()  # each environment plays exactly one episode
    run.train_step(rollout, 0.01)
    magnitude = run.ppo.last_advantages.abs().numpy().mean(axis=0)
    sampler = run.level_samplers[3]
    for env, level in enumerate(levels):
        assert sampler.scores[sampler.levels.index(level)] == pytest.approx(magnitude[env])
    assert not run.pending_gae


def test_avoidable_score_counts_missed_minus_undeliverable_packets():
    from dsa_period.capacity import best_alone_bits, evaluated_packets, undeliverable

    settings = SamplerSettings(kind="plr", score="avoidable")
    run = trainer(settings, env_ues=(3,), episode_slots=40, rollout=10)
    env = run.envs[0]
    while not env.truncated:  # all OFF: every packet of the window misses
        env.step(np.zeros((3, 2), dtype=np.int64))
    packets = evaluated_packets(run.traffic[0], 0, 40)
    lost = undeliverable(packets, best_alone_bits(run.scenarios[0], env.channel.gain_block(0, 40)))
    count = np.bincount(packets["ue"], minlength=3)
    lost_by_ue = np.bincount(packets["ue"], weights=lost, minlength=3)
    expected = max((count - lost_by_ue) / np.maximum(count, 1))
    assert run._avoidable_score(0) == pytest.approx(expected)
    assert 0 < expected <= 1


# ------------------------------------------------------------------ per-N runs
def test_per_n_config_equals_r0_apart_from_the_environments():
    raw = tomllib.loads((ROOT / "configs" / "exp009-step1-p0.toml").read_text())
    r0 = tomllib.loads((ROOT / "configs" / "exp008-round1-r0.toml").read_text())
    assert raw["experiment"]["per_n"] == [2, 4, 6, 8, 10, 12, 16, 20]
    assert raw["experiment"]["seeds"] == [101, 202] and raw["experiment"]["condition"] == "P0"
    for section in ("model", "ppo", "update", "constraint"):
        assert raw[section] == r0[section], section
    training = resolve_training(raw, 8)
    assert training["env_ues"] == [8] * 14
    assert {k: v for k, v in training.items() if k != "env_ues"} == {
        k: v for k, v in r0["training"].items() if k != "env_ues"
    }
    assert settings_from(raw, "fixed", 8)[4] == SamplerSettings()
    with pytest.raises(ValueError):
        resolve_training(raw, None)
    with pytest.raises(ValueError):
        resolve_training(raw, 14)
    with pytest.raises(ValueError):
        resolve_training(r0, 8)


# ------------------------------------------------------------------------ sets
SETS = json.loads((ROOT / "docs" / "research" / "exp009-sets.json").read_text())


def test_exp009_set_manifest_matches_the_code():
    assert load_script("register_exp008_sets").build("EXP-DSA05-009") == SETS


def test_v2_confirm2_is_separate_and_needs_its_switch():
    assert list(splits_v2.indices("v2-confirm2")) == list(range(30_500, 31_000))
    assert not set(splits_v2.indices("v2-confirm1")) & set(splits_v2.indices("v2-confirm2"))
    assert splits_v2.evaluation_repeats("v2-confirm2") == 1
    sequence = np.random.SeedSequence([splits_v2.EXECUTION_SEED_BASE, 3, 30_600, 0]).spawn(4)
    manual = [np.random.Generator(np.random.PCG64(s)).random() for s in sequence]
    assert [
        g.random() for g in splits_v2.execution_generators("v2-confirm2", 4, 30_600, 0)
    ] == manual
    for argv in (["--split", "v2-confirm2"], ["--split", "v2-confirm2", "--confirm-c1"]):
        with pytest.raises(SystemExit):
            evaluate_capacity.main(["--config", str(CONFIG), "--policies", "best_local", *argv])


def test_exp009_screening_set_is_guarded():
    assert splits_v2.screening_instances("exp009-screen256") == list(range(3000, 3256))
    with pytest.raises(ValueError):
        evaluate_capacity.instances_for("dev", 3200, 100)
    with pytest.raises(SystemExit):  # not formal
        evaluate_capacity.main(
            ["--config", str(CONFIG), "--policies", "best_local", "--split", "dev",
             "--screen", "exp009-screen256", "--grid", "8"]
        )  # fmt: skip


# ------------------------------------------------------------------- selection
def _run(tmp: Path, name: str, condition: str, seed: int, num_ues) -> Path:
    run = tmp / name
    (run / "checkpoints").mkdir(parents=True)
    manifest = {"experiment": {"id": "x", "condition": condition}, "seed": seed}
    manifest["num_ues"] = num_ues
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_per_n_selection_uses_own_n_and_breaks_ties(tmp_path):
    screen = {}
    for seed in (101, 202):
        mixed = _run(tmp_path, f"m-{seed}", "R0", seed, None)
        for k in range(1, 6):
            spec = f"marl:{mixed}/checkpoints/step-00{k}000000.pt"
            screen[spec] = {n: {"F01": 0.9, "F05": 0.9 + 0.01 * k} for n in sel.GRID}
        for n in sel.GRID:
            run = _run(tmp_path, f"p-{n}-{seed}", "P0", seed, n)
            for k in range(1, 6):
                spec = f"marl:{run}/checkpoints/step-00{k}000000.pt"
                f01 = 0.95 if k in (2, 3) else 0.9
                screen[spec] = {m: {"F01": 0.0, "F05": 0.0} for m in sel.GRID if m != n}
                screen[spec][n] = {"F01": f01, "F05": 0.9}
    chosen = sel.select(screen, {"R0": "M0"})
    assert set(chosen) == {"M0", "P0"}
    p0 = chosen["P0"]["101"]["8"]
    assert p0["slots"] == 3_000_000 and p0["F01"] == 0.95  # tie on F01 and F05: the later
    assert all(c["F01"] > 0 for c in p0["checkpoints"])  # other-N rows of per-N runs ignored
    m0 = chosen["M0"]["202"]["6"]
    assert m0["slots"] == 5_000_000  # F01 ties: higher F05
    assert p0["end_F01"] == [0.9, 0.9]
    assert sel.layers(chosen)["P0"]["101"]["8"] == p0["selected"]


def _layer(f: dict[int, float], end=None) -> dict:
    return {
        str(n): {"F01": f[n], "F05": f[n], "end_F01": [end if end is not None else f[n]] * 2}
        for n in sel.GRID
    }


def _cond(f: dict[int, float], f202=None, end=None) -> dict:
    return {"101": _layer(f, end), "202": _layer(f202 or f, end)}


BASE = {2: 1.0, 4: 0.98, 6: 0.95, 8: 0.92, 10: 0.85, 12: 0.7, 16: 0.2, 20: 0.0}


def test_step1_rules_and_choice_of_b():
    better = {**BASE, 6: 0.97, 8: 0.95}
    assert sel.step1({"M0": _cond(BASE), "P0": _cond(better)})["B"] == "P0"
    worse_low = {**better, 4: 0.96}  # rule 2: N = 4 drops 0.02
    result = sel.step1({"M0": _cond(BASE), "P0": _cond(worse_low)})
    assert result["B"] == "M0" and not result["P0"]["rules"]["rule2_other_n_no_drop"]
    dip = {**BASE, 6: 0.94}  # seed 202: S_near -0.005; mean +0.01 but one seed down
    split = sel.step1({"M0": _cond(BASE), "P0": _cond(better, f202=dip)})
    assert split["P0"]["delta_S_near"] == pytest.approx(0.01)
    assert split["P0"]["rules"]["rule1_improvement"] is False
    traded = {**BASE, 6: 0.93, 8: 0.99}  # S_near +0.025 but F(6) -0.02
    assert not sel.step1({"M0": _cond(BASE), "P0": _cond(traded)})["P0"]["rules"][
        "rule5_near_per_n"
    ]


def test_step2_choice_prefers_the_simpler_condition_within_the_tie_band():
    table = {
        "P0": _cond(BASE),
        "S1": _cond({**BASE, 6: 0.97, 8: 0.95}),  # S_near +0.025
        "P1": _cond({**BASE, 6: 0.98, 8: 0.96}),  # +0.035
        "P2": _cond({**BASE, 6: 0.96, 8: 0.95}, end=0.0),  # fails the end-segment rule
    }
    result = sel.step2(table, "P0")
    assert result["retained"] == ["S1", "P1"] and result["M"] == "P1"  # 0.01 apart: not tied
    table["P1"] = _cond({**BASE, 6: 0.975, 8: 0.95})  # +0.0275, within 0.005 of S1
    assert sel.step2(table, "P0")["M"] == "S1"
    assert sel.step2({"P0": _cond(BASE)}, "P0")["M"] == "P0"


# ---------------------------------------------------------- confirmation analysis
def _episodes(shares: dict[int, float], instances: int, seed: int) -> dict[int, list]:
    rng = np.random.default_rng(seed)
    out = {}
    for n, share in shares.items():
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


def test_confirmation_analysis_reads_seed_layers():
    analyze = load_script("analyze_exp008_confirm")
    grid = (2, 4, 6, 8, 10, 12, 16, 20, 24)
    m_f = {2: 1.0, 4: 1.0, 6: 1.0, 8: 0.975, 10: 0.9, 12: 0.7, 16: 0.3, 20: 0.1, 24: 0.0}
    a_f = {**m_f, 8: 0.85, 10: 0.8}
    episodes, layers = {}, {"M": [], "A_mix": []}
    for name, shares in (("M", m_f), ("A_mix", a_f)):
        for s in range(3):
            layer = {}
            for n in grid:
                model = 20 if n >= 24 else n  # N >= 24 use the N = 20 model
                spec = f"marl:{name}-{s}-n{model}"
                episodes.setdefault(spec, {}).update(
                    _episodes({n: shares[n]}, 40, 100 * s + n + (0 if name == "M" else 7))
                )
                layer[str(n)] = spec
            layers[name].append(layer)
    episodes["best_local"] = _episodes({**a_f, 6: 0.9}, 40, 999)
    episodes["central_edf"] = _episodes({n: 1.0 for n in grid}, 40, 998)
    methods = {
        "primary": "M",
        "baseline": "A_mix",
        "M": layers["M"],
        "A_mix": layers["A_mix"],
        "best_local": "best_local",
        "rules": ["best_local"],
        "edf": ["central_edf"],
    }
    result = analyze.analyze(episodes, methods, list(range(40)), 200)
    t = result["tests"]["0.01"]
    assert t["N_star"] == {"M": "8", "A_mix": "6", "best_local": "4"}
    assert result["verdict_eps_0.01"] == {"H1": True, "H2": True, "H3": True, "all": True}
    assert "M-A_mix" in t["joint_capacity_bootstrap"]
    assert result["grid"] == list(grid)


def test_step2_configs_are_mixed_r0_with_the_designed_samplers():
    r0 = tomllib.loads((ROOT / "configs" / "exp008-round1-r0.toml").read_text())
    expected = {
        "s1": {"kind": "stratified", "hard_share": 0.5, "tau": TAU},
        "p1": {"kind": "plr", "score": "gae"},
        "p2": {"kind": "plr", "score": "avoidable"},
    }
    plr = {"replay_probability": 0.5, "buffer_size": 500, "min_fill": 50, "beta": 0.1, "rho": 0.1}
    for key, sampler in expected.items():
        raw = tomllib.loads((ROOT / "configs" / f"exp009-step2-{key}.toml").read_text())
        experiment = raw["experiment"]
        assert (experiment["step"], experiment["condition"], experiment["base"]) == (
            2,
            key.upper(),
            "M0",
        )
        assert experiment["seeds"] == [101, 202]
        for section in ("training", "model", "ppo", "update", "constraint"):
            assert raw[section] == r0[section], (key, section)
        assert raw["sampler"] == (sampler | plr if sampler["kind"] == "plr" else sampler)
        assert settings_from(raw, "fixed")[4] == SamplerSettings(**raw["sampler"])
