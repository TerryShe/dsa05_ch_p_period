"""EXP-DSA05-011B: resumed training, stress set, starvation guard, selection, analysis, driver."""

import dataclasses
import importlib.util
import io
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from dsa_period import splits_v2, starvation
from dsa_period.capacity import run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.experiments import evaluate_capacity, select_exp011b, train_capacity_mappo
from dsa_period.marl.features import stack
from dsa_period.marl.level_sampler import LevelSampler, SamplerSettings
from dsa_period.marl.ppo import PPOSettings
from dsa_period.marl.trainer_v2 import (
    CapacityConstraint,
    CapacityTrainer,
    CapacityTrainingSettings,
    entropy_coef,
)
from dsa_period.rules_v2 import make_policy_v2
from dsa_period.urgency import SummarySettings, parse_wrapper

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "scenario-v2.toml"
KEYS = ("m", "u", "o", "l", "u0")
SOURCES = {"m": "configs/exp009-step2-s1.toml", **{k: f"configs/exp011-{k}.toml" for k in KEYS[1:]}}


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = _module("analyze_exp011b", "scripts/analyze_exp011b.py")
driver = _module("run_exp011b", "scripts/run_exp011b.py")
diagnose = _module("diagnose_exp011b", "scripts/diagnose_exp011b.py")


# ----------------------------------------------------------------- resumed training
def test_entropy_schedule_is_kept_by_the_resume_configs():
    original = CapacityTrainingSettings(budget_slots=5_000_000, entropy_decay_fraction=0.5)
    resumed = CapacityTrainingSettings(budget_slots=10_000_000, entropy_decay_fraction=0.25)
    for slots in (1_792, 1_000_000, 2_499_840, 2_500_000, 5_001_472, 9_999_000):
        assert entropy_coef(original, slots) == entropy_coef(resumed, slots)
    assert entropy_coef(resumed, 5_001_472) == 0.0


def test_level_sampler_state_round_trips():
    settings = SamplerSettings(kind="stratified", tau={"4": 10.0})
    first = LevelSampler(settings, 4, 7, lambda index: 0.0)
    for _ in range(5):
        first.next_level(lambda: 3)
    second = LevelSampler(settings, 4, 99, lambda index: 0.0)
    second.load_state_dict(first.state_dict())
    assert second.state_dict() == first.state_dict()
    assert [second.next_level(lambda: 3) for _ in range(3)] == [
        first.next_level(lambda: 3) for _ in range(3)
    ]


TAU = tomllib.loads((ROOT / "configs/exp011b-u.toml").read_text())["sampler"]["tau"]


def _tiny(seed: int = 5, budget: int = 64, fraction: float = 0.5, hidden: int = 16):
    settings = CapacityTrainingSettings(
        budget_slots=budget, entropy_decay_fraction=fraction, env_ues=(4, 6),
        rollout_slots=16, episode_slots=60, checkpoint_every_slots=32, hidden=hidden,
    )  # fmt: skip
    return CapacityTrainer(
        load_scenario(CONFIG, 6), settings, PPOSettings(), CapacityConstraint("fixed"), seed,
        sampler=SamplerSettings(kind="stratified", tau={"4": TAU["4"], "6": TAU["6"]}),
        information=SummarySettings("urgency"),
    )  # fmt: skip


@pytest.fixture(scope="module")
def checkpoint():
    trainer = _tiny()
    for slots in (32, 64):
        rollout, _ = trainer.collect()
        trainer.train_step(rollout, entropy_coef(trainer.settings, slots))
    buffer = io.BytesIO()
    torch.save(trainer.checkpoint(64, 2), buffer)
    return buffer.getvalue()


def _load(data: bytes) -> dict:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def test_resume_restores_the_run_and_continues_deterministically(checkpoint):
    runs = []
    for _ in range(2):
        trainer = _tiny(budget=128, fraction=0.25)
        assert trainer.resume(_load(checkpoint)) == (64, 2)
        stored = _load(checkpoint)["ppo"]
        for name in ("actor", "critic"):
            live = trainer.ppo.state_dict()[name]
            assert all(torch.equal(live[k], stored[name][k]) for k in live)
        rollout, _ = trainer.collect()
        trainer.train_step(rollout, entropy_coef(trainer.settings, 96))
        runs.append((rollout.tokens.copy(), trainer.ppo.actor.state_dict()))
    assert np.array_equal(runs[0][0], runs[1][0])
    assert all(torch.equal(runs[0][1][k], runs[1][1][k]) for k in runs[0][1])


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": 6, "budget": 128, "fraction": 0.25},  # another run
        {"budget": 64, "fraction": 0.5},  # no budget left
        {"budget": 128, "fraction": 0.5},  # a longer entropy decay
        {"budget": 128, "fraction": 0.25, "hidden": 8},  # another network
    ],
)
def test_resume_refuses_a_different_run(checkpoint, changes):
    with pytest.raises(ValueError):
        _tiny(**changes).resume(_load(checkpoint))


def _tiny_config(tmp_path: Path, name: str, budget: int, fraction: float) -> Path:
    text = (ROOT / "configs/exp011b-u-base.toml").read_text()
    for old, new in (
        ("budget_slots = 5000000", f"budget_slots = {budget}"),
        ("entropy_decay_fraction = 0.5", f"entropy_decay_fraction = {fraction}"),
        ("env_ues = [4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 16, 16, 20, 20]", "env_ues = [4, 6]"),
        ("rollout_slots = 128", "rollout_slots = 16"),
        ("episode_slots = 600", "episode_slots = 60"),
        ("checkpoint_every_slots = 1000000", "checkpoint_every_slots = 32"),
    ):
        assert text.count(old) == 1
        text = text.replace(old, new)
    path = tmp_path / name
    path.write_text(text)
    return path


def test_the_cli_resumes_a_run_into_a_new_directory(tmp_path):
    base = _tiny_config(tmp_path, "base.toml", 64, 0.5)
    resume = _tiny_config(tmp_path, "resume.toml", 128, 0.25)
    common = ["--scenario", str(CONFIG), "--variant", "fixed", "--seed", "404"]
    train_capacity_mappo.main(["--config", str(base), *common, "--output", str(tmp_path / "src")])
    source = tmp_path / "src" / "checkpoints" / "step-000000064.pt"
    outputs = []
    for name in ("a", "b"):
        train_capacity_mappo.main(["--config", str(resume), *common, "--resume", str(source),
                                   "--output", str(tmp_path / name)])  # fmt: skip
        outputs.append(tmp_path / name)
    manifest = json.loads((outputs[0] / "manifest.json").read_text())
    assert manifest["resume"]["slots"] == 64 and manifest["resume"]["iteration"] == 2
    assert len(manifest["resume"]["sha256"]) == 64
    assert sorted(p.name for p in (outputs[0] / "checkpoints").iterdir()) == [
        "step-000000096.pt", "step-000000128.pt",
    ]  # fmt: skip
    assert json.loads((outputs[0] / "result.json").read_text())["iterations"] == 4
    first = json.loads((outputs[0] / "training.jsonl").read_text().splitlines()[0])
    assert (first["iteration"], first["slots"]) == (3, 96)
    a, b = (_load((o / "checkpoints/step-000000128.pt").read_bytes()) for o in outputs)
    assert all(torch.equal(a["ppo"]["actor"][k], b["ppo"]["actor"][k]) for k in a["ppo"]["actor"])
    with pytest.raises(SystemExit):  # the base config's budget is used up
        train_capacity_mappo.main(["--config", str(base), *common, "--resume", str(source),
                                   "--output", str(tmp_path / "c")])  # fmt: skip
    assert not (tmp_path / "c").exists()


def _sections(path: str) -> dict:
    raw = tomllib.loads((ROOT / path).read_text())
    raw.pop("experiment")
    return raw


@pytest.mark.parametrize("key", KEYS)
def test_configs_are_the_original_recipe(key):
    original = _sections(SOURCES[key])
    assert _sections(f"configs/exp011b-{key}-base.toml") == original
    resumed = _sections(f"configs/exp011b-{key}.toml")
    assert resumed["training"].pop("budget_slots") == 10_000_000
    assert resumed["training"].pop("entropy_decay_fraction") == 0.25
    original["training"].pop("budget_slots")
    original["training"].pop("entropy_decay_fraction")
    assert resumed == original
    for stage, seeds in (("-base", [404, 505]), ("", [101, 202, 303, 404, 505])):
        experiment = tomllib.loads((ROOT / f"configs/exp011b-{key}{stage}.toml").read_text())[
            "experiment"
        ]
        assert experiment["seeds"] == seeds
        assert experiment["condition"] == driver.KEYS[key]


# ---------------------------------------------------------------- sets and starvation
def test_sets_are_registered_nested_and_guarded():
    register = _module("register_sets", "scripts/register_exp008_sets.py")
    manifest = json.loads((ROOT / "docs/research/exp011b-sets.json").read_text())
    assert manifest == json.loads(json.dumps(register.build("EXP-DSA05-011B")))
    stress = manifest["screening"]["exp011b-stress"]["instances"]
    assert {n: len(v) for n, v in stress.items()} == {"8": 55, "12": 81, "16": 107}
    assert set(stress["8"]) <= set(stress["12"]) <= set(stress["16"])
    assert all(7000 <= i < 10_000 for v in stress.values() for i in v)
    assert splits_v2.screening_instances("exp011b-eval512") == list(range(6000, 6512))
    with pytest.raises(ValueError):
        splits_v2.screening_instances("exp011b-stress")
    for index in (6000, 6511, 7000, 9999):
        with pytest.raises(ValueError):
            splits_v2.check_dev_range([index])
    with pytest.raises(SystemExit):
        evaluate_capacity.main(["--config", str(CONFIG), "--split", "dev", "--screen",
                                "exp011b-stress", "--policies", "best_local", "--formal"])  # fmt: skip


def test_mean_snr_is_the_channel_mean_and_starvation_needs_a_strong_ue():
    scenario = load_scenario(CONFIG, 8)
    topology, traffic = scenario.instance(1000)
    env = scenario.make_env(topology, traffic, splits_v2.fading_seeds("dev", 8, 1000, 0))
    mean = np.asarray(env.channel.mean_gain_linear)
    mean = mean.max(axis=1) if mean.ndim == 2 else mean
    expected = 10 * np.log10(scenario.menu.power_w[-1] * mean / scenario.physical.noise_power_w)
    assert np.allclose(starvation.mean_snr_db(scenario, 1000), expected)
    assert list(starvation.starved([0.9, 0.9, 0.4], [25.0, 15.0, 40.0])) == [True, False, False]
    snr = np.array(starvation.dev_snr_db(8, 1000))
    strong, weak = int(np.argmax(snr)), int(np.argmin(snr))
    rates = np.zeros(8)
    rates[weak] = 0.9
    row = {"instance": 1000, "violation_rate": rates.tolist(), "distance_m": [1.0] * 8}
    assert starvation.cells([row], 8) == [] or snr[weak] >= 20
    rates[strong] = 0.6
    row["violation_rate"] = rates.tolist()
    found = starvation.cells([row], 8)
    assert snr[strong] >= 20 and [c["ue"] for c in found] == [strong]


# ------------------------------------------------------------------------ selection
def _run(tmp_path: Path, name: str, condition: str, seed: int) -> Path:
    run = tmp_path / name
    (run / "checkpoints").mkdir(parents=True)
    manifest = {"experiment": {"id": "x", "condition": condition}, "seed": seed, "num_ues": None}
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_guarded_selection_excludes_starving_checkpoints_at_every_n(tmp_path):
    u, m = _run(tmp_path, "u", "U", 101), _run(tmp_path, "m", "S1", 101)
    spec = {f"{r.name}{k}": f"marl:{r}/checkpoints/step-00{k}000000.pt" for r in (u, m)
            for k in (1, 2, 6)}  # fmt: skip
    screen, cells = {}, {}
    for name, s in spec.items():
        p2 = {"1": 0.001, "2": 0.004, "6": 0.003}[name[-1]]
        screen[s] = {n: {"P2": p2, "F01": 0.9} for n in (6, 8, 10, 12, 16, 20)}
        cells[s] = {"screen": {n: [] for n in (6, 8, 10, 12, 16, 20)},
                    "stress": {n: [] for n in (8, 12, 16)}}  # fmt: skip
    cells[spec["u1"]]["stress"][16] = [{"instance": 7022}]  # U 1M starves at N = 16 only
    cells[spec["m1"]]["screen"][20] = [{"instance": 3001}]  # outside M's evaluation grid
    chosen = select_exp011b.select(screen, cells, {"S1": "M"})
    assert {c["selected"] for c in chosen["U"]["101"]["cells"].values()} == {spec["u6"]}
    assert {c["selected"] for c in chosen["M"]["101"]["cells"].values()} == {spec["m1"]}
    early = select_exp011b.select(screen, cells, {"S1": "M"}, max_slots=5_100_000)
    assert {c["selected"] for c in early["U"]["101"]["cells"].values()} == {spec["u2"]}
    effect = select_exp011b.budget_effect(chosen, early)
    assert effect["U"]["8"] == pytest.approx(0.003 - 0.004)
    for s in (spec["u2"], spec["u6"]):
        cells[s]["stress"][8] = [{"instance": 7022}, {"instance": 7061}]
    fallback = select_exp011b.select(screen, cells, {"S1": "M"})["U"]["101"]["cells"]["8"]
    assert fallback["fallback"] and fallback["selected"] == spec["u1"]


def test_methods_need_five_seeds_of_every_condition():
    entry = {"cells": {"8": {"selected": "marl:x.pt"}}}
    selection = {
        c: {str(s): entry for s in (101, 202, 303, 404, 505)} for c in driver.KEYS.values()
    }
    out = select_exp011b.methods(selection)
    assert len(out["U"]) == 5 and out["cg_opt"] == "configured_grant_opt"
    del selection["L"]["505"]
    with pytest.raises(ValueError):
        select_exp011b.methods(selection)


# ------------------------------------------------------------------------- analysis
def _episodes(plan: dict, instances: list[int], w: dict, starve: tuple | None = None) -> dict:
    out: dict = {}
    for name, group in plan["groups"].items():
        for layer, member in enumerate(group if isinstance(group, list) else [group]):
            for n in plan["grids"][name]:
                spec = member if isinstance(member, str) else member[str(n)]
                rows = []
                for i in instances:
                    rates = np.zeros(n)
                    rates[0] = w[name] * (1 + 0.1 * ((i + layer) % 3))
                    if starve == (name, layer, n, i):
                        snr = np.array(starvation.dev_snr_db(n, i))
                        rates[int(np.argmax(snr))] = 0.9
                    rows.append({
                        "instance": i, "worst_violation_rate": float(rates.max()),
                        "violation_rate": rates.tolist(), "feasible": {"0.01": rates.max() <= 0.01},
                        "infeasible_for_all": {"0.01": False}, "distance_m": [10.0] * n,
                    })  # fmt: skip
                out.setdefault(spec, {})[n] = rows
    return out


W = {"U": 0.002, "M": 0.01, "M+Y": 0.006, "O": 0.008, "L": 0.009, "U-hide-all": 0.01,
     "U-hide-nonurgent": 0.009, "best_local": 0.02, "CG-opt": 0.03, "N_EDF": 0.001,
     "U0": 0.005, "U@0": 0.007}  # fmt: skip


def test_the_analysis_applies_the_pre_registered_rules():
    layers = lambda c: [{str(n): f"marl:{c}{s}.pt" for n in (6, 8, 10, 12, 16)}  # noqa: E731
                        for s in range(5)]  # fmt: skip
    plan = driver.plan(
        {c: layers(c) for c in driver.KEYS.values()}
        | {"best_local": "best_local", "cg_opt": "configured_grant_opt", "edf": "central_edf"}
    )
    assert len(set(driver.pairs_of(plan))) == len(driver.pairs_of(plan))
    for spec, _ in driver.pairs_of(plan):
        if "@" in spec:
            assert parse_wrapper(spec) is not None
    instances = list(range(16))
    result = analysis.analyze(_episodes(plan, instances, W), plan, instances, 200)
    assert result["decision"] == "effective" and result["content"]["supported"]
    assert result["coordination"]["learned_response_better"]
    assert result["proposed_method"]["U_is_the_proposed_method"]
    assert result["mechanism"][10]["H_M2"]
    starving = _episodes(plan, instances, W, starve=("U", 2, 8, 3))
    result = analysis.analyze(starving, plan, instances, 200)
    assert result["starvation_cells"]["U"][8] == 1
    assert not result["proposed_method"]["conditions"]["U_without_starvation_cells"]
    assert not result["proposed_method"]["U_is_the_proposed_method"]


# --------------------------------------------------------------------------- driver
def test_stages_need_their_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "OUT", tmp_path)
    with pytest.raises(SystemExit):
        driver.check_gate("stage1")
    (tmp_path / "gates.json").write_text(json.dumps({"0": {}, "A": {"branch": "T"}}))
    driver.check_gate("stage1")
    with pytest.raises(SystemExit):
        driver.check_gate("stage2")
    with pytest.raises(SystemExit):
        driver.check_gate("stage3")


def test_driver_labels_runs_uniquely():
    early, late = ("step-001001728", "step-005001472"), ("step-006001408", "step-010001152")
    runs = {"s1-seed101": early, "u-seed101": early, "m-seed404-base": early,
            "m-seed101": late, "u-seed101b": late, "m-seed404": late}  # fmt: skip
    names = [driver.label(Path(f"/x/{run}/checkpoints/{step}.pt"))
             for run, steps in runs.items() for step in steps]  # fmt: skip
    assert len(set(names)) == len(names) and "m-seed101-step-001001728" in names
    with pytest.raises(SystemExit):
        driver.run_queue([("a", [], Path("/x/o"), 1), ("b", [], Path("/x/o"), 1)], Path("/x"))


# ------------------------------------------------------------------- counterfactual
def test_the_gain_probe_leaves_the_episode_unchanged(checkpoint, tmp_path):
    path = tmp_path / "step-000000064.pt"
    path.write_bytes(checkpoint)
    scenario = dataclasses.replace(load_scenario(CONFIG, 6), horizon_slots=160, warmup_slots=40)
    spec = f"marl:{path}"
    plain = run_capacity_episode(
        scenario, make_policy_v2(spec, scenario.physical, scenario.menu), "dev", 3
    )
    probe = diagnose.GainProbe(make_policy_v2(spec, scenario.physical, scenario.menu), 1,
                               [0.0, -30.0])  # fmt: skip
    probed = run_capacity_episode(scenario, probe, "dev", 3)
    assert probed["violations"] == plain["violations"]
    values = np.array(probe.p_off)
    assert values.shape[1] == 2 and np.all((values >= 0) & (values <= 1))


# --------------------------------------------------------------------- input probes
def _probe_policy(checkpoint, tmp_path):
    path = tmp_path / "step-000000064.pt"
    path.write_bytes(checkpoint)
    scenario = dataclasses.replace(load_scenario(CONFIG, 6), horizon_slots=160, warmup_slots=40)
    return scenario, lambda: make_policy_v2(f"marl:{path}", scenario.physical, scenario.menu)


def test_input_probes_leave_the_episode_unchanged(checkpoint, tmp_path):
    scenario, policy = _probe_policy(checkpoint, tmp_path)
    gain = diagnose.GainProbe(policy(), 2, [0.0, *diagnose.GAIN_DB])
    plain = run_capacity_episode(scenario, gain, "dev", 3)
    names = diagnose.variants(scenario, summary=True)
    probe = diagnose.InputProbe(policy(), 2, names)
    probed = run_capacity_episode(scenario, probe, "dev", 3)
    assert probed["violations"] == plain["violations"]
    assert names == ["actual", "period=10", "period=20", "period=40", "hidden=0",
                     "gain=-10", "gain=-20", "gain=-30", "broadcast=0", "broadcast=1"]  # fmt: skip
    values = np.array(probe.p_off)
    assert len(values) == len(probe.urgent) == len(probe.marked) == len(probe.sends) > 0
    # "actual" is the policy's own P(OFF); UE 2 is slow (40 slots), so its own period is a no-op
    np.testing.assert_allclose(values[:, 0], np.array(gain.p_off)[:, 0])
    np.testing.assert_allclose(values[:, names.index("period=40")], values[:, 0])
    # the gain variants are the gain counterfactual's values
    np.testing.assert_allclose(values[:, 5:8], np.array(gain.p_off)[:, 1:])
    assert probe.sends == gain.sends
    # pending slots are numbered by release: one number per packet, in order
    steps = np.diff(probe.packet)
    assert len(probe.packet) == len(values) and set(steps) <= {0, 1} and steps.sum() > 0
    assert diagnose.variants(scenario, summary=False)[-1] == "gain=-30"


def test_per_packet_products_and_never_sent_packets():
    p_off = np.array([[0.5, 1.0], [0.5, 0.2], [0.9, 0.9], [0.8, 0.1]])
    packet, sends = np.array([1, 1, 1, 2]), np.array([False, False, True, False])
    silent, never = diagnose.per_packet(p_off, packet, sends)
    np.testing.assert_allclose(silent, [[0.225, 0.18], [0.8, 0.1]])
    assert never.tolist() == [False, True]


def test_each_variant_replaces_one_input_group():
    context, seen = np.zeros((3, 14), np.float32), np.zeros((3, 4), bool)
    hidden = torch.ones(3, 4, 8)
    varied, shown, state = diagnose.edit("period=20", context, seen, hidden, 1)
    assert varied[1, diagnose.PERIOD_INPUT] == 0.5 and np.count_nonzero(varied) == 1
    assert torch.equal(state, hidden) and not shown.any()
    varied, shown, state = diagnose.edit("hidden=0", context, seen, hidden, 1)
    assert state[1].abs().sum() == 0 and state[[0, 2]].eq(1).all() and hidden.eq(1).all()
    varied, shown, state = diagnose.edit("broadcast=1", context, seen, hidden, 1)
    assert shown[1].all() and not shown[[0, 2]].any() and not seen.any()
    with pytest.raises(ValueError):
        diagnose.edit("broadcast=1", context, None, hidden, 1)
    with pytest.raises(ValueError):
        diagnose.edit("gain=3", context, seen, hidden, 1)


def test_hidden_reset_clears_the_ue_at_each_of_its_releases(checkpoint, tmp_path):
    scenario, policy = _probe_policy(checkpoint, tmp_path)
    inner, at_release = policy(), []
    act = inner.act

    def recording(observations):
        obs = stack(observations)
        if obs.to_deadline[2] == obs.period[2]:
            at_release.append(float(inner._hidden[2].abs().sum()))
        return act(observations)

    inner.act = recording
    reset = diagnose.HiddenReset(inner, 2)
    run_capacity_episode(scenario, reset, "dev", 3)
    assert len(at_release) == reset.resets > 0 and max(at_release) == 0.0
    assert 0 < reset.pending and 0 <= reset.sent <= reset.pending


def test_the_probe_cell_is_the_first_starved_ue_or_the_worst_near_one(monkeypatch, tmp_path):
    monkeypatch.setattr(starvation, "dev_snr_db", lambda n, i: (30.0, 10.0))
    rows = [{"instance": 7, "violation_rate": [0.1, 0.9], "distance_m": [3.0, 60.0]},
            {"instance": 9, "violation_rate": [0.2, 0.0], "distance_m": [4.0, 2.0]}]  # fmt: skip
    assert diagnose.probe_cell(rows, 2) == {"instance": 9, "ue": 0, "kind": "worst near-AP UE"}
    rows[1]["violation_rate"][0] = 0.6
    assert diagnose.probe_cell(rows, 2) == {"instance": 9, "ue": 0, "kind": "starved"}
    with pytest.raises(SystemExit):  # replays are formal only
        diagnose.main(["probes", "--pairs", "a=b", "--output", str(tmp_path / "p.json")])
