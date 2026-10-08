"""EXP-DSA05-010 phase 0: EDF imitation trainer, checkpoints, evaluation slice, decision."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dsa_period import splits_v2
from dsa_period.capacity import run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.experiments import evaluate_capacity, train_imitation
from dsa_period.experiments import select_exp009 as sel
from dsa_period.marl.features import decode_actions
from dsa_period.marl.imitation import (
    ImitationSettings,
    ImitationTrainer,
    segment_loss,
    teacher_labels,
)
from dsa_period.marl.trainer_v2 import build_actor, load_capacity_policy
from dsa_period.rules_v2 import CentralEdfPolicy

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "scenario-v2.toml"
TINY = ImitationSettings(
    iterations=2, episodes_per_env=1, episode_slots=20, num_envs=2, segment_slots=8, epochs=1,
    batch_episodes=2, hidden=8,
)  # fmt: skip


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observations(n=6, slots=6):
    scenario = load_scenario(CONFIG, n)
    topology, traffic = scenario.instance(0)
    env = scenario.make_env(topology, traffic, splits_v2.fading_seeds("dev", n, 0, 0), 50)
    obs = env.reset()
    for _ in range(slots):
        obs = env.step(np.zeros((n, 2), dtype=np.int64)).observations
    return scenario, obs


def test_teacher_labels_are_the_edf_actions_in_student_indices():
    scenario, obs = observations()
    teacher = CentralEdfPolicy(scenario.physical, scenario.menu)
    labels = teacher_labels(teacher, obs, scenario.menu)
    assert np.array_equal(decode_actions(labels, scenario.menu), teacher.act(obs))
    waiting = np.array([o.remaining_bits > 0 for o in obs])
    assert np.all(labels[~waiting] == 0)
    served = labels > 0
    assert served.sum() == min(waiting.sum(), scenario.physical.num_subchannels)
    channels = decode_actions(labels[served], scenario.menu)[:, 0]
    assert len(set(channels)) == len(channels)  # one UE per subchannel, max power
    assert np.all(
        decode_actions(labels[served], scenario.menu)[:, 1] == scenario.menu.num_levels - 1
    )


def _batch(steps=12, ues=3, actions=21):
    torch.manual_seed(0)
    legal = torch.ones(steps, ues, actions, dtype=torch.bool)
    active = torch.ones(steps, ues, dtype=torch.bool)
    active[:, 0] = False
    legal[:, 0, 1:] = False
    labels = torch.randint(1, actions, (steps, ues))
    return {
        "tokens": torch.randn(steps, ues, 4, 14),
        "context": torch.randn(steps, ues, 13),
        "legal": legal,
        "labels": labels,  # UE 0 has no packet: its label is illegal and must be ignored
        "active": active,
    }


def test_inactive_ue_slots_stay_out_of_the_loss():
    scenario = load_scenario(CONFIG, 3)
    torch.manual_seed(1)
    actor = build_actor(scenario.physical, scenario.menu, 8)
    loss, _ = segment_loss(actor, _batch(), 0, 12, actor.initial_hidden(3))
    assert torch.isfinite(loss)


def test_truncated_segments_carry_the_hidden_state():
    scenario = load_scenario(CONFIG, 3)
    torch.manual_seed(2)
    actor = build_actor(scenario.physical, scenario.menu, 8)
    batch = _batch()
    starts = torch.zeros(12, 3, dtype=torch.bool)
    full, _ = actor.sequence(batch["tokens"], batch["context"], actor.initial_hidden(3), starts)
    hidden, parts = actor.initial_hidden(3), []
    for start in (0, 5, 10):
        stop = min(start + 5, 12)
        logits, hidden = actor.sequence(
            batch["tokens"][start:stop], batch["context"][start:stop], hidden, starts[start:stop]
        )
        hidden = hidden.detach()
        parts.append(logits)
    assert torch.allclose(full, torch.cat(parts), atol=1e-6)


def test_dagger_iterations_checkpoints_and_execution(tmp_path):
    trainer = ImitationTrainer(load_scenario(CONFIG, 3), TINY, 3, seed=7)
    result = trainer.run(tmp_path / "run", log=lambda *_: None)
    lines = [json.loads(x) for x in (tmp_path / "run" / "training.jsonl").read_text().splitlines()]
    assert [x["executed_by"] for x in lines] == ["teacher", "student"]
    assert result["episodes"] == 4 and len(trainer.data.items) == 4  # aggregated over iterations
    assert trainer.data.items[0]["tokens"].dtype == np.float16
    checkpoint = sorted((tmp_path / "run" / "checkpoints").glob("*.pt"))[-1]
    assert checkpoint.name == "step-000000080.pt"
    scenario = load_scenario(CONFIG, 5)
    policy = load_capacity_policy(checkpoint, scenario.physical, scenario.menu)
    short = scenario.__class__(**{**scenario.__dict__, "horizon_slots": 200, "warmup_slots": 40})
    assert run_capacity_episode(short, policy, "dev", 0)["num_ues"] == 5


def test_teacher_iteration_executes_the_teacher_actions(monkeypatch):
    trainer = ImitationTrainer(load_scenario(CONFIG, 3), TINY, 3, seed=7)
    executed, labelled = [], []
    original = trainer.collect.__func__

    import dsa_period.marl.imitation as module

    real = module.decode_actions

    def spy(indices, menu):
        executed.append(np.asarray(indices).copy())
        return real(indices, menu)

    real_labels = module.teacher_labels

    def label_spy(teacher, raw, menu):
        out = real_labels(teacher, raw, menu)
        labelled.append(out.copy())
        return out

    monkeypatch.setattr(module, "decode_actions", spy)
    monkeypatch.setattr(module, "teacher_labels", label_spy)
    original(trainer, by_teacher=True)
    assert np.array_equal(np.concatenate(executed), np.concatenate(labelled))


def test_imitation_cli_checks_n_and_seed(tmp_path):
    with pytest.raises(SystemExit):
        train_imitation.main(["--seed", "101", "--num-ues", "12", "--output", str(tmp_path / "a")])
    with pytest.raises(SystemExit):
        train_imitation.main(
            ["--seed", "999", "--num-ues", "8", "--formal", "--output", str(tmp_path / "b")]
        )


# ------------------------------------------------------------- evaluation slice
def test_exp010_slice_is_registered_and_guarded():
    manifest = json.loads((ROOT / "docs" / "research" / "exp010-sets.json").read_text())
    assert load_script("register_exp008_sets").build("EXP-DSA05-010") == manifest
    assert splits_v2.screening_instances("exp010-eval512") == list(range(4000, 4512))
    with pytest.raises(ValueError):
        evaluate_capacity.instances_for("dev", 4100, 10)
    with pytest.raises(SystemExit):
        evaluate_capacity.main(
            ["--config", str(CONFIG), "--policies", "best_local", "--split", "dev",
             "--screen", "exp010-eval512", "--grid", "8"]
        )  # fmt: skip


def test_selection_accepts_the_exp010_grid(tmp_path):
    screen = {}
    for n in (6, 8, 10):
        run = tmp_path / f"il-n{n}"
        (run / "checkpoints").mkdir(parents=True)
        (run / "manifest.json").write_text(
            json.dumps({"experiment": {"id": "x", "condition": "IL"}, "seed": 101, "num_ues": n})
        )
        for k in (1, 2):
            screen[f"marl:{run}/checkpoints/step-00{k}000000.pt"] = {n: {"F01": 0.9, "F05": 0.9}}
    with pytest.raises(ValueError):
        sel.select(screen)  # the EXP-009 grid needs every training N
    chosen = sel.select(screen, grid=(6, 8, 10))
    assert chosen["IL"]["101"]["8"]["slots"] == 2_000_000


# --------------------------------------------------------------------- decision
def _episodes(share: float, instances: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    feasible = np.zeros(instances, bool)
    feasible[rng.permutation(instances)[: round(share * instances)]] = True
    return [
        {"instance": i, "feasible": {"0.01": bool(f), "0.05": True}} for i, f in enumerate(feasible)
    ]


@pytest.mark.parametrize(
    ("il8", "amix8", "decision"),
    [
        (0.98, 0.92, "optimization-limited"),
        (0.94, 0.85, "partly optimization-limited"),
        (0.93, 0.93, "no evidence"),
    ],
)
def test_decision_rule(il8, amix8, decision):
    analyze = load_script("analyze_exp010")
    instances = list(range(400))
    episodes, methods = {}, {"IL": [], "A_mix": [], "M": []}
    for group, f8, seeds in (("IL", il8, 2), ("A_mix", amix8, 3), ("M", amix8, 3)):
        for s in range(seeds):
            layer = {}
            for n in (6, 8, 10):
                spec = f"marl:{group}-{s}-{n}"
                episodes[spec] = {
                    n: _episodes(f8 if n == 8 else 0.97, 400, 10 * s + n + len(group))
                }
                layer[str(n)] = spec
            methods[group].append(layer)
    for spec in ("best_local", "central_edf"):
        episodes[spec] = {n: _episodes(0.9, 400, n) for n in (6, 8, 10)}
    methods.update(best_local="best_local", edf="central_edf")
    result = analyze.analyze(episodes, methods, instances, 300)
    assert result["decision"] == decision
    assert result["D8"]["difference"] == pytest.approx(il8 - amix8, abs=0.003)
