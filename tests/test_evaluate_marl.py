import json
from pathlib import Path

import pytest

from dsa_period.config import load_scenario
from dsa_period.experiments import evaluate_marl
from dsa_period.marl.ppo import PPOSettings
from dsa_period.marl.trainer import ConstraintSettings, Trainer, TrainingSettings

CONFIGS = Path(__file__).parents[1] / "configs"
N4 = str(CONFIGS / "scenario-v1-n4.toml")


def trained_run(tmp_path, formal=True, scenario=N4):
    sc = load_scenario(N4)
    settings = TrainingSettings(
        budget_slots=32,
        episode_slots=40,
        num_envs=2,
        rollout_slots=16,
        validation_every_slots=32,
        hidden=16,
        critic_hidden=32,
    )
    constraint = ConstraintSettings(
        "local", eta=2.0, mu_max=10.0, mu0=2.0, mu0_candidates=(0.0, 4.0)
    )
    run = tmp_path / "run"
    Trainer(sc, settings, PPOSettings(), constraint, seed=11, validation_indices=[10_000]).run(
        run, log=lambda *_: None
    )
    manifest = {
        "formal": formal,
        "provenance": {"commit": "test", "dirty": False},
        "scenario": scenario,
        "variant": "local",
        "profile": "local-test",
        "seed": 11,
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_runner_reproduces_training_validation_episodes(tmp_path):
    run = trained_run(tmp_path)
    output = tmp_path / "eval.json"
    evaluate_marl.main(
        [
            "--scenario",
            N4,
            "--runs",
            str(run),
            "--baselines",
            "paced_sticky:margin=0",
            "--reference",
            "paced_sticky:margin=0",
            "--split",
            "validation",
            "--instances",
            "1",
            "--workers",
            "2",
            "--output",
            str(output),
        ]
    )
    record = json.loads(output.read_text())
    result = json.loads((run / "result.json").read_text())
    for choice, checkpoint in (
        ("selected", result["selected_checkpoint"]),
        ("final", result["final_checkpoint"]),
    ):
        stored = json.loads((run / "validation" / f"{checkpoint}.json").read_text())
        entry = next(e for e in stored["by_mu0"] if e["mu0"] == result["selected_mu0"])
        assert record["results"][f"local-test:{choice}"]["episodes"]["11"] == entry["episodes"]
    assert "paired_vs_reference" in record["results"]["local-test:selected"]
    sources = record["results"]["local-test:final"]["sources"][0]
    assert len(sources["checkpoint_sha256"]) == 64


def test_runner_guards_test_split_informal_runs_and_scenario_mismatch(tmp_path):
    with pytest.raises(SystemExit):
        evaluate_marl.main(["--scenario", N4, "--baselines", "random_max", "--split", "test"])
    informal = trained_run(tmp_path / "a", formal=False)
    with pytest.raises(SystemExit):
        evaluate_marl.main(
            ["--scenario", N4, "--runs", str(informal), "--split", "validation", "--instances", "1"]
        )
    other = trained_run(tmp_path / "b", scenario=str(CONFIGS / "scenario-v1-n8.toml"))
    with pytest.raises(SystemExit):
        evaluate_marl.main(
            ["--scenario", N4, "--runs", str(other), "--split", "validation", "--instances", "1"]
        )
    with pytest.raises(SystemExit):
        evaluate_marl.main(
            [
                "--scenario",
                N4,
                "--baselines",
                "random_max",
                "--reference",
                "sticky_max",
                "--split",
                "validation",
                "--instances",
                "1",
            ]
        )
