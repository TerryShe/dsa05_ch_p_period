"""Train one EXP-DSA05-006 run: one constraint variant (fixed = A, local = C) and seed.

``--formal`` requires a clean committed tree and no overrides; ``--budget-slots``
and ``--set`` exist only for smoke tests and throughput checks. The output holds
manifest.json, training.jsonl, checkpoints/ and result.json. Validation of the
checkpoints runs afterwards (``evaluate_capacity`` with ``marl:`` specs).
"""

import argparse
import hashlib
import json
import platform
import tomllib
from pathlib import Path

import torch

from dsa_period import splits_v2
from dsa_period.config import load_scenario
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.marl.ppo import PPOSettings
from dsa_period.marl.trainer_v2 import (
    CapacityConstraint,
    CapacityTrainer,
    CapacityTrainingSettings,
)


def settings_from(raw: dict, variant: str):
    training = CapacityTrainingSettings(
        **{k: tuple(v) if k == "env_ues" else v for k, v in raw["training"].items()},
        **raw["model"],
    )
    return (
        training,
        PPOSettings(**raw["ppo"]),
        CapacityConstraint(variant, **raw["constraint"][variant]),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp006-capacity-mappo.toml"))
    parser.add_argument("--scenario", type=Path, default=Path("configs/scenario-v2.toml"))
    parser.add_argument("--variant", choices=("fixed", "local"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--budget-slots", type=int, help="non-formal override")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="non-formal override",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    provenance = git_provenance()
    if args.formal and (provenance["dirty"] or args.budget_slots is not None or args.set):
        parser.error("--formal requires a clean tree and no overrides")
    raw = tomllib.loads(args.config.read_text())
    changes = {}
    if args.budget_slots is not None:
        changes["training.budget_slots"] = args.budget_slots
    for item in args.set:
        key, _, value = item.partition("=")
        changes[key] = float(value) if "." in value or "e" in value else int(value)
    for key, value in changes.items():
        section, _, name = key.partition(".")
        table = raw["constraint"][args.variant] if section == "constraint" else raw[section]
        if name not in table:
            parser.error(f"unknown override {key}")
        table[name] = value
    training, ppo, constraint = settings_from(raw, args.variant)
    # Any N works: the trainer derives one scenario per environment from it.
    scenario = load_scenario(args.scenario, max(training.env_ues))
    args.output.mkdir(parents=True)
    manifest = {
        "experiment": raw["experiment"],
        "protocol_id": splits_v2.PROTOCOL_ID,
        "formal": args.formal,
        "provenance": provenance,
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "scenario": str(args.scenario),
        "scenario_sha256": hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        "variant": args.variant,
        "seed": args.seed,
        "overrides": changes,
        "resolved": {
            "training": raw["training"],
            "model": raw["model"],
            "ppo": raw["ppo"],
            "constraint": raw["constraint"][args.variant],
        },
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    trainer = CapacityTrainer(scenario, training, ppo, constraint, args.seed)
    result = trainer.run(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
