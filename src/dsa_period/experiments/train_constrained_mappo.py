"""Train one EXP-DSA05-001 run: one scenario variant, constraint variant and seed.

``--formal`` requires a clean committed tree and the configured budget and full
validation split; the overrides exist only for smoke tests and throughput checks.
The output directory holds manifest.json, training.jsonl, validation/, the
checkpoints of every validated actor and result.json with the protocol selection.
"""

import argparse
import hashlib
import json
import platform
import tomllib
from pathlib import Path

import torch

from dsa_period import splits
from dsa_period.config import load_scenario
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.marl.ppo import PPOSettings
from dsa_period.marl.trainer import ConstraintSettings, Trainer, TrainingSettings


def load_experiment(
    path: Path, variant: str
) -> tuple[dict, TrainingSettings, PPOSettings, ConstraintSettings]:
    raw = tomllib.loads(path.read_text())
    training = TrainingSettings(**raw["training"], **raw["model"])
    ppo = PPOSettings(**raw["ppo"])
    constraint = ConstraintSettings(variant=variant, **raw["constraint"][variant])
    return raw, training, ppo, constraint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/exp001-constrained-mappo.toml")
    )
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--variant", choices=("fixed", "global", "local"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--budget-slots", type=int, help="non-formal override")
    parser.add_argument("--validation-instances", type=int, help="non-formal: first K only")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="non-formal override of a training/ppo/constraint value",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    provenance = git_provenance()
    overrides = args.budget_slots is not None or args.validation_instances is not None or args.set
    if args.formal and (provenance["dirty"] or overrides):
        parser.error("--formal requires a clean tree and no overrides")
    raw, training, ppo, constraint = load_experiment(args.config, args.variant)
    changes = {}
    if args.budget_slots is not None:
        changes["training.budget_slots"] = args.budget_slots
    for item in args.set:
        key, _, value = item.partition("=")
        changes[key] = float(value) if "." in value or "e" in value else int(value)
    if changes:
        for key, value in changes.items():
            section, _, name = key.partition(".")
            table = raw["constraint"][args.variant] if section == "constraint" else raw[section]
            if name not in table:
                parser.error(f"unknown override {key}")
            table[name] = value
        training = TrainingSettings(**raw["training"], **raw["model"])
        ppo = PPOSettings(**raw["ppo"])
        constraint = ConstraintSettings(variant=args.variant, **raw["constraint"][args.variant])
    scenario = load_scenario(args.scenario)
    validation = list(splits.indices("validation"))
    if args.validation_instances is not None:
        validation = validation[: args.validation_instances]
    args.output.mkdir(parents=True)
    manifest = {
        "experiment": raw["experiment"],
        "protocol_id": splits.PROTOCOL_ID,
        "formal": args.formal,
        "provenance": provenance,
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "scenario": str(args.scenario),
        "scenario_sha256": hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        "variant": args.variant,
        "seed": args.seed,
        "overrides": changes
        | (
            {"validation_instances": args.validation_instances}
            if args.validation_instances is not None
            else {}
        ),
        "resolved": {
            "training": raw["training"],
            "model": raw["model"],
            "ppo": raw["ppo"],
            "constraint": raw["constraint"][args.variant],
        },
        "validation_instances": validation,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    trainer = Trainer(scenario, training, ppo, constraint, args.seed, validation)
    result = trainer.run(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
