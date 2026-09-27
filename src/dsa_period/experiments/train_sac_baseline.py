"""Train one EXP-DSA05-005 run: an XR-SBPC-style SAC-Discrete baseline profile and seed.

A profile fixes the reward (v1, v1s or v2) and the discount gamma. ``--formal``
requires a clean committed tree, the configured budget and the full validation
split; the overrides exist only for smoke tests and throughput checks. The output
directory holds manifest.json, training.jsonl, validation/, the policy checkpoint
of every validation and result.json with the protocol selection, in the same
layout as ``train_constrained_mappo``.
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
from dsa_period.marl.sac import SacSettings
from dsa_period.marl.sac_trainer import SacTrainer, SacTrainingSettings
from dsa_period.marl.xr_sbpc import RewardSettings


def resolve(raw: dict, profile: str) -> tuple[SacTrainingSettings, SacSettings, RewardSettings]:
    table = dict(raw.get("profiles", {}).get(profile, {}))
    if set(table) != {"reward", "gamma"}:
        raise ValueError(f"profile {profile!r} must set exactly reward and gamma")
    training = SacTrainingSettings(**raw["training"])
    sac = SacSettings(**(raw["sac"] | {"gamma": table["gamma"]}))
    reward = RewardSettings(kind=table["reward"], **raw["reward"])
    return training, sac, reward


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp005-xr-sbpc-baseline.toml"))
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--profile", required=True)
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
        help="non-formal override of a training/sac/reward value",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    provenance = git_provenance()
    overrides = args.budget_slots is not None or args.validation_instances is not None or args.set
    if args.formal and (provenance["dirty"] or overrides):
        parser.error("--formal requires a clean tree and no overrides")
    raw = tomllib.loads(args.config.read_text())
    if args.profile not in raw.get("profiles", {}):
        parser.error(f"unknown profile {args.profile}")
    changes: dict[str, float | int] = {}
    if args.budget_slots is not None:
        changes["training.budget_slots"] = args.budget_slots
    for item in args.set:
        key, _, value = item.partition("=")
        changes[key] = float(value) if "." in value or "e" in value else int(value)
    for key, value in changes.items():
        section, _, name = key.partition(".")
        if section not in ("training", "sac", "reward") or name not in raw[section]:
            parser.error(f"unknown override {key}")
        raw[section][name] = value
    try:
        training, sac, reward = resolve(raw, args.profile)
    except ValueError as error:
        parser.error(str(error))
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
        "variant": "sac",
        "profile": args.profile,
        "seed": args.seed,
        "overrides": changes
        | (
            {"validation_instances": args.validation_instances}
            if args.validation_instances is not None
            else {}
        ),
        "resolved": {
            "training": raw["training"],
            "sac": raw["sac"] | {"gamma": sac.gamma},
            "reward": raw["reward"] | {"kind": reward.kind},
        },
        "validation_instances": validation,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    trainer = SacTrainer(scenario, training, sac, reward, args.seed, validation)
    result = trainer.run(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
