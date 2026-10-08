"""Train one EXP-DSA05-010 run: DAgger imitation of N_EDF at one N and seed.

``--formal`` requires a clean committed tree, a seed and N declared by the
config and no overrides; ``--set imitation.KEY=VALUE`` exists only for smoke
tests. The output holds manifest.json, training.jsonl, checkpoints/ (one per
iteration, named by the cumulative environment slots) and result.json.
"""

import argparse
import hashlib
import json
import platform
import tomllib
from dataclasses import asdict
from pathlib import Path

import torch

from dsa_period import splits_v2
from dsa_period.config import load_scenario
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.experiments.train_capacity_mappo import parse_value
from dsa_period.marl.imitation import ImitationSettings, ImitationTrainer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp010-il.toml"))
    parser.add_argument("--scenario", type=Path, default=Path("configs/scenario-v2.toml"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-ues", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="imitation.KEY=VALUE")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    provenance = git_provenance()
    if args.formal and (provenance["dirty"] or args.set):
        parser.error("--formal requires a clean tree and no overrides")
    raw = tomllib.loads(args.config.read_text())
    declared = raw["experiment"]
    if args.num_ues not in declared["per_n"]:
        parser.error(f"{args.config} trains at N = {declared['per_n']}")
    if args.formal and args.seed not in declared["seeds"]:
        parser.error(f"{args.config} declares seeds {declared['seeds']}")
    table = dict(raw["imitation"])
    for item in args.set:
        key, _, value = item.partition("=")
        section, _, name = key.partition(".")
        if section != "imitation" or name not in ImitationSettings.__dataclass_fields__:
            parser.error(f"unknown override {key}")
        table[name] = parse_value(value)
    settings = ImitationSettings(**table)
    scenario = load_scenario(args.scenario, args.num_ues)
    args.output.mkdir(parents=True)
    manifest = {
        "experiment": declared,
        "protocol_id": splits_v2.PROTOCOL_ID,
        "formal": args.formal,
        "provenance": provenance,
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "scenario": str(args.scenario),
        "scenario_sha256": hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        "variant": "imitation",
        "seed": args.seed,
        "num_ues": args.num_ues,
        "overrides": args.set,
        "resolved": {"imitation": asdict(settings)},
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    result = ImitationTrainer(scenario, settings, args.num_ues, args.seed).run(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
