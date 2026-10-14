"""Train one EXP-DSA05-006/008 run: one constraint variant (fixed = A, local = C) and seed.

The optional ``[update]`` section of the config holds the EXP-DSA05-008 update
options (``trainer_v2.UpdateSettings``); without it the EXP-006 update runs. The
optional ``[sampler]`` section holds the EXP-DSA05-009 deployment sampler
(``level_sampler.SamplerSettings``). A per-N config (``experiment.per_n`` lists
the allowed N, ``training.num_envs`` the environments) is run once per N with
``--num-ues N``: all environments then have N UEs (EXP-009, protocol v2 §4).
The optional ``[information]`` section adds the EXP-DSA05-011 urgency state
summary (``urgency.SummarySettings``); without it the actor sees L0 only.
EXP-DSA05-012 adds the optional ``[survival]`` (``survival.SurvivalSettings``)
and ``[branching]`` (``trainer_v2.BranchSettings``) sections. A config that
declares ``experiment.scenario`` runs only on that scenario file.
``--resume CHECKPOINT`` continues that checkpoint's run (EXP-DSA05-011B design
§2.2) into a new output directory: the config must equal the run's except a
larger budget with the same entropy decay length; a formal resume needs a formal
source run. The manifest records the checkpoint and its SHA-256.

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
from dataclasses import asdict
from pathlib import Path

import torch

from dsa_period import splits_v2
from dsa_period.config import load_scenario
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.marl.level_sampler import SamplerSettings
from dsa_period.marl.ppo import PPOSettings
from dsa_period.marl.trainer_v2 import (
    BranchSettings,
    CapacityConstraint,
    CapacityTrainer,
    CapacityTrainingSettings,
    UpdateSettings,
)
from dsa_period.survival import SurvivalSettings
from dsa_period.urgency import SummarySettings


def resolve_training(raw: dict, num_ues: int | None) -> dict:
    """The [training] table with env_ues filled in for a per-N config."""
    per_n = raw["experiment"].get("per_n")
    table = dict(raw["training"])
    if per_n is None:
        if num_ues is not None:
            raise ValueError("--num-ues needs a per-N config (experiment.per_n)")
        return table
    if num_ues not in per_n or "env_ues" in table:
        raise ValueError(f"a per-N config runs at one N of {per_n}, given by --num-ues")
    table["env_ues"] = [num_ues] * table.pop("num_envs")
    return table


def settings_from(raw: dict, variant: str, num_ues: int | None = None):
    """Training, PPO, constraint, update and sampler settings of one run."""
    table = resolve_training(raw, num_ues)
    training = CapacityTrainingSettings(
        **{k: tuple(v) if k == "env_ues" else v for k, v in table.items()},
        **raw["model"],
    )
    return (
        training,
        PPOSettings(**raw["ppo"]),
        CapacityConstraint(variant, **raw["constraint"][variant]),
        UpdateSettings(**raw.get("update", {})),
        SamplerSettings(**raw.get("sampler", {})),
    )


def information_from(raw: dict) -> SummarySettings | None:
    """The EXP-011 summary of a config ([information]); None for L0 configs.

    ``theta_db = "all"`` means every uplink bit is read.
    """
    table = raw.get("information")
    if table is None:
        return None
    table = dict(table)
    if table.get("theta_db") == "all":
        table["theta_db"] = None
    return SummarySettings(**table)


def survival_from(raw: dict) -> tuple[SurvivalSettings | None, BranchSettings | None]:
    """The EXP-012 survival settings and branch episodes of a config (None when absent)."""
    survival = raw.get("survival")
    branching = raw.get("branching")
    return (
        None if survival is None else SurvivalSettings(**survival),
        None if branching is None else BranchSettings(**branching),
    )


def parse_value(text: str) -> int | float | str:
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp006-capacity-mappo.toml"))
    parser.add_argument("--scenario", type=Path, default=Path("configs/scenario-v2.toml"))
    parser.add_argument("--variant", choices=("fixed", "local"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-ues", type=int, help="N of a per-N config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, metavar="CHECKPOINT", help="continue its run")
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
    declared = raw["experiment"]
    if declared.get("variant", args.variant) != args.variant:
        parser.error(f"{args.config} declares variant {declared['variant']}")
    if args.formal and args.seed not in declared.get("seeds", [args.seed]):
        parser.error(f"{args.config} declares seeds {declared['seeds']}")
    if "scenario" in declared and Path(declared["scenario"]) != args.scenario:
        parser.error(f"{args.config} runs on {declared['scenario']}, not {args.scenario}")
    changes = {}
    if args.budget_slots is not None:
        changes["training.budget_slots"] = args.budget_slots
    for item in args.set:
        key, _, value = item.partition("=")
        changes[key] = parse_value(value)
    for key, value in changes.items():
        section, _, name = key.partition(".")
        if section == "update":
            if name not in UpdateSettings.__dataclass_fields__:
                parser.error(f"unknown override {key}")
            raw.setdefault("update", {})[name] = value
            continue
        if section == "information":
            if name not in SummarySettings.__dataclass_fields__:
                parser.error(f"unknown override {key}")
            raw.setdefault("information", {})[name] = value
            continue
        if section in ("survival", "branching"):
            fields = SurvivalSettings if section == "survival" else BranchSettings
            if name not in fields.__dataclass_fields__:
                parser.error(f"unknown override {key}")
            raw.setdefault(section, {})[name] = value
            continue
        table = raw["constraint"][args.variant] if section == "constraint" else raw[section]
        if name not in table:
            parser.error(f"unknown override {key}")
        table[name] = value
    try:
        training, ppo, constraint, update, sampler = settings_from(raw, args.variant, args.num_ues)
        information = information_from(raw)
        survival, branching = survival_from(raw)
    except ValueError as exc:
        parser.error(str(exc))
    resume = None
    if args.resume is not None:
        source = json.loads((args.resume.parent.parent / "manifest.json").read_text())
        if args.formal and not source["formal"]:
            parser.error(f"{args.resume} comes from an informal run")
        resume = {
            "checkpoint": str(args.resume),
            "sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest(),
            "source_config": source["config"],
            "source_provenance": source["provenance"],
        }
    # Any N works: the trainer derives one scenario per environment from it.
    scenario = load_scenario(args.scenario, max(training.env_ues))
    trainer = CapacityTrainer(
        scenario, training, ppo, constraint, args.seed, update, sampler, information,
        survival, branching,
    )  # fmt: skip
    start = (0, 0)
    if args.resume is not None:
        try:
            start = trainer.resume(torch.load(args.resume, map_location="cpu", weights_only=False))
        except ValueError as exc:
            parser.error(f"{args.resume}: {exc}")
        resume.update(slots=start[0], iteration=start[1])
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
        "num_ues": args.num_ues,
        "overrides": changes,
        "resume": resume,
        "resolved": {
            "training": resolve_training(raw, args.num_ues),
            "model": raw["model"],
            "ppo": raw["ppo"],
            "constraint": raw["constraint"][args.variant],
            "update": asdict(update),
            "sampler": asdict(sampler),
            "information": None if information is None else asdict(information),
            "survival": None if survival is None else asdict(survival),
            "branching": None if branching is None else asdict(branching),
        },
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    result = trainer.run(args.output, start=start)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
