"""Evaluate policies over the N grid of scenario v2.0 under protocol v2.

Every (policy, N, instance) episode is independent and runs in a process pool.
For each policy the output holds F(N) with its interval, per-UE violation
statistics, the capacity N* at every eps and the bootstrap distribution of N*.
``dev`` takes an explicit index range that must not touch a registered
screening set; ``--screen NAME`` reads exactly such a set (its instances and
grid or a subset of it, ``--formal`` only). Test evaluation needs
``--confirm-test`` and ``--formal`` (protocol v2 reads the test split only for
frozen methods); the confirmation sets need ``--formal`` and their own switch:
v2-confirm1 (EXP-DSA05-008) ``--confirm-c1``, v2-confirm2 (EXP-DSA05-009)
``--confirm-c2``.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dsa_period import splits_v2
from dsa_period.capacity import (
    EPSILONS,
    capacity_label,
    eps_key,
    run_capacity_episode,
    summarize,
)
from dsa_period.config import load_scenario
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.rules_v2 import make_policy_v2


def _episode(job: tuple) -> tuple:
    config, spec, n, split, instance = job
    if spec.startswith("marl:"):
        import torch

        torch.set_num_threads(1)  # one core per worker process
    scenario = load_scenario(config, n)
    policy = make_policy_v2(spec, scenario.physical, scenario.menu)
    return spec, n, run_capacity_episode(scenario, policy, split, instance)


def instances_for(
    split: str, first: int | None, count: int | None, screen: str | None = None
) -> list[int]:
    if screen is not None:
        if split != "dev" or first is not None or count is not None:
            raise ValueError("--screen reads its registered dev instances, nothing else")
        return splits_v2.screening_instances(screen)
    if split == "dev":
        if first is None or count is None:
            raise ValueError("dev evaluation needs --first-instance and --instances")
        chosen = [splits_v2.require("dev", i) for i in range(first, first + count)]
        splits_v2.check_dev_range(chosen)
        return chosen
    if first is not None or count is not None:
        raise ValueError("validation/test/confirmation sets always use the whole set")
    return list(splits_v2.indices(split))


def print_summary(name: str, summary: dict) -> None:
    for eps in EPSILONS:
        block = summary["per_eps"][eps_key(eps)]
        point = block["capacity"]
        star = capacity_label(point, summary["grid"])
        shares = " ".join(f"{r['num_ues']}:{r['feasible_share']:.2f}" for r in block["rows"])
        print(f"  {name:<44} eps={eps:<5} N*={star:<4} F(N) {shares}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--split",
        choices=("dev", "validation", "test", *splits_v2.CONFIRMATION_SETS),
        required=True,
    )
    parser.add_argument("--first-instance", type=int, help="dev split only")
    parser.add_argument("--instances", type=int, help="dev split only")
    parser.add_argument("--grid", type=int, nargs="+", help="default: the config's capacity grid")
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--confirm-c1", action="store_true", help="read v2-confirm1 (EXP-008)")
    parser.add_argument("--confirm-c2", action="store_true", help="read v2-confirm2 (EXP-009)")
    parser.add_argument("--screen", choices=tuple(splits_v2.SCREENING_SETS))
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"output {args.output} already exists")
    provenance = git_provenance()
    if args.formal and provenance["dirty"]:
        parser.error("--formal requires a clean, committed working tree")
    if args.split == "test" and not (args.confirm_test and args.formal):
        parser.error("test evaluation requires --confirm-test and --formal")
    switches = {"v2-confirm1": args.confirm_c1, "v2-confirm2": args.confirm_c2}
    if args.split in switches and not (switches[args.split] and args.formal):
        flag = "--confirm-c1" if args.split == "v2-confirm1" else "--confirm-c2"
        parser.error(f"{args.split} evaluation requires {flag} and --formal")
    if args.screen is not None and not args.formal:
        parser.error("a screening set is read only with --formal")
    try:
        chosen = instances_for(args.split, args.first_instance, args.instances, args.screen)
    except ValueError as exc:
        parser.error(str(exc))
    base = load_scenario(args.config, 4)
    grid = sorted(args.grid or base.capacity["grid"])
    if args.screen is not None:
        registered = list(splits_v2.SCREENING_SETS[args.screen]["grid"])
        if args.grid is not None and not set(grid) <= set(registered):
            parser.error(f"the screening set {args.screen} is registered at N = {registered}")
        grid = grid if args.grid is not None else registered
    for spec in args.policies:
        make_policy_v2(spec, base.physical, base.menu)  # validate before running
    jobs = [
        (args.config, spec, n, args.split, i)
        for spec in args.policies
        for n in grid
        for i in chosen
    ]
    results: dict = {spec: {n: {"-": []} for n in grid} for spec in args.policies}
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for done, (spec, n, episode) in enumerate(pool.map(_episode, jobs, chunksize=1), 1):
            results[spec][n]["-"].append(episode)
            if done % max(1, len(jobs) // 20) == 0:
                print(f"{done}/{len(jobs)} episodes, {time.time() - start:.0f}s", flush=True)
    print(f"{base.experiment['id']} {args.split}: {len(chosen)} instances, grid {grid}")
    summaries = {}
    for spec in args.policies:
        summaries[spec] = summarize(results[spec])
        print_summary(spec, summaries[spec])
    if args.output is not None:
        args.output.mkdir(parents=True)
        record = {
            "protocol_id": splits_v2.PROTOCOL_ID,
            "formal": args.formal,
            "provenance": provenance,
            "config": args.config,
            "split": args.split,
            "screen": args.screen,
            "instances": chosen,
            "grid": grid,
            "policies": args.policies,
            "summaries": summaries,
        }
        (args.output / "summary.json").write_text(json.dumps(record, indent=2))
        episodes = {spec: {str(n): results[spec][n]["-"] for n in grid} for spec in args.policies}
        (args.output / "episodes.json").write_text(json.dumps(episodes))


if __name__ == "__main__":
    main()
