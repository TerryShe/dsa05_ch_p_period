"""Run policies on the near-UE stress set of EXP-DSA05-011B (design §2.3, §3.1, §3.3).

Every (spec, N, deployment) of ``exp011b-stress`` (``starvation.stress_instances``
at N = 8, 12, 16) runs one protocol episode with the seeds of
``evaluate_capacity`` (dev, repeat 0). The output directory holds summary.json
(provenance, the instance lists, and per spec and N the starvation cells:
a UE with mean SNR >= 20 dB and violation rate >= 0.5) and episodes.json (the
per-UE records). The stress set is read only with ``--formal`` on a clean
committed tree; ``--instances FIRST COUNT`` runs informal smoke tests on other,
unregistered dev instances without the geometric filter.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dsa_period import splits_v2, starvation
from dsa_period.capacity import run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.rules_v2 import make_policy_v2

CONFIG = starvation.CONFIG
KEEP = ("instance", "packets", "violations", "violation_rate", "worst_violation_rate",
        "distance_m", "shadowing_db", "class_index")  # fmt: skip


def _episode(job: tuple) -> tuple:
    spec, n, instance = job
    if "marl:" in spec:  # wrapped specs too (4c8cc4f)
        import torch

        torch.set_num_threads(1)
    scenario = load_scenario(CONFIG, n)
    policy = make_policy_v2(spec, scenario.physical, scenario.menu)
    record = run_capacity_episode(scenario, policy, "dev", instance)
    row = {key: record[key] for key in KEEP}
    row["snr_db"] = starvation.mean_snr_db(scenario, instance).tolist()
    return spec, n, row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", nargs="+", required=True)
    parser.add_argument("--grid", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--instances", type=int, nargs=2, metavar=("FIRST", "COUNT"),
                        help="informal smoke runs on unregistered dev instances")  # fmt: skip
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    provenance = git_provenance()
    registered = splits_v2.SCREENING_SETS[starvation.STRESS_SET]["grid"]
    if not set(args.grid) <= set(registered):
        parser.error(f"{starvation.STRESS_SET} is registered at N = {list(registered)}")
    if args.instances is None:
        if not args.formal or provenance["dirty"]:
            parser.error("the stress set is read only with --formal on a clean committed tree")
        chosen = {
            n: starvation.stress_instances(load_scenario(CONFIG, n)) for n in sorted(args.grid)
        }
    else:
        if args.formal:
            parser.error("--instances is for informal smoke runs")
        first, count = args.instances
        plain = [splits_v2.require("dev", i) for i in range(first, first + count)]
        try:
            splits_v2.check_dev_range(plain)
        except ValueError as exc:
            parser.error(str(exc))
        chosen = {n: plain for n in sorted(args.grid)}
    for spec in args.specs:
        make_policy_v2(spec, load_scenario(CONFIG, 4).physical, load_scenario(CONFIG, 4).menu)
    jobs = [(spec, n, i) for spec in args.specs for n in chosen for i in chosen[n]]
    episodes: dict = {spec: {str(n): [] for n in chosen} for spec in args.specs}
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for done, (spec, n, row) in enumerate(pool.map(_episode, jobs, chunksize=1), 1):
            episodes[spec][str(n)].append(row)
            if done % max(1, len(jobs) // 10) == 0:
                print(f"{done}/{len(jobs)} episodes, {time.time() - start:.0f}s", flush=True)
    summaries = {
        spec: {
            n: {"deployments": len(rows), "cells": starvation.cells(rows, int(n))}
            for n, rows in by_n.items()
        }
        for spec, by_n in episodes.items()
    }
    args.output.mkdir(parents=True)
    record = {
        "protocol_id": splits_v2.PROTOCOL_ID,
        "set": starvation.STRESS_SET if args.instances is None else "smoke",
        "formal": args.formal,
        "provenance": provenance,
        "definition": {
            "strong_snr_db": starvation.STRONG_SNR_DB,
            "starved_rate": starvation.STARVED_RATE,
        },  # fmt: skip
        "instances": {str(n): v for n, v in chosen.items()},
        "summaries": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(record, indent=2))
    (args.output / "episodes.json").write_text(json.dumps(episodes))
    for spec, by_n in summaries.items():
        counts = {n: len(v["cells"]) for n, v in by_n.items()}
        print(f"{spec}: starvation cells {counts}")


if __name__ == "__main__":
    main()
