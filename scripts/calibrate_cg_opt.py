"""Budget calibration of CG-opt (design configured-grant-optimized_20261010.md §4.4).

Dev instances 0-63 at N in {6, 8, 10, 16, 24, 32}, allocation only: no
episodes and no F. Every deployment is optimized without a node or evaluation
limit (time safety T only); the rules of §4.4 then fix

* M: the smallest power of 10 not below the 99th percentile of the node counts
  of the stage-1 solves (both solves of every deployment pooled);
* E: the smallest power of 10, at least 10^4, within which at least 99% of the
  searches reach their local optimum.

Also kept per deployment: the solve records, the predicted outages on bank A
of the greedy, stage 1 and the final allocation, and on the CG bank B those of
CG-opt and of the CG greedy (its own prediction), for the out-of-sample check
of §5 layer 1.

    python scripts/calibrate_cg_opt.py --workers 16 \
        --output docs/results/cg-opt-calibration_20261010/calibration.json
"""

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

from dsa_period import cg_opt
from dsa_period.config import load_scenario
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.rules_v2 import CG_BANK_SAMPLES, CG_BANK_SEED, allocate_grants, fading_bank

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "scenario-v2.toml"
INSTANCES = range(64)
GRID = (6, 8, 10, 16, 24, 32)
QUANTILE = 0.99
MIN_EVALUATIONS = 10**4
TARGET = 0.01


@cache
def _scenario(n: int):
    return load_scenario(CONFIG, n)


def run(n: int, instance: int) -> dict:
    scenario = _scenario(n)
    physical, fading = scenario.physical, scenario.fading
    topology, traffic = scenario.instance(instance)
    snr = (
        scenario.menu.power_w[-1] * np.asarray(topology.ue_ap_gain_linear) / physical.noise_power_w
    )
    bits = np.asarray(traffic.packet_bits, dtype=np.float64)
    period = np.asarray(traffic.period_slots, dtype=np.int64)
    phase = np.asarray(traffic.phase_slots, dtype=np.int64)
    a, b = (
        fading_bank(
            physical.num_subchannels, fading.doppler_hz, fading.sinusoids,
            physical.slot_duration_s, int(period.max()), samples, seed,
        )
        for samples, seed in (
            (cg_opt.CG_OPT_BANK_SAMPLES, cg_opt.CG_OPT_BANK_SEED),
            (CG_BANK_SAMPLES, CG_BANK_SEED),
        )
    )  # fmt: skip
    result = cg_opt.optimize_grants(
        physical, snr, bits, period, phase, a, max_nodes=None, max_evaluations=None
    )
    on_b = cg_opt.deployment(physical, snr, bits, period, phase, b)
    greedy_b = allocate_grants(physical, snr, bits, period, phase, b)
    return {
        "num_ues": n,
        "instance": instance,
        "record": result.record,
        "outage_a": list(result.allocation.outage),
        "outage_b": cg_opt.predicted_outage(on_b, result.allocation.occasions),
        "cg_outage_b": list(greedy_b.outage),
        "occasions": [len(o) for o in result.allocation.occasions],
        "cg_occasions": [len(o) for o in greedy_b.occasions],
        "assigned_share": result.allocation.assigned_share,
        "blocked": result.allocation.blocked,
        "cg_blocked": greedy_b.blocked,
    }


def power_of_ten_at_least(x: float) -> int:
    return 10 ** max(0, math.ceil(math.log10(x))) if x > 1 else 1


def budget(rows: list[dict]) -> dict:
    solves = [s for r in rows for s in r["record"]["stage1"]]
    nodes = [s["nodes"] for s in solves if s["nodes"] is not None]  # None: killed at T
    evaluations = np.array([r["record"]["stage2"]["evaluations"] for r in rows])
    m = power_of_ten_at_least(float(np.quantile(nodes, QUANTILE)))
    e = MIN_EVALUATIONS
    while np.mean(evaluations <= e) < QUANTILE:
        e *= 10
    return {
        "max_nodes": m,
        "max_evaluations": e,
        "nodes_quantile_0.99": float(np.quantile(nodes, QUANTILE)),
        "nodes_max": int(max(nodes)),
        "solves": len(solves),
        "solves_killed": len(solves) - len(nodes),
        "solves_above_max_nodes": int(sum(x > m for x in nodes)),
        "time_limit_hits": int(
            sum(s["time_limit_hit"] for r in rows for s in r["record"]["stage1"])
        ),
        "evaluations_quantile_0.99": float(np.quantile(evaluations, QUANTILE)),
        "evaluations_max": int(evaluations.max()),
        "searches_within_max_evaluations": float(np.mean(evaluations <= e)),
        "searches_not_at_local_optimum": int(
            sum(not r["record"]["stage2"]["local_optimum"] for r in rows)
        ),
    }


def per_n(rows: list[dict]) -> dict:
    out = {}
    for n in GRID:
        group = [r for r in rows if r["num_ues"] == n]
        misses = lambda name: [  # noqa: E731
            max(r["record"]["misses"][name]) / cg_opt.CG_OPT_BANK_SAMPLES
            for r in group
            if name in r["record"]["misses"]
        ]
        gap = np.concatenate([np.subtract(r["outage_b"], r["outage_a"]) for r in group])
        seconds = [r["record"]["seconds"] for r in group]
        out[n] = {
            "deployments": len(group),
            "start_milp": sum(r["record"]["start"] == "milp" for r in group),
            "mean_max_outage_a": {
                name: float(np.mean(misses(name))) for name in ("greedy", "milp", "final")
            },
            "mean_max_outage_b": {
                "CG": float(np.mean([max(r["cg_outage_b"]) for r in group])),
                "CG_opt": float(np.mean([max(r["outage_b"]) for r in group])),
            },
            "predicted_all_within_0.01_b": {
                "CG": sum(max(r["cg_outage_b"]) <= TARGET for r in group),
                "CG_opt": sum(max(r["outage_b"]) <= TARGET for r in group),
            },
            "ues_above_0.01_b": {
                "CG": sum(sum(q > TARGET for q in r["cg_outage_b"]) for r in group),
                "CG_opt": sum(sum(q > TARGET for q in r["outage_b"]) for r in group),
            },
            "outage_b_minus_a": {
                "mean": float(gap.mean()),
                "max": float(gap.max()),
                "min": float(gap.min()),
            },
            "occasions": {
                "CG": sum(sum(r["cg_occasions"]) for r in group),
                "CG_opt": sum(sum(r["occasions"]) for r in group),
            },
            "blocked": {
                "CG": sum(r["cg_blocked"] for r in group),
                "CG_opt": sum(r["blocked"] for r in group),
            },
            "seconds": {"mean": float(np.mean(seconds)), "max": float(np.max(seconds))},
            "nodes_max": max(s["nodes"] or 0 for r in group for s in r["record"]["stage1"]),
            "evaluations_max": max(r["record"]["stage2"]["evaluations"] for r in group),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    provenance = git_provenance()
    began = time.perf_counter()
    jobs = [(n, i) for n in GRID for i in INSTANCES]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, *zip(*jobs)))
    result = {
        "design": "docs/experiment-designs/configured-grant-optimized_20261010.md §4.4",
        "provenance": provenance,
        "instances": [INSTANCES.start, INSTANCES.stop - 1],
        "grid": list(GRID),
        "time_limit_s": cg_opt.TIME_LIMIT_S,
        "wall_seconds": time.perf_counter() - began,
        "budget": budget(rows),
        "per_n": per_n(rows),
        "deployments": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["budget"], indent=1))
    for n, row in result["per_n"].items():
        print(
            f"N={n:2d} start milp {row['start_milp']:2d}/64 | max q^A greedy "
            f"{row['mean_max_outage_a']['greedy']:.4f} final {row['mean_max_outage_a']['final']:.4f}"
            f" | all<=0.01 on B: CG {row['predicted_all_within_0.01_b']['CG']:2d} "
            f"opt {row['predicted_all_within_0.01_b']['CG_opt']:2d} | B-A mean "
            f"{row['outage_b_minus_a']['mean']:+.5f} | {row['seconds']['mean']:.1f}s "
            f"(max {row['seconds']['max']:.1f}) nodes<= {row['nodes_max']} evals<= "
            f"{row['evaluations_max']}"
        )


if __name__ == "__main__":
    main()
