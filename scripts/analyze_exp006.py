"""EXP-DSA05-006 phase-2 analysis on the validation split (descriptive; H1/H2 are test-only).

Pools the selected checkpoints of the three seeds (F averaged over seeds per
instance, two-level bootstrap over seeds and instances), compares them per N
with the best calibrated local rule (``best_local``, whose validation episodes
come from the calibration runs) by paired differences, and lists N_EDF. Also
reports per-UE failure shapes by class and distance.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dsa_period.capacity import EPSILONS, capacity, capacity_label, eps_key, summarize
from dsa_period.evaluation import _bootstrap, _interval
from dsa_period.rules_v2 import best_local_spec

CLASSES = ("fast", "medium", "slow")


def matrix(episodes_by_seed: dict, key: str, instances: list[int]) -> np.ndarray:
    rows = []
    for seed in sorted(episodes_by_seed):
        cells = {e["instance"]: float(e["feasible"][key]) for e in episodes_by_seed[seed]}
        rows.append([cells[i] for i in instances])
    return np.array(rows)


def failure_shape(episodes: list[dict]) -> dict:
    rates = np.concatenate([e["violation_rate"] for e in episodes])
    cls = np.concatenate([e["class_index"] for e in episodes])
    dist = np.concatenate([e["distance_m"] for e in episodes])
    bad = rates > 0.01
    return {
        "ues_over_1pct": float(bad.mean()),
        "by_class": {CLASSES[c]: float(bad[cls == c].mean()) for c in range(3) if (cls == c).any()},
        "far_ge_80m": float(bad[dist >= 80].mean()),
        "near_lt_50m": float(bad[dist < 50].mean()),
        "occupancy": float(np.mean([np.sum(e["duty"]) / 4 for e in episodes])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marl", type=Path, required=True, help="full-grid validation output")
    parser.add_argument("--rules", type=Path, nargs="+", required=True, help="calibration runs")
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    marl_eps = json.loads((args.marl / "episodes.json").read_text())
    rules_eps = {}
    for path in args.rules:
        rules_eps.update(json.loads((path / "episodes.json").read_text()))
    edf = json.loads((args.edf / "summary.json").read_text())["summaries"]["central_edf"]
    specs = sorted(marl_eps)
    grid = sorted(int(n) for n in marl_eps[specs[0]])
    pooled = summarize({n: {spec: marl_eps[spec][str(n)] for spec in specs} for n in grid})
    local = {n: best_local_spec(n) for n in grid}
    local_summary = summarize({n: {"-": rules_eps[local[n]][str(n)]} for n in grid})
    result = {"grid": grid, "marl_specs": specs, "best_local": local, "per_eps": {}}
    for eps in EPSILONS:
        key = eps_key(eps)
        rows = []
        for n in grid:
            instances = sorted(e["instance"] for e in marl_eps[specs[0]][str(n)])
            m = matrix({s: marl_eps[s][str(n)] for s in specs}, key, instances)
            lo = matrix({"-": rules_eps[local[n]][str(n)]}, key, instances)
            diff = _bootstrap(m, lo, 10_000, 20_260_930)
            marl_row = next(r for r in pooled["per_eps"][key]["rows"] if r["num_ues"] == n)
            local_row = next(r for r in local_summary["per_eps"][key]["rows"] if r["num_ues"] == n)
            edf_row = next((r for r in edf["per_eps"][key]["rows"] if r["num_ues"] == n), None)
            rows.append(
                {
                    "num_ues": n,
                    "marl": marl_row["feasible_share"],
                    "marl_ci95": marl_row["ci95"],
                    "marl_per_seed": m.mean(axis=1).tolist(),
                    "best_local": local_row["feasible_share"],
                    "difference": float(m.mean() - lo.mean()),
                    "difference_ci95": _interval(diff),
                    "edf": edf_row["feasible_share"] if edf_row else None,
                }
            )
        marl_point = pooled["per_eps"][key]["capacity"]
        local_point = capacity(grid, [r["best_local"] for r in rows])
        result["per_eps"][key] = {
            "rows": rows,
            "marl_capacity": marl_point,
            "marl_capacity_bootstrap": pooled["per_eps"][key]["capacity_bootstrap"],
            "best_local_capacity": local_point,
            "edf_capacity": edf["per_eps"][key]["capacity"],
        }
        print(
            f"eps={eps}: N* MARL {capacity_label(marl_point, grid)} "
            f"(bootstrap {pooled['per_eps'][key]['capacity_bootstrap']}), "
            f"best local {capacity_label(local_point, grid)}, "
            f"EDF {capacity_label(edf['per_eps'][key]['capacity'], grid)}"
        )
        for r in rows:
            print(
                f"  N={r['num_ues']:<3} MARL {r['marl']:.3f} [{r['marl_ci95'][0]:.2f},{r['marl_ci95'][1]:.2f}] "
                f"seeds {[round(x, 2) for x in r['marl_per_seed']]} local {r['best_local']:.3f} "
                f"diff {r['difference']:+.3f} [{r['difference_ci95'][0]:+.3f},{r['difference_ci95'][1]:+.3f}] "
                f"EDF {r['edf']}"
            )
    result["failure_shape"] = {
        str(n): {
            "marl": failure_shape([e for s in specs for e in marl_eps[s][str(n)]]),
            "best_local": failure_shape(rules_eps[local[n]][str(n)]),
        }
        for n in grid
        if n in (6, 8, 10, 12, 16)
    }
    for n, row in result["failure_shape"].items():
        print(f"  shape N={n}: MARL {row['marl']}\n           local {row['best_local']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
