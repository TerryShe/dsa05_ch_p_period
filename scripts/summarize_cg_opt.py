"""Archive the optimized static configured grant CG-opt (design configured-grant-optimized_20261010.md §7).

Inputs: formal ``evaluate_capacity`` outputs of ``configured_grant_opt`` on the
validation split (full grid) and on exp010-eval512 (N = 6, 8, 10); the formal
CG outputs of RES-DSA05-CG-BASELINE (``configured_grant``, same sets); the
stored validation references (N_EDF and the calibrated local rules) and the
EXP-010 evaluations of A_mix, M, best_local and N_EDF, as for CG.

Writes into --output:

* validation.json: F_eps(N) and N* of CG-opt next to CG, N_EDF, every
  calibrated rule family and best_local; CG-opt - CG paired by instance;
* eval512.json: F_eps at N = 6, 8, 10 for CG-opt, CG, A_mix, M, best_local and
  N_EDF; paired differences CG-opt - other (deployments paired, each group's
  training seeds resampled on their own, 10,000 draws, seed 20260930); the
  pre-registered reading at N = 8 (§7.3) and the main-table rule for N_CG;
* diagnostics.json: per set and N, the CG diagnostics of both allocations
  (prediction on the bank each optimized on), CG-opt's prediction on the CG
  bank B against the realized violation rate, bank A against bank B (§5 layer
  1) and the solver record (§4.4: statuses, nodes, gaps, time-limit hits,
  search evaluations and moves).
"""

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from dsa_period.capacity import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, EPSILONS, eps_key
from dsa_period.config import load_scenario
from dsa_period.evaluation import _bootstrap, _interval

ROOT = Path(__file__).parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cgb = _script("summarize_cg_baseline")
exp010 = cgb.exp010

SPECS = {"CG_opt": "configured_grant_opt", "CG": "configured_grant"}
OTHERS = ("CG", "A_mix", "M", "best_local", "edf")
TARGET = 0.01


def _feasible(rows: list[dict], key: str, instances: list[int]) -> np.ndarray:
    cells = {e["instance"]: float(e["feasible"][key]) for e in rows}
    return np.array([[cells[i] for i in instances]])


def validation_table(record: dict, cg_record: dict, episodes: dict, cg_episodes: dict, draws: int):
    if record["instances"] != cg_record["instances"] or record["grid"] != cg_record["grid"]:
        raise ValueError("CG-opt and CG were evaluated on different validation deployments")
    table = cgb.validation_table(cg_record)
    del table["rows"]["CG_single"]
    summary = record["summaries"][SPECS["CG_opt"]]
    table["rows"] = {
        "CG_opt": {
            key: {
                "F": {str(r["num_ues"]): r["feasible_share"] for r in block["rows"]},
                "ci95": {str(r["num_ues"]): r["ci95"] for r in block["rows"]},
                "capacity": block["capacity"],
            }
            for key, block in summary["per_eps"].items()
        }
    } | table["rows"]
    instances = record["instances"]
    opt, cg = episodes[SPECS["CG_opt"]], cg_episodes[SPECS["CG"]]
    table["CG_opt_minus_CG"] = {
        key: {
            n: {
                "difference": float(
                    _feasible(opt[n], key, instances).mean()
                    - _feasible(cg[n], key, instances).mean()
                ),
                "ci95": _interval(
                    _bootstrap(
                        _feasible(opt[n], key, instances),
                        _feasible(cg[n], key, instances),
                        draws,
                        BOOTSTRAP_SEED,
                    )
                ),
            }
            for n in opt
        }
        for key in map(eps_key, EPSILONS)
    }
    return table


def eval512_table(paths: list[Path], formal: bool, draws: int) -> dict:
    """CG-opt against CG and the EXP-010 groups on exp010-eval512, paired by deployment."""
    episodes, instances = exp010.load(paths, formal)
    methods = json.loads(cgb.EXP010_METHODS.read_text())
    groups = {name: [spec] for name, spec in SPECS.items()}
    groups.update({g: list(methods[g]) for g in ("A_mix", "M")})
    groups.update({"best_local": [methods["best_local"]], "edf": [methods["edf"]]})
    out: dict = {"instances": len(instances), "F": {}, "differences": {}}
    mats = {}
    for eps in EPSILONS:
        key = eps_key(eps)
        mats[key] = {
            g: {n: exp010.matrix(episodes, m, n, key, instances) for n in exp010.GRID}
            for g, m in groups.items()
        }
        out["F"][key] = {
            g: {
                str(n): {
                    "F": float(mats[key][g][n].mean()),
                    "ci95": _interval(_bootstrap(mats[key][g][n], None, draws, BOOTSTRAP_SEED)),
                }
                for n in exp010.GRID
            }
            for g in groups
        }
    m01 = mats[eps_key(0.01)]
    out["differences"] = {
        other: {
            str(n): {
                "difference": float(m01["CG_opt"][n].mean() - m01[other][n].mean()),
                "ci95": _interval(
                    _bootstrap(m01["CG_opt"][n], m01[other][n], draws, BOOTSTRAP_SEED)
                ),
            }
            for n in exp010.GRID
        }
        for other in OTHERS
    }
    main = out["differences"]["A_mix"]["8"]
    lower, upper = main["ci95"]
    if lower > 0:
        reading = "CG-opt above the learned policies"
    elif upper < 0:
        reading = "CG-opt below the learned policies"
    else:
        reading = "indistinguishable"
    gain = out["differences"]["CG"]["8"]
    out["reading_n8"] = {
        "F8_CG_opt": out["F"][eps_key(0.01)]["CG_opt"]["8"]["F"],
        "CG_opt_minus_A_mix": main,
        "reading": reading,
        "CG_opt_minus_CG": gain,
        "main_table_N_CG": "CG-opt and CG" if gain["ci95"][1] < 0 else "CG-opt",
    }
    return out


def bank_check(rows: list[dict]) -> dict:
    """Prediction on the optimization bank A against the CG bank B (§5 layer 1)."""
    a = np.concatenate([e["policy_diagnostics"]["predicted_outage"] for e in rows])
    b = np.concatenate([e["policy_diagnostics"]["predicted_outage_bank_b"] for e in rows])
    labels = ("<0.001", "0.001-0.005", "0.005-0.01", "0.01-0.02", "0.02-0.05")
    labels += ("0.05-0.1", ">=0.1")
    bins = np.searchsorted(cgb.OUTAGE_BINS, a, side="right")
    return {
        "ues": int(a.size),
        "b_minus_a": {
            "mean": float(np.mean(b - a)),
            "mean_abs": float(np.mean(np.abs(b - a))),
            "max": float(np.max(b - a)),
            "min": float(np.min(b - a)),
        },
        "by_outage_a": {
            labels[k]: {
                "ues": int(np.sum(bins == k)),
                "outage_a_mean": float(a[bins == k].mean()),
                "outage_b_mean": float(b[bins == k].mean()),
            }
            for k in sorted(set(bins.tolist()))
        },
        "within_0.01_on_a_not_on_b": int(np.sum((a <= TARGET) & (b > TARGET))),
        "within_0.01_on_b_not_on_a": int(np.sum((b <= TARGET) & (a > TARGET))),
    }


def realized_on_b(rows: list[dict], periods: list[int]) -> dict:
    """The CG predicted-vs-realized table with CG-opt's prediction on bank B."""
    swapped = [
        e
        | {
            "policy_diagnostics": e["policy_diagnostics"]
            | {"predicted_outage": e["policy_diagnostics"]["predicted_outage_bank_b"]}
        }
        for e in rows
    ]
    return cgb.diagnostics(swapped, periods)["predicted_vs_realized"]


def solver(rows: list[dict]) -> dict:
    records = [(e["instance"], e["policy_diagnostics"]["solver"]) for e in rows]
    solves = [(i, k, s) for i, r in records for k, s in enumerate(r["stage1"])]
    search = [r["stage2"] for _, r in records]
    return {
        "budget": records[0][1]["budget"],
        "start": dict(Counter(r["start"] for _, r in records)),
        "stage1_status": dict(Counter(s["status"] for _, _, s in solves)),
        "stage1_without_solution": sum(len(r["stage1"]) == 1 for _, r in records),
        "nodes_max": max(s["nodes"] or 0 for _, _, s in solves),
        "gap_max": max(s["gap"] or 0.0 for _, _, s in solves),
        "killed": [[i, k] for i, k, s in solves if s["status"] == "Killed at the time limit"],
        # design §11.8: a crashed solve is retried once without restarts
        "crashed_and_retried": [
            [i, k, s["status"]] for i, k, s in solves if "crashed_attempt" in s
        ],
        "solve_seconds": {
            "mean": float(np.mean([s["seconds"] for _, _, s in solves])),
            "max": float(np.max([s["seconds"] for _, _, s in solves])),
        },
        "time_limit_hits": [[i, k] for i, k, s in solves if s["time_limit_hit"]],
        "node_limit_hits": [[i, k] for i, k, s in solves if s["node_limit_hit"]],
        "evaluations": {
            "mean": float(np.mean([s["evaluations"] for s in search])),
            "max": int(max(s["evaluations"] for s in search)),
        },
        "accepted": {
            move: sum(s["accepted"][move] for s in search) for move in search[0]["accepted"]
        },
        "not_at_local_optimum": [i for i, r in records if not r["stage2"]["local_optimum"]],
        "configure_seconds": {
            "mean": float(np.mean([r["seconds"] for _, r in records])),
            "max": float(np.max([r["seconds"] for _, r in records])),
        },
    }


def diagnostics(rows: list[dict], cg_rows: list[dict], periods: list[int]) -> dict:
    if sorted(e["instance"] for e in rows) != sorted(e["instance"] for e in cg_rows):
        raise ValueError("CG-opt and CG diagnostics are not paired")
    return {
        "CG_opt": cgb.diagnostics(rows, periods),
        "CG": cgb.diagnostics(cg_rows, periods),
        "CG_opt_predicted_on_b_vs_realized": realized_on_b(rows, periods),
        "bank_a_vs_b": bank_check(rows),
        "solver": solver(rows),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--eval512", type=Path, required=True)
    parser.add_argument("--cg-validation", type=Path, required=True)
    parser.add_argument("--cg-eval512", type=Path, required=True)
    parser.add_argument(
        "--exp010", type=Path, nargs="+", required=True, help="EXP-010 evaluation directories"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--allow-informal", action="store_true", help="smoke tests only")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    if args.resamples != BOOTSTRAP_RESAMPLES and not args.allow_informal:
        parser.error("the formal analysis uses the protocol's bootstrap size")
    formal = not args.allow_informal
    record, episodes = cgb.read(args.validation, "validation", None, formal)
    cg_record, cg_episodes = cgb.read(args.cg_validation, "validation", None, True)
    periods = [c.period_slots for c in load_scenario(ROOT / record["config"], 4).traffic_classes]
    validation = validation_table(record, cg_record, episodes, cg_episodes, args.resamples)
    evaluation = eval512_table(
        [args.eval512, args.cg_eval512, *args.exp010], formal, args.resamples
    )
    _, eval_episodes = cgb.read(args.eval512, "dev", exp010.SET, formal)
    _, cg_eval_episodes = cgb.read(args.cg_eval512, "dev", exp010.SET, True)
    opt, cg = SPECS["CG_opt"], SPECS["CG"]
    diag = {
        "validation": {
            n: diagnostics(rows, cg_episodes[cg][n], periods) for n, rows in episodes[opt].items()
        },
        "exp010-eval512": {
            n: diagnostics(rows, cg_eval_episodes[cg][n], periods)
            for n, rows in eval_episodes[opt].items()
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "validation.json").write_text(json.dumps(validation, indent=1))
    (args.output / "eval512.json").write_text(json.dumps(evaluation, indent=1))
    (args.output / "diagnostics.json").write_text(json.dumps(diag, indent=1))
    for key in map(eps_key, EPSILONS):
        print(f"validation eps={key}")
        for name, row in validation["rows"].items():
            shares = " ".join(f"{n}:{f:.2f}" for n, f in row[key]["F"].items())
            star = row[key].get("capacity", {}).get("n_star", "-")
            print(f"  {name:<18} N*={star!s:<3} {shares}")
    for group, by_n in evaluation["F"][eps_key(0.01)].items():
        print(f"eval512 {group:<11} " + " ".join(f"F({n}) {v['F']:.3f}" for n, v in by_n.items()))
    r = evaluation["reading_n8"]
    print(f"N=8: CG-opt - A_mix {r['CG_opt_minus_A_mix']} -> {r['reading']}")
    print(f"N=8: CG-opt - CG {r['CG_opt_minus_CG']} -> main table N_CG: {r['main_table_N_CG']}")
    for name, by_n in diag.items():
        for n, d in by_n.items():
            s = d["solver"]
            print(
                f"{name} N={n}: start {s['start']} time-limit {len(s['time_limit_hits'])} "
                f"crashed {s['crashed_and_retried']} "
                f"node-limit {len(s['node_limit_hits'])} not-local-opt "
                f"{len(s['not_at_local_optimum'])} B-A {d['bank_a_vs_b']['b_minus_a']['mean']:+.5f}"
            )


if __name__ == "__main__":
    main()
