"""Archive the configured-grant reference N_CG (design configured-grant-baseline_20261009.md §5).

Inputs: formal ``evaluate_capacity`` outputs of ``configured_grant`` and
``configured_grant:single_channel=1`` on the validation split (full grid) and
on exp010-eval512 (N = 6, 8, 10); the stored validation references
(docs/results/v2-calibration_20261001: N_EDF and the per-N calibrated local
rules); the EXP-010 evaluations of A_mix, M, best_local and N_EDF on
exp010-eval512 with their methods file.

Writes into --output:

* validation.json: F_eps(N) and N* of both CG variants next to N_EDF, every
  calibrated rule family and best_local (validation F as calibrated);
* eval512.json: F_eps at N = 6, 8, 10 for CG, CG-single, A_mix, M, best_local
  and N_EDF; paired differences CG - other (deployments paired, each group's
  training seeds resampled on their own, 10,000 draws, seed 20260930, as in
  EXP-010) and the pre-registered reading at N = 8 (design §5.3);
* diagnostics.json: per variant, set and N, the CG allocation (occasions per
  UE by class and mean-SNR bin, assigned and used share of the occasions,
  placements blocked by fragmentation), predicted outage against the realized
  violation rate, and the failing UEs.
"""

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from dsa_period import splits_v2
from dsa_period.capacity import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EPSILONS,
    allowance,
    capacity,
    eps_key,
)
from dsa_period.config import load_scenario
from dsa_period.evaluation import _bootstrap, _interval
from dsa_period.rules_v2 import CALIBRATION_TABLE

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "analyze_exp010", ROOT / "scripts" / "analyze_exp010.py"
)
exp010 = importlib.util.module_from_spec(_spec)
sys.modules["analyze_exp010"] = exp010
_spec.loader.exec_module(exp010)

CG = {"CG": "configured_grant", "CG_single": "configured_grant:single_channel=1"}
CALIBRATION = ROOT / "docs" / "results" / "v2-calibration_20261001"
EXP010_METHODS = ROOT / "docs" / "results" / "exp010_20261008" / "methods-exp010.json"
OTHERS = ("A_mix", "M", "best_local", "edf")
THRESHOLD = 0.95
SNR_BINS_DB = (10.0, 20.0, 30.0)
OUTAGE_BINS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1)


def read(path: Path, split: str, screen: str | None, formal: bool) -> tuple[dict, dict]:
    record = json.loads((path / "summary.json").read_text())
    ok = (
        record["protocol_id"] == splits_v2.PROTOCOL_ID
        and record["split"] == split
        and record.get("screen") == screen
        and (record["formal"] or not formal)
    )
    if not ok:
        raise ValueError(f"{path}: not a formal {screen or split} evaluation")
    return record, json.loads((path / "episodes.json").read_text())


def validation_table(record: dict) -> dict:
    """F and N* on the validation split: both CG variants and the stored references."""
    edf = json.loads((CALIBRATION / "edf-summary.json").read_text())
    calibration = json.loads((CALIBRATION / "calibration.json").read_text())
    if edf["instances"] != record["instances"] or calibration["instances"] != record["instances"]:
        raise ValueError("the references were evaluated on other validation instances")
    rows: dict = {}
    for name, spec in {**CG, "N_EDF": None}.items():
        summary = record["summaries"][spec] if spec else edf["summaries"]["central_edf"]
        rows[name] = {
            key: {
                "F": {str(r["num_ues"]): r["feasible_share"] for r in block["rows"]},
                "ci95": {str(r["num_ues"]): r["ci95"] for r in block["rows"]},
                "capacity": block["capacity"],
            }
            for key, block in summary["per_eps"].items()
        }
    families = calibration["families"]
    for family, entry in families.items():
        rows[family] = {
            key: {
                "F": {n: row["feasible_share"][key] for n, row in entry["per_n"].items()},
                "capacity": entry["capacity"][key],
            }
            for key in map(eps_key, EPSILONS)
        }
    best = {}
    order = list(CALIBRATION_TABLE)
    for n in calibration["grid"]:

        def rank(family: str, n: int = n) -> tuple:
            row = families[family]["per_n"][str(n)]
            share = row["feasible_share"]
            return (-share["0.01"], -share["0.05"], row["worst_ue_violation_rate_mean"])

        best[str(n)] = min(order, key=lambda f: (*rank(f), order.index(f)))
    rows["best_local"] = {}
    for key in map(eps_key, EPSILONS):
        shares = {n: rows[f][key]["F"][n] for n, f in best.items()}
        rows["best_local"][key] = {
            "F": shares,
            "family": best,
            "capacity": capacity([int(n) for n in shares], list(shares.values())),
        }
    return {"instances": len(record["instances"]), "grid": record["grid"], "rows": rows}


def eval512_table(paths: list[Path], formal: bool, draws: int) -> dict:
    """CG against the EXP-010 groups on exp010-eval512, paired by deployment."""
    episodes, instances = exp010.load(paths, formal)
    methods = json.loads(EXP010_METHODS.read_text())
    groups = {name: [spec] for name, spec in CG.items()}
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
    for name in CG:
        out["differences"][name] = {
            other: {
                str(n): {
                    "difference": float(m01[name][n].mean() - m01[other][n].mean()),
                    "ci95": _interval(
                        _bootstrap(m01[name][n], m01[other][n], draws, BOOTSTRAP_SEED)
                    ),
                }
                for n in exp010.GRID
            }
            for other in (*OTHERS, *(c for c in CG if c != name))
        }
    main = out["differences"]["CG"]["A_mix"]["8"]
    f8 = out["F"][eps_key(0.01)]["CG"]["8"]["F"]
    lower, upper = main["ci95"]
    if f8 >= THRESHOLD and lower > 0:
        reading = "CG crosses the threshold"
    elif upper < 0:
        reading = "CG below the learned policies"
    else:
        reading = "indistinguishable"
    out["reading_n8"] = {"F8_CG": f8, "CG_minus_A_mix": main, "reading": reading}
    return out


def _bin(value: float, edges: tuple) -> int:
    return int(np.searchsorted(edges, value, side="right"))


def diagnostics(episodes: list[dict], periods: list[int]) -> dict:
    """Allocation, prediction and failures of one CG variant at one N.

    ``periods[c]`` is the period of traffic class c.
    """
    by_class = defaultdict(lambda: {"ues": 0, "occasions": 0.0, "window_share": 0.0})
    by_snr = defaultdict(lambda: {"ues": 0, "window_share": 0.0, "outage": 0.0, "v": 0, "p": 0})
    by_outage = defaultdict(lambda: {"ues": 0, "outage": 0.0, "v": 0, "p": 0})
    failing = {"ues": 0, "classes": defaultdict(int), "mean_snr_db": [], "window_share": []}
    failing["predicted_outage"] = []
    offered = sent = blocked_total = blocked_episodes = 0
    assigned = []
    for e in episodes:
        d = e["policy_diagnostics"]
        assigned.append(d["assigned_share"])
        offered += sum(d["offered"])
        sent += sum(d["sent"])
        blocked_total += d["blocked"]
        blocked_episodes += d["blocked"] > 0
        for ue, cls in enumerate(e["class_index"]):
            period = periods[cls]
            share = d["occasions"][ue] / period
            q, snr = d["predicted_outage"][ue], d["mean_snr_db"][ue]
            v, p = e["violations"][ue], e["packets"][ue]
            c = by_class[cls]
            c["ues"] += 1
            c["occasions"] += d["occasions"][ue]
            c["window_share"] += share
            s = by_snr[_bin(snr, SNR_BINS_DB)]
            s["ues"] += 1
            s["window_share"] += share
            s["outage"] += q
            s["v"] += v
            s["p"] += p
            o = by_outage[_bin(q, OUTAGE_BINS)]
            o["ues"] += 1
            o["outage"] += q
            o["v"] += v
            o["p"] += p
            if v > allowance(0.01, p):
                failing["ues"] += 1
                failing["classes"][str(cls)] += 1
                failing["mean_snr_db"].append(snr)
                failing["window_share"].append(share)
                failing["predicted_outage"].append(q)
    snr_labels = ("<10", "10-20", "20-30", ">=30")
    outage_labels = ("<0.001", "0.001-0.005", "0.005-0.01", "0.01-0.02", "0.02-0.05")
    outage_labels += ("0.05-0.1", ">=0.1")
    return {
        "episodes": len(episodes),
        "infeasible_for_all": sum(e["infeasible_for_all"]["0.01"] for e in episodes),
        "assigned_share_mean": float(np.mean(assigned)),
        "used_share": sent / offered if offered else 0.0,
        "blocked_per_episode": blocked_total / len(episodes),
        "episodes_with_blocked": blocked_episodes,
        "by_class": {
            str(k): {
                "ues": c["ues"],
                "occasions_mean": c["occasions"] / c["ues"],
                "window_share_mean": c["window_share"] / c["ues"],
            }
            for k, c in sorted(by_class.items())
        },
        "by_mean_snr_db": {
            snr_labels[k]: {
                "ues": s["ues"],
                "window_share_mean": s["window_share"] / s["ues"],
                "predicted_outage_mean": s["outage"] / s["ues"],
                "violation_rate": s["v"] / s["p"],
            }
            for k, s in sorted(by_snr.items())
        },
        "predicted_vs_realized": {
            outage_labels[k]: {
                "ues": o["ues"],
                "predicted_outage_mean": o["outage"] / o["ues"],
                "violation_rate": o["v"] / o["p"],
            }
            for k, o in sorted(by_outage.items())
        },
        "failing_ues": {
            "count": failing["ues"],
            "classes": dict(failing["classes"]),
            "mean_snr_db_median": _median(failing["mean_snr_db"]),
            "window_share_median": _median(failing["window_share"]),
            "predicted_outage_median": _median(failing["predicted_outage"]),
        },
    }


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--eval512", type=Path, required=True)
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
    record, episodes = read(args.validation, "validation", None, formal)
    periods = [c.period_slots for c in load_scenario(ROOT / record["config"], 4).traffic_classes]
    validation = validation_table(record)
    evaluation = eval512_table([args.eval512, *args.exp010], formal, args.resamples)
    _, eval_episodes = read(args.eval512, "dev", exp010.SET, formal)
    diag = {
        name: {
            "validation": {n: diagnostics(rows, periods) for n, rows in episodes[spec].items()},
            "exp010-eval512": {
                n: diagnostics(rows, periods) for n, rows in eval_episodes[spec].items()
            },
        }
        for name, spec in CG.items()
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
    print(f"N=8: CG - A_mix {r['CG_minus_A_mix']} -> {r['reading']}")


if __name__ == "__main__":
    main()
