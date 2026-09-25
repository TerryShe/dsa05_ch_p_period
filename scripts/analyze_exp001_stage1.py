"""EXP-DSA05-001 phase 1: tuning selection (1a) and the long-training decision (1b).

Reads formal training runs (their manifests, validation records and results),
verifies provenance, and applies the rules written in the design's §9:

* 1a: per variant, pool every validation candidate of every tuning setting and
  apply the protocol selection rule; the winner's setting is the variant's.
* 1b: per variant, the three seeds' selected candidates; the variant proceeds
  to long training iff mean F >= 0.95 and mean E/UB < the baseline's.

Writes a compact JSON (curves and selections) and, if requested, a figure.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dsa_period.evaluation import FEASIBLE_EPISODE_RATE, select

BASELINE_E_OVER_UB = {8: 2.871}  # paced_sticky margin 0.5, validation calibration


def load_run(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    result = json.loads((path / "result.json").read_text())
    curve = []
    for record_path in sorted((path / "validation").glob("step-*.json")):
        record = json.loads(record_path.read_text())
        for entry in record["by_mu0"]:
            metrics = entry["summary"]["metrics"]
            curve.append(
                {
                    "checkpoint": record["checkpoint"],
                    "slots": record["slots"],
                    "mu0": entry["mu0"],
                    "summary": entry["summary"],
                    "feasible": metrics["feasible"]["mean"],
                    "e_over_ub": metrics["e_over_ub"]["mean"],
                    "worst_ue_ocr": metrics["worst_ue_ocr"]["mean"],
                    "mw_per_ue": metrics["mw_per_ue"]["mean"],
                }
            )
    return {
        "path": str(path),
        "manifest": manifest,
        "result": result,
        "curve": curve,
        "result_sha256": hashlib.sha256((path / "result.json").read_bytes()).hexdigest(),
    }


def candidate_key(run: dict, point: dict) -> str:
    suffix = f"@mu0={point['mu0']:g}" if run["manifest"]["variant"] == "local" else ""
    return (
        f"{run['manifest']['profile']}/seed{run['manifest']['seed']}/{point['checkpoint']}{suffix}"
    )


def point_of(run: dict, candidate: str) -> dict:
    for point in run["curve"]:
        name = point["checkpoint"]
        if run["manifest"]["variant"] == "local":
            name += f"@mu0={point['mu0']:g}"
        if name == candidate:
            return point
    raise KeyError(candidate)


def brief(point: dict) -> dict:
    keys = ("checkpoint", "slots", "mu0", "feasible", "e_over_ub", "worst_ue_ocr", "mw_per_ue")
    return {key: point[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    runs = [load_run(path) for path in args.runs]
    for run in runs:
        manifest = run["manifest"]
        if not manifest["formal"] or manifest["provenance"]["dirty"]:
            raise SystemExit(f"{run['path']} is not a formal clean run")
    num_ues = {int(run["manifest"]["scenario"].split("-n")[-1].split(".")[0]) for run in runs}
    if len(num_ues) != 1:
        raise SystemExit("runs mix scenario variants")
    n = num_ues.pop()
    baseline = BASELINE_E_OVER_UB[n]

    by_variant: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_variant[run["manifest"]["variant"]].append(run)

    tuning, decision = {}, {}
    for variant, members in sorted(by_variant.items()):
        seed101 = [run for run in members if run["manifest"]["seed"] == 101]
        pooled = {candidate_key(run, p): p["summary"] for run in seed101 for p in run["curve"]}
        winner = select(pooled)
        profile = winner.split("/")[0]
        tuning[variant] = {
            "settings": {
                run["manifest"]["profile"]: {
                    "selected": run["result"]["selected"],
                    "selected_point": brief(point_of(run, run["result"]["selected"])),
                }
                for run in seed101
            },
            "pooled_winner": winner,
            "selected_profile": profile,
        }
        chosen = [run for run in members if run["manifest"]["profile"] == profile]
        seeds = sorted(run["manifest"]["seed"] for run in chosen)
        points = [point_of(run, run["result"]["selected"]) for run in chosen]
        if len(chosen) >= 1:
            feasible = float(np.mean([p["feasible"] for p in points]))
            energy = float(np.mean([p["e_over_ub"] for p in points]))
            decision[variant] = {
                "profile": profile,
                "seeds": seeds,
                "per_seed": {
                    str(run["manifest"]["seed"]): brief(point)
                    for run, point in zip(chosen, points, strict=True)
                },
                "mean_feasible": feasible,
                "mean_e_over_ub": energy,
                "complete": seeds == [101, 202, 303],
                "passes": feasible >= FEASIBLE_EPISODE_RATE and energy < baseline,
            }

    summary = {
        "num_ues": n,
        "baseline_e_over_ub": baseline,
        "runs": [
            {
                "path": run["path"],
                "variant": run["manifest"]["variant"],
                "profile": run["manifest"]["profile"],
                "seed": run["manifest"]["seed"],
                "commit": run["manifest"]["provenance"]["commit"],
                "result_sha256": run["result_sha256"],
                "wall_s": run["result"]["wall_s"],
                "selected": run["result"]["selected"],
                "curve": [brief(point) for point in run["curve"]],
            }
            for run in runs
        ],
        "tuning": tuning,
        "decision": decision,
    }
    for variant, entry in decision.items():
        print(
            f"{variant:<7} {entry['profile']:<14} seeds {entry['seeds']} "
            f"F {entry['mean_feasible']:.3f} E/UB {entry['mean_e_over_ub']:.3f} "
            f"(baseline {baseline}) passes={entry['passes']} complete={entry['complete']}"
        )
    if args.output is not None:
        if args.output.exists():
            parser.error(f"output {args.output} already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if args.figure is not None:
        plot(runs, baseline, args.figure)


def plot(runs: list[dict], baseline: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {"fixed": "#4C72B0", "global": "#DD8452", "local": "#55A868"}
    for run in sorted(runs, key=lambda r: (r["manifest"]["variant"], r["manifest"]["profile"])):
        variant = run["manifest"]["variant"]
        mu0 = run["result"].get("selected_mu0", 0.0)
        points = [p for p in run["curve"] if variant != "local" or p["mu0"] == mu0]
        slots = [p["slots"] / 1e6 for p in points]
        label = f"{run['manifest']['profile']} s{run['manifest']['seed']}"
        style = dict(color=colors[variant], alpha=0.8, marker="o", markersize=3, label=label)
        axes[0].plot(slots, [p["feasible"] for p in points], **style)
        axes[1].plot(slots, [p["e_over_ub"] for p in points], **style)
    axes[0].axhline(FEASIBLE_EPISODE_RATE, color="gray", linestyle="--", linewidth=1)
    axes[0].set(xlabel="training slots (M)", ylabel="validation feasibility F", ylim=(-0.02, 1.02))
    axes[1].axhline(baseline, color="gray", linestyle="--", linewidth=1, label="paced_sticky")
    axes[1].set(xlabel="training slots (M)", ylabel="validation E/UB")
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
