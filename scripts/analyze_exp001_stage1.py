"""EXP-DSA05-001 phase 1: tuning selection (1a) and the long-training decision (1b).

Reads formal training runs (their manifests, validation records and results),
verifies provenance, and applies the rules written in the design's §9:

* 1a: per variant, pool every validation candidate of every tuning setting and
  apply the protocol selection rule; the winner's setting is the variant's.
* 1b: per variant, the three seeds' selected candidates; the variant proceeds
  to long training iff mean F >= 0.95 and mean E/UB < the baseline's.

Writes a compact JSON (curves and selections) and, if requested, two figures.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dsa_period.config import load_scenario
from dsa_period.evaluation import FEASIBLE_EPISODE_RATE, select

# Strongest feasible local baseline per N (validation calibration, 2026-09-24):
# paced_sticky with margin 0 (N=4, exactly optimal), 0.25 (N=6), 0.5 (N=8), 8 (N=12).
BASELINE_E_OVER_UB = {4: 1.000, 6: 1.782, 8: 2.871, 12: 5.498}
FAR_M = 75.0
WINDOW_SLOTS = 1800


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


def violation_breakdown(run: dict, scenario) -> dict:
    """Where the final checkpoint's validation violations fall, by class and distance.

    ``packet_weighted`` is what a single global multiplier sees (every settled
    packet counts once); ``per_ue_mean`` weights every UE equally, as the
    per-UE constraint does.
    """
    path = Path(run["path"]) / "validation" / f"{run['result']['final_checkpoint']}.json"
    record = json.loads(path.read_text())
    mu0 = run["result"].get("selected_mu0", 0.0)
    entry = next(e for e in record["by_mu0"] if e["mu0"] == mu0)
    rows = []
    for episode in entry["episodes"]:
        _, traffic = scenario.instance(episode["instance"])
        for ue, rate in enumerate(episode["per_ue_violation_rate"]):
            rows.append((int(traffic.period_slots[ue]), episode["distance_m"][ue], rate))
    rows = np.array(rows)
    weights = WINDOW_SLOTS / rows[:, 0]
    by_period = {}
    for period in sorted(set(rows[:, 0].astype(int))):
        mask = rows[:, 0] == period
        far = mask & (rows[:, 1] >= FAR_M)
        by_period[str(period)] = {
            "ue_episodes": int(mask.sum()),
            "mean_violation": float(rows[mask, 2].mean()),
            "far_mean_violation": float(rows[far, 2].mean()) if far.any() else None,
            "starved_ue_episodes": int((rows[mask, 2] == 1.0).sum()),
        }
    return {
        "checkpoint": run["result"]["final_checkpoint"],
        "mu0": mu0,
        "per_ue_mean": float(rows[:, 2].mean()),
        "packet_weighted": float(np.sum(weights * rows[:, 2]) / np.sum(weights)),
        "far_mean": float(rows[rows[:, 1] >= FAR_M, 2].mean()),
        "near_mean": float(rows[rows[:, 1] < FAR_M, 2].mean()),
        "by_period": by_period,
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
    parser.add_argument("--figure", type=Path, help="selected settings, all seeds")
    parser.add_argument("--tuning-figure", type=Path, help="stage 1a settings, seed 101")
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
    scenario = load_scenario(runs[0]["manifest"]["scenario"])

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
                "final_violation_breakdown": violation_breakdown(run, scenario),
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
        plot(runs, decision, baseline, args.figure, n)
    if args.tuning_figure is not None:
        plot_tuning(runs, baseline, args.tuning_figure)


# Reference categorical palette (dataviz skill), validated: the first three slots
# all-pairs and the first four adjacent-pairs on the light surface.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
VARIANT_ORDER = ("fixed", "global", "local")
VARIANT_LABEL = {
    "fixed": "A fixed penalty",
    "global": "B global Lagrangian",
    "local": "C local duals",
}


def _points(run: dict) -> list[dict]:
    """Validation curve at the run's selected mu0 (local variant) or its only curve."""
    mu0 = run["result"].get("selected_mu0", 0.0)
    local = run["manifest"]["variant"] == "local"
    return [p for p in run["curve"] if not local or p["mu0"] == mu0]


def _axes_style(ax, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("training slots (M)", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def _reference_lines(axes, baseline: float) -> None:
    axes[0].axhline(FEASIBLE_EPISODE_RATE, color=MUTED, linestyle="--", linewidth=1)
    axes[0].annotate(
        "F = 0.95",
        (0.02, FEASIBLE_EPISODE_RATE),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        color=MUTED,
        fontsize=7,
    )
    axes[1].axhline(baseline, color=MUTED, linestyle="--", linewidth=1)
    axes[1].annotate(
        f"paced_sticky {baseline:.3f}",
        (0.98, baseline),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        color=MUTED,
        fontsize=7,
        ha="right",
    )


def plot(runs: list[dict], decision: dict, baseline: float, path: Path, num_ues: int) -> None:
    """Selected setting of each variant: seed curves (thin) and seed mean (thick)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for variant in (v for v in VARIANT_ORDER if v in decision):
        slot = VARIANT_ORDER.index(variant)  # color follows the variant across figures
        profile = decision[variant]["profile"]
        chosen = [r for r in runs if r["manifest"]["profile"] == profile]
        curves = [_points(r) for r in chosen]
        slots = np.array([p["slots"] / 1e6 for p in curves[0]])
        color = SERIES[slot]
        for column, key in enumerate(("feasible", "e_over_ub")):
            values = np.array([[p[key] for p in curve] for curve in curves])
            for row in values:
                axes[column].plot(slots, row, color=color, linewidth=1, alpha=0.45)
            mean = values.mean(axis=0)
            axes[column].plot(
                slots, mean, color=color, linewidth=2, label=f"{VARIANT_LABEL[variant]} ({profile})"
            )
            axes[column].annotate(
                VARIANT_LABEL[variant].split()[0],
                (slots[-1], mean[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                color=INK,
                fontsize=8,
                va="center",
            )
    _reference_lines(axes, baseline)
    _axes_style(axes[0], "validation feasibility F")
    _axes_style(axes[1], "validation E / UB")
    axes[0].set_ylim(-0.02, 1.04)
    axes[1].set_ylim(1.0, 6.0)
    axes[1].legend(fontsize=8, frameon=False, labelcolor=INK, loc="center right")
    fig.suptitle(
        f"N={num_ues}, selected setting per variant: seeds 101/202/303 (thin) and their mean (thick)",
        color=INK,
        fontsize=10,
    )
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def plot_tuning(runs: list[dict], baseline: float, path: Path) -> None:
    """Stage 1a tuning at seed 101: one column per variant, one line per setting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, grid = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for column, variant in enumerate(VARIANT_ORDER):
        members = sorted(
            (
                r
                for r in runs
                if r["manifest"]["variant"] == variant and r["manifest"]["seed"] == 101
            ),
            # Numeric order of the tuned value: fixed-w1 < fixed-w3 < fixed-w10 < fixed-w30.
            key=lambda r: float(re.sub(r"[^0-9.]", "", r["manifest"]["profile"].split("-")[-1])),
        )
        for slot, run in enumerate(members):
            points = _points(run)
            slots = [p["slots"] / 1e6 for p in points]
            label = run["manifest"]["profile"]
            if variant == "local":
                label += f" (mu0={run['result']['selected_mu0']:g})"
            for row, key in enumerate(("feasible", "e_over_ub")):
                grid[row, column].plot(
                    slots,
                    [p[key] for p in points],
                    color=SERIES[slot],
                    linewidth=1.5,
                    marker="o",
                    markersize=3,
                    label=label,
                )
        grid[0, column].set_title(VARIANT_LABEL[variant], color=INK, fontsize=10)
        _reference_lines(grid[:, column], baseline)
        _axes_style(grid[0, column], "validation feasibility F")
        _axes_style(grid[1, column], "validation E / UB")
        grid[0, column].set_ylim(-0.02, 1.04)
        grid[1, column].set_ylim(0, 6.0)
        grid[0, column].legend(fontsize=7, frameon=False, labelcolor=INK, loc="best")
    fig.suptitle(
        "Stage 1a tuning, N=8, seed 101 (local variant shown at its selected mu0)",
        color=INK,
        fontsize=10,
    )
    fig.savefig(path, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
