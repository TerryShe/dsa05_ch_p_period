"""EXP-DSA05-002: periodic memory versus phase 1 of EXP-DSA05-001 (pre-registered §4).

For each variant (fixed-w10, local-eta2) compares the protocol-selected validation
candidates of the periodic-memory runs with the phase-1 runs of the same setting
and seeds, and applies the adoption rule:

1. mean F over the three seeds >= 0.95;
2. mean E/UB at least 5% (relative) below the phase-1 mean of the same variant;
3. mean collision share (from the efficiency diagnosis) below phase 1's.

Also writes paired per-seed differences and a validation-curve figure.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dsa_period.evaluation import FEASIBLE_EPISODE_RATE

PROFILES = {"fixed": "fixed-w10", "local": "local-eta2"}
LABEL = {"fixed": "A fixed penalty (w=10)", "local": "C local duals (eta=2)"}
SEEDS = (101, 202, 303)
MIN_RELATIVE_REDUCTION = 0.05
SERIES = ("#2a78d6", "#eb6834")  # reference palette slots 1-2: phase 1, periodic memory
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"


def load(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    if not manifest["formal"] or manifest["provenance"]["dirty"]:
        raise SystemExit(f"{path} is not a formal clean run")
    result = json.loads((path / "result.json").read_text())
    curve = []
    for record_path in sorted((path / "validation").glob("step-*.json")):
        record = json.loads(record_path.read_text())
        entry = next(e for e in record["by_mu0"] if e["mu0"] == result.get("selected_mu0", 0.0))
        m = entry["summary"]["metrics"]
        curve.append((record["slots"], m["feasible"]["mean"], m["e_over_ub"]["mean"]))
    selected = result["candidates"][result["selected"]]
    return {
        "path": str(path),
        "commit": manifest["provenance"]["commit"],
        "lags": manifest["resolved"]["model"].get("periodic_lags", []),
        "result_sha256": hashlib.sha256((path / "result.json").read_bytes()).hexdigest(),
        "selected": result["selected"],
        "feasible": selected["feasible"],
        "e_over_ub": selected["e_over_ub"],
        "curve": curve,
    }


def by_setting(paths: list[Path]) -> dict:
    """Index runs by (variant, seed); require exactly the pre-registered settings."""
    runs = {}
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text())
        key = (manifest["variant"], manifest["seed"])
        if manifest["profile"] != PROFILES.get(manifest["variant"]) or key in runs:
            raise SystemExit(f"unexpected or duplicate run {path}")
        runs[key] = load(path)
    expected = {(variant, seed) for variant in PROFILES for seed in SEEDS}
    if set(runs) != expected:
        raise SystemExit(f"runs cover {sorted(runs)}, expected {sorted(expected)}")
    return runs


def diagnosis(path: Path) -> dict:
    record = json.loads(path.read_text())
    return {name: r for name, r in record["results"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, nargs="+", required=True, help="phase-1 run dirs")
    parser.add_argument("--after", type=Path, nargs="+", required=True, help="periodic-memory dirs")
    parser.add_argument("--diagnosis-before", type=Path, required=True)
    parser.add_argument("--diagnosis-after", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    before, after = by_setting(args.before), by_setting(args.after)
    if any(r["lags"] for r in before.values()) or any(not r["lags"] for r in after.values()):
        raise SystemExit("before runs must have no periodic memory and after runs must have it")
    diag_before, diag_after = diagnosis(args.diagnosis_before), diagnosis(args.diagnosis_after)
    tag = {"fixed": "A", "local": "C"}
    summary = {"runs": {}, "variants": {}}
    for variant in PROFILES:
        rows = []
        for seed in SEEDS:
            b, a = before[(variant, seed)], after[(variant, seed)]
            key = f"{tag[variant]}-s{seed}"
            rows.append(
                {
                    "seed": seed,
                    "before": {k: b[k] for k in ("selected", "feasible", "e_over_ub")},
                    "after": {k: a[k] for k in ("selected", "feasible", "e_over_ub")},
                    "e_over_ub_difference": a["e_over_ub"] - b["e_over_ub"],
                    "collision_before": diag_before[key]["collision_share"],
                    "collision_after": diag_after[key]["collision_share"],
                    "interference_loss_before": diag_before[key]["interference_loss"],
                    "interference_loss_after": diag_after[key]["interference_loss"],
                    "duty_before": diag_before[key]["duty"],
                    "duty_after": diag_after[key]["duty"],
                }
            )
            summary["runs"][f"{variant}-before-{seed}"] = b
            summary["runs"][f"{variant}-after-{seed}"] = a
        mean = {
            name: float(np.mean([r[name] for r in rows]))
            for name in (
                "collision_before",
                "collision_after",
                "interference_loss_before",
                "interference_loss_after",
                "duty_before",
                "duty_after",
            )
        }
        f_after = float(np.mean([r["after"]["feasible"] for r in rows]))
        e_before = float(np.mean([r["before"]["e_over_ub"] for r in rows]))
        e_after = float(np.mean([r["after"]["e_over_ub"] for r in rows]))
        reduction = 1.0 - e_after / e_before
        checks = {
            "feasible": f_after >= FEASIBLE_EPISODE_RATE,
            "energy": reduction >= MIN_RELATIVE_REDUCTION,
            "collisions": mean["collision_after"] < mean["collision_before"],
        }
        if all(checks.values()):
            verdict = "adopt"
        elif not checks["feasible"]:
            verdict = "hurts feasibility"
        elif checks["collisions"]:
            verdict = "mechanism works, gain insufficient"
        else:
            verdict = "not adopted"
        summary["variants"][variant] = {
            "per_seed": rows,
            "mean_feasible_before": float(np.mean([r["before"]["feasible"] for r in rows])),
            "mean_feasible_after": f_after,
            "mean_e_over_ub_before": e_before,
            "mean_e_over_ub_after": e_after,
            "relative_reduction": reduction,
            **mean,
            "checks": checks,
            "verdict": verdict,
        }
        print(
            f"{variant:<6} F {summary['variants'][variant]['mean_feasible_before']:.3f} -> {f_after:.3f}  "
            f"E/UB {e_before:.3f} -> {e_after:.3f} ({-100 * reduction:+.1f}%)  "
            f"collisions {mean['collision_before']:.3f} -> {mean['collision_after']:.3f}  "
            f"I-loss {mean['interference_loss_before']:.3f} -> {mean['interference_loss_after']:.3f}  "
            f"=> {verdict}"
        )
    if args.output is not None:
        if args.output.exists():
            parser.error(f"output {args.output} already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if args.figure is not None:
        plot(before, after, args.figure)


def plot(before: dict, after: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for row, variant in enumerate(PROFILES):
        for slot, (runs, label) in enumerate(((before, "phase 1"), (after, "periodic memory"))):
            curves = [runs[(variant, seed)]["curve"] for seed in SEEDS]
            slots = np.array([c[0] for c in curves[0]]) / 1e6
            for column in range(2):
                values = np.array([[c[column + 1] for c in curve] for curve in curves])
                for line in values:
                    axes[row, column].plot(slots, line, color=SERIES[slot], linewidth=1, alpha=0.4)
                axes[row, column].plot(
                    slots, values.mean(axis=0), color=SERIES[slot], linewidth=2, label=label
                )
        axes[row, 0].axhline(FEASIBLE_EPISODE_RATE, color=MUTED, linestyle="--", linewidth=1)
        axes[row, 1].axhline(2.871, color=MUTED, linestyle="--", linewidth=1)
        axes[row, 1].annotate(
            "paced_sticky 2.871",
            (0.98, 2.871),
            xycoords=("axes fraction", "data"),
            xytext=(0, 3),
            textcoords="offset points",
            color=MUTED,
            fontsize=7,
            ha="right",
        )
        axes[row, 0].set_ylim(-0.02, 1.04)
        axes[row, 1].set_ylim(1.5, 6.0)
        axes[row, 0].set_title(f"{LABEL[variant]}: validation F", color=INK, fontsize=10)
        axes[row, 1].set_title(f"{LABEL[variant]}: validation E / UB", color=INK, fontsize=10)
        axes[row, 1].legend(fontsize=8, frameon=False, labelcolor=INK)
        for ax in axes[row]:
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("training slots (M)", color=MUTED, fontsize=9)
    fig.suptitle(
        "N=8, seeds 101/202/303 (thin) and mean (thick): phase 1 vs periodic memory",
        color=INK,
        fontsize=10,
    )
    fig.savefig(path, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
