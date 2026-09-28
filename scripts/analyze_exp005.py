"""EXP-DSA05-005: validation-side analysis of the 18 SAC runs (design §3.1, §4 items 5 and 9).

Checks that every run is formal, clean, on one commit and complete, then reports:

* each run's protocol selection and final checkpoint (validation F, E/UB);
* the tuned gamma of each reward: the protocol rule applied to the pooled
  validation candidates of the two seed-101 runs of that reward (design §3.1);
* sample efficiency: the first validation with F >= 0.95 and E/UB below the
  N=8 baseline (2.871), and the first with E/UB <= 1.5, also for the reference
  runs (EXP-DSA05-001 phase 2b A and C) when given.

Validation only: nothing here reads confirmation or test data.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dsa_period.evaluation import FEASIBLE_EPISODE_RATE, select

REWARDS = ("v1", "v1s", "v2")
GAMMAS = {"g05": 0.5, "g099": 0.99}
BASELINE_E_OVER_UB = 2.871  # paced_sticky margin=0.5 at N=8 (validation calibration)
BUDGET_SLOTS = 20_000_000
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
GAMMA_COLOR = {0.5: SERIES[1], 0.99: SERIES[2]}


def load_run(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    result = json.loads((path / "result.json").read_text())
    curve = []
    summaries = {}
    for record_path in sorted((path / "validation").glob("step-*.json")):
        record = json.loads(record_path.read_text())
        entry = record["by_mu0"][0] if manifest["variant"] == "sac" else None
        if entry is None:
            mu0 = result.get("selected_mu0", 0.0)
            entry = next(e for e in record["by_mu0"] if e["mu0"] == mu0)
        metrics = entry["summary"]["metrics"]
        summaries[record["checkpoint"]] = entry["summary"]
        curve.append(
            {
                "checkpoint": record["checkpoint"],
                "slots": record["slots"],
                "feasible": metrics["feasible"]["mean"],
                "e_over_ub": metrics["e_over_ub"]["mean"],
                "mw_per_ue": metrics["mw_per_ue"]["mean"],
            }
        )
    return {
        "path": str(path),
        "manifest": manifest,
        "result": result,
        "curve": curve,
        "summaries": summaries,
        "result_sha256": hashlib.sha256((path / "result.json").read_bytes()).hexdigest(),
    }


def check_sac(runs: list[dict]) -> str:
    commits = {r["manifest"]["provenance"]["commit"] for r in runs}
    for r in runs:
        m, res = r["manifest"], r["result"]
        if not m["formal"] or m["provenance"]["dirty"] or m["variant"] != "sac":
            raise SystemExit(f"{r['path']} is not a formal clean SAC run")
        if res["slots"] != BUDGET_SLOTS or len(r["curve"]) != 21:
            raise SystemExit(f"{r['path']} is incomplete")
    if len(commits) != 1:
        raise SystemExit(f"runs come from several commits: {sorted(commits)}")
    grid = {(r["manifest"]["profile"], r["manifest"]["seed"]) for r in runs}
    expected = {(f"{w}-{g}", s) for w in REWARDS for g in GAMMAS for s in (101, 202, 303)}
    if grid != expected:
        raise SystemExit(f"run matrix differs from the design: missing {sorted(expected - grid)}")
    return commits.pop()


def first_slots(curve: list[dict], condition) -> int | None:
    hits = [p["slots"] for p in curve if condition(p)]
    return hits[0] if hits else None


def efficiency(run: dict) -> dict:
    return {
        "feasible_below_baseline": first_slots(
            run["curve"],
            lambda p: (
                p["feasible"] >= FEASIBLE_EPISODE_RATE and p["e_over_ub"] < BASELINE_E_OVER_UB
            ),
        ),
        "e_over_ub_at_most_1_5": first_slots(run["curve"], lambda p: p["e_over_ub"] <= 1.5),
    }


def tuned_gamma(runs: list[dict], reward: str) -> dict:
    """Design §3.1: pool the seed-101 candidates of both gammas; the protocol rule picks."""
    pooled = {}
    for r in runs:
        profile = r["manifest"]["profile"]
        if profile.split("-")[0] == reward and r["manifest"]["seed"] == 101:
            for name, summary in r["summaries"].items():
                pooled[f"{profile}/{name}"] = summary
    winner = select(pooled)
    profile = winner.split("/")[0]
    m = pooled[winner]["metrics"]
    return {
        "profile": profile,
        "gamma": GAMMAS[profile.split("-")[1]],
        "candidate": winner,
        "feasible": m["feasible"]["mean"],
        "e_over_ub": m["e_over_ub"]["mean"],
    }


def run_row(r: dict) -> dict:
    res = r["result"]
    by_name = {p["checkpoint"]: p for p in r["curve"]}
    sel, fin = by_name[res["selected_checkpoint"]], by_name[res["final_checkpoint"]]
    return {
        "path": r["path"],
        "profile": r["manifest"]["profile"],
        "seed": r["manifest"]["seed"],
        "commit": r["manifest"]["provenance"]["commit"],
        "result_sha256": r["result_sha256"],
        "wall_h": res.get("wall_s", 0.0) / 3600,
        "selected": res["selected_checkpoint"],
        "selected_feasible": sel["feasible"],
        "selected_e_over_ub": sel["e_over_ub"],
        "final_feasible": fin["feasible"],
        "final_e_over_ub": fin["e_over_ub"],
        "efficiency": efficiency(r),
        "curve": [{k: p[k] for k in ("slots", "feasible", "e_over_ub")} for p in r["curve"]],
    }


def plot(runs: list[dict], references: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, grid = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    ref = [r for r in references if r["manifest"]["profile"] == "fixed-w10"]
    for column, reward in enumerate(REWARDS):
        for gamma_key, gamma in GAMMAS.items():
            chosen = [r for r in runs if r["manifest"]["profile"] == f"{reward}-{gamma_key}"]
            slots = np.array([p["slots"] / 1e6 for p in chosen[0]["curve"]])
            for row, key in enumerate(("feasible", "e_over_ub")):
                values = np.array([[p[key] for p in r["curve"]] for r in chosen])
                ax = grid[row, column]
                for line in values:
                    ax.plot(slots, line, color=GAMMA_COLOR[gamma], linewidth=1, alpha=0.4)
                ax.plot(
                    slots,
                    values.mean(axis=0),
                    color=GAMMA_COLOR[gamma],
                    linewidth=2,
                    label=f"SAC, gamma={gamma}",
                )
        if ref:
            slots = np.array([p["slots"] / 1e6 for p in ref[0]["curve"]])
            for row, key in enumerate(("feasible", "e_over_ub")):
                values = np.array([[p[key] for p in r["curve"]] for r in ref]).mean(axis=0)
                grid[row, column].plot(
                    slots, values, color=INK, linestyle="--", linewidth=1.5, label="A (MAPPO) mean"
                )
        grid[0, column].set_title(f"reward {reward}", color=INK, fontsize=10)
        grid[0, column].axhline(FEASIBLE_EPISODE_RATE, color=MUTED, linestyle=":", linewidth=1)
        grid[1, column].axhline(BASELINE_E_OVER_UB, color=MUTED, linestyle=":", linewidth=1)
        grid[1, column].annotate(
            "paced_sticky 2.871",
            (0.98, BASELINE_E_OVER_UB),
            xycoords=("axes fraction", "data"),
            xytext=(0, 3),
            textcoords="offset points",
            color=MUTED,
            fontsize=7,
            ha="right",
        )
        grid[1, column].set_ylim(0.8, 12.0)
        grid[0, column].set_ylim(-0.02, 1.04)
    for ax in grid.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    for column in range(3):
        grid[1, column].set_xlabel("training slots (M)", color=MUTED, fontsize=9)
    handles, labels = grid[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        labelcolor=INK,
        fontsize=8,
    )
    grid[0, 0].set_ylabel("validation feasibility F", color=MUTED, fontsize=9)
    grid[1, 0].set_ylabel("validation E / UB", color=MUTED, fontsize=9)
    fig.suptitle(
        "N=8 validation curves: seeds 101/202/303 (thin) and their mean (thick); A for reference",
        color=INK,
        fontsize=10,
    )
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="the 18 SAC run directories")
    parser.add_argument("--reference", type=Path, nargs="*", default=[], help="phase-2b A/C runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    for path in (args.output, args.figure):
        if path is not None and path.exists():
            parser.error(f"output {path} already exists")
    runs = [load_run(p) for p in args.runs]
    commit = check_sac(runs)
    references = [load_run(p) for p in args.reference]
    rows = [run_row(r) for r in runs]
    tuned = {reward: tuned_gamma(runs, reward) for reward in REWARDS}
    reference_rows = [
        {
            "path": r["path"],
            "profile": r["manifest"]["profile"],
            "seed": r["manifest"]["seed"],
            "efficiency": efficiency(r),
        }
        for r in references
    ]
    record = {
        "commit": commit,
        "baseline_e_over_ub": BASELINE_E_OVER_UB,
        "runs": rows,
        "tuned": tuned,
        "reference_efficiency": reference_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    if args.figure is not None:
        plot(runs, references, args.figure)
    print(f"all 18 runs formal and clean on {commit[:7]}")
    print(
        f"{'run':<22} {'selected':>15} {'F':>6} {'E/UB':>7} {'final F':>8} {'E/UB':>7} {'feas<2.871':>10} {'<=1.5':>6}"
    )
    for row in rows:
        eff = row["efficiency"]
        print(
            f"{row['profile'] + '/' + str(row['seed']):<22} {row['selected']:>15} "
            f"{row['selected_feasible']:6.3f} {row['selected_e_over_ub']:7.3f} "
            f"{row['final_feasible']:8.3f} {row['final_e_over_ub']:7.3f} "
            f"{str(eff['feasible_below_baseline']):>10} {str(eff['e_over_ub_at_most_1_5']):>6}"
        )
    for reward, t in tuned.items():
        print(
            f"tuned {reward}: gamma {t['gamma']} ({t['candidate']}, F {t['feasible']:.3f}, E/UB {t['e_over_ub']:.3f})"
        )
    for row in reference_rows:
        print(f"reference {row['profile']}/{row['seed']}: {row['efficiency']}")


if __name__ == "__main__":
    main()
