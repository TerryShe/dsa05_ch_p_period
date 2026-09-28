"""Build manuscript assets from completed, archived experiments only.

This script reads existing JSON summaries; it never runs policies or opens C1
evaluation data. Sources and their SHA-256 digests accompany the copied data.
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
SOURCES = {
    "test": "docs/results/exp004-test_20260926/summary.json",
    "learning_n8": "docs/results/exp001-phase2b_20260925/summary.json",
    "diagnosis_long": "docs/results/exp001-phase2b_20260925/diagnosis.json",
    "diagnosis_short": "docs/results/exp001-efficiency-diagnosis_20260925/summary.json",
    "periodic_memory": "docs/results/exp002-periodic-memory_20260925/summary.json",
    "references": "docs/results/protocol-v1-references_20260924/test.json",
}
COLORS = {"fixed": "#0072B2", "local": "#D55E00", "paced": "#009E73"}
NS = [4, 6, 8, 12]


def save(fig, name):
    fig.savefig(HERE / "figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(HERE / "figures" / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    for name in ("figures", "generated", "data"):
        (HERE / name).mkdir(exist_ok=True)
    records, sources = {}, {}
    for name, rel in SOURCES.items():
        raw = (ROOT / rel).read_bytes()
        records[name] = json.loads(raw)
        sources[name] = {"path": rel, "sha256": hashlib.sha256(raw).hexdigest()}
    archive = {
        "description": "Frozen manuscript evidence snapshot; no EXP-005 outcome included.",
        "sources": sources,
        "records": records,
    }
    (HERE / "data" / "evidence.json").write_text(json.dumps(archive, indent=2) + "\n")
    test = records["test"]["long"]
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.25,
    })
    methods = [
        ("fixed-w10:selected", "MAPPO-F", "fixed", "o"),
        ("local-eta2:selected", "MAPPO-D", "local", "s"),
        (None, "Paced-sticky", "paced", "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.05))
    x = np.arange(len(NS))
    for offset, (key, label, color, marker) in zip((-.13, 0, .13), methods):
        rows = [test[str(n)]["methods"][key or test[str(n)]["reference"]] for n in NS]
        for ax, metric in zip(axes, ("e_over_ub", "feasible")):
            means = np.array([r[metric]["mean"] for r in rows])
            limits = np.array([r[metric]["ci95"] for r in rows])
            ax.errorbar(x + offset, means, yerr=np.maximum(0, np.vstack((means-limits[:, 0], limits[:, 1]-means))),
                        label=label, color=COLORS[color], marker=marker, ms=4,
                        capsize=2, linestyle="-")
    axes[0].set_ylabel(r"Mean packet energy / $U_0$")
    axes[0].axhline(1, color="0.5", lw=.7, ls=":")
    axes[0].set_ylim(.8, 5.95)
    axes[1].set_ylabel(r"Feasible-episode fraction $F$")
    axes[1].axhline(.95, color="0.35", lw=.8, ls="--")
    axes[1].set_ylim(.925, 1.008)
    for ax in axes:
        ax.set_xticks(x, NS)
        ax.set_xlabel(r"Number of UEs $N$")
        ax.grid(axis="y", alpha=.18)
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=.5, w_pad=1.5)
    save(fig, "test_performance")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.0))
    runs = records["learning_n8"]["runs"]
    for variant, label in (("fixed", "MAPPO-F"), ("local", "MAPPO-D")):
        group = [r for r in runs if r["variant"] == variant]
        for ax, metric in zip(axes, ("e_over_ub", "feasible")):
            ys = []
            for r in group:
                xs = np.array([p["slots"] for p in r["curve"]]) / 1e6
                y = np.array([p[metric] for p in r["curve"]])
                ys.append(y)
                ax.plot(xs, y, color=COLORS[variant], lw=.6, alpha=.35)
            ax.plot(xs, np.mean(ys, axis=0), color=COLORS[variant], label=label, lw=1.6)
    axes[0].axhline(records["learning_n8"]["baseline_e_over_ub"], color=COLORS["paced"],
                    ls="--", lw=.8, label="Paced-sticky")
    axes[0].set_ylabel(r"Validation energy / $U_0$")
    axes[0].set_ylim(1, 5.8)
    axes[1].set_ylabel(r"Validation $F$")
    axes[1].axhline(.95, color="0.35", lw=.8, ls="--")
    axes[1].set_ylim(.48, 1.015)
    for ax in axes:
        ax.axvline(10, color="0.5", lw=.8, ls=":")
        ax.set_xlabel("Training slots (millions)")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(axis="y", alpha=.18)
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=.5, w_pad=1.4)
    save(fig, "learning_n8")

    short = records["diagnosis_short"]["results"]
    long = records["diagnosis_long"]["results"]
    groups = [
        [short[f"A-s{s}"] for s in (101, 202, 303)],
        [long[f"A-s{s}"] for s in (101, 202, 303)],
        [long[f"C-s{s}"] for s in (101, 202, 303)],
        [long["paced_sticky:margin=0.5"]],
        [long["central-upper-bound"]],
    ]
    labels = ["F: 2M", "F: 20M", "D: 20M", "Paced", "Central"]
    fig, ax = plt.subplots(figsize=(3.42, 1.82))
    x = np.arange(len(labels))
    for offset, metric, label, color in [(-.17, "collision_share", "Co-channel overlap", "#0072B2"),
                                        (.17, "interference_loss", "MI loss", "#E69F00")]:
        values = [[r[metric] for r in group] for group in groups]
        ax.bar(x+offset, [np.mean(v) for v in values], .31, color=color, label=label, alpha=.85)
        for i, v in enumerate(values):
            ax.scatter(i+offset+np.linspace(-.06, .06, len(v)), v, s=9, color="0.15", zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, .87)
    ax.set_ylabel("Fraction")
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, loc="upper right", fontsize=6.5)
    fig.tight_layout(pad=.5)
    save(fig, "coordination_n8")

    names = [("fixed-w10:selected", "MAPPO-F"), ("local-eta2:selected", "MAPPO-D"),
             (None, "Paced-sticky"), ("random_max", "Random-max"),
             ("sticky_max", "Sticky-max"), ("lazy_pacing", "Lazy-pacing")]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule", r"Method & $N=4$ & $N=6$ & $N=8$ & $N=12$ \\", r"\midrule"]
    for key, label in names:
        cells = []
        for n in NS:
            row = test[str(n)]["methods"][key or test[str(n)]["reference"]]
            f, e = row["feasible"]["mean"], row["e_over_ub"]["mean"]
            suffix = r"$^\dagger$" if f < .95 else ""
            cells.append(f"{f:.3f}/{e:.2f}" + suffix)
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "generated" / "all_methods.tex").write_text("\n".join(lines)+"\n")

    macros = []
    for n, word in zip(NS, ("Four", "Six", "Eight", "Twelve")):
        row = test[str(n)]["methods"]["fixed-w10:selected"]
        ref = test[str(n)]["methods"][test[str(n)]["reference"]]
        saving = 100*(1-row["e_over_ub"]["mean"]/ref["e_over_ub"]["mean"])
        for stem, value, digits in [("Saving", saving, 1), ("Energy", row["e_over_ub"]["mean"], 3),
                                     ("Feasible", row["feasible"]["mean"], 3), ("Power", row["mw_per_ue"]["mean"], 2)]:
            macros.append(f"\\newcommand{{\\{stem}{word}}}{{{value:.{digits}f}}}")
    (HERE / "generated" / "numbers.tex").write_text("\n".join(macros)+"\n")
    with (HERE / "data" / "test_results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["N", "method", "metric", "mean", "ci95_lower", "ci95_upper"])
        for n in NS:
            for key, row in test[str(n)]["methods"].items():
                for metric in ("feasible", "e_over_ub", "e_over_lb", "mw_per_ue"):
                    entry = row[metric]
                    writer.writerow([n, key, metric, entry["mean"], *entry["ci95"]])
    print("Generated three vector figures, results table, numerical macros, CSV, and evidence snapshot.")


if __name__ == "__main__":
    main()
