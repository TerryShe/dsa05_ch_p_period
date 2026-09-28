"""Build manuscript assets from completed, archived experiments only.

This script reads completed EXP-004 and EXP-005 archives, including the frozen
C1 comparison; it never executes policies. Source hashes accompany the data.
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
    "confirm1": "docs/results/exp005-xr-sbpc_20260927/confirm1-summary.json",
    "learning_sac": "docs/results/exp005-xr-sbpc_20260927/validation-analysis.json",
    "diagnosis_confirm_a": "docs/results/exp005-xr-sbpc_20260927/diagnosis-A.json",
    "diagnosis_sac_v1": "docs/results/exp005-xr-sbpc_20260927/diagnosis-v1.json",
    "diagnosis_sac_v1s": "docs/results/exp005-xr-sbpc_20260927/diagnosis-v1s.json",
    "diagnosis_sac_v2": "docs/results/exp005-xr-sbpc_20260927/diagnosis-v2.json",
    "references_confirm1": "docs/results/confirm1-references_20260927/confirm1.json",
}
COLORS = {"fixed": "#0072B2", "local": "#D55E00", "paced": "#009E73", "sac": "#8C56A3"}
NS = [4, 6, 8, 12]
CONFIRM_METHODS = [
    ("fixed-w10:selected", "MAPPO-F", "0.99"),
    ("local-eta2:selected", "MAPPO-D", "0.99"),
    ("paced_sticky:margin=0.5", "Paced-sticky", "--"),
    ("v1-g099:selected", "SAC-V1", "0.99"),
    ("v1s-g099:selected", "SAC-V1s", "0.99"),
    ("v2-g099:selected", "SAC-V2", "0.99"),
    ("v1-g05:selected", "SAC-V1", "0.5"),
    ("v1s-g05:selected", "SAC-V1s", "0.5"),
    ("v2-g05:selected", "SAC-V2", "0.5"),
]


def interval(entry):
    low, high = entry["ci95"]
    return f"{entry['mean']:.3f} [{low:.3f}, {high:.3f}]"


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
        "description": "Completed EXP-004 test and EXP-005 C1 evidence; validation diagnostics are separate sources.",
        "sources": sources,
        "records": records,
    }
    (HERE / "data" / "evidence.json").write_text(json.dumps(archive, indent=2) + "\n")
    test = records["test"]["long"]
    confirm = records["confirm1"]
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
    curve_groups = [
        ("fixed", "MAPPO-F", [r for r in runs if r["variant"] == "fixed"]),
        ("local", "MAPPO-D", [r for r in runs if r["variant"] == "local"]),
        ("sac", r"SAC-V2 ($\gamma=0.99$)", [
            r for r in records["learning_sac"]["runs"] if r["profile"] == "v2-g099"
        ]),
    ]
    for variant, label, group in curve_groups:
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
    axes[0].set_ylim(bottom=1)
    axes[1].set_ylabel(r"Validation $F$")
    axes[1].axhline(.95, color="0.35", lw=.8, ls="--")
    axes[1].set_ylim(0, 1.025)
    for ax in axes:
        ax.axvline(10, color="0.5", lw=.8, ls=":")
        ax.set_xlabel("Training slots (millions)")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(axis="y", alpha=.18)
    axes[0].legend(frameon=False, loc="upper right", ncol=2, fontsize=6)
    fig.tight_layout(pad=.5, w_pad=1.4)
    save(fig, "learning_n8")

    short = records["diagnosis_short"]["results"]
    long = records["diagnosis_confirm_a"]["results"]
    sac = records["diagnosis_sac_v2"]["results"]
    groups = [
        [short[f"A-s{s}"] for s in (101, 202, 303)],
        [long[f"fixed-w10-s{s}"] for s in (101, 202, 303)],
        [sac[f"v2-g099-s{s}"] for s in (101, 202, 303)],
        [long["paced_sticky:margin=0.5"]],
        [long["central-upper-bound"]],
    ]
    labels = ["F: 2M", "F: 20M", "SAC-V2", "Paced", "Central"]
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

    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Method & $\gamma$ & $F$ & Mean $E/U_0$ [95\% CI] \\", r"\midrule"]
    for i, (key, label, gamma) in enumerate(CONFIRM_METHODS):
        if i in (3, 6):
            lines.append(r"\midrule")
        row = confirm["methods"][key]
        suffix = r"$^\dagger$" if not row["satisfies_constraint"] else ""
        lines.append(f"{label}{suffix} & {gamma} & {row['feasible']['mean']:.3f} & "
                     + interval(row["e_over_ub"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "generated" / "confirm1_methods.tex").write_text("\n".join(lines)+"\n")

    macros = []
    for n, word in zip(NS, ("Four", "Six", "Eight", "Twelve")):
        row = test[str(n)]["methods"]["fixed-w10:selected"]
        ref = test[str(n)]["methods"][test[str(n)]["reference"]]
        saving = 100*(1-row["e_over_ub"]["mean"]/ref["e_over_ub"]["mean"])
        for stem, value, digits in [("Saving", saving, 1), ("Energy", row["e_over_ub"]["mean"], 3),
                                     ("Feasible", row["feasible"]["mean"], 3), ("Power", row["mw_per_ue"]["mean"], 2)]:
            macros.append(f"\\newcommand{{\\{stem}{word}}}{{{value:.{digits}f}}}")
    c1_f = confirm["methods"]["fixed-w10:selected"]
    c1_v2 = confirm["methods"]["v2-g099:selected"]
    c1_paced = confirm["methods"]["paced_sticky:margin=0.5"]
    for stem, value, digits in [
        ("ConfirmEnergy", c1_f["e_over_ub"]["mean"], 3),
        ("ConfirmFeasible", c1_f["feasible"]["mean"], 3),
        ("ConfirmSacEnergy", c1_v2["e_over_ub"]["mean"], 3),
        ("ConfirmSavingSac", 100*(1-c1_f["e_over_ub"]["mean"]/c1_v2["e_over_ub"]["mean"]), 1),
        ("ConfirmSavingPaced", 100*(1-c1_f["e_over_ub"]["mean"]/c1_paced["e_over_ub"]["mean"]), 1),
    ]:
        macros.append(f"\\newcommand{{\\{stem}}}{{{value:.{digits}f}}}")
    for stem, entry in [
        ("ConfirmGapSac", confirm["q2_main"]["e_over_ub"]),
        ("ConfirmGapVOne", confirm["q1_main"]["pairs"]["v1_minus_v2"]["e_over_ub"]),
        ("ConfirmGapVSparse", confirm["q1_main"]["pairs"]["v1s_minus_v2"]["e_over_ub"]),
    ]:
        macros.append(f"\\newcommand{{\\{stem}}}{{{interval(entry)}}}")
    (HERE / "generated" / "numbers.tex").write_text("\n".join(macros)+"\n")
    with (HERE / "data" / "test_results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["N", "method", "metric", "mean", "ci95_lower", "ci95_upper"])
        for n in NS:
            for key, row in test[str(n)]["methods"].items():
                for metric in ("feasible", "e_over_ub", "e_over_lb", "mw_per_ue"):
                    entry = row[metric]
                    writer.writerow([n, key, metric, entry["mean"], *entry["ci95"]])
    with (HERE / "data" / "confirm1_results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "N", "method", "satisfies_constraint", "metric", "mean", "ci95_lower", "ci95_upper"])
        for key, row in confirm["methods"].items():
            for metric in ("feasible", "e_over_ub", "e_over_lb", "mw_per_ue"):
                entry = row[metric]
                writer.writerow(["confirm1", 8, key, row["satisfies_constraint"], metric,
                                 entry["mean"], *entry["ci95"]])
    comparisons = {"training_scheme": confirm["q2_main"],
                   "mappo_f_minus_d": confirm["replication"]["a_minus_c"]}
    for analysis in ("q1_main", "q1_sensitivity"):
        comparisons.update({f"{analysis}:{key}": row for key, row in confirm[analysis]["pairs"].items()})
    with (HERE / "data" / "confirm1_comparisons.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", "a", "b", "energy_comparison_valid", "metric", "mean", "ci95_lower", "ci95_upper"])
        for name, row in comparisons.items():
            for metric in ("feasible", "e_over_ub"):
                writer.writerow([name, row["a"], row["b"], row["energy_read"], metric,
                                 row[metric]["mean"], *row[metric]["ci95"]])
    print("Generated three vector figures, two tables, macros, three CSVs, and a 13-source evidence snapshot.")


if __name__ == "__main__":
    main()
