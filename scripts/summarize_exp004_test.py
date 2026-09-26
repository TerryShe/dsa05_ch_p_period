"""EXP-DSA05-004: summary of the test evaluation of the frozen MARL runs.

Reads the ``evaluate_marl`` records (one per N for the long runs, one for the
N=8 short runs), checks that each is a formal test evaluation on a clean tree
covering the whole split, and applies the reading rules written in the plan's §3:

1. a method satisfies the constraint iff its test F >= 0.95;
2. among methods that satisfy it, a method saves energy significantly iff the
   upper end of the 95% interval of its paired E/UB difference to the reference
   baseline is < 0 (not read at N=4, where the reference is exactly optimal);
3. A vs C: paired E/UB difference, "no significant difference" if the interval
   contains 0;
4. gap to the optimum: mean E/UB and E/LB with intervals;
5. the short-training B: test F < 0.95 confirms its validation failure.

Writes a compact JSON (no per-episode data) and, if requested, a figure.
"""

import argparse
import hashlib
import json
from pathlib import Path

from dsa_period import splits
from dsa_period.config import load_scenario
from dsa_period.evaluation import FEASIBLE_EPISODE_RATE, paired_difference

KEEP = ("feasible", "e_over_ub", "e_over_lb", "worst_ue_ocr", "ocr", "mw_per_ue")
# Reference baseline per N: paced_sticky as selected by the validation calibration.
REFERENCE = {
    4: "paced_sticky:margin=0",
    6: "paced_sticky:margin=0.25",
    8: "paced_sticky:margin=0.5",
    12: "paced_sticky:margin=8",
}
LONG = {"A": "fixed-w10", "C": "local-eta2"}
SHORT = {"A": "fixed-w10", "B": "global-eta20", "C": "local-eta2"}


def load(path: Path) -> tuple[dict, str, int]:
    """A formal, clean, whole-split test record, its SHA-256 and its N."""
    raw = path.read_bytes()
    record = json.loads(raw)
    if not record["formal"] or record["provenance"]["dirty"] or record["split"] != "test":
        raise SystemExit(f"{path} is not a formal clean test evaluation")
    if record["instances"] != list(splits.indices("test")):
        raise SystemExit(f"{path} does not cover the whole test split")
    num_ues = load_scenario(record["scenario"]).physical.num_ues
    return record, hashlib.sha256(raw).hexdigest(), num_ues


def method_summary(result: dict) -> dict:
    summary = result["summary"]
    out = {
        "training_seeds": summary["training_seeds"],
        "episodes": summary["episodes"],
        "satisfies_constraint": summary["satisfies_constraint"],
        "per_ue_violation_rate": summary["per_ue_violation_rate"],
        **{
            metric: {key: summary["metrics"][metric][key] for key in ("mean", "ci95", "per_seed")}
            for metric in KEEP
            if metric in summary["metrics"]
        },
    }
    if "paired_vs_reference" in result:
        out["paired_vs_reference"] = result["paired_vs_reference"]
    if "sources" in result:
        out["sources"] = result["sources"]
    return out


def savings(method: dict, reference: dict, num_ues: int) -> dict:
    """Rule 2: significant energy saving against the reference baseline."""
    if num_ues == 4:
        return {"read": False, "reason": "reference baseline is exactly optimal at N=4"}
    if not method["satisfies_constraint"]:
        return {"read": False, "reason": "method violates the constraint"}
    if not reference["satisfies_constraint"]:
        return {"read": False, "reason": "reference baseline violates the constraint"}
    upper = method["paired_vs_reference"]["e_over_ub"]["ci95"][1]
    return {"read": True, "significant": upper < 0}


def compare(a: dict, b: dict, metric: str = "e_over_ub") -> dict:
    """Paired difference mean(a) - mean(b) and whether its interval excludes 0."""
    diff = paired_difference(a["episodes"], b["episodes"], metric)
    low, high = diff["ci95"]
    return diff | {"significant": high < 0 or low > 0}


def summarize_long(path: Path) -> tuple[int, dict, dict]:
    record, sha, num_ues = load(path)
    results = record["results"]
    reference = REFERENCE[num_ues]
    if record["reference"] != reference or record["checkpoints"] != "both":
        raise SystemExit(f"{path}: expected reference {reference} and both checkpoints")
    methods = {name: method_summary(result) for name, result in results.items()}
    judged = {}
    for label, profile in LONG.items():
        for choice in ("selected", "final"):
            name = f"{profile}:{choice}"
            if len(methods[name]["training_seeds"]) != 3:
                raise SystemExit(f"{path}: {name} does not have three training seeds")
            judged[f"{label}:{choice}"] = {
                "method": name,
                "satisfies_constraint": methods[name]["satisfies_constraint"],
                "savings_vs_reference": savings(methods[name], methods[reference], num_ues),
            }
    a_vs_c = {
        choice: compare(results[f"{LONG['A']}:{choice}"], results[f"{LONG['C']}:{choice}"])
        for choice in ("selected", "final")
    }
    summary = {
        "source": str(path),
        "source_sha256": sha,
        "provenance": record["provenance"],
        "scenario": record["scenario"],
        "scenario_sha256": record["scenario_sha256"],
        "instances": len(record["instances"]),
        "reference": reference,
        "methods": methods,
        "judgments": judged,
        "a_minus_c_e_over_ub": a_vs_c,
    }
    return num_ues, summary, results


def summarize_short(path: Path, long_n8: dict | None) -> dict:
    record, sha, num_ues = load(path)
    if num_ues != 8 or record["checkpoints"] != "selected":
        raise SystemExit(f"{path}: expected the N=8 short runs with selected checkpoints")
    results = record["results"]
    reference = REFERENCE[8]
    methods = {name: method_summary(result) for name, result in results.items()}
    judged = {
        label: {
            "method": f"{profile}:selected",
            "satisfies_constraint": methods[f"{profile}:selected"]["satisfies_constraint"],
            "savings_vs_reference": savings(methods[f"{profile}:selected"], methods[reference], 8),
        }
        for label, profile in SHORT.items()
    }
    out = {
        "source": str(path),
        "source_sha256": sha,
        "provenance": record["provenance"],
        "instances": len(record["instances"]),
        "reference": reference,
        "methods": methods,
        "judgments": judged,
        "b_failure_confirmed": not judged["B"]["satisfies_constraint"],
    }
    if long_n8 is not None:
        # Baselines are deterministic: both calls must produce identical episodes.
        out["reference_matches_long_call"] = (
            results[reference]["episodes"] == long_n8[reference]["episodes"]
        )
        out["long_minus_short_e_over_ub"] = {
            label: compare(long_n8[f"{LONG[label]}:selected"], results[f"{SHORT[label]}:selected"])
            for label in ("A", "C")
        }
    return out


# Reference categorical palette (dataviz skill), same slots as the EXP-001 figures:
# A fixed penalty = slot 0, C local duals = slot 2; the baseline is a neutral reference.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"


def plot(long: dict[int, dict], path: Path) -> None:
    """Test F and E/UB per N: reference baseline, A and C (selected checkpoints)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted(long)
    rows = [
        ("reference paced_sticky", lambda n: REFERENCE[n], MUTED, "s", "--"),
        ("A fixed penalty (w=10)", lambda n: f"{LONG['A']}:selected", SERIES[0], "o", "-"),
        ("C local duals (eta=2)", lambda n: f"{LONG['C']}:selected", SERIES[2], "D", "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for offset, (label, name, color, marker, style) in zip((-0.12, 0.0, 0.12), rows, strict=True):
        for ax, metric in zip(axes, ("feasible", "e_over_ub"), strict=True):
            values = [long[n]["methods"][name(n)][metric] for n in ns]
            x = [i + offset for i in range(len(ns))]
            mean = [v["mean"] for v in values]
            err = [
                [m - v["ci95"][0] for m, v in zip(mean, values, strict=True)],
                [v["ci95"][1] - m for m, v in zip(mean, values, strict=True)],
            ]
            ax.errorbar(
                x,
                mean,
                yerr=err,
                color=color,
                marker=marker,
                markersize=6,
                linestyle=style,
                linewidth=1.5,
                capsize=3,
                label=label,
            )
    for ax, ylabel in zip(axes, ("test feasibility F", "test E / UB"), strict=True):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xticks(range(len(ns)), [f"N={n}" for n in ns])
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    axes[0].axhline(FEASIBLE_EPISODE_RATE, color=MUTED, linestyle=":", linewidth=1)
    axes[0].annotate(
        "F = 0.95",
        (0.02, FEASIBLE_EPISODE_RATE),
        xycoords=("axes fraction", "data"),
        xytext=(0, -10),
        textcoords="offset points",
        color=MUTED,
        fontsize=7,
    )
    axes[1].axhline(1.0, color=MUTED, linestyle=":", linewidth=1)
    axes[1].annotate(
        "central upper bound",
        (0.98, 1.0),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        color=MUTED,
        fontsize=7,
        ha="right",
    )
    axes[1].legend(fontsize=8, frameon=False, labelcolor=INK, loc="upper left")
    fig.suptitle(
        "Test split (200 instances x 2 runs): mean over training seeds, "
        "two-level bootstrap 95% intervals",
        color=INK,
        fontsize=10,
    )
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("long", type=Path, nargs="+", help="one long-run record per N")
    parser.add_argument("--short", type=Path, help="N=8 short-run record")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    for path in (args.output, args.figure):
        if path is not None and path.exists():
            parser.error(f"output {path} already exists")
    long, raw = {}, {}
    for path in args.long:
        n, summary, results = summarize_long(path)
        if n in long:
            parser.error(f"two long-run records for N={n}")
        long[n], raw[n] = summary, results
    record = {"long": {str(n): long[n] for n in sorted(long)}}
    if args.short is not None:
        record["short_n8"] = summarize_short(args.short, raw.get(8))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    if args.figure is not None:
        plot(long, args.figure)

    def row(name: str, m: dict) -> str:
        f, e, lb = m["feasible"], m["e_over_ub"], m["e_over_lb"]
        pair = m.get("paired_vs_reference", {}).get("e_over_ub")
        pair_text = (
            f"{pair['mean']:+.3f} [{pair['ci95'][0]:+.3f},{pair['ci95'][1]:+.3f}]" if pair else "-"
        )
        return (
            f"  {name:<26} F {f['mean']:.3f} [{f['ci95'][0]:.3f},{f['ci95'][1]:.3f}]"
            f"  E/UB {e['mean']:.3f} [{e['ci95'][0]:.3f},{e['ci95'][1]:.3f}]"
            f"  E/LB {lb['mean']:.3f}  vs ref {pair_text}"
        )

    for n in sorted(long):
        print(f"N={n} (reference {long[n]['reference']})")
        for name, m in long[n]["methods"].items():
            print(row(name, m))
        for key, j in long[n]["judgments"].items():
            print(
                f"  {key:<11} constraint {j['satisfies_constraint']}  {j['savings_vs_reference']}"
            )
        for choice, d in long[n]["a_minus_c_e_over_ub"].items():
            print(f"  A-C {choice:<8} {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")
    if "short_n8" in record:
        short = record["short_n8"]
        print("N=8 short training")
        for name, m in short["methods"].items():
            print(row(name, m))
        print(f"  B failure confirmed: {short['b_failure_confirmed']}")
        if "reference_matches_long_call" in short:
            print(
                f"  reference episodes identical to the long call: "
                f"{short['reference_matches_long_call']}"
            )
            for label, d in short["long_minus_short_e_over_ub"].items():
                print(
                    f"  long-short {label} {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]"
                )


if __name__ == "__main__":
    main()
