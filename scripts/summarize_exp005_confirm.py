"""EXP-DSA05-005: the pre-registered reading (design §4) of the confirmation-set evaluation.

Reads one ``evaluate_marl --split confirm1`` record holding the 18 SAC runs,
EXP-DSA05-001 phase-2b A (fixed-w10) and C (local-eta2), and the local baselines,
plus the validation analysis of ``scripts/analyze_exp005.py`` (tuned gammas), and
applies §4:

1. constraint: C1 F >= 0.95;
2. saving against the reference paced_sticky (margin 0.5): as EXP-DSA05-004 §3.2;
3. Q1 main (gamma 0.99) and 4. Q1 sensitivity (gamma 0.5): V1 - V2, V1s - V2 and
   V1 - V1s, paired E/UB and F differences; an energy difference is read only
   when both methods satisfy the constraint (protocol §5);
5. tuned performance: each reward at its tuned gamma (descriptive);
6. Q2: tuned V2 - A (main) and gamma-0.99 V2 - A (supplement), read only when
   both satisfy the constraint;
7. Q3 is the V1 - V1s comparison of items 3 and 4;
8. replication of EXP-DSA05-004 on C1: A and C against the reference and A - C.

All comparisons use selected checkpoints; final checkpoints are listed only.
``--dry-run`` relaxes the formal/confirm1 checks to exercise the script on a
validation record; its output is marked and is never a result.
"""

import argparse
import hashlib
import json
from pathlib import Path

from dsa_period import splits
from dsa_period.evaluation import paired_difference

REFERENCE = "paced_sticky:margin=0.5"
BASELINES = (REFERENCE, "random_max", "sticky_max", "lazy_pacing")
A, C = "fixed-w10", "local-eta2"
REWARDS = ("v1", "v1s", "v2")
GAMMA_KEYS = {0.99: "g099", 0.5: "g05"}
KEEP = ("feasible", "e_over_ub", "e_over_lb", "worst_ue_ocr", "mw_per_ue")
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"


def method(profile: str, choice: str = "selected") -> str:
    return f"{profile}:{choice}"


def summary_of(result: dict) -> dict:
    s = result["summary"]
    return {
        "training_seeds": s["training_seeds"],
        "satisfies_constraint": s["satisfies_constraint"],
        "per_ue_violation_rate": s["per_ue_violation_rate"],
        **{m: {k: s["metrics"][m][k] for k in ("mean", "ci95", "per_seed")} for m in KEEP},
        **({"sources": result["sources"]} if "sources" in result else {}),
    }


def compare(results: dict, a: str, b: str) -> dict:
    """Paired E/UB and F differences a - b; the energy difference is read only when
    both satisfy the constraint (protocol §5)."""
    ea, eb = results[a]["episodes"], results[b]["episodes"]
    energy = paired_difference(ea, eb, "e_over_ub")
    feasible = paired_difference(ea, eb, "feasible")
    both = all(results[x]["summary"]["satisfies_constraint"] for x in (a, b))
    low, high = energy["ci95"]
    verdict = None
    if both:
        verdict = "a_better" if high < 0 else "b_better" if low > 0 else "no_significant_difference"
    return {
        "a": a,
        "b": b,
        "e_over_ub": energy,
        "feasible": feasible,
        "energy_read": both,
        "verdict": verdict,
    }


def reward_effect(results: dict, gamma: float) -> dict:
    g = GAMMA_KEYS[gamma]
    v1, v1s, v2 = (method(f"{r}-{g}") for r in REWARDS)
    pairs = {
        "v1_minus_v2": compare(results, v1, v2),
        "v1s_minus_v2": compare(results, v1s, v2),
        "v1_minus_v1s": compare(results, v1, v1s),
    }
    a, b = pairs["v1_minus_v2"]["verdict"], pairs["v1s_minus_v2"]["verdict"]
    if a is None or b is None:
        attribution = "not_read: a method violates the constraint"
    elif a == b and a != "no_significant_difference":
        attribution = (
            f"consistent: both reliability rewards {'beat' if a == 'a_better' else 'lose to'} v2"
        )
    else:
        attribution = "depends_on_shaping_or_not_significant"
    return {
        "gamma": gamma,
        "feasible": {
            r: results[method(f"{r}-{g}")]["summary"]["satisfies_constraint"] for r in REWARDS
        },
        "pairs": pairs,
        "attribution": attribution,
    }


def check(record: dict, dry_run: bool) -> None:
    if dry_run:
        return
    if not record["formal"] or record["provenance"]["dirty"] or record["split"] != "confirm1":
        raise SystemExit("not a formal clean confirm1 evaluation")
    if record["instances"] != list(splits.indices("confirm1")):
        raise SystemExit("the record does not cover the whole confirmation set")
    if record["reference"] != REFERENCE or record["checkpoints"] != "both":
        raise SystemExit(f"expected reference {REFERENCE} and both checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="evaluate_marl output on confirm1")
    parser.add_argument("--analysis", type=Path, required=True, help="analyze_exp005 output")
    parser.add_argument("--exp004", type=Path, help="EXP-DSA05-004 summary.json, for comparison")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for path in (args.output, args.figure):
        if path is not None and path.exists():
            parser.error(f"output {path} already exists")
    raw = args.record.read_bytes()
    record = json.loads(raw)
    check(record, args.dry_run)
    results = record["results"]
    expected = [
        method(f"{r}-{g}", c)
        for r in REWARDS
        for g in GAMMA_KEYS.values()
        for c in ("selected", "final")
    ]
    expected += [method(p, c) for p in (A, C) for c in ("selected", "final")] + list(BASELINES)
    missing = sorted(set(expected) - set(results))
    if missing:
        raise SystemExit(f"missing methods {missing}")
    for name in expected:
        seeds = len(results[name]["summary"]["training_seeds"])
        if seeds != (1 if name in BASELINES else 3):
            raise SystemExit(f"{name} has {seeds} training seeds")
    analysis = json.loads(args.analysis.read_text())
    tuned = {r: analysis["tuned"][r]["gamma"] for r in REWARDS}

    def saving(name: str) -> dict:
        if not results[name]["summary"]["satisfies_constraint"]:
            return {"read": False, "reason": "method violates the constraint"}
        if not results[REFERENCE]["summary"]["satisfies_constraint"]:
            return {"read": False, "reason": "reference violates the constraint"}
        upper = results[name]["paired_vs_reference"]["e_over_ub"]["ci95"][1]
        return {"read": True, "significant": upper < 0}

    methods = {name: summary_of(results[name]) for name in expected}
    for name in expected:
        if name != REFERENCE:
            methods[name]["paired_vs_reference"] = results[name]["paired_vs_reference"]
            methods[name]["saving_vs_reference"] = saving(name)
    tuned_v2 = method(f"v2-{GAMMA_KEYS[tuned['v2']]}")
    out = {
        "source": str(args.record),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "dry_run": args.dry_run,
        "provenance": record["provenance"],
        "split": record["split"],
        "instances": len(record["instances"]),
        "tuned_gamma": tuned,
        "methods": methods,
        "q1_main": reward_effect(results, 0.99),
        "q1_sensitivity": reward_effect(results, 0.5),
        "tuned_performance": {
            r: {
                "method": method(f"{r}-{GAMMA_KEYS[tuned[r]]}"),
                **{
                    m: methods[method(f"{r}-{GAMMA_KEYS[tuned[r]]}")][m]
                    for m in ("feasible", "e_over_ub")
                },
            }
            for r in REWARDS
        },
        "q2_main": compare(results, tuned_v2, method(A)),
        "q2_supplement": compare(results, method("v2-g099"), method(A)),
        "replication": {
            "a_minus_c": compare(results, method(A), method(C)),
            "a_saving": saving(method(A)),
            "c_saving": saving(method(C)),
        },
    }
    if args.exp004 is not None:
        n8 = json.loads(args.exp004.read_text())["long"]["8"]["methods"]
        out["exp004_test_n8"] = {
            name: {m: n8[name][m]["mean"] for m in ("feasible", "e_over_ub")}
            for name in (method(A), method(C), REFERENCE)
        }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
    if args.figure is not None:
        plot(out, args.figure)
    report(out)


def fmt_pair(p: dict) -> str:
    e, f = p["e_over_ub"], p["feasible"]
    return (
        f"dE/UB {e['mean']:+.3f} [{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}]  "
        f"dF {f['mean']:+.3f} [{f['ci95'][0]:+.3f},{f['ci95'][1]:+.3f}]  "
        f"{p['verdict'] if p['energy_read'] else 'energy not read'}"
    )


def report(out: dict) -> None:
    if out["dry_run"]:
        print("DRY RUN: not a result")
    print(f"split {out['split']}, {out['instances']} instances; tuned gamma {out['tuned_gamma']}")
    for name, m in out["methods"].items():
        f, e = m["feasible"], m["e_over_ub"]
        print(
            f"  {name:<24} F {f['mean']:.3f} [{f['ci95'][0]:.3f},{f['ci95'][1]:.3f}]  "
            f"E/UB {e['mean']:.3f} [{e['ci95'][0]:.3f},{e['ci95'][1]:.3f}]  "
            f"{m.get('saving_vs_reference', '')}"
        )
    for key in ("q1_main", "q1_sensitivity"):
        q = out[key]
        print(f"{key} (gamma {q['gamma']}): feasible {q['feasible']} -> {q['attribution']}")
        for pair, p in q["pairs"].items():
            print(f"  {pair:<14} {fmt_pair(p)}")
    for key in ("q2_main", "q2_supplement"):
        print(f"{key}: {out[key]['a']} - {out[key]['b']}: {fmt_pair(out[key])}")
    rep = out["replication"]
    print(
        f"replication A - C: {fmt_pair(rep['a_minus_c'])}; A {rep['a_saving']}; C {rep['c_saving']}"
    )


def plot(out: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        (REFERENCE, "paced_sticky", MUTED, "o"),
        (method(A), "A (MAPPO)", SERIES[0], "o"),
        (method(C), "C (MAPPO, duals)", SERIES[0], "s"),
        (method("v1-g05"), "SAC v1, g=0.5", SERIES[1], "o"),
        (method("v1-g099"), "SAC v1, g=0.99", SERIES[1], "o"),
        (method("v1s-g05"), "SAC v1s, g=0.5", SERIES[3], "o"),
        (method("v1s-g099"), "SAC v1s, g=0.99", SERIES[3], "o"),
        (method("v2-g05"), "SAC v2, g=0.5", SERIES[2], "o"),
        (method("v2-g099"), "SAC v2, g=0.99", SERIES[2], "o"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, metric, label in zip(
        axes, ("feasible", "e_over_ub"), ("C1 feasibility F", "C1 E / UB"), strict=True
    ):
        for y, (name, _, color, marker) in enumerate(rows):
            m = out["methods"][name][metric]
            ax.errorbar(
                m["mean"],
                y,
                xerr=[[m["mean"] - m["ci95"][0]], [m["ci95"][1] - m["mean"]]],
                fmt=marker,
                color=color,
                markersize=6,
                capsize=3,
                markerfacecolor=color if out["methods"][name]["satisfies_constraint"] else SURFACE,
            )
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel(label, color=MUTED, fontsize=9)
    axes[0].set_yticks(range(len(rows)), [r[1] for r in rows])
    axes[0].invert_yaxis()
    axes[0].axvline(0.95, color=MUTED, linestyle=":", linewidth=1)
    axes[1].axvline(1.0, color=MUTED, linestyle=":", linewidth=1)
    fig.suptitle(
        "Confirmation set C1 (N=8, 200 instances x 2 runs), selected checkpoints; "
        "hollow marker = violates the constraint",
        color=INK,
        fontsize=10,
    )
    fig.savefig(path, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
