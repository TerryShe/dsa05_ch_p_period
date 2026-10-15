"""EXP-DSA05-012 analysis (design §2, §4).

``pilot``: stage 1 on exp012-pilot256 (dev 1400-1655, scenario v2.1, N = 8, 10,
12) from the formal ``evaluate_capacity`` outputs and the plan written by
``scripts/run_exp012.py``. Per method and N (learned methods average their seed
layers):

* Q_U: deployment mean of the worst UE's unavailability (primary, s = 1);
* Q_O: deployment mean of the worst UE's outage count;
* the all-UE mean unavailability, the share of deployments with an outage, P2,
  the all-UE mean violation rate and P(miss | previous packet missed);
* s = 2 as sensitivity; paired two-level bootstrap intervals of SC against every
  other method (descriptive: two seeds).

An optional group M-S+S (M-S with the M+S yield rule; design §10.4, descriptive)
is analysed when the plan lists it; it never enters gate A.

``pilot1b``: stage 1b after gate A = adjust (design §10.5): the same quantities with
SC-B at lambda_d = 0.1 and 0.3 added, each compared with every other method. The
lambda_d carried forward is, among the candidates with Q_U <= 0.8 x the best rule
(the three rules of gate A) and P2 <= 1.10 x P2(M+S) at N = 10 (point estimates),
the one with the lowest Q_U; if none qualifies, none is chosen. Both modes also
count misses, isolated misses and misses in runs of two or more.

``pilot1c``: stage 1c on exp012-pilot2-256 (dev 1656-1911; design revision 4,
§11.4): SP and SP-B (four seeds each) against M, M+S, M+Y, R17-like (on the
pilot's M seeds 101 and 202) and SC-B (descriptive). Gate A2 (N = 10, point
estimates), per variant: Q_U <= 0.8 x the best rule, P2 <= 1.10 x P2(M+S) and every
seed's P2 <= 1.25 x P2(M+S). Continue with the passing variant of lower Q_U; stop
iff both variants have Q_U >= Q_U(M+S); otherwise adjust.

Gate A (design §4.1, point estimates at N = 10): continue iff Q_U(SC) is at least
20% below the best rule (lowest Q_U of M+S, M+Y, R17-like) and below M-S, and
P2(SC) <= 1.10 P2(M+S); stop iff Q_U(SC) >= Q_U(M); otherwise adjust.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from dsa_period import splits_v2

ROOT = Path(__file__).parents[1]
PILOT_SET = "exp012-pilot256"
PILOT2_SET = "exp012-pilot2-256"
DRAWS = 10_000
RULES = ("M+S", "M+Y", "R17-like")
ORDER = ("SC", "SC-B", "M-S", "M", "M+S", "M+Y", "R17-like")
EXTRA = ("M-S+S",)  # descriptive only (design §10.4)
CANDIDATES = ("SC-B[0.1]", "SC-B[0.3]")  # stage 1b (design §10.5)
COUNTS = ("misses", "isolated_misses", "misses_in_runs")
SP_VARIANTS = ("SP", "SP-B")  # stage 1c (design §11.4)
ORDER_1C = ("SP", "SP-B", "M", "M+S", "M+Y", "R17-like", "SC-B")


def _exp011():
    spec = importlib.util.spec_from_file_location(
        "analyze_exp011", ROOT / "scripts/analyze_exp011.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _exp011()


def per_deployment(episode: dict, s: int) -> dict[str, float]:
    """Per-deployment quantities of one episode at survival time s."""
    record = episode["survival"][str(s)]
    unavailable = np.asarray(record["unavailable_slots"], dtype=float) / episode["survival_window"]
    outages = np.asarray(record["outages"], dtype=float)
    pairs = after = misses = isolated = in_runs = 0
    for missed, total in zip(episode["missed_packets"], episode["packets"]):
        seq = np.zeros(total, dtype=np.bool_)
        seq[np.asarray(missed, dtype=np.int64)] = True
        pairs += int(np.sum(seq[1:] & seq[:-1]))
        after += int(seq[:-1].sum())
        edges = np.diff(np.concatenate(([0], seq.astype(np.int8), [0])))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        misses += int(seq.sum())
        isolated += int(np.sum(lengths == 1))
        in_runs += int(lengths[lengths >= 2].sum())
    return {
        "worst_unavailability": float(unavailable.max()),
        "worst_outages": float(outages.max()),
        "mean_unavailability": float(unavailable.mean()),
        "any_outage": float(outages.sum() > 0),
        "w": float(episode["worst_violation_rate"]),
        "mean_rate": float(np.mean(episode["violation_rate"])),
        "pairs": float(pairs),
        "after_miss": float(after),
        "misses": float(misses),
        "isolated_misses": float(isolated),
        "misses_in_runs": float(in_runs),
    }


def matrix(episodes: dict, layers: list[dict], n: int, field: str, instances, s: int):
    """(seed layers, deployments) of one per-deployment quantity."""
    rows = []
    for layer in layers:
        by_instance = {e["instance"]: e for e in episodes[layer[str(n)]][n]}
        rows.append([per_deployment(by_instance[i], s)[field] for i in instances])
    return np.array(rows)


def pilot(
    paths: list[Path],
    plan: dict,
    focus: tuple[str, ...] = ("SC",),
    screen: str = PILOT_SET,
    order: tuple[str, ...] | None = None,
) -> dict:
    """Per-method quantities, and each method in ``focus`` against every other one."""
    episodes, instances = base.load(paths, screen=screen)
    groups, grid = plan["groups"], plan["grid"]
    names = order or ORDER + tuple(name for name in EXTRA + CANDIDATES if name in groups)
    out: dict = {"instances": len(instances), "grid": grid, "methods": {}}
    out |= {f"versus_{name}": {} for name in focus}
    for s in (1, 2):
        for name in names:
            for n in grid:
                values = {
                    field: matrix(episodes, groups[name], n, field, instances, s)
                    for field in ("worst_unavailability", "worst_outages", "mean_unavailability",
                                  "any_outage", "w", "mean_rate", "pairs", "after_miss", *COUNTS)
                }  # fmt: skip
                pairs, after = values.pop("pairs").sum(), values.pop("after_miss").sum()
                counts = {field: values.pop(field) for field in COUNTS}
                row = {k: float(v.mean()) for k, v in values.items()}
                row["p_miss_after_miss"] = float(pairs / after) if after else None
                # totals over the deployments, averaged over the seed layers
                row |= {field: float(v.sum(1).mean()) for field, v in counts.items()}
                row["by_seed_worst_unavailability"] = (
                    values["worst_unavailability"].mean(1).tolist()
                )
                row["by_seed_w"] = values["w"].mean(1).tolist()
                out["methods"].setdefault(f"s={s}", {}).setdefault(name, {})[str(n)] = row
        for target in focus:
            for name in names:
                if name == target:
                    continue
                for n in grid:
                    comparison = {}
                    for field in ("worst_unavailability", "worst_outages", "w"):
                        a = matrix(episodes, groups[target], n, field, instances, s)
                        b = matrix(episodes, groups[name], n, field, instances, s)
                        comparison[field] = base.compare(a, b, DRAWS)
                    versus = out[f"versus_{target}"].setdefault(f"s={s}", {})
                    versus.setdefault(name, {})[str(n)] = comparison
    return out


def gate_a(methods: dict, n: str = "10") -> dict:
    q = {name: methods[name][n]["worst_unavailability"] for name in ORDER}
    best_rule = min(RULES, key=lambda name: q[name])
    p2_sc, p2_ref = methods["SC"][n]["w"], methods["M+S"][n]["w"]
    checks = {
        "SC_vs_best_rule": q["SC"] <= 0.8 * q[best_rule],
        "SC_vs_M-S": q["SC"] <= 0.8 * q["M-S"],
        "P2_SC_vs_M+S": p2_sc <= 1.10 * p2_ref,
    }
    if q["SC"] >= q["M"]:
        decision = "stop"
    elif all(checks.values()):
        decision = "continue"
    else:
        decision = "adjust"
    return {
        "n": int(n),
        "Q_U": q,
        "best_rule": best_rule,
        "ratio_SC_to_best_rule": q["SC"] / q[best_rule] if q[best_rule] else None,
        "ratio_SC_to_M-S": q["SC"] / q["M-S"] if q["M-S"] else None,
        "ratio_P2_SC_to_M+S": p2_sc / p2_ref if p2_ref else None,
        "checks": checks,
        "decision": decision,
    }


def select_1b(methods: dict, n: str = "10") -> dict:
    """Stage 1b (design §10.5): the lambda_d carried forward (point estimates at N = 10)."""
    q = {name: methods[name][n]["worst_unavailability"] for name in methods}
    best_rule = min(RULES, key=lambda name: q[name])
    p2_ref = methods["M+S"][n]["w"]
    candidates = {}
    for name in CANDIDATES:
        p2 = methods[name][n]["w"]
        candidates[name] = {
            "Q_U": q[name],
            "ratio_to_best_rule": q[name] / q[best_rule] if q[best_rule] else None,
            "ratio_P2_to_M+S": p2 / p2_ref if p2_ref else None,
            "ratio_to_M-S": q[name] / q["M-S"] if q["M-S"] else None,  # descriptive
            "checks": {"Q_U_vs_best_rule": q[name] <= 0.8 * q[best_rule],
                       "P2_vs_M+S": p2 <= 1.10 * p2_ref},
        }  # fmt: skip
    eligible = [name for name in CANDIDATES if all(candidates[name]["checks"].values())]
    chosen = min(eligible, key=lambda name: q[name]) if eligible else None
    return {
        "n": int(n),
        "best_rule": best_rule,
        "candidates": candidates,
        "eligible": eligible,
        "chosen": chosen,
        "decision": "carry forward" if chosen else "none eligible",
    }


def gate_a2(methods: dict, n: str = "10") -> dict:
    """Gate A2 of stage 1c (design §11.4; point estimates at N = 10)."""
    q = {name: methods[name][n]["worst_unavailability"] for name in methods}
    best_rule = min(RULES, key=lambda name: q[name])
    p2_ref = methods["M+S"][n]["w"]
    variants = {}
    for name in SP_VARIANTS:
        row = methods[name][n]
        variants[name] = {
            "Q_U": q[name],
            "ratio_to_best_rule": q[name] / q[best_rule] if q[best_rule] else None,
            "ratio_to_M+S": q[name] / q["M+S"] if q["M+S"] else None,
            "ratio_P2_to_M+S": row["w"] / p2_ref if p2_ref else None,
            "ratio_P2_by_seed": [w / p2_ref for w in row["by_seed_w"]] if p2_ref else None,
            "checks": {
                "Q_U_vs_best_rule": q[name] <= 0.8 * q[best_rule],
                "P2_vs_M+S": row["w"] <= 1.10 * p2_ref,
                "P2_every_seed": all(w <= 1.25 * p2_ref for w in row["by_seed_w"]),
            },
        }
    passing = [name for name in SP_VARIANTS if all(variants[name]["checks"].values())]
    if passing:
        decision, carried = "continue", min(passing, key=lambda name: q[name])
    elif all(q[name] >= q["M+S"] for name in SP_VARIANTS):
        decision, carried = "stop", None
    else:
        decision, carried = "adjust", None
    return {
        "n": int(n),
        "best_rule": best_rule,
        "variants": variants,
        "passing": passing,
        "carried": carried,
        "decision": decision,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "pilot1b", "pilot1c"))
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    for path in args.evaluations:
        record = json.loads((path / "summary.json").read_text())
        if not record["formal"] or record["provenance"]["dirty"]:
            parser.error(f"{path} is not a formal evaluation on a clean tree")
        if record["config"] != "configs/scenario-v2.1.toml":
            parser.error(f"{path} did not run on scenario v2.1")
    plan = json.loads(args.plan.read_text())
    if args.mode == "pilot":
        result = pilot(args.evaluations, plan)
        result["gate_A"] = gate_a(result["methods"]["s=1"])
    elif args.mode == "pilot1b":
        result = pilot(args.evaluations, plan, CANDIDATES)
        result["selection_1b"] = select_1b(result["methods"]["s=1"])
    else:
        result = pilot(args.evaluations, plan, SP_VARIANTS, PILOT2_SET, ORDER_1C)
        result["gate_A2"] = gate_a2(result["methods"]["s=1"])
    result["protocol_id"] = splits_v2.PROTOCOL_ID
    args.output.write_text(json.dumps(result, indent=2))
    if args.mode == "pilot":
        gate = result["gate_A"]
        print(f"gate A (N = {gate['n']}): {gate['decision']}; Q_U {gate['Q_U']}; "
              f"best rule {gate['best_rule']}")  # fmt: skip
        print(f"checks {gate['checks']}")
    elif args.mode == "pilot1c":
        gate = result["gate_A2"]
        print(f"gate A2 (N = {gate['n']}): {gate['decision']}, carried {gate['carried']}; "
              f"best rule {gate['best_rule']}")  # fmt: skip
        for name, row in gate["variants"].items():
            print(f"{name}: Q_U {row['Q_U']:.6f}, x best rule {row['ratio_to_best_rule']:.3f}, "
                  f"P2 x M+S {row['ratio_P2_to_M+S']:.3f}, checks {row['checks']}")  # fmt: skip
    else:
        selection = result["selection_1b"]
        print(f"stage 1b (N = {selection['n']}): {selection['decision']}, chosen "
              f"{selection['chosen']}; best rule {selection['best_rule']}")  # fmt: skip
        for name, row in selection["candidates"].items():
            print(f"{name}: Q_U {row['Q_U']:.6f}, x best rule {row['ratio_to_best_rule']:.3f}, "
                  f"P2 x M+S {row['ratio_P2_to_M+S']:.3f}, checks {row['checks']}")  # fmt: skip


if __name__ == "__main__":
    main()
