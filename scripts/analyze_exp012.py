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
DRAWS = 10_000
RULES = ("M+S", "M+Y", "R17-like")
ORDER = ("SC", "SC-B", "M-S", "M", "M+S", "M+Y", "R17-like")
EXTRA = ("M-S+S",)  # descriptive only (design §10.4)


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
    pairs = after = 0
    for missed, total in zip(episode["missed_packets"], episode["packets"]):
        seq = np.zeros(total, dtype=np.bool_)
        seq[np.asarray(missed, dtype=np.int64)] = True
        pairs += int(np.sum(seq[1:] & seq[:-1]))
        after += int(seq[:-1].sum())
    return {
        "worst_unavailability": float(unavailable.max()),
        "worst_outages": float(outages.max()),
        "mean_unavailability": float(unavailable.mean()),
        "any_outage": float(outages.sum() > 0),
        "w": float(episode["worst_violation_rate"]),
        "mean_rate": float(np.mean(episode["violation_rate"])),
        "pairs": float(pairs),
        "after_miss": float(after),
    }


def matrix(episodes: dict, layers: list[dict], n: int, field: str, instances, s: int):
    """(seed layers, deployments) of one per-deployment quantity."""
    rows = []
    for layer in layers:
        by_instance = {e["instance"]: e for e in episodes[layer[str(n)]][n]}
        rows.append([per_deployment(by_instance[i], s)[field] for i in instances])
    return np.array(rows)


def pilot(paths: list[Path], plan: dict) -> dict:
    episodes, instances = base.load(paths, screen=PILOT_SET)
    groups, grid = plan["groups"], plan["grid"]
    names = ORDER + tuple(name for name in EXTRA if name in groups)
    out: dict = {"instances": len(instances), "grid": grid, "methods": {}, "versus_SC": {}}
    for s in (1, 2):
        for name in names:
            for n in grid:
                values = {
                    field: matrix(episodes, groups[name], n, field, instances, s)
                    for field in ("worst_unavailability", "worst_outages", "mean_unavailability",
                                  "any_outage", "w", "mean_rate", "pairs", "after_miss")
                }  # fmt: skip
                pairs, after = values.pop("pairs").sum(), values.pop("after_miss").sum()
                row = {k: float(v.mean()) for k, v in values.items()}
                row["p_miss_after_miss"] = float(pairs / after) if after else None
                row["by_seed_worst_unavailability"] = (
                    values["worst_unavailability"].mean(1).tolist()
                )
                out["methods"].setdefault(f"s={s}", {}).setdefault(name, {})[str(n)] = row
        for name in names[1:]:
            for n in grid:
                comparison = {}
                for field in ("worst_unavailability", "worst_outages", "w"):
                    a = matrix(episodes, groups["SC"], n, field, instances, s)
                    b = matrix(episodes, groups[name], n, field, instances, s)
                    comparison[field] = base.compare(a, b, DRAWS)
                out["versus_SC"].setdefault(f"s={s}", {}).setdefault(name, {})[str(n)] = comparison
    out["gate_A"] = gate_a(out["methods"]["s=1"])
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot",))
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
    result = pilot(args.evaluations, json.loads(args.plan.read_text()))
    result["protocol_id"] = splits_v2.PROTOCOL_ID
    args.output.write_text(json.dumps(result, indent=2))
    gate = result["gate_A"]
    print(
        f"gate A (N = {gate['n']}): {gate['decision']}; Q_U {gate['Q_U']}; best rule {gate['best_rule']}"
    )
    print(f"checks {gate['checks']}")


if __name__ == "__main__":
    main()
