"""Aggregate a central-reference run into the compact summary cited by its report.

Reads the runner's summary.json, prints the report tables and writes a compact
JSON (aggregates plus per-instance reference values) with the source SHA-256.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

POLICIES = ("random_max", "sticky_max", "lazy_pacing")
DISTANCE_BINS = ((0, 50), (50, 75), (75, 100.0001))
LEVELS_DBM = (5, 10, 15, 19, 23)


def _stats(values) -> dict:
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def summarize(record: dict) -> dict:
    runs = record["runs"]
    plans = [plan for run in runs for plan in run["plans"]]
    distance = np.array([p["distance_m"] for p in plans])
    period = np.array([p["period_slots"] for p in plans])
    energy = np.array([p["energy_j"] for p in plans])
    bound = np.array([p["continuous_bound_j"] for p in plans])
    counts = np.array([[p["counts_by_power_id"][str(i)] for i in range(1, 6)] for p in plans])
    slot = 1e-3
    optimal_power = energy / (period * slot)
    witness_power = np.array(
        [row["average_power_w"] for run in runs for row in run["witness"]["summary"]["per_ue"]]
    )
    heuristic_power = {
        policy: np.array(
            [w for run in runs for w in run["heuristics"][policy]["per_ue_average_power_w"]]
        )
        for policy in POLICIES
    }
    by_distance = []
    for lo, hi in DISTANCE_BINS:
        mask = (distance >= lo) & (distance < hi)
        by_distance.append(
            {
                "distance_m": [lo, min(hi, 100.0)],
                "ues": int(mask.sum()),
                "optimal_energy_share": float(energy[mask].sum() / energy.sum()),
                "optimal_mw_per_ue": float(1e3 * witness_power[mask].mean()),
                **{
                    f"{policy}_mw_per_ue": float(1e3 * heuristic_power[policy][mask].mean())
                    for policy in POLICIES
                },
            }
        )
    by_class = {}
    for p in sorted(set(period.tolist())):
        mask = period == p
        by_class[str(p)] = {
            "ues": int(mask.sum()),
            "optimal_energy_per_packet_uj": float(1e6 * energy[mask].mean()),
            "mean_duty_cycle": float((counts[mask].sum(1) / p).mean()),
        }
    reference = [run["reference"] for run in runs]
    heuristics = {}
    for policy in POLICIES:
        h = [run["heuristics"][policy] for run in runs]
        heuristics[policy] = {
            "ocr_mean": float(np.mean([x["ocr"] for x in h])),
            "worst_ue_ocr_mean": float(np.mean([x["worst_ue_ocr"] for x in h])),
            "worst_ue_ocr_min": float(np.min([x["worst_ue_ocr"] for x in h])),
            "instances_worst_ue_below_095": int(sum(x["worst_ue_ocr"] < 0.95 for x in h)),
            "mw_per_ue_mean": float(1e3 * np.mean([x["average_power_per_ue_w"] for x in h])),
            "energy_ratio_to_strict_reference": _stats(
                [x["energy_ratio_to_strict_reference"] for x in h]
            ),
        }
    return {
        "instances": len(runs),
        "window_packets": int(sum(run["window_packets"] for run in runs)),
        "witness_checks_passed": all(all(run["witness"]["checks"].values()) for run in runs),
        "witness_ocr_all_one": all(run["witness"]["summary"]["ocr"] == 1.0 for run in runs),
        "reference": {
            "energy_per_packet_uj": _stats(
                [1e6 * x["strict_energy_per_packet_j"] for x in reference]
            ),
            "mw_per_ue_mean": float(
                1e3 * np.mean([x["average_power_per_ue_w"] for x in reference])
            ),
            "capped_over_strict": float(
                sum(x["capped_j"] for x in reference) / sum(x["strict_j"] for x in reference)
            ),
            "slot_share_by_level_dbm": dict(
                zip(map(str, LEVELS_DBM), (counts.sum(0) / counts.sum()).tolist(), strict=True)
            ),
            "ue_plans_using_15dbm_or_more": int((counts[:, 2:].sum(1) > 0).sum()),
            "ue_plans": len(plans),
            "duty_cycle": _stats(counts.sum(1) / period),
            "discrete_over_continuous_energy_weighted": float(energy.sum() / bound.sum()),
            "discrete_over_continuous_per_ue": _stats(energy / bound),
            "long_run_vs_window_power_max_rel_diff": float(
                np.max(np.abs(optimal_power - witness_power) / witness_power)
            ),
        },
        "by_distance": by_distance,
        "by_period_slots": by_class,
        "heuristics": heuristics,
        "per_instance": [
            {
                "instance": run["instance"],
                "strict_j": run["reference"]["strict_j"],
                "capped_j": run["reference"]["capped_j"],
                "mw_per_ue": 1e3 * run["reference"]["average_power_per_ue_w"],
                **{
                    f"{policy}_energy_ratio": run["heuristics"][policy][
                        "energy_ratio_to_strict_reference"
                    ]
                    for policy in POLICIES
                },
            }
            for run in runs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    record = json.loads(raw)
    summary = {
        "source": str(args.source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "formal": record["formal"],
        "provenance": record["provenance"],
        "config": record["config"],
        "first_instance": record["first_instance"],
        "policy_seed": record["policy_seed"],
        **summarize(record),
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_instance"}, indent=2))
    if args.output is not None:
        if args.output.exists():
            parser.error(f"output {args.output} already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
