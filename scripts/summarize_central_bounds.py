"""Aggregate central-bounds runs (one per N) into the compact summary cited by the report.

Energies are converted to mean power per UE over the evaluation window,
P = E_window / ((H - W) * tau * N), so different N are comparable.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

POLICIES = ("random_max", "sticky_max", "lazy_pacing")
POWER_DBM = (5, 10, 15, 19, 23)
DISTANCE_BINS = ((0, 50), (50, 75), (75, 100.0001))
SLOT_S = 1e-3
WINDOW_SLOTS = 1800


def _stats(values) -> dict | None:
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if x.size == 0:
        return None
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def summarize(path: Path) -> dict:
    raw = path.read_bytes()
    record = json.loads(raw)
    runs = record["runs"]
    num_ues = len(runs[0]["lower"]["per_ue_packet_energy_j"])
    scale = 1e3 / (WINDOW_SLOTS * SLOT_S * num_ues)  # J over window -> mW per UE
    solved = [r for r in runs if r["upper"] is not None]
    shares = np.mean(
        [[r["upper"]["slot_share_by_power_id"][str(i)] for i in range(1, 6)] for r in solved],
        axis=0,
    )
    # Per-UE cycle energy: lower bound from the solo plans, upper bound from the schedule.
    # Power ID 0 is OFF: exactly 0 W (numeric 0 dBm would be an active 1 mW).
    watts = np.concatenate(([0.0], 10 ** ((np.asarray(POWER_DBM, dtype=float) - 30) / 10)))
    distance, lower_ue, upper_ue, duty = [], [], [], []
    for r in solved:
        ids = np.asarray(r["upper"]["power_ids"])
        cycle = ids.shape[0]
        for ue in range(num_ues):
            period = r["traffic"]["period_slots"][ue]
            distance.append(r["topology"]["ue_ap_distances_m"][ue])
            lower_ue.append(r["lower"]["per_ue_packet_energy_j"][ue] * cycle / period)
            upper_ue.append(SLOT_S * watts[ids[:, ue]].sum())
            duty.append(float(np.mean(ids[:, ue] > 0)))
    distance, lower_ue, upper_ue = map(np.asarray, (distance, lower_ue, upper_ue))
    duty = np.asarray(duty)
    extra = upper_ue - lower_ue
    by_distance = []
    for lo, hi in DISTANCE_BINS:
        mask = (distance >= lo) & (distance < hi)
        by_distance.append(
            {
                "distance_m": [lo, min(hi, 100.0)],
                "ues": int(mask.sum()),
                "upper_over_lower": float(upper_ue[mask].sum() / lower_ue[mask].sum()),
                "share_of_extra_energy": float(extra[mask].sum() / extra.sum())
                if extra.sum() > 0
                else 0.0,
                "upper_duty_cycle": float(duty[mask].mean()),
            }
        )
    heuristics = {}
    for policy in POLICIES:
        h = [r["heuristics"][policy] for r in runs]
        heuristics[policy] = {
            "ocr_mean": float(np.mean([x["ocr"] for x in h])),
            "worst_ue_ocr_mean": float(np.mean([x["worst_ue_ocr"] for x in h])),
            "instances_worst_ue_below_095": int(sum(x["worst_ue_ocr"] < 0.95 for x in h)),
            "mw_per_ue": float(1e3 * np.mean([x["average_power_per_ue_w"] for x in h])),
            "energy_over_lower": _stats([x["energy_over_lower"] for x in h]),
            "energy_over_upper": _stats([x["energy_over_upper"] for x in h]),
        }
    return {
        "source": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "formal": record["formal"],
        "provenance": record["provenance"],
        "config": record["config"],
        "first_instance": record["first_instance"],
        "policy_seed": record["policy_seed"],
        "num_ues": num_ues,
        "instances": len(runs),
        "upper_solved": len(solved),
        "exact_instances": int(sum(r["exact"] for r in solved)),
        "all_checks_passed": all(all(r["upper"]["checks"].values()) for r in solved),
        "mip_status_counts": {
            str(s): sum(r["upper"]["mip_status"] == s for r in solved)
            for s in sorted({r["upper"]["mip_status"] for r in solved})
        },
        "mip_seconds": _stats([r["upper"]["solve_seconds"] for r in solved]),
        "early_completions": int(sum(r["upper"]["early_completions"] for r in solved)),
        "lower_mw_per_ue": _stats([scale * r["lower"]["strict_j"] for r in runs]),
        "upper_mw_per_ue": _stats([scale * r["upper"]["strict_j"] for r in solved]),
        "upper_over_lower": _stats([r["upper_over_lower"] for r in solved]),
        "capped_over_strict_lower": float(
            sum(r["lower"]["capped_j"] for r in runs) / sum(r["lower"]["strict_j"] for r in runs)
        ),
        "capped_over_strict_upper": float(
            sum(r["upper"]["capped_j"] for r in solved)
            / sum(r["upper"]["strict_j"] for r in solved)
        ),
        "upper_slot_utilization": _stats([r["upper"]["slot_utilization"] for r in solved]),
        "upper_slot_share_by_power_dbm": dict(
            zip(map(str, POWER_DBM), shares.tolist(), strict=True)
        ),
        "upper_vs_lower_by_distance": by_distance,
        "heuristics": heuristics,
        "per_instance": [
            {
                "instance": r["instance"],
                "lower_j": r["lower"]["strict_j"],
                "upper_j": None if r["upper"] is None else r["upper"]["strict_j"],
                "exact": r.get("exact"),
                "mip_gap": None if r["upper"] is None else r["upper"]["mip_gap"],
            }
            for r in runs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summaries = [summarize(path) for path in args.sources]
    for s in summaries:
        brief = {k: v for k, v in s.items() if k != "per_instance"}
        print(json.dumps(brief, indent=1))
    if args.output is not None:
        if args.output.exists():
            parser.error(f"output {args.output} already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"runs": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
