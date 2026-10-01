"""Per-N calibration of the local rules of scenario v2.0 (protocol v2 §5).

Reads the validation outputs of ``evaluate_capacity`` (one or several runs,
which together cover every spec of ``rules_v2.CALIBRATION_TABLE`` on the same
instances and grid) and selects, for each rule family and each N, one parameter
by ``capacity.select_per_n``. The selection is written to ``calibration.json``
and frozen before any test evaluation.
"""

import argparse
import json
from pathlib import Path

from dsa_period.capacity import EPSILONS, capacity, capacity_label, eps_key, select_per_n
from dsa_period.rules_v2 import CALIBRATION_TABLE


def merge(records: list[dict]) -> dict:
    """One record from several runs on the same split, instances and grid."""
    first = records[0]
    for record in records[1:]:
        for key in ("protocol_id", "split", "instances", "grid", "config", "formal"):
            if record[key] != first[key]:
                raise ValueError(f"runs differ in {key}")
    summaries: dict = {}
    for record in records:
        overlap = set(summaries) & set(record["summaries"])
        if overlap:
            raise ValueError(f"specs evaluated twice: {sorted(overlap)}")
        summaries.update(record["summaries"])
    return {
        **{key: first[key] for key in ("protocol_id", "split", "instances", "grid", "config")},
        "formal": all(r["formal"] for r in records),
        "provenance": [r["provenance"] for r in records],
        "summaries": summaries,
    }


def calibrate(record: dict) -> dict:
    """Selections per family and N, the calibrated F(N) and N* of each family."""
    if record["split"] != "validation":
        raise ValueError("calibration reads the validation split only")
    summaries = record["summaries"]
    missing = [s for specs in CALIBRATION_TABLE.values() for s in specs if s not in summaries]
    if missing:
        raise ValueError(f"missing specs: {missing}")
    families = {}
    for family, specs in CALIBRATION_TABLE.items():
        chosen = select_per_n(summaries, specs)
        grid = sorted(chosen)
        families[family] = {
            "per_n": {str(n): chosen[n] for n in grid},
            "capacity": {
                eps_key(eps): capacity(
                    grid, [chosen[n]["feasible_share"][eps_key(eps)] for n in grid]
                )
                for eps in EPSILONS
            },
        }
    return families


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation", type=Path, nargs="+", required=True, help="evaluate_capacity outputs"
    )
    parser.add_argument("--output", type=Path, required=True, help="calibration.json to write")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    record = merge([json.loads((path / "summary.json").read_text()) for path in args.evaluation])
    families = calibrate(record)
    grid = sorted(int(n) for n in next(iter(families.values()))["per_n"])
    print(f"validation calibration, {len(record['instances'])} instances, grid {grid}")
    for family, result in families.items():
        stars = {k: capacity_label(v, grid) for k, v in result["capacity"].items()}
        print(f"  {family:<18} N* {stars}")
        for n in grid:
            row = result["per_n"][str(n)]
            shares = " ".join(f"{k}:{v:.2f}" for k, v in row["feasible_share"].items())
            print(f"    N={n:<3} {row['spec']:<40} F {shares}")
    output = {
        "protocol_id": record["protocol_id"],
        "formal": record["formal"],
        "evaluation_provenance": record["provenance"],
        "evaluation_config": record["config"],
        "split": record["split"],
        "instances": record["instances"],
        "grid": grid,
        "table": {family: list(specs) for family, specs in CALIBRATION_TABLE.items()},
        "families": families,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
