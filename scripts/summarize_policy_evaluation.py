"""Compact summary of ``evaluate_policies`` runs (one per N) for result reports.

Keeps each policy's protocol metrics (means, 95% intervals), the per-UE violation
distribution and the protocol selection, plus the SHA-256 of every source file.
"""

import argparse
import hashlib
import json
from pathlib import Path

KEEP = ("feasible", "e_over_ub", "e_over_lb", "worst_ue_ocr", "ocr", "mw_per_ue")


def summarize(path: Path) -> dict:
    raw = path.read_bytes()
    record = json.loads(raw)
    policies = {}
    for spec, result in record["results"].items():
        summary = result["summary"]
        policies[spec] = {
            "satisfies_constraint": summary["satisfies_constraint"],
            "episodes": summary["episodes"],
            "per_ue_violation_rate": summary["per_ue_violation_rate"],
            **{
                metric: {
                    "mean": summary["metrics"][metric]["mean"],
                    "ci95": summary["metrics"][metric]["ci95"],
                }
                for metric in KEEP
                if metric in summary["metrics"]
            },
        }
    return {
        "source": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "formal": record["formal"],
        "provenance": record["provenance"],
        "config": record["config"],
        "split": record["split"],
        "instances": len(record["instances"]),
        "selected_by_protocol_rule": record["selected_by_protocol_rule"],
        "policies": policies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    runs = [summarize(path) for path in args.sources]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    for run in runs:
        print(run["config"], "->", run["selected_by_protocol_rule"])


if __name__ == "__main__":
    main()
