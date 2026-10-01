"""Checkpoint and variant selection of EXP-DSA05-006 (design §4.1, §5).

Reads ``evaluate_capacity`` validation outputs whose policies are ``marl:``
checkpoint specs. The score of a checkpoint is S = mean F at eps = 0.01 over
``SELECTION_GRID``. Per run (training output directory) the checkpoint with the
highest S is selected, ties going to the later one; the last checkpoint is
reported as well. Per variant the selected scores are averaged over seeds; the
phase-1 rule prefers C (local) unless A beats it by at least ``MARGIN``. The
best calibrated local rule's S on the same grid is the learning-signal bar.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dsa_period.capacity import eps_key
from dsa_period.rules_v2 import CALIBRATION_PATH, CALIBRATION_TABLE

SELECTION_GRID = (4, 6, 8, 10, 12, 16)
MARGIN = 0.02


def score(summary: dict) -> float:
    rows = {r["num_ues"]: r for r in summary["per_eps"][eps_key(0.01)]["rows"]}
    missing = [n for n in SELECTION_GRID if n not in rows]
    if missing:
        raise ValueError(f"the evaluation lacks N = {missing}")
    return sum(rows[n]["feasible_share"] for n in SELECTION_GRID) / len(SELECTION_GRID)


def local_bar(calibration: dict) -> float:
    """S of the best calibrated local rule at each N (validation, same grid)."""
    total = 0.0
    for n in SELECTION_GRID:
        total += max(
            calibration["families"][family]["per_n"][str(n)]["feasible_share"]["0.01"]
            for family in CALIBRATION_TABLE
        )
    return total / len(SELECTION_GRID)


def select(summaries: dict[str, dict]) -> dict:
    """Per run: scores of all checkpoints, the selected one and the last one."""
    runs: dict[Path, dict[int, tuple[str, float]]] = defaultdict(dict)
    for spec, summary in summaries.items():
        kind, _, path = spec.partition(":")
        if kind != "marl":
            continue
        checkpoint = Path(path)
        slots = int(checkpoint.stem.split("-")[1])
        runs[checkpoint.parent.parent][slots] = (spec, score(summary))
    result = {}
    for run, by_slots in sorted(runs.items()):
        order = sorted(by_slots)
        best = max(order, key=lambda slots: (by_slots[slots][1], slots))
        manifest = json.loads((run / "manifest.json").read_text())
        result[str(run)] = {
            "variant": manifest["variant"],
            "seed": manifest["seed"],
            "scores": {by_slots[s][0]: by_slots[s][1] for s in order},
            "selected": by_slots[best][0],
            "selected_score": by_slots[best][1],
            "last": by_slots[order[-1]][0],
            "last_score": by_slots[order[-1]][1],
        }
    return result


def choose_variant(runs: dict) -> dict:
    """Phase-1 rule: mean selected S per variant; C unless A is ahead by MARGIN."""
    by_variant: dict[str, list[float]] = defaultdict(list)
    for run in runs.values():
        by_variant[run["variant"]].append(run["selected_score"])
    means = {variant: sum(v) / len(v) for variant, v in by_variant.items()}
    if set(means) != {"fixed", "local"}:
        return {"means": means, "chosen": None}
    chosen = "fixed" if means["fixed"] >= means["local"] + MARGIN else "local"
    return {"means": means, "chosen": chosen}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    summaries, splits_seen = {}, set()
    for path in args.evaluation:
        record = json.loads((path / "summary.json").read_text())
        splits_seen.add(record["split"])
        summaries.update(record["summaries"])
    if splits_seen != {"validation"}:
        parser.error("selection reads validation outputs only")
    runs = select(summaries)
    bar = local_bar(json.loads(args.calibration.read_text()))
    variants = choose_variant(runs)
    for run, row in runs.items():
        print(
            f"{run}: {row['variant']} seed {row['seed']} selected {Path(row['selected']).stem} "
            f"S {row['selected_score']:.3f} (last {row['last_score']:.3f})"
        )
    print(f"variant means {variants['means']}, chosen {variants['chosen']}")
    print(f"best calibrated local rule S = {bar:.3f} (learning-signal bar, design §4.1)")
    output = {
        "selection_grid": list(SELECTION_GRID),
        "runs": runs,
        "variants": variants,
        "local_rule_score": bar,
        "learning_signal": {
            variant: mean > bar for variant, mean in variants["means"].items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
