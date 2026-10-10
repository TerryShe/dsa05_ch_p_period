"""Per-N checkpoint selection by P2 and the methods file of EXP-DSA05-011.

Design: docs/experiment-designs/011-urgency-information-coordination_20261010.md
§4.1. Every learned method is selected the same way, on exp009-screen256:

* ``select``: reads formal ``evaluate_capacity --screen exp009-screen256``
  outputs. A run contributes every checkpoint at every N it was screened at
  (all EXP-011 runs and M = S1 are mixed-N). For each (condition, seed, N) the
  checkpoint with the lowest P2 (deployment mean of the worst UE's violation
  rate) is selected, then the higher F_0.01(N), then the later one. Conditions
  come from the run manifests (U, O, L, U0; ``--rename S1=M`` for EXP-009's S1).
  The re-selection of M reads EXP-009's existing formal screening outputs; no
  new evaluation is needed.
* ``methods``: the evaluation methods file: the seed layers of U, O, L, U0 and M
  (re-selected), EXP-009's frozen M layers at the evaluation N (descriptive,
  "M_exp009"), best_local, CG-opt and N_EDF.

Validation, test, confirmation sets and other dev ranges are refused.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dsa_period import splits_v2
from dsa_period.capacity import eps_key

SCREEN = "exp009-screen256"
GRID = (6, 8, 10, 12, 16)  # evaluation N (design §4.2)
CORE = (8, 10, 12)  # main rule and receive-condition grid (design §3.5, §5.2)
GRIDS = {"U": GRID, "O": GRID, "L": GRID, "M": GRID, "U0": CORE}
EXP009_METHODS = Path("docs/results/exp009-confirm_methods/methods-v2-confirm2.json")


def load_screen(paths: list[Path], formal: bool = True) -> dict[str, dict[int, dict]]:
    """spec -> N -> {"P2", "F01"} from formal exp009-screen256 outputs only."""
    found: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in paths:
        record = json.loads((path / "summary.json").read_text())
        ok = (
            record["protocol_id"] == splits_v2.PROTOCOL_ID
            and record["split"] == "dev"
            and record.get("screen") == SCREEN
            and record["instances"] == splits_v2.screening_instances(SCREEN)
            and (record["formal"] or not formal)
        )
        if not ok:
            raise ValueError(f"{path}: not a formal {SCREEN} evaluation")
        for spec, summary in record["summaries"].items():
            for row in summary["per_eps"][eps_key(0.01)]["rows"]:
                n = row["num_ues"]
                if n in found[spec]:
                    raise ValueError(f"{spec} at N = {n} appears in more than one evaluation")
                found[spec][n] = {
                    "P2": row["worst_ue_violation_rate_mean"],
                    "F01": row["feasible_share"],
                }
    return dict(found)


def best(rows: list[dict]) -> dict:
    """Lowest P2, then highest F_0.01, then the later checkpoint (design §4.1)."""
    return min(rows, key=lambda r: (r["P2"], -r["F01"], -r["slots"]))


def select(screen: dict, rename: dict[str, str] | None = None, grids: dict | None = None) -> dict:
    """condition -> seed -> N -> {selected, P2, F01, slots, checkpoints}."""
    rename, grids = rename or {}, grids or GRIDS
    cells: dict[tuple, list] = defaultdict(list)
    for spec, by_n in screen.items():
        kind, _, path = spec.partition(":")
        if kind != "marl":
            raise ValueError(f"screening selects checkpoints only, not {spec}")
        checkpoint = Path(path)
        manifest = json.loads((checkpoint.parent.parent / "manifest.json").read_text())
        if manifest.get("num_ues") is not None:
            raise ValueError(f"{spec}: EXP-011 selects among mixed-N runs only")
        experiment = manifest["experiment"]
        condition = experiment.get("condition", experiment["id"])
        condition = rename.get(condition, condition)
        if condition not in grids:
            raise ValueError(f"{spec}: unknown condition {condition}")
        slots = int(checkpoint.stem.split("-")[1])
        for n, shares in by_n.items():
            if n in grids[condition]:
                cells[(condition, manifest["seed"], n)].append(
                    {"spec": spec, "slots": slots, **shares}
                )
    result: dict = defaultdict(lambda: defaultdict(dict))
    for (condition, seed, n), rows in sorted(cells.items()):
        rows.sort(key=lambda r: r["slots"])
        chosen = best(rows)
        result[condition][str(seed)][str(n)] = {
            "selected": chosen["spec"],
            "P2": chosen["P2"],
            "F01": chosen["F01"],
            "slots": chosen["slots"],
            "checkpoints": rows,
        }
    for condition, seeds in result.items():
        for seed, by_n in seeds.items():
            missing = [n for n in grids[condition] if str(n) not in by_n]
            if missing:
                raise ValueError(f"{condition} seed {seed} lacks N = {missing}")
            counts = {len(cell["checkpoints"]) for cell in by_n.values()}
            if len(counts) != 1:
                raise ValueError(f"{condition} seed {seed}: unequal checkpoint counts per N")
    return {c: dict(s) for c, s in result.items()}


def methods(selection: dict, exp009: dict) -> dict:
    """The evaluation methods file (design §3.1; M+Y and the wrappers are built from it)."""
    out: dict = {}
    for condition in ("U", "O", "L", "U0", "M"):
        seeds = selection.get(condition)
        if seeds is None:
            raise ValueError(f"the selection lacks {condition}")
        out[condition] = [
            {n: cell["selected"] for n, cell in sorted(by_n.items(), key=lambda x: int(x[0]))}
            for _, by_n in sorted(seeds.items(), key=lambda x: int(x[0]))
        ]
    out["M_exp009"] = [{str(n): layer[str(n)] for n in GRID} for layer in exp009["M"]]
    out.update(best_local="best_local", cg_opt="configured_grant_opt", edf="central_edf")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("select")
    pick.add_argument("--screen", type=Path, nargs="+", required=True)
    pick.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW")
    pick.add_argument("--output", type=Path, required=True)
    pick.add_argument("--allow-informal", action="store_true", help="smoke tests only")
    build = sub.add_parser("methods")
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--exp009-methods", type=Path, default=EXP009_METHODS)
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    if args.command == "select":
        rename = dict(item.split("=", 1) for item in args.rename)
        screen = load_screen(args.screen, not args.allow_informal)
        record = {
            "screen": SCREEN,
            "rule": "per (condition, seed, N): min P2, then max F_0.01, then later checkpoint",
            "sources": [str(p) for p in args.screen],
            "rename": rename,
            "selection": select(screen, rename),
        }
    else:
        selection = json.loads(args.selection.read_text())["selection"]
        record = methods(selection, json.loads(args.exp009_methods.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
