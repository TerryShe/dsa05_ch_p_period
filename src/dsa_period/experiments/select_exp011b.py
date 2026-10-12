"""Per-N checkpoint selection with the anti-starvation guard (EXP-DSA05-011B).

Design: docs/experiment-designs/011b-guarded-selection-extended-training_20261012.md
§3.3. Every learned method (M, U, O, L, U0; seeds 101-505; checkpoints 1M-10M) is
selected the same way:

* guard: a checkpoint with a starvation cell (``starvation.cells``: a UE with
  mean SNR >= 20 dB and violation rate >= 0.5) at any N of the stress set
  ``exp011b-stress`` or at any evaluation N of its condition on
  exp009-screen256 is not selected at any N (EXP-009's M screening also covers
  N = 2, 4, 20, outside the evaluation grid, and is not counted there);
* rule: per (condition, seed, N), among the guarded checkpoints the lowest P2
  on exp009-screen256, then the higher F_0.01(N), then the later checkpoint
  (``select_exp011.best``);
* fallback: if no checkpoint of a (condition, seed) passes, the one with the
  fewest starvation cells, then the rule; reported as a fallback;
* budget effect (descriptive): the same selection with candidates up to 5M.

``methods`` writes the evaluation methods file (design §3.4). Conditions come
from the run manifests (``--rename S1=M`` for EXP-009's S1 runs).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dsa_period import splits_v2, starvation
from dsa_period.experiments.select_exp011 import best, load_screen

SCREEN = "exp009-screen256"
GRID = (6, 8, 10, 12, 16)
CORE = (8, 10, 12)
GRIDS = {"U": GRID, "O": GRID, "L": GRID, "M": GRID, "U0": CORE}
SEEDS = (101, 202, 303, 404, 505)
BUDGET_5M = 5_100_000


def load_cells(screen_dirs: list[Path], stress_dirs: list[Path]) -> dict[str, dict]:
    """spec -> {"screen": {N: cells}, "stress": {N: cells}} from formal outputs."""
    found: dict = defaultdict(lambda: {"screen": {}, "stress": {}})
    registered = {str(n) for n in splits_v2.SCREENING_SETS[starvation.STRESS_SET]["grid"]}
    for kind, paths in (("screen", screen_dirs), ("stress", stress_dirs)):
        for path in paths:
            record = json.loads((path / "summary.json").read_text())
            wanted = SCREEN if kind == "screen" else starvation.STRESS_SET
            if not record["formal"] or record.get("screen", record.get("set")) != wanted:
                raise ValueError(f"{path}: not a formal {wanted} output")
            if kind == "stress" and set(record["instances"]) != registered:
                raise ValueError(f"{path}: the stress set must cover N = {sorted(registered)}")
            for spec, by_n in json.loads((path / "episodes.json").read_text()).items():
                for n, rows in by_n.items():
                    if int(n) in found[spec][kind]:
                        raise ValueError(f"{spec} at N = {n} appears twice in the {kind} outputs")
                    found[spec][kind][int(n)] = starvation.cells(rows, int(n))
    return dict(found)


def manifest_of(spec: str, rename: dict[str, str]) -> tuple[str, int, int]:
    """(condition, seed, slots) of a checkpoint spec, from its run manifest."""
    kind, _, path = spec.partition(":")
    if kind != "marl":
        raise ValueError(f"selection is among checkpoints only, not {spec}")
    checkpoint = Path(path)
    manifest = json.loads((checkpoint.parent.parent / "manifest.json").read_text())
    if manifest.get("num_ues") is not None:
        raise ValueError(f"{spec}: EXP-011B selects among mixed-N runs only")
    experiment = manifest["experiment"]
    condition = experiment.get("condition", experiment["id"])
    return rename.get(condition, condition), int(manifest["seed"]), int(checkpoint.stem[5:])


def select(
    screen: dict,
    cells: dict,
    rename: dict[str, str] | None = None,
    grids: dict | None = None,
    max_slots: int | None = None,
) -> dict:
    """condition -> seed -> N -> the guarded choice and every candidate's record."""
    rename, grids = rename or {}, grids or GRIDS
    runs: dict[tuple, dict] = defaultdict(dict)  # (condition, seed) -> spec -> row
    for spec, by_n in screen.items():
        condition, seed, slots = manifest_of(spec, rename)
        if condition not in grids:
            raise ValueError(f"{spec}: unknown condition {condition}")
        if max_slots is not None and slots > max_slots:
            continue
        if spec not in cells or not cells[spec]["stress"]:
            raise ValueError(f"{spec} has no stress-set output")
        count = sum(len(v) for v in cells[spec]["stress"].values()) + sum(
            len(v) for n, v in cells[spec]["screen"].items() if n in grids[condition]
        )
        runs[(condition, seed)][spec] = {"spec": spec, "slots": slots, "starvation": count,
                                         "by_n": by_n}  # fmt: skip
    result: dict = defaultdict(dict)
    for (condition, seed), specs in sorted(runs.items()):
        clean = [s for s, row in specs.items() if row["starvation"] == 0]
        fallback = not clean
        if fallback:
            fewest = min(row["starvation"] for row in specs.values())
            clean = [s for s, row in specs.items() if row["starvation"] == fewest]
        per_n = {}
        for n in grids[condition]:
            candidates = [
                {"spec": s, "slots": specs[s]["slots"], **specs[s]["by_n"][n]}
                for s in clean
                if n in specs[s]["by_n"]
            ]
            if not candidates:
                raise ValueError(f"{condition} seed {seed} has no screened checkpoint at N = {n}")
            chosen = best(candidates)
            per_n[str(n)] = {
                "selected": chosen["spec"],
                "P2": chosen["P2"],
                "F01": chosen["F01"],
                "slots": chosen["slots"],
                "fallback": fallback,
            }
        result[condition][str(seed)] = {
            "cells": per_n,
            "checkpoints": [
                {
                    "spec": s,
                    "slots": row["slots"],
                    "starvation": row["starvation"],
                    **{f"P2@{n}": row["by_n"][n]["P2"] for n in sorted(row["by_n"])},
                }
                for s, row in sorted(specs.items(), key=lambda x: x[1]["slots"])
            ],  # fmt: skip
        }
    return dict(result)


def budget_effect(full: dict, early: dict) -> dict:
    """Per condition and N: mean over seeds of P2(selected up to 10M) - P2(up to 5M)."""
    out: dict = {}
    for condition, seeds in full.items():
        out[condition] = {}
        for n in seeds[next(iter(seeds))]["cells"]:
            diffs = [
                seeds[s]["cells"][n]["P2"] - early[condition][s]["cells"][n]["P2"] for s in seeds
            ]
            out[condition][n] = sum(diffs) / len(diffs)
    return out


def methods(selection: dict) -> dict:
    """The evaluation methods file (design §3.4; wrappers are built by the driver)."""
    out: dict = {}
    for condition in ("U", "O", "L", "U0", "M"):
        seeds = selection.get(condition)
        if seeds is None or sorted(map(int, seeds)) != list(SEEDS):
            raise ValueError(f"the selection of {condition} needs seeds {SEEDS}")
        out[condition] = [
            {n: cell["selected"] for n, cell in sorted(entry["cells"].items(), key=lambda x: int(x[0]))}
            for _, entry in sorted(seeds.items(), key=lambda x: int(x[0]))
        ]  # fmt: skip
    out.update(best_local="best_local", cg_opt="configured_grant_opt", edf="central_edf")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("select")
    pick.add_argument("--screen", type=Path, nargs="+", required=True)
    pick.add_argument("--stress", type=Path, nargs="+", required=True)
    pick.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW")
    pick.add_argument("--output", type=Path, required=True)
    build = sub.add_parser("methods")
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    if args.command == "select":
        rename = dict(item.split("=", 1) for item in args.rename)
        screen = load_screen(args.screen)
        cells = load_cells(args.screen, args.stress)
        full = select(screen, cells, rename)
        early = select(screen, cells, rename, max_slots=BUDGET_5M)
        record = {
            "screen": SCREEN,
            "stress": starvation.STRESS_SET,
            "rule": "guard: no starvation cell at any N (stress set, screen); then per "
            "(condition, seed, N) min P2, max F_0.01, later checkpoint",
            "sources": {
                "screen": [str(p) for p in args.screen],
                "stress": [str(p) for p in args.stress],
            },  # fmt: skip
            "rename": rename,
            "selection": full,
            "selection_up_to_5M": early,
            "budget_effect_P2": budget_effect(full, early),
        }
    else:
        record = methods(json.loads(args.selection.read_text())["selection"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
