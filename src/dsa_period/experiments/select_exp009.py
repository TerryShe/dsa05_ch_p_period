"""Per-N checkpoint selection and the pre-registered step rules of EXP-DSA05-009.

Design: docs/experiment-designs/009-per-n-training-hard-deployments_20261004.md,
§3-§5. A learned method is one algorithm trained and evaluated per N (protocol
v2 §4, 2026-10-04 supplement).

* ``select``: reads ``evaluate_capacity --screen exp009-screen256`` outputs. A
  per-N run (manifest ``num_ues``) contributes its checkpoints at its own N only;
  a mixed-N run (no ``num_ues``) contributes every checkpoint at every N. For
  each (condition, seed, N) the checkpoint with the highest F_0.01(N) is
  selected, then the higher F_0.05(N), then the later one. A seed layer is the
  selected models of one (condition, seed) over all training N.
* ``step``: the five rules of §4 for a candidate condition against a baseline
  (step 1: P0 against M0; step 2: S1, P1, P2 against U), the step-1 choice of B
  and the step-2 choice of M.
* ``methods``: the methods file of the confirmation analysis (§6), frozen before
  v2-confirm2 is read: M = the seed layers of the selected condition, M_last =
  each of its runs' last checkpoint (one policy for every N), A_mix = the seed
  layers of the EXP-006 long runs, best_local, the six calibrated rules and both
  N_EDF versions. N >= 24 use each layer's N = 20 model (descriptive only).

Only formal screening outputs of exp009-screen256 are read; validation, test,
confirmation sets and other dev ranges are refused.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dsa_period import splits_v2
from dsa_period.capacity import eps_key

SCREEN = "exp009-screen256"
GRID = tuple(splits_v2.SCREENING_SETS[SCREEN]["grid"])
NEAR = (6, 8)
OTHER = (2, 4, 10, 12)  # rule 2
IMPROVEMENT, DROP, F2_FLOOR, TIE, TOL = 0.01, 0.01, 0.95, 0.005, 1e-12
STEP2_ORDER = ("S1", "P1", "P2")
EXTRA_N = (24, 28, 32)  # evaluated with the N = 20 model (protocol v2 §4 supplement)
RULES = tuple(
    f"calibrated:{f}"
    for f in (
        "random_max",
        "sticky_max",
        "p_persistent",
        "opportunistic",
        "reservation",
        "paced_sticky_csi",
    )
)
EDF = ("central_edf", "central_edf:defer_hopeless=1")


def load_screen(paths: list[Path], formal: bool = True) -> dict[str, dict[int, dict]]:
    """spec -> N -> {"F01", "F05"} from formal exp009-screen256 outputs only."""
    found: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in paths:
        record = json.loads((path / "summary.json").read_text())
        ok = (
            record["protocol_id"] == splits_v2.PROTOCOL_ID
            and record["split"] == "dev"
            and record.get("screen") == SCREEN
            and record["instances"] == splits_v2.screening_instances(SCREEN)
            and set(record["grid"]) <= set(GRID)
            and (record["formal"] or not formal)
        )
        if not ok:
            raise ValueError(f"{path}: not a formal {SCREEN} evaluation")
        for spec, summary in record["summaries"].items():
            rows = {
                key: {r["num_ues"]: r["feasible_share"] for r in summary["per_eps"][key]["rows"]}
                for key in (eps_key(0.01), eps_key(0.05))
            }
            for n in rows[eps_key(0.01)]:
                if n in found[spec]:
                    raise ValueError(f"{spec} at N = {n} appears in more than one evaluation")
                found[spec][n] = {"F01": rows[eps_key(0.01)][n], "F05": rows[eps_key(0.05)][n]}
    return dict(found)


def select(screen: dict, rename: dict[str, str] | None = None, grid=GRID) -> dict:
    """Per (condition, seed, N): every checkpoint's F, the selected one and the end segment.

    ``grid`` lists the N every (condition, seed) must cover (EXP-010 uses 6, 8, 10).
    """
    rename = rename or {}
    cells: dict[tuple, list] = defaultdict(list)
    for spec, by_n in screen.items():
        kind, _, path = spec.partition(":")
        if kind != "marl":
            raise ValueError(f"screening selects checkpoints only, not {spec}")
        checkpoint = Path(path)
        manifest = json.loads((checkpoint.parent.parent / "manifest.json").read_text())
        experiment = manifest["experiment"]
        condition = experiment.get("condition", experiment["id"])
        condition = rename.get(condition, condition)
        own = manifest.get("num_ues")
        slots = int(checkpoint.stem.split("-")[1])
        for n, shares in by_n.items():
            if own is not None and n != own:
                continue  # a per-N model is evaluated at its own N only (cross-N: future work)
            cells[(condition, manifest["seed"], n)].append({"spec": spec, "slots": slots, **shares})
    result: dict = defaultdict(lambda: defaultdict(dict))
    for (condition, seed, n), rows in sorted(cells.items()):
        rows.sort(key=lambda r: r["slots"])
        best = max(rows, key=lambda r: (r["F01"], r["F05"], r["slots"]))
        result[condition][str(seed)][str(n)] = {
            "selected": best["spec"],
            "F01": best["F01"],
            "F05": best["F05"],
            "slots": best["slots"],
            "end_F01": [r["F01"] for r in rows[-2:]],
            "checkpoints": rows,
        }
    for condition, seeds in result.items():
        for seed, by_n in seeds.items():
            missing = [n for n in grid if str(n) not in by_n]
            if missing:
                raise ValueError(f"{condition} seed {seed} lacks N = {missing}")
    return {c: dict(s) for c, s in result.items()}


def layers(selection: dict) -> dict:
    """condition -> seed -> {N: selected spec} (the seed layers of the analysis)."""
    return {
        condition: {
            seed: {n: cell["selected"] for n, cell in by_n.items()} for seed, by_n in seeds.items()
        }
        for condition, seeds in selection.items()
    }


def merge_selections(paths: list[Path]) -> dict:
    """condition -> seed -> N -> cell over several selection files (seeds must not repeat)."""
    merged: dict = defaultdict(dict)
    for path in paths:
        for condition, seeds in json.loads(path.read_text())["selection"].items():
            for seed, cells in seeds.items():
                if seed in merged[condition]:
                    raise ValueError(f"{condition} seed {seed} appears in more than one selection")
                merged[condition][seed] = cells
    return dict(merged)


def methods(selection: dict, primary: str, baseline: str = "A_mix") -> dict:
    """The frozen methods file of the confirmation analysis (§6)."""

    def layer(cells: dict) -> dict:
        out = {n: cells[n]["selected"] for n in sorted(cells, key=int)}
        out.update({str(n): cells["20"]["selected"] for n in EXTRA_N})
        return out

    def last(cells: dict) -> str:
        rows = max(cells.values(), key=lambda c: c["checkpoints"][-1]["slots"])["checkpoints"]
        return rows[-1]["spec"]

    seeds = sorted(selection[primary])
    return {
        "primary": "M",
        "baseline": baseline,
        "M": [layer(selection[primary][s]) for s in seeds],
        "M_last": [last(selection[primary][s]) for s in seeds],
        baseline: [layer(selection[baseline][s]) for s in sorted(selection[baseline])],
        "best_local": "best_local",
        "rules": list(RULES),
        "edf": list(EDF),
        "_source": {"condition": primary, "seeds": seeds},
    }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def summary(entry: dict) -> dict:
    """Seed means of the selected F_0.01(N), S_near and the end segment at N = 6, 8."""
    seeds = entry.values()
    out = {f"F{n}": _mean(s[str(n)]["F01"] for s in seeds) for n in GRID}
    out["S_near"] = _mean(_mean(s[str(n)]["F01"] for n in NEAR) for s in seeds)
    out["end"] = _mean(f for s in seeds for n in NEAR for f in s[str(n)]["end_F01"])
    return out


def rules(candidate: dict, baseline: dict) -> dict:
    """The five §4 rules of a condition against its baseline; seeds paired by number."""
    if sorted(candidate) != sorted(baseline):
        raise ValueError("a condition and its baseline need the same training seeds")
    c, b = summary(candidate), summary(baseline)
    per_seed = {
        seed: _mean(candidate[seed][str(n)]["F01"] for n in NEAR)
        - _mean(baseline[seed][str(n)]["F01"] for n in NEAR)
        for seed in candidate
    }
    checks = {
        "rule1_improvement": c["S_near"] - b["S_near"] >= IMPROVEMENT - TOL
        and all(d >= -TOL for d in per_seed.values()),
        "rule2_other_n_no_drop": all(c[f"F{n}"] >= b[f"F{n}"] - DROP - TOL for n in OTHER),
        "rule3_end_segment": c["end"] >= b["end"] - TOL,
        "rule4_low_load": c["F2"] >= F2_FLOOR - TOL,
        "rule5_near_per_n": all(c[f"F{n}"] >= b[f"F{n}"] - DROP - TOL for n in NEAR),
    }
    return {
        "means": c,
        "baseline_means": b,
        "delta_S_near": c["S_near"] - b["S_near"],
        "delta_S_near_per_seed": per_seed,
        "rules": checks,
        "retained": all(checks.values()),
    }


def step1(selection: dict, candidate: str = "P0", baseline: str = "M0") -> dict:
    """§4: B = P0 if it passes every rule against M0, else M0."""
    check = rules(selection[candidate], selection[baseline])
    return {candidate: check, "B": candidate if check["retained"] else baseline}


def step2(selection: dict, baseline: str) -> dict:
    """§5.4: rules against U; the highest S_near among retained, ties to the simpler one."""
    checks = {c: rules(selection[c], selection[baseline]) for c in STEP2_ORDER if c in selection}
    retained = [c for c in STEP2_ORDER if c in checks and checks[c]["retained"]]
    choice = baseline
    if retained:
        best = max(checks[c]["means"]["S_near"] for c in retained)
        close = [c for c in retained if checks[c]["means"]["S_near"] >= best - TIE - TOL]
        choice = min(close, key=STEP2_ORDER.index)
    return {"checks": checks, "retained": retained, "M": choice}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select")
    s.add_argument("--screen", type=Path, nargs="+", required=True)
    s.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW")
    s.add_argument("--grid", type=int, nargs="+", default=list(GRID))
    for name in ("step1", "step2", "methods"):
        r = sub.add_parser(name)
        r.add_argument("--selection", type=Path, nargs="+", required=True)
    sub.choices["methods"].add_argument("--primary", required=True)
    sub.choices["methods"].add_argument("--baseline", default="A_mix")
    sub.choices["step1"].add_argument("--candidate", default="P0")
    sub.choices["step1"].add_argument("--baseline", default="M0")
    sub.choices["step2"].add_argument("--baseline", required=True, help="the U condition label")
    for p in sub.choices.values():
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--allow-informal", action="store_true", help="smoke tests only")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    if args.command == "select":
        rename = dict(item.split("=", 1) for item in args.rename)
        chosen = select(load_screen(args.screen, not args.allow_informal), rename, tuple(args.grid))
        result = {"screen": SCREEN, "selection": chosen, "layers": layers(chosen)}
        for condition, seeds in chosen.items():
            for seed, by_n in seeds.items():
                cells = " ".join(f"{n}:{c['F01']:.3f}" for n, c in by_n.items())
                print(f"{condition} seed {seed}: {cells}")
    elif args.command == "methods":
        result = methods(merge_selections(args.selection), args.primary, args.baseline)
        print(f"M = {args.primary} seeds {result['_source']['seeds']}, baseline {args.baseline}")
    else:
        selection: dict = {}
        for path in args.selection:
            for condition, seeds in json.loads(path.read_text())["selection"].items():
                if condition in selection:
                    raise ValueError(f"{condition} appears in more than one selection")
                selection[condition] = seeds
        if args.command == "step1":
            result = step1(selection, args.candidate, args.baseline)
            print(f"{args.candidate}: {result[args.candidate]['rules']} -> B = {result['B']}")
        else:
            result = step2(selection, args.baseline)
            print(f"retained {result['retained']} -> M = {result['M']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
