"""Checkpoint selection and the pre-registered screening rules of EXP-DSA05-008.

Design: docs/experiment-designs/008-training-signal_20261002.md, §4.1-§4.3 and
§5.3. The EXP-006 selection (``select_capacity_checkpoints``) is left untouched.

* ``select``: reads ``evaluate_capacity --screen exp008-screen256`` outputs (every
  checkpoint of every run at N = 6, 8). Per run, the checkpoint with the highest
  S_near = (F(6) + F(8)) / 2 at eps = 0.01 is selected, ties going to the later
  one; the last checkpoint and the end segment (the last two checkpoints) are
  reported. The selected specs are then evaluated once on the validation split
  at N in {2, 4, 6, 8, 10, 12, 16}; a failed guard never re-selects.
* ``round1``: retention of F1-F5 against R0 (§4.2, five conditions) and the
  choice of B1 (§4.3), including the COMB check when two or more are retained.
* ``round2``: the choice of B2 among A-600, A-2400, C-600, C-2400 (§5.3).

Screening and validation outputs are checked for their set, instances, grid and
protocol; anything else (dev ranges, test, confirmation sets) is refused.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dsa_period import splits_v2
from dsa_period.capacity import eps_key

SCREEN = "exp008-screen256"
NEAR_GRID = (6, 8)
THR_GRID = (6, 8, 10, 12)
S006_GRID = (4, 6, 8, 10, 12, 16)
GUARD_GRID = (2, 4, 6, 8, 10, 12, 16)
IMPROVEMENT = 0.01  # §4.2 rule 1
DROP = 0.01  # §4.2 rules 2, 4, 5
F2_FLOOR = 0.95  # §4.2 rule 4
TIE = 0.005  # §4.3, §5.3
TOL = 1e-12
FACTOR_ORDER = ("R0", "F1", "F2", "F3", "F4", "F5", "COMB")
ROUND2_ORDER = ("A-600", "A-2400", "C-600", "C-2400")


def _shares(summary: dict) -> dict[int, float]:
    return {r["num_ues"]: r["feasible_share"] for r in summary["per_eps"][eps_key(0.01)]["rows"]}


def load_evaluations(paths: list[Path], kind: str, formal: bool = True) -> dict[str, dict]:
    """spec -> F(N) at eps = 0.01, from screening or validation outputs only."""
    found: dict[str, dict] = {}
    for path in paths:
        record = json.loads((path / "summary.json").read_text())
        if record["protocol_id"] != splits_v2.PROTOCOL_ID:
            raise ValueError(f"{path}: not a protocol-v2 evaluation")
        if formal and not record["formal"]:
            raise ValueError(f"{path}: not a formal evaluation")
        if kind == "screen":
            ok = (
                record["split"] == "dev"
                and record.get("screen") == SCREEN
                and record["instances"] == splits_v2.screening_instances(SCREEN)
                and record["grid"] == list(NEAR_GRID)
            )
        elif kind == "validation":
            ok = record["split"] == "validation" and record["instances"] == list(
                splits_v2.indices("validation")
            )
        else:
            raise ValueError(f"unknown evaluation kind {kind}")
        if not ok:
            raise ValueError(f"{path}: not a {kind} evaluation of EXP-DSA05-008")
        for spec, summary in record["summaries"].items():
            if spec in found:
                raise ValueError(f"{spec} appears in more than one {kind} evaluation")
            found[spec] = _shares(summary)
    return found


def _s_near(shares: dict[int, float]) -> float:
    return sum(shares[n] for n in NEAR_GRID) / len(NEAR_GRID)


def select(screen: dict[str, dict]) -> dict[str, dict]:
    """Per run directory: scores of every checkpoint, the selected and the last one."""
    runs: dict[Path, dict[int, str]] = defaultdict(dict)
    for spec in screen:
        kind, _, path = spec.partition(":")
        if kind != "marl":
            raise ValueError(f"screening selects checkpoints only, not {spec}")
        checkpoint = Path(path)
        runs[checkpoint.parent.parent][int(checkpoint.stem.split("-")[1])] = spec
    result = {}
    for run, by_slots in sorted(runs.items()):
        order = sorted(by_slots)
        if len(order) < 2:
            raise ValueError(f"{run}: the end segment needs at least two checkpoints")
        scores = {s: _s_near(screen[by_slots[s]]) for s in order}
        best = max(order, key=lambda s: (scores[s], s))
        manifest = json.loads((run / "manifest.json").read_text())
        experiment = manifest["experiment"]
        end = order[-2:]
        result[str(run)] = {
            "condition": experiment.get("condition", experiment["id"]),
            "round": experiment.get("round"),
            "variant": manifest["variant"],
            "seed": manifest["seed"],
            "checkpoints": [
                {
                    "spec": by_slots[s],
                    "slots": s,
                    "F6": screen[by_slots[s]][6],
                    "F8": screen[by_slots[s]][8],
                    "S_near": scores[s],
                }
                for s in order
            ],
            "selected": by_slots[best],
            "selected_slots": best,
            "selected_S_near": scores[best],
            "selected_F": {f"F{n}": screen[by_slots[best]][n] for n in NEAR_GRID},
            "last": by_slots[order[-1]],
            "last_S_near": scores[order[-1]],
            "end_slots": end,
            "end_S_near": [scores[s] for s in end],
        }
    return result


def _validation_metrics(shares: dict[int, float]) -> dict:
    missing = [n for n in GUARD_GRID if n not in shares]
    if missing:
        raise ValueError(f"the validation evaluation lacks N = {missing}")
    return {
        **{f"F{n}": shares[n] for n in GUARD_GRID},
        "S_thr": sum(shares[n] for n in THR_GRID) / len(THR_GRID),
        "S_006": sum(shares[n] for n in S006_GRID) / len(S006_GRID),
    }


def conditions(selection: dict, validation: dict[str, dict]) -> dict[str, dict]:
    """Per condition: its runs by seed with screening and validation metrics."""
    out: dict[str, dict] = defaultdict(lambda: {"runs": {}})
    for run, row in selection.items():
        if row["selected"] not in validation:
            raise ValueError(f"no validation evaluation of the selected {row['selected']}")
        entry = out[row["condition"]]
        if row["seed"] in entry["runs"]:
            raise ValueError(f"two runs of {row['condition']} with seed {row['seed']}")
        entry["runs"][row["seed"]] = {
            "run": run,
            "selected": row["selected"],
            "S_near": row["selected_S_near"],
            "screen_F": row["selected_F"],
            "end_slots": row["end_slots"],
            "end_S_near": row["end_S_near"],
            "validation": _validation_metrics(validation[row["selected"]]),
        }
    for entry in out.values():
        runs = entry["runs"].values()
        entry["mean"] = {
            "S_near": _mean(r["S_near"] for r in runs),
            "F6_screen": _mean(r["screen_F"]["F6"] for r in runs),
            "F8_screen": _mean(r["screen_F"]["F8"] for r in runs),
            "end_S_near": _mean(s for r in runs for s in r["end_S_near"]),
            **{key: _mean(r["validation"][key] for r in runs) for key in _VALIDATION_KEYS},
        }
    return dict(out)


_VALIDATION_KEYS = (*(f"F{n}" for n in GUARD_GRID), "S_thr", "S_006")


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _guards(cand: dict, base: dict) -> dict[str, bool]:
    """§4.2 rules 2, 4 and 5 of a condition against its baseline (means over seeds)."""
    c, b = cand["mean"], base["mean"]
    return {
        "rule2_validation_no_drop": all(
            c[k] >= b[k] - DROP - TOL for k in ("F4", "F6", "F8", "S_thr")
        ),
        "rule4_validation_F2": c["F2"] >= F2_FLOOR - TOL and c["F2"] >= b["F2"] - DROP - TOL,
        "rule5_screen_per_n_no_drop": all(
            c[k] >= b[k] - DROP - TOL for k in ("F6_screen", "F8_screen")
        ),
    }


def retention(cand: dict, base: dict) -> dict:
    """All five §4.2 rules of a condition against R0; seeds are paired by number."""
    if sorted(cand["runs"]) != sorted(base["runs"]):
        raise ValueError("a condition and R0 need the same training seeds")
    for seed, run in cand["runs"].items():
        if run["end_slots"] != base["runs"][seed]["end_slots"]:
            raise ValueError("the end segments of a condition and R0 differ")
    per_seed = {
        seed: run["S_near"] - base["runs"][seed]["S_near"] for seed, run in cand["runs"].items()
    }
    delta = cand["mean"]["S_near"] - base["mean"]["S_near"]
    rules = {
        "rule1_screen_improvement": delta >= IMPROVEMENT - TOL
        and all(d >= -TOL for d in per_seed.values()),
        **_guards(cand, base),
        "rule3_end_segment": cand["mean"]["end_S_near"] - base["mean"]["end_S_near"] >= -TOL,
    }
    return {
        "delta_S_near": delta,
        "delta_S_near_per_seed": per_seed,
        "delta_end_S_near": cand["mean"]["end_S_near"] - base["mean"]["end_S_near"],
        "rules": dict(sorted(rules.items())),
        "retained": all(rules.values()),
    }


def _pick(candidates: dict[str, dict], key) -> str:
    best = max(c["mean"]["S_near"] for c in candidates.values())
    close = [name for name, c in candidates.items() if c["mean"]["S_near"] >= best - TIE - TOL]
    return min(close, key=key)


def round1(table: dict[str, dict], comb_factors: list[str] | None = None) -> dict:
    """§4.2 retention of F1-F5 and the §4.3 choice of B1."""
    base = table["R0"]
    factors = [f for f in FACTOR_ORDER[1:6] if f in table]
    checks = {f: retention(table[f], base) for f in factors}
    retained = [f for f in factors if checks[f]["retained"]]
    out: dict = {"retention": checks, "retained": retained}
    if len(retained) >= 2 and "COMB" not in table:
        out.update(comb_required=retained, B1=None)
        return out
    candidates = {"R0": base, **{f: table[f] for f in retained}}
    sizes = {"R0": 0, **{f: 1 for f in retained}}
    if len(retained) >= 2:
        if sorted(comb_factors or []) != sorted(retained):
            raise ValueError(f"COMB must combine exactly the retained factors {retained}")
        comb = retention(table["COMB"], base)
        out["comb"] = {"factors": retained, **comb}
        if comb["retained"]:
            candidates["COMB"] = table["COMB"]
            sizes["COMB"] = len(retained)
    out["candidates"] = {
        name: {"S_near": c["mean"]["S_near"], "S_thr": c["mean"]["S_thr"], "factors": sizes[name]}
        for name, c in candidates.items()
    }
    out["B1"] = _pick(
        candidates,
        lambda name: (sizes[name], -candidates[name]["mean"]["S_thr"], FACTOR_ORDER.index(name)),
    )
    return out


def round2(table: dict[str, dict], a600: str) -> dict:
    """§5.3: guards 2, 4, 5 against A-600 (the B1 runs), then the highest S_near."""
    named = {"A-600": table[a600], **{c: table[c] for c in ROUND2_ORDER[1:] if c in table}}
    base = named["A-600"]
    guards = {c: _guards(named[c], base) for c in ROUND2_ORDER[1:] if c in named}
    candidates = {"A-600": base}
    candidates.update({c: named[c] for c, g in guards.items() if all(g.values())})
    return {
        "a600_condition": a600,
        "guards": guards,
        "means": {c: named[c]["mean"] for c in named},
        "candidates": sorted(candidates),
        "B2": _pick(candidates, ROUND2_ORDER.index),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select", help="per-run checkpoint selection on the screening set")
    s.add_argument("--screen", type=Path, nargs="+", required=True)
    for name in ("round1", "round2"):
        r = sub.add_parser(name)
        r.add_argument("--selection", type=Path, nargs="+", required=True)
        r.add_argument("--validation", type=Path, nargs="+", required=True)
    sub.choices["round1"].add_argument("--comb-factors", nargs="*")
    sub.choices["round2"].add_argument("--a600", required=True, help="the B1 condition label")
    for p in sub.choices.values():
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--allow-informal", action="store_true", help="smoke tests only")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    formal = not args.allow_informal
    if args.command == "select":
        result = {"screen": SCREEN, "runs": select(load_evaluations(args.screen, "screen", formal))}
        for run, row in result["runs"].items():
            print(
                f"{run}: {row['condition']} seed {row['seed']} selected "
                f"{Path(row['selected']).stem} S_near {row['selected_S_near']:.4f} "
                f"(last {row['last_S_near']:.4f}, end {row['end_S_near']})"
            )
        print("validate:", " ".join(r["selected"] for r in result["runs"].values()))
    else:
        selection: dict = {}
        for path in args.selection:
            selection.update(json.loads(path.read_text())["runs"])
        validation = load_evaluations(args.validation, "validation", formal)
        table = conditions(selection, validation)
        if args.command == "round1":
            result = {"conditions": table, **round1(table, args.comb_factors)}
            print(f"retained {result['retained']}; B1 {result['B1']}")
            if result.get("comb_required"):
                print(f"COMB required: {result['comb_required']}")
        else:
            result = {"conditions": table, **round2(table, args.a600)}
            print(f"B2 {result['B2']} (candidates {result['candidates']})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
