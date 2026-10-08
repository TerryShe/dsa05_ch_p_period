"""EXP-DSA05-010 decision analysis exactly as pre-registered (design §3.2).

Inputs: formal ``evaluate_capacity --screen exp010-eval512`` outputs (512 dev
deployments, N = 6, 8, 10) and the methods file of the driver:

    {"IL": [layer, layer], "A_mix": [3 layers], "M": [3 layers],
     "best_local": "best_local", "edf": "central_edf"}

where a layer maps "N" to the checkpoint spec used at N. Learned groups average
F over their seeds; bootstrap differences pair deployments and resample each
group's training seeds on their own (``evaluation._bootstrap``), 10,000 draws,
seed 20260930.

* D8 = F_0.01(8; IL) - F_0.01(8; A_mix) and its 95% interval decide:
  "optimization-limited" if the lower bound > 0 and F_0.01(8; IL) >= 0.95;
  "partly optimization-limited" if the lower bound > 0 but F_0.01(8; IL) < 0.95;
  "no evidence" if the lower bound <= 0.
* Descriptive: every group's F at N = 6, 8, 10 (eps 0.01 and 0.05), IL against
  A_mix, M and best_local at each N, N_EDF.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dsa_period import splits_v2
from dsa_period.capacity import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, EPSILONS, eps_key
from dsa_period.evaluation import _bootstrap, _interval

SET = "exp010-eval512"
GRID = (6, 8, 10)
LEARNED = ("IL", "A_mix", "M")
THRESHOLD = 0.95


def load(paths: list[Path], formal: bool = True) -> tuple[dict, list[int]]:
    """spec -> {N: episodes} over formal exp010-eval512 outputs."""
    episodes: dict[str, dict] = {}
    instances = splits_v2.screening_instances(SET)
    for path in paths:
        record = json.loads((path / "summary.json").read_text())
        ok = (
            record["protocol_id"] == splits_v2.PROTOCOL_ID
            and record["split"] == "dev"
            and record.get("screen") == SET
            and record["instances"] == instances
            and (record["formal"] or not formal)
        )
        if not ok:
            raise ValueError(f"{path}: not a formal {SET} evaluation")
        for spec, by_n in json.loads((path / "episodes.json").read_text()).items():
            for n, rows in by_n.items():
                if int(n) in episodes.get(spec, {}):
                    raise ValueError(f"{spec} at N = {n} was evaluated twice")
                episodes.setdefault(spec, {})[int(n)] = rows
    return episodes, instances


def matrix(episodes: dict, members: list, n: int, key: str, instances: list[int]) -> np.ndarray:
    rows = []
    for member in members:
        spec = member if isinstance(member, str) else member[str(n)]
        cells = {e["instance"]: float(e["feasible"][key]) for e in episodes[spec][n]}
        rows.append([cells[i] for i in instances])
    return np.array(rows)


def decide(lower: float, f8: float) -> str:
    if lower > 0:
        return "optimization-limited" if f8 >= THRESHOLD else "partly optimization-limited"
    return "no evidence"


def analyze(episodes: dict, methods: dict, instances: list[int], draws: int) -> dict:
    groups = {g: list(methods[g]) for g in LEARNED}
    groups["best_local"] = [methods["best_local"]]
    groups["edf"] = [methods["edf"]]
    result: dict = {"methods": methods, "instances": len(instances), "F": {}, "per_seed": {}}
    mats = {}
    for eps in EPSILONS:
        key = eps_key(eps)
        mats[key] = {
            g: {n: matrix(episodes, m, n, key, instances) for n in GRID} for g, m in groups.items()
        }
        result["F"][key] = {g: {n: float(mats[key][g][n].mean()) for n in GRID} for g in groups}
        result["per_seed"][key] = {
            g: [
                [float(row.mean()) for row in [mats[key][g][n][s] for n in GRID]]
                for s in range(len(m))
            ]
            for g, m in groups.items()
            if g in LEARNED
        }
    m01 = mats[eps_key(0.01)]
    d8 = _bootstrap(m01["IL"][8], m01["A_mix"][8], draws, BOOTSTRAP_SEED)
    interval = _interval(d8)
    f8 = float(m01["IL"][8].mean())
    result["D8"] = {
        "difference": float(m01["IL"][8].mean() - m01["A_mix"][8].mean()),
        "ci95": interval,
        "F8_IL": f8,
        "F8_IL_ci95": _interval(_bootstrap(m01["IL"][8], None, draws, BOOTSTRAP_SEED)),
    }
    result["decision"] = decide(interval[0], f8)
    result["differences"] = {
        other: {
            str(n): {
                "difference": float(m01["IL"][n].mean() - m01[other][n].mean()),
                "ci95": _interval(_bootstrap(m01["IL"][n], m01[other][n], draws, BOOTSTRAP_SEED)),
            }
            for n in GRID
        }
        for other in ("A_mix", "M", "best_local", "edf")
    }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--methods", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--allow-informal", action="store_true", help="smoke tests only")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    if args.resamples != BOOTSTRAP_RESAMPLES and not args.allow_informal:
        parser.error("the formal analysis uses the protocol's bootstrap size")
    episodes, instances = load(args.evaluations, not args.allow_informal)
    result = analyze(episodes, json.loads(args.methods.read_text()), instances, args.resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for group, row in result["F"][eps_key(0.01)].items():
        print(f"{group:<10} " + " ".join(f"F({n}) {f:.3f}" for n, f in row.items()))
    d = result["D8"]
    print(f"D8 {d['difference']:+.4f} {d['ci95']}  F8(IL) {d['F8_IL']:.4f} -> {result['decision']}")


if __name__ == "__main__":
    main()
