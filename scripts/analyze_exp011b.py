"""EXP-DSA05-011B analysis exactly as pre-registered (design §5).

Inputs (amendment 2, design §15): formal ``evaluate_capacity --screen
exp011b-eval128`` outputs, formal ``stress_exp011b.py --set exp011b-near``
outputs (the near-UE deployments of dev 6128-6511, used for the starvation count
only) and the evaluation plan written by ``scripts/run_exp011b.py`` (same format as EXP-011:
{"groups": {name: [layer, ...] | spec}, "grids": {name: [N, ...]}}). Groups: U,
M, M+Y, O, L, U-hide-all, U-hide-nonurgent, best_local, CG-opt, N_EDF at every
N; U0 and U@0 (U evaluated at a 0 dB reading threshold) at N = 8, 10, 12. Each
learned group has five seed layers (101-505).

Statistics as in EXP-011 (``analyze_exp011``): w per deployment, P2 its mean,
two-level bootstrap with 10,000 draws and seed 20,260,930.

* Main (§5.2): D_N = P2(U) - P2(M) at N = 8, 10, 12, the EXP-011 rule.
* Secondary (§5.3): U - O and U - L (U better than a control: 95% upper bound
  < 0 at two or more of N = 8, 10, 12; content supported: better than both);
  E_N = P2(U) - P2(M+Y) (learned response better at two or more N; M+Y
  significantly better where the lower bound > 0); starvation cells of every
  group (``starvation.cells``) on exp011b-eval128 and exp011b-near; H-M1, H-M2; U0 - M and U0 - U@0; D_N over all
  N; budget effect (from the selection) and the EXP-011 slice (descriptive).
* Proposed method (§5.4): U if the main rule is at least "partly effective", U
  has no starvation cell and M+Y is not significantly better at two or more N.
* Minimum distance (amendment 2, descriptive, decides nothing): P2, D_N and the
  secondary comparisons again on the deployments whose UEs are all at least 5 m
  and 10 m from the AP at that N (scenario geometry only).
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from dsa_period import starvation
from dsa_period.capacity import BOOTSTRAP_RESAMPLES

SET = "exp011b-eval128"
NEAR_SET = "exp011b-near"
MIN_DISTANCES_M = (5.0, 10.0)
SUBSET_KEYS = ("instances", "P2", "D_N", "decision", "harmful_N", "content", "coordination",
               "mechanism")  # fmt: skip
GRID = (6, 8, 10, 12, 16)
CORE = (8, 10, 12)
HIDES = ("all", "nonurgent")
EXP011_ANALYSIS = Path("docs/results/exp011_20261012/analysis.json")


def _exp011():
    path = Path(__file__).parent / "analyze_exp011.py"
    spec = importlib.util.spec_from_file_location("analyze_exp011", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _exp011()


def starvation_cells(episodes: dict, group, n: int) -> int:
    """(seed layer, deployment) cells of a group at N with a starved strong UE."""
    members = group if isinstance(group, list) else [group]
    total = 0
    for member in members:
        spec = member if isinstance(member, str) else member[str(n)]
        total += len({c["instance"] for c in starvation.cells(episodes[spec][n], n)})
    return total


def load_near(paths: list[Path], formal: bool = True) -> dict:
    """spec -> {N: episodes} over formal exp011b-near outputs (``stress_exp011b.py``)."""
    near: dict = {}
    for path in paths:
        record = json.loads((path / "summary.json").read_text())
        if record.get("set") != NEAR_SET or (formal and not record["formal"]):
            raise ValueError(f"{path}: not a formal {NEAR_SET} output")
        for n, listed in record["instances"].items():
            if listed != starvation.stress_instances(starvation._scenario(int(n)), NEAR_SET):
                raise ValueError(f"{path}: not the registered {NEAR_SET} deployments at N = {n}")
        for spec, by_n in json.loads((path / "episodes.json").read_text()).items():
            for n, rows in by_n.items():
                if int(n) in near.get(spec, {}):
                    raise ValueError(f"{spec} at N = {n} appears twice in the {NEAR_SET} outputs")
                near.setdefault(spec, {})[int(n)] = rows
    return near


def min_distances(instances: list[int], grid=GRID) -> dict[int, dict[int, float]]:
    """N -> instance -> distance of the deployment's nearest UE to the AP (geometry only)."""
    return {
        n: {
            i: float(starvation._scenario(n).instance(i)[0].ue_ap_distances_m.min())
            for i in instances
        }
        for n in grid
    }


def analyze(
    episodes: dict,
    plan: dict,
    instances: list[int] | dict[int, list[int]],
    draws: int,
    exp011: dict | None = None,
    near: dict | None = None,
    distances: dict[int, dict[int, float]] | None = None,
) -> dict:
    """The pre-registered analysis; ``instances`` may differ per N (a subset analysis)."""
    groups, grids = plan["groups"], {k: tuple(v) for k, v in plan["grids"].items()}
    every = {n: instances[n] if isinstance(instances, dict) else instances for n in GRID}

    def mat(name: str, n: int, field: str = "w") -> np.ndarray:
        return base.matrix(episodes, groups[name], n, field, every[n])

    def compare(a: np.ndarray, b: np.ndarray) -> dict:
        return base.compare(a, b, draws)

    out: dict = {"plan": plan, "instances": {n: len(every[n]) for n in GRID}, "delta": base.DELTA}
    out["P2"] = {g: {n: float(mat(g, n).mean()) for n in grids[g]} for g in groups}
    out["P1"] = {g: {n: float(mat(g, n, "feasible").mean()) for n in grids[g]} for g in groups}
    out["N_star_on_grid"] = {
        g: base.capacity_on(GRID, [out["P1"][g][n] for n in GRID])
        for g in groups
        if grids[g] == GRID
    }
    out["all_ue_mean_rate"] = {
        g: {n: float(mat(g, n, "mean_rate").mean()) for n in grids[g]} for g in groups
    }
    out["p90_w"] = {
        g: {n: float(np.percentile(mat(g, n), 90, axis=1).mean()) for n in grids[g]} for g in groups
    }
    by_set = {SET: {g: {n: starvation_cells(episodes, groups[g], n) for n in grids[g]}
                    for g in groups}}  # fmt: skip
    if near is not None:
        by_set[NEAR_SET] = {
            g: {n: starvation_cells(near, groups[g], n) for n in grids[g]} for g in groups
        }
    out["starvation_cells_by_set"] = by_set
    out["starvation_cells"] = {
        g: {n: sum(cells[g][n] for cells in by_set.values()) for n in grids[g]} for g in groups
    }

    # Main rule (§5.2) and the load profile (§5.3 item 6).
    d = {n: compare(mat("U", n), mat("M", n)) for n in GRID}
    out["D_N"] = d
    out["decision"] = base.main_decision({n: d[n]["ci95"][1] for n in CORE})
    out["harmful_N"] = [n for n in GRID if d[n]["ci95"][0] > 0]

    # Content (§5.3 item 1).
    content: dict = {}
    for other in ("O", "L"):
        rows = {n: compare(mat("U", n), mat(other, n)) for n in GRID}
        better = sum(rows[n]["ci95"][1] < 0 for n in CORE)
        content[other] = {"rows": rows, "better_at": better, "U_better": better >= 2}
    content["supported"] = content["O"]["U_better"] and content["L"]["U_better"]
    out["content"] = content

    # Use of the summary (§5.3 item 2).
    e = {n: compare(mat("U", n), mat("M+Y", n)) for n in CORE}
    learned = sum(e[n]["ci95"][1] < 0 for n in CORE)
    rule = sum(e[n]["ci95"][0] > 0 for n in CORE)
    out["coordination"] = {
        "E_N": e,
        "learned_better_at": learned,
        "learned_response_better": learned >= 2,
        "rule_better_at": rule,
        "core_mean_P2": {g: float(np.mean([out["P2"][g][n] for n in CORE])) for g in ("U", "M+Y")},
    }

    # Mechanism (§5.3 item 4).
    mech: dict = {}
    for n in GRID:
        u = mat("U", n)
        gain = float(mat("M", n).mean() - u.mean())
        row = {"gain_P2_M_minus_U": gain}
        for hide in HIDES:
            row[hide] = compare(mat(f"U-hide-{hide}", n), u)
        lost = row["nonurgent"]["difference"]
        row["H_M2"] = row["all"]["ci95"][0] > 0
        row["H_M1"] = row["nonurgent"]["ci95"][0] > 0 and gain > 0 and lost >= 0.5 * gain
        mech[n] = row
    out["mechanism"] = mech

    # 0 dB (§5.3 item 5).
    out["zero_db"] = {
        n: {
            "U0_minus_M": compare(mat("U0", n), mat("M", n)),
            "U0_minus_U_at_0dB": compare(mat("U0", n), mat("U@0", n)),
        }
        for n in CORE
    }

    # Sensitivity and guard (§5.1).
    keep = {n: mat("U", n, "hopeless")[0] == 0 for n in GRID}
    out["sensitivity"] = {
        n: {
            "all_ue_mean_rate_U_minus_M": compare(
                mat("U", n, "mean_rate"), mat("M", n, "mean_rate")
            ),
            "log_w_U_minus_M": compare(mat("U", n, "log_w"), mat("M", n, "log_w")),
            "excluding_infeasible_for_all": {
                "deployments": int(keep[n].sum()),
                **compare(mat("U", n)[:, keep[n]], mat("M", n)[:, keep[n]]),
            },
            "p90_U_minus_M": out["p90_w"]["U"][n] - out["p90_w"]["M"][n],
        }
        for n in GRID
    }

    # Proposed method (§5.4).
    conditions = {
        "main_rule_at_least_partly_effective": out["decision"] != "no evidence",
        "U_without_starvation_cells": sum(out["starvation_cells"]["U"].values()) == 0,
        "M+Y_not_significantly_better_at_two_or_more_N": rule < 2,
    }
    out["proposed_method"] = {
        "conditions": conditions,
        "U_is_the_proposed_method": all(conditions.values()),
    }

    # The EXP-011 slice (§5.3 item 7, descriptive).
    if exp011 is not None:
        out["exp011_slice"] = {
            n: {"exp011_D_N": exp011["D_N"][str(n)], "exp011b_D_N": d[n]} for n in GRID
        }

    # Minimum UE-AP distance (amendment 2, descriptive; decides nothing).
    if distances is not None:
        out["min_distance"] = {}
        for d_min in MIN_DISTANCES_M:
            kept = {n: [i for i in every[n] if distances[n][i] >= d_min] for n in GRID}
            subset = analyze(episodes, plan, kept, draws)
            out["min_distance"][str(d_min)] = {key: subset[key] for key in SUBSET_KEYS}
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--near", type=Path, nargs="+", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--exp011", type=Path, default=EXP011_ANALYSIS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--allow-informal", action="store_true", help="smoke tests only")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    if args.resamples != BOOTSTRAP_RESAMPLES and not args.allow_informal:
        parser.error("the formal analysis uses the protocol's bootstrap size")
    episodes, instances = base.load(args.evaluations, not args.allow_informal, SET)
    near = load_near(args.near, not args.allow_informal)
    exp011 = json.loads(args.exp011.read_text()) if args.exp011.exists() else None
    result = analyze(episodes, json.loads(args.plan.read_text()), instances, args.resamples,
                     exp011, near, min_distances(instances))  # fmt: skip
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for n, row in result["D_N"].items():
        print(f"N={n:<3} D_N {row['difference']:+.5f} {row['ci95']}")
    print(f"decision: {result['decision']}; content supported: {result['content']['supported']}; "
          f"U is the proposed method: {result['proposed_method']['U_is_the_proposed_method']}")  # fmt: skip


if __name__ == "__main__":
    main()
