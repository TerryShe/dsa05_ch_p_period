"""EXP-DSA05-011 analysis exactly as pre-registered (design §5).

Inputs: formal ``evaluate_capacity --screen exp011-eval512`` outputs and the
evaluation plan written by ``scripts/run_exp011.py``:

    {"groups": {name: [layer, ...] | spec}, "grids": {name: [N, ...]}}

where a layer maps "N" to the spec used at N (learned methods, one layer per
training seed) and a plain spec is a method without training seeds. Group
names: U, M, M+Y, O, L, U-hide-all, U-hide-nonurgent, U-hide-urgent,
best_local, best_local+Y, CG-opt, N_EDF, M_exp009 at every N; U@θ and M+Y@θ for
θ in {all, -15, -5, 0} and U0 at N = 8, 10, 12 (U and M+Y are the -10 dB points).

Per deployment: w = the worst UE's violation rate (P2 is its mean), feasibility
at eps = 0.01 (P1), the all-UE mean violation rate (guard) and log10(w + delta),
delta = 1e-4 (design §5.1). Differences pair deployments and resample each
group's seed layers on their own (``evaluation._bootstrap``), 10,000 draws,
seed 20,260,930.

* Main (§5.2): D_N = P2(U) - P2(M) at N = 8, 10, 12: "effective" if every 95%
  upper bound < 0, "partly effective" if one or two, "no evidence" if none.
* Secondary (§5.3): E_N = P2(U) - P2(M+Y) and the method entering the final
  confirmation (lower mean P2 over N = 8, 10, 12); U - O and U - L; H-M2 (U with
  the broadcast hidden from all UEs) and H-M1 (hidden from non-urgent UEs loses
  at least half of the gain P2(M) - P2(U)); the threshold boundary of U and M+Y
  (the harshest θ whose difference to M has a 95% upper bound < 0) and U0 at
  0 dB; D_N over all N; the frontier table; P1 and the sensitivity analyses.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dsa_period import splits_v2
from dsa_period.capacity import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, FEASIBLE_EPISODE_SHARE
from dsa_period.evaluation import _bootstrap, _interval

SET = "exp011-eval512"
GRID = (6, 8, 10, 12, 16)
CORE = (8, 10, 12)
THETAS = ("all", "-15", "-10", "-5", "0")  # most lenient first; -10 dB is the main condition
DELTA = 1e-4
HIDES = ("all", "nonurgent", "urgent")
BITS = {  # information per slot, design §2.6 (downlink bits; uplink bit per transmission)
    "CG-opt": "grants at configuration only",
    "M": "K RSSI values + own outcome (L0)",
    "U": "L0 + K = 4 bits down + 1 bit up per transmission",
    "N_EDF": "per-slot grants from a central scheduler with every UE's state",
}


def load(paths: list[Path], formal: bool = True, screen: str = SET) -> tuple[dict, list[int]]:
    """spec -> {N: episodes} over formal outputs of an evaluation slice (default exp011-eval512)."""
    episodes: dict[str, dict] = {}
    instances = splits_v2.screening_instances(screen)
    for path in paths:
        record = json.loads((path / "summary.json").read_text())
        ok = (
            record["protocol_id"] == splits_v2.PROTOCOL_ID
            and record["split"] == "dev"
            and record.get("screen") == screen
            and record["instances"] == instances
            and (record["formal"] or not formal)
        )
        if not ok:
            raise ValueError(f"{path}: not a formal {screen} evaluation")
        for spec, by_n in json.loads((path / "episodes.json").read_text()).items():
            for n, rows in by_n.items():
                if int(n) in episodes.get(spec, {}):
                    raise ValueError(f"{spec} at N = {n} was evaluated twice")
                episodes.setdefault(spec, {})[int(n)] = rows
    return episodes, instances


FIELDS = {
    "w": lambda e: float(e["worst_violation_rate"]),
    "feasible": lambda e: float(e["feasible"]["0.01"]),
    "mean_rate": lambda e: float(np.mean(e["violation_rate"])),
    "log_w": lambda e: float(np.log10(e["worst_violation_rate"] + DELTA)),
    "hopeless": lambda e: float(e["infeasible_for_all"]["0.01"]),
}


def matrix(episodes: dict, group, n: int, field: str, instances: list[int]) -> np.ndarray:
    """(seed layers, deployments) of one per-deployment quantity."""
    members = group if isinstance(group, list) else [group]
    rows = []
    for member in members:
        spec = member if isinstance(member, str) else member[str(n)]
        cells = {e["instance"]: FIELDS[field](e) for e in episodes[spec][n]}
        rows.append([cells[i] for i in instances])
    return np.array(rows)


def compare(a: np.ndarray, b: np.ndarray, draws: int) -> dict:
    return {
        "difference": float(a.mean() - b.mean()),
        "ratio": float(a.mean() / b.mean()) if b.mean() > 0 else None,
        "ci95": _interval(_bootstrap(a, b, draws, BOOTSTRAP_SEED)),
    }


def main_decision(upper: dict) -> str:
    significant = sum(upper[n] < 0 for n in CORE)
    if significant == len(CORE):
        return "effective"
    return "partly effective" if significant else "no evidence"


def boundary(rows: dict) -> str | None:
    """The harshest threshold whose difference to M has a 95% upper bound < 0."""
    passed = [theta for theta in THETAS if rows[theta]["ci95"][1] < 0]
    return passed[-1] if passed else None


def capacity_on(grid, shares: list[float]) -> int | None:
    """N* on the evaluation grid: the point before the first F < 0.95 (descriptive)."""
    for i, share in enumerate(shares):
        if share < FEASIBLE_EPISODE_SHARE:
            return grid[i - 1] if i else None
    return grid[-1]


def analyze(episodes: dict, plan: dict, instances: list[int], draws: int) -> dict:
    groups, grids = plan["groups"], {k: tuple(v) for k, v in plan["grids"].items()}

    def mat(name: str, n: int, field: str = "w") -> np.ndarray:
        return matrix(episodes, groups[name], n, field, instances)

    out: dict = {"plan": plan, "instances": len(instances), "delta": DELTA}
    out["P2"] = {g: {n: float(mat(g, n).mean()) for n in grids[g]} for g in groups}
    out["P1"] = {g: {n: float(mat(g, n, "feasible").mean()) for n in grids[g]} for g in groups}
    out["N_star_on_grid"] = {
        g: capacity_on(GRID, [out["P1"][g][n] for n in GRID]) for g in groups if grids[g] == GRID
    }
    out["all_ue_mean_rate"] = {
        g: {n: float(mat(g, n, "mean_rate").mean()) for n in grids[g]} for g in groups
    }
    # 90% quantile of w per seed layer, then the mean over layers (framing §3.3).
    out["p90_w"] = {
        g: {n: float(np.percentile(mat(g, n), 90, axis=1).mean()) for n in grids[g]} for g in groups
    }

    # Main rule (§5.2) and the load profile (§5.3 item 5).
    d = {n: compare(mat("U", n), mat("M", n), draws) for n in GRID}
    out["D_N"] = d
    out["decision"] = main_decision({n: d[n]["ci95"][1] for n in CORE})
    out["harmful_N"] = [n for n in GRID if d[n]["ci95"][0] > 0]

    # Coordination (§5.3 item 1).
    e = {n: compare(mat("U", n), mat("M+Y", n), draws) for n in CORE}
    learned_better = sum(e[n]["ci95"][1] < 0 for n in CORE)
    mean_core = {g: float(np.mean([out["P2"][g][n] for n in CORE])) for g in ("U", "M+Y")}
    out["coordination"] = {
        "E_N": e,
        "learned_better_at": learned_better,
        "learned_response_better": learned_better >= 2,
        "core_mean_P2": mean_core,
        "enters_final_confirmation": min(mean_core, key=mean_core.get),
    }

    # Content (§5.3 item 2).
    out["content"] = {
        other: {n: compare(mat("U", n), mat(other, n), draws) for n in GRID} for other in ("O", "L")
    }

    # Mechanism (§5.3 item 3).
    mech: dict = {}
    for n in GRID:
        u, m = mat("U", n), mat("M", n)
        gain = float(m.mean() - u.mean())
        row = {"gain_P2_M_minus_U": gain}
        for hide in HIDES:
            row[hide] = compare(mat(f"U-hide-{hide}", n), u, draws)
        lost = row["nonurgent"]["difference"]
        row["H_M2"] = row["all"]["ci95"][0] > 0
        row["H_M1"] = row["nonurgent"]["ci95"][0] > 0 and gain > 0 and lost >= 0.5 * gain
        mech[n] = row
    out["mechanism"] = mech

    # Receive conditions (§3.5, §5.3 item 4) at N = 8, 10, 12.
    receive: dict = {}
    for method in ("U", "M+Y"):
        receive[method] = {}
        for n in CORE:
            rows = {
                theta: compare(
                    mat(method if theta == "-10" else f"{method}@{theta}", n), mat("M", n), draws
                )
                for theta in THETAS
            }
            receive[method][n] = {"by_theta": rows, "boundary": boundary(rows)}
    receive["U0"] = {
        n: {
            "U0_minus_M": compare(mat("U0", n), mat("M", n), draws),
            "U0_minus_U_at_0dB": compare(mat("U0", n), mat("U@0", n), draws),
        }
        for n in CORE
    }
    out["receive"] = receive

    # Frontier table (§5.3 item 6) and the guard and sensitivity analyses (§5.1).
    out["frontier"] = {
        n: {g: {"P2": out["P2"][g][n], "information": BITS[g]} for g in BITS} for n in GRID
    }
    keep = {n: mat("U", n, "hopeless")[0] == 0 for n in GRID}
    out["sensitivity"] = {
        n: {
            "all_ue_mean_rate_U_minus_M": compare(
                mat("U", n, "mean_rate"), mat("M", n, "mean_rate"), draws
            ),
            "log_w_U_minus_M": compare(mat("U", n, "log_w"), mat("M", n, "log_w"), draws),
            "excluding_infeasible_for_all": {
                "deployments": int(keep[n].sum()),
                **compare(mat("U", n)[:, keep[n]], mat("M", n)[:, keep[n]], draws),
            },
            "p90_U_minus_M": out["p90_w"]["U"][n] - out["p90_w"]["M"][n],
        }
        for n in GRID
    }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--allow-informal", action="store_true", help="smoke tests only")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    if args.resamples != BOOTSTRAP_RESAMPLES and not args.allow_informal:
        parser.error("the formal analysis uses the protocol's bootstrap size")
    episodes, instances = load(args.evaluations, not args.allow_informal)
    result = analyze(episodes, json.loads(args.plan.read_text()), instances, args.resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for n, row in result["D_N"].items():
        print(f"N={n:<3} D_N {row['difference']:+.5f} {row['ci95']}")
    print(f"decision: {result['decision']}; enters final confirmation: "
          f"{result['coordination']['enters_final_confirmation']}")  # fmt: skip


if __name__ == "__main__":
    main()
