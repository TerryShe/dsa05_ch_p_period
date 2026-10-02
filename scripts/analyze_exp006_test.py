"""EXP-DSA05-006 test analysis exactly as pre-registered (design §6 and §13).

Inputs: the three ``evaluate_capacity --split test`` outputs (learned policies,
calibrated local rules with ``best_local``, N_EDF) and the phase-2 selection.

* Main table: F(N) and N* of every method at eps = 0.01 and 0.05; the learned
  policy pools its seeds (F averaged over seeds per instance), for the selected
  and for the last checkpoints.
* H1: N*_0.01(MARL, selected) > N*_0.01(best_local); also the bootstrap
  distribution of both N* and the share of draws with N*_MARL > N*_local
  (instances resampled jointly over N and methods, seeds over N).
* H2: for N in {6, 8, 10, 12}, the 95% interval of F_0.01(MARL) - F_0.01(best_local)
  has a lower bound > 0 (two-level paired bootstrap).
* Descriptive: share of the gap closed, eps = 0.05, last checkpoints.
* Supplement (§13): S = mean F_0.01 over N in {4, 6, 8, 10, 12, 16}, paired
  difference MARL - best_local; failure composition at N in {4, 6, 8}.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dsa_period.capacity import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EPSILONS,
    allowance,
    capacity,
    capacity_label,
    eps_key,
    summarize,
)
from dsa_period.evaluation import _bootstrap, _interval

H2_GRID = (6, 8, 10, 12)
S_GRID = (4, 6, 8, 10, 12, 16)
FAILURE_GRID = (4, 6, 8)
CLASSES = ("fast", "medium", "slow")


def feasible(episodes: list[dict], key: str, instances: list[int]) -> np.ndarray:
    cells = {e["instance"]: float(e["feasible"][key]) for e in episodes}
    return np.array([cells[i] for i in instances])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marl", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {
        name: json.loads((path / "summary.json").read_text())
        for name, path in (("marl", args.marl), ("local", args.local), ("edf", args.edf))
    }
    if {r["split"] for r in records.values()} != {"test"} or not all(
        r["formal"] for r in records.values()
    ):
        raise SystemExit("the analysis reads the formal test evaluations only")
    episodes = {
        name: json.loads((path / "episodes.json").read_text())
        for name, path in (("marl", args.marl), ("local", args.local), ("edf", args.edf))
    }
    selection = json.loads(args.selection.read_text())
    selected = [r["selected"] for r in selection["runs"].values()]
    last = [r["last"] for r in selection["runs"].values()]
    grid = sorted(int(n) for n in episodes["marl"][selected[0]])
    instances = sorted(e["instance"] for e in episodes["marl"][selected[0]][str(grid[0])])
    rng_seed, draws = BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES

    def pooled(specs: list[str]) -> dict:
        return summarize({n: {s: episodes["marl"][s][str(n)] for s in specs} for n in grid})

    methods = {"MARL (selected)": pooled(selected), "MARL (last)": pooled(last)}
    for spec in records["local"]["policies"]:
        methods[spec] = records["local"]["summaries"][spec]
    for spec in records["edf"]["policies"]:
        methods[spec] = records["edf"]["summaries"][spec]
    result: dict = {
        "commits": {k: r["provenance"] for k, r in records.items()},
        "selected": selected,
        "last": last,
        "instances": len(instances),
        "main": {},
    }
    for eps in EPSILONS:
        key = eps_key(eps)
        table = {}
        for name, summary in methods.items():
            block = summary["per_eps"][key]
            table[name] = {
                "capacity": capacity_label(
                    block["capacity"], [r["num_ues"] for r in block["rows"]]
                ),
                "capacity_bootstrap": block["capacity_bootstrap"],
                "F": {r["num_ues"]: r["feasible_share"] for r in block["rows"]},
                "ci95": {r["num_ues"]: r["ci95"] for r in block["rows"]},
            }
        result["main"][key] = table

    # Matrices: MARL [seeds, instances] per N; best_local [1, instances] per N.
    def mats(key: str) -> tuple[dict, dict]:
        marl = {
            n: np.stack([feasible(episodes["marl"][s][str(n)], key, instances) for s in selected])
            for n in grid
        }
        local = {
            n: feasible(episodes["local"]["best_local"][str(n)], key, instances)[None, :]
            for n in grid
        }
        return marl, local

    tests = {}
    for eps in EPSILONS:
        key = eps_key(eps)
        marl, local = mats(key)
        star_marl = capacity(grid, [marl[n].mean() for n in grid])
        star_local = capacity(grid, [local[n].mean() for n in grid])
        rng = np.random.default_rng(rng_seed)
        ahead = 0
        dist_marl: dict[str, int] = {}
        dist_local: dict[str, int] = {}
        for _ in range(draws):
            inst = rng.integers(len(instances), size=len(instances))
            seeds = rng.integers(len(selected), size=len(selected))
            m = capacity(grid, [marl[n][seeds][:, inst].mean() for n in grid])
            lo = capacity(grid, [local[n][:, inst].mean() for n in grid])
            ahead += m["n_star"] > lo["n_star"]
            for point, dist in ((m, dist_marl), (lo, dist_local)):
                label = capacity_label(point, grid)
                dist[label] = dist.get(label, 0) + 1
        h2 = {}
        for n in H2_GRID:
            diff = _bootstrap(marl[n], local[n], draws, rng_seed)
            h2[n] = {
                "difference": float(marl[n].mean() - local[n].mean()),
                "ci95": _interval(diff),
                "lower_bound_positive": bool(_interval(diff)[0] > 0),
            }
        edf_star = records["edf"]["summaries"]["central_edf"]["per_eps"][key]["capacity"]["n_star"]
        gap = edf_star - star_local["n_star"]
        s_marl = np.mean([marl[n] for n in S_GRID], axis=0)  # [seeds, instances]
        s_local = np.mean([local[n] for n in S_GRID], axis=0)
        s_diff = _bootstrap(s_marl, s_local, draws, rng_seed)
        tests[key] = {
            "N_star_marl": capacity_label(star_marl, grid),
            "N_star_best_local": capacity_label(star_local, grid),
            "H1_marl_greater": star_marl["n_star"] > star_local["n_star"],
            "bootstrap_share_marl_greater": ahead / draws,
            "bootstrap_N_star_marl": {k: v / draws for k, v in sorted(dist_marl.items())},
            "bootstrap_N_star_best_local": {k: v / draws for k, v in sorted(dist_local.items())},
            "H2": h2,
            "gap_closed": (star_marl["n_star"] - star_local["n_star"]) / gap if gap else None,
            "S": {
                "marl": float(s_marl.mean()),
                "best_local": float(s_local.mean()),
                "difference": float(s_marl.mean() - s_local.mean()),
                "ci95": _interval(s_diff),
            },
        }
    result["tests"] = tests

    failures = {}
    sources = {f"MARL {s.split('fixed-')[1].split('/')[0]}": ("marl", s) for s in selected}
    sources["best_local"] = ("local", "best_local")
    sources["central_edf"] = ("edf", "central_edf")
    for n in FAILURE_GRID:
        rows = {}
        for label, (group, spec) in sources.items():
            failed = []
            for e in episodes[group][spec][str(n)]:
                if e["feasible"]["0.01"]:
                    continue
                ues = []
                for u, rate in enumerate(e["violation_rate"]):
                    if rate > 0.01:
                        share = e["undeliverable"][u] / max(allowance(0.01, e["packets"][u]), 1)
                        ues.append(
                            {
                                "ue": u,
                                "class": CLASSES[e["class_index"][u]],
                                "distance_m": round(e["distance_m"][u], 1),
                                "shadowing_db": round(e["shadowing_db"][u], 1),
                                "undeliverable": e["undeliverable"][u],
                                "undeliverable_share_of_allowance": round(share, 3),
                                "violation_rate": round(rate, 4),
                            }
                        )
                failed.append({"instance": e["instance"], "ues": ues})
            heavy = [
                f
                for f in failed
                if any(u["undeliverable_share_of_allowance"] >= 0.3 for u in f["ues"])
            ]
            rows[label] = {
                "failed": failed,
                "share_with_u_over_A_ge_0.3": len(heavy) / len(failed) if failed else None,
            }
        failures[str(n)] = rows
    result["failure_composition"] = failures

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for key, t in tests.items():
        print(
            f"eps={key}: N* MARL {t['N_star_marl']} vs best_local {t['N_star_best_local']} "
            f"(H1 {t['H1_marl_greater']}, bootstrap P(MARL>local) {t['bootstrap_share_marl_greater']:.3f}); "
            f"gap closed {t['gap_closed']}"
        )
        print(
            f"    N* bootstrap MARL {t['bootstrap_N_star_marl']} local {t['bootstrap_N_star_best_local']}"
        )
        for n, h in t["H2"].items():
            print(
                f"    H2 N={n}: {h['difference']:+.3f} [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}] -> {h['lower_bound_positive']}"
            )
        s = t["S"]
        print(
            f"    S MARL {s['marl']:.3f} local {s['best_local']:.3f} diff {s['difference']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
        )
    for key, table in result["main"].items():
        print(f"main table eps={key}")
        for name, row in table.items():
            cells = " ".join(f"{n}:{f:.2f}" for n, f in row["F"].items())
            print(f"  {name[:40]:<40} N* {row['capacity']:<4} {cells}")
    for n, rows in failures.items():
        print(f"failures N={n}")
        for label, row in rows.items():
            print(
                f"  {label:<12} {[f['instance'] for f in row['failed']]} heavy share {row['share_with_u_over_A_ge_0.3']}"
            )


if __name__ == "__main__":
    main()
