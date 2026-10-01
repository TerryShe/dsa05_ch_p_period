"""Summarize the scenario v2.0 phase-0 magnitude check (dev instances).

Reads the outputs of ``evaluate_capacity`` (central scheduler, local rules) and
``evaluate_capacity_bounds`` and prints, per eps, F(N) and N* of every method,
the per-N best local rule (the envelope a per-N calibration could reach at most
on these instances), the LP bounds, undeliverable-packet statistics, the check
criteria of scenario v2.0 §11 and the LP solve times.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dsa_period.capacity import EPSILONS, capacity, eps_key

LOCAL_EXCLUDED = ("central_edf",)


def load(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def star(point: dict) -> str:
    if point["at_least"]:
        return f">={point['n_star']}"
    return "<4" if point["n_star"] == 0 else str(point["n_star"])


def table(rows: dict[str, dict], grid: list[int], key: str) -> None:
    header = " ".join(f"{n:>5}" for n in grid)
    print(f"  {'method':<42} {'N*':>5}  {header}")
    for name, summary in rows.items():
        block = summary["per_eps"][key]
        shares = {r["num_ues"]: r["feasible_share"] for r in block["rows"]}
        cells = " ".join(f"{shares[n]:5.2f}" if n in shares else f"{'':>5}" for n in grid)
        print(f"  {name:<42} {star(block['capacity']):>5}  {cells}")


def envelope(rules: dict[str, dict], key: str) -> tuple[list[int], list[float], list[str]]:
    grid = sorted({r["num_ues"] for s in rules.values() for r in s["per_eps"][key]["rows"]})
    best, names = [], []
    for n in grid:
        candidates = {
            name: next(r["feasible_share"] for r in s["per_eps"][key]["rows"] if r["num_ues"] == n)
            for name, s in rules.items()
        }
        name = max(candidates, key=candidates.get)
        best.append(candidates[name])
        names.append(name)
    return grid, best, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--bounds", type=Path)
    args = parser.parse_args()
    edf, rules = load(args.edf), load(args.rules)
    bounds = load(args.bounds) if args.bounds else None
    grid = sorted(set(edf["grid"]) | set(rules["grid"]) | set(bounds["grid"] if bounds else []))
    print(f"instances: {edf['split']} {edf['instances'][0]}-{edf['instances'][-1]}")
    local = {k: v for k, v in rules["summaries"].items() if not k.startswith(LOCAL_EXCLUDED)}
    for eps in EPSILONS:
        key = eps_key(eps)
        print(f"\n=== eps = {eps}: F(N) and N* (first-drop rule, F >= 0.95) ===")
        rows = dict(edf["summaries"])
        if bounds:
            rows.update({f"LP bound {k}": v for k, v in bounds["summaries"].items()})
        rows.update(local)
        table(rows, grid, key)
        env_grid, best, names = envelope(local, key)
        point = capacity(env_grid, best)
        print(f"  {'best local rule per N (envelope)':<42} {star(point):>5}  ", end="")
        print(" ".join(f"{b:5.2f}" for b in best))
        for n, name in zip(env_grid, names, strict=True):
            print(f"    N={n}: {name}")
        edf_star = edf["summaries"]["central_edf"]["per_eps"][key]["capacity"]
        print(
            f"  check (a) 4 < N_EDF <= 32: N_EDF = {star(edf_star)}; "
            f"(b) best local rule at least one grid point lower: envelope N* = {star(point)}"
        )
        for name, summary in edf["summaries"].items():
            for row in summary["per_eps"][key]["rows"]:
                if row["infeasible_for_all_share"] > 0:
                    print(
                        f"  {name} N={row['num_ues']}: infeasible-for-all share "
                        f"{row['infeasible_for_all_share']:.3f}"
                    )
    episodes = json.loads((args.edf / "episodes.json").read_text())["central_edf"]
    lost = sum(sum(e["undeliverable"]) for eps_list in episodes.values() for e in eps_list)
    total = sum(sum(e["packets"]) for eps_list in episodes.values() for e in eps_list)
    print(f"\nundeliverable packets over all EDF episodes: {lost} of {total}")
    if bounds:
        times = [
            (int(n), row["solve_s"], kind)
            for r in bounds["per_instance"]
            for kind, rows in r["kinds"].items()
            for n, row in rows.items()
            if "solve_s" in row
        ]
        for kind in ("mac", "orth"):
            sel = [(n, t) for n, t, k in times if k == kind]
            if sel:
                largest = max(n for n, _ in sel)
                top = [t for n, t in sel if n == largest]
                print(
                    f"LP {kind}: {len(sel)} solves, median {np.median([t for _, t in sel]):.1f}s, "
                    f"N={largest} median {np.median(top):.1f}s max {max(top):.1f}s"
                )


if __name__ == "__main__":
    main()
