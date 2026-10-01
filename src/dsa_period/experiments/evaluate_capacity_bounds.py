"""LP capacity bounds N_UB (``mac``) and N_orth (``orth``) of scenario v2.0.

One job per instance. The instance's topology, traffic and fading are nested
across N (protocol v2), so the gains of the largest N are computed once and the
first n UEs give the deployment with n UEs. ``orth`` is monotone in N, so a
binary search over the grid finds, per eps, the largest feasible grid point;
unsolved grid points are implied by monotonicity and marked as such. ``mac`` is
not monotone in N, but s*_mac <= s*_orth, so it is solved only at grid points
where ``orth`` is infeasible at some eps (elsewhere it is implied feasible).

An LP that ends without an optimal solution (e.g. at the time limit) counts as
feasible, which keeps the bound valid; such solves are flagged and counted.

Each finished instance is written at once to ``<output>/instances/``; with
``--resume`` a stopped run continues from the instances already written, under
the same settings and commit (``<output>/run.json``).
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dsa_period import splits_v2
from dsa_period.capacity import EPSILONS, eps_key, summarize
from dsa_period.capacity_bounds import KINDS, MONOTONE_KINDS, feasible_at, solve_bound
from dsa_period.config import load_scenario
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.experiments.evaluate_capacity import instances_for, print_summary


def _verdict(s_star: float | None, eps: float) -> bool:
    """Feasible at eps; an unsolved LP counts as feasible (the bound stays valid)."""
    ok = feasible_at(s_star, eps)
    return True if ok is None else ok


def _instance(job: tuple) -> dict:
    config, split, instance, grid, kinds, time_limit = job
    top = load_scenario(config, grid[-1])
    topology, _ = top.instance(instance)
    channel = top.make_channel(topology, splits_v2.fading_seeds(split, grid[-1], instance, 0))
    gains = channel.gain_block(0, top.horizon_slots)
    solved: dict = {}

    def solve(kind: str, n: int) -> dict:
        key = (kind, n)
        if key not in solved:
            scenario = top.with_num_ues(n)
            _, traffic = scenario.instance(instance)
            solved[key] = solve_bound(scenario, traffic, gains[:, :n, :], kind, time_limit)
        return solved[key]

    record: dict = {"instance": instance, "kinds": {}}
    feasible_by_kind: dict = {}
    for kind in (k for k in MONOTONE_KINDS if k in kinds):
        feasible: dict = {}
        for eps in EPSILONS:
            # Largest feasible position by binary search; s* is nondecreasing in N.
            low, high = -1, len(grid)  # grid[low] feasible, grid[high] infeasible
            while high - low > 1:
                middle = (low + high) // 2
                ok = _verdict(solve(kind, grid[middle])["s_star"], eps)
                low, high = (middle, high) if ok else (low, middle)
            feasible[eps_key(eps)] = {n: position <= low for position, n in enumerate(grid)}
        feasible_by_kind[kind] = feasible
        record["kinds"][kind] = {
            str(n): {
                **solved.get((kind, n), {"s_star": None, "implied": True}),
                "feasible": {key: feasible[key][n] for key in feasible},
            }
            for n in grid
        }
    orth_feasible = feasible_by_kind.get("orth", {})
    if "mac" in kinds:
        rows = {}
        for n in grid:
            # s*_mac <= s*_orth: where orth is feasible at every eps, so is mac.
            if orth_feasible and all(orth_feasible[key][n] for key in orth_feasible):
                rows[n] = {"s_star": None, "implied_by_orth": True}
                rows[n]["feasible"] = {key: True for key in orth_feasible}
            else:
                row = solve("mac", n)
                rows[n] = {
                    **row,
                    "feasible": {eps_key(e): _verdict(row["s_star"], e) for e in EPSILONS},
                }
        record["kinds"]["mac"] = {str(n): row for n, row in rows.items()}
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("dev", "validation", "test"), required=True)
    parser.add_argument("--first-instance", type=int, help="dev split only")
    parser.add_argument("--instances", type=int, help="dev split only")
    parser.add_argument("--grid", type=int, nargs="+", help="default: capacity grid + extra")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=["mac", "orth"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--time-limit", type=float, default=3600.0, help="seconds per LP")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="continue a stopped run")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--confirm-test", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.resume:
        parser.error(f"output {args.output} already exists (use --resume to continue it)")
    provenance = git_provenance()
    if args.formal and provenance["dirty"]:
        parser.error("--formal requires a clean, committed working tree")
    if args.split == "test" and not (args.confirm_test and args.formal):
        parser.error("test evaluation requires --confirm-test and --formal")
    try:
        chosen = instances_for(args.split, args.first_instance, args.instances)
    except ValueError as exc:
        parser.error(str(exc))
    base = load_scenario(args.config, 4)
    grid = sorted(args.grid or [*base.capacity["grid"], *base.capacity["reference_extra"]])
    settings = {
        "protocol_id": splits_v2.PROTOCOL_ID,
        "formal": args.formal,
        "commit": provenance["commit"],
        "config": args.config,
        "split": args.split,
        "instances": chosen,
        "grid": grid,
        "kinds": list(args.kinds),
        "time_limit_s": args.time_limit,
    }
    folder = args.output / "instances"
    if args.resume and args.output.exists():
        previous = json.loads((args.output / "run.json").read_text())
        if previous != settings:
            parser.error("--resume needs the same settings and commit as the stopped run")
    else:
        folder.mkdir(parents=True)
        (args.output / "run.json").write_text(json.dumps(settings, indent=2))
    records = [json.loads(f.read_text()) for f in sorted(folder.glob("*.json"))]
    finished = {r["instance"] for r in records}
    jobs = [
        (args.config, args.split, i, grid, tuple(args.kinds), args.time_limit)
        for i in chosen
        if i not in finished
    ]
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_instance, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            (folder / f"{record['instance']}.json").write_text(json.dumps(record))
            records.append(record)
            print(
                f"{len(records)}/{len(chosen)} instances (last {record['instance']}, "
                f"{done} this run), {time.time() - start:.0f}s",
                flush=True,
            )
    records.sort(key=lambda r: r["instance"])
    summaries = {}
    never = {eps_key(e): False for e in EPSILONS}
    for kind in args.kinds:
        # "infeasible_for_all" is the undeliverable-packet test of the method
        # evaluation, not a bound's own verdict; bounds report it as False.
        results = {
            n: {
                "-": [
                    {
                        "instance": r["instance"],
                        "feasible": r["kinds"][kind][str(n)]["feasible"],
                        "infeasible_for_all": never,
                    }
                    for r in records
                ]
            }
            for n in grid
        }
        summaries[kind] = summarize(results)
        print_summary(f"LP bound {kind}", summaries[kind])
    solve_times = [
        row["solve_s"]
        for r in records
        for rows in r["kinds"].values()
        for row in rows.values()
        if "solve_s" in row
    ]
    unsolved = sum(
        1
        for r in records
        for rows in r["kinds"].values()
        for row in rows.values()
        if "solve_s" in row and row["s_star"] is None
    )
    if solve_times:
        print(
            f"LP solves: {len(solve_times)}, max {max(solve_times):.1f}s, "
            f"total {sum(solve_times):.0f}s, unsolved (counted feasible) {unsolved}"
        )
    record = {
        **settings,
        "provenance": provenance,
        "summaries": summaries,
        "per_instance": records,
    }
    (args.output / "summary.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
