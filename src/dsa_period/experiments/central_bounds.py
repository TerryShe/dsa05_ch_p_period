"""Lower and upper bounds on the minimum energy for any N, verified in the environment.

Lower bound: per-UE interference-free optimum (``central.solo_reference``), valid
for every schedule including co-channel sharing. Upper bound: minimum-energy
orthogonal cyclic time-sharing schedule (``central_timeshare``), replayed in
PeriodicUplinkEnv; its measured energy is the bound. When the two coincide the
optimum is exact. Local heuristics run on the same instances for comparison.
``--formal`` refuses a dirty Git tree and records the commit.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dsa_period.baselines import POLICIES
from dsa_period.central import solo_reference
from dsa_period.central_timeshare import act_cycle, solve_orthogonal_cycle
from dsa_period.config import load_scenario
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.experiments.scenario_smoke import run_policy
from dsa_period.metrics import window_summary
from dsa_period.splits import indices

RTOL = 1e-9
EXACT_RTOL = 1e-9


def _capped(energies_by_ue: list[list[float]], cap: float) -> float:
    """Drop each UE's floor(cap * n_i) most expensive packets (orthogonal: no coupling)."""
    total = 0.0
    for energies in energies_by_ue:
        keep = len(energies) - int(np.floor(cap * len(energies) + 1e-9))
        total += float(np.sum(np.sort(energies)[:keep]))
    return total


def evaluate_instance(
    scenario, index: int, policy_seed: int, time_limit_s: float, heuristics: bool = True
) -> dict:
    config, menu = scenario.physical, scenario.menu
    cap = scenario.objective["max_violation_rate"]
    topology, traffic = scenario.instance(index)
    due = traffic.packets_due(scenario.warmup_slots, scenario.horizon_slots)
    lower = solo_reference(config, menu, topology, traffic)
    lower_energy = lower.window_energy_j(due, cap)
    schedule = solve_orthogonal_cycle(config, menu, topology, traffic, time_limit_s=time_limit_s)
    row = {
        "instance": index,
        "topology": topology.to_record(),
        "traffic": traffic.to_record(),
        "window_packets": int(due.sum()),
        "lower": {
            "strict_j": lower_energy["strict_j"],
            "capped_j": lower_energy["capped_j"],
            "per_ue_packet_energy_j": [plan.energy_j for plan in lower.plans],
        },
    }
    if schedule is None:
        row["upper"] = None
    else:
        env = scenario.make_env(topology, traffic)
        env.reset()
        while not env.truncated:
            env.step(act_cycle(schedule, env.elapsed_steps))
        jobs = env.job_records()
        witness = window_summary(
            jobs,
            env.energy_by_slot(),
            warmup_slots=scenario.warmup_slots,
            horizon_slots=scenario.horizon_slots,
            slot_duration_s=config.slot_duration_s,
            num_ues=config.num_ues,
        )
        settled = [job for job in jobs if job["success"] is not None]
        window = [
            job
            for job in jobs
            if scenario.warmup_slots < job["deadline_slot"] <= scenario.horizon_slots
        ]
        planned = [
            schedule.planned_energy_j(
                job["ue"],
                job["arrival_slot"],
                int(traffic.period_slots[job["ue"]]),
                int(traffic.phase_slots[job["ue"]]),
            )
            for job in settled
        ]
        strict = witness["packet_energy_j"]
        checks = {
            "all_settled_on_time": all(job["success"] for job in settled),
            "at_most_k_transmitters": bool(
                np.all((schedule.power_ids > 0).sum(axis=1) <= config.num_subchannels)
            ),
            "job_energy_within_plan": all(
                job["energy_j"] <= energy * (1 + RTOL)
                for job, energy in zip(settled, planned, strict=True)
            ),
            "window_packet_counts_match": [r["packets"] for r in witness["per_ue"]] == due.tolist(),
            "lower_not_above_upper": lower_energy["strict_j"] <= strict * (1 + RTOL),
        }
        if not all(checks.values()):
            raise AssertionError(f"instance {index}: bound check failed {checks}")
        by_ue = [[j["energy_j"] for j in window if j["ue"] == ue] for ue in range(config.num_ues)]
        levels = schedule.power_ids[schedule.power_ids > 0]
        row["upper"] = {
            "strict_j": strict,
            "capped_j": _capped(by_ue, cap),
            "cycle_slots": schedule.cycle_slots,
            "cycle_energy_j": schedule.cycle_energy_j,
            "power_ids": schedule.power_ids.tolist(),
            "mip_status": schedule.mip_status,
            "mip_gap": schedule.mip_gap,
            "mip_dual_bound_j": schedule.mip_dual_bound_j,
            "solve_seconds": schedule.solve_seconds,
            "slot_utilization": float((schedule.power_ids > 0).sum() / schedule.power_ids.shape[0])
            / config.num_subchannels,
            "slot_share_by_power_id": {
                str(level): float(np.mean(levels == level)) for level in range(1, menu.num_levels)
            },
            "early_completions": int(
                sum(
                    job["energy_j"] < energy * (1 - RTOL)
                    for job, energy in zip(settled, planned, strict=True)
                )
            ),
            "checks": checks,
        }
        row["upper_over_lower"] = strict / lower_energy["strict_j"]
        row["exact"] = strict <= lower_energy["strict_j"] * (1 + EXACT_RTOL)
    results = {}
    for policy in POLICIES if heuristics else ():
        s = run_policy(scenario, index, policy, policy_seed)["summary"]
        results[policy] = {
            "ocr": s["ocr"],
            "worst_ue_ocr": s["worst_ue_ocr"],
            "packet_energy_j": s["packet_energy_j"],
            "average_power_per_ue_w": s["average_power_per_ue_w"],
            "energy_over_lower": s["packet_energy_j"] / lower_energy["strict_j"],
            "energy_over_upper": (
                None if row["upper"] is None else s["packet_energy_j"] / row["upper"]["strict_j"]
            ),
        }
    row["heuristics"] = results
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--first-instance", type=int, default=0)
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument(
        "--split",
        choices=("validation", "test", "confirm1"),
        help="use a frozen protocol split or confirmation set instead",
    )
    parser.add_argument(
        "--skip-heuristics", action="store_true", help="bounds only; do not run baselines"
    )
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--time-limit-s", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"output {args.output} already exists")
    provenance = git_provenance()
    if args.formal and provenance["dirty"]:
        parser.error("--formal requires a clean, committed working tree")
    scenario = load_scenario(args.config)
    if args.split is not None:
        chosen = indices(args.split)
        args.first_instance, args.instances = chosen.start, len(chosen)
    rows = [
        evaluate_instance(
            scenario, index, args.policy_seed, args.time_limit_s, not args.skip_heuristics
        )
        for index in range(args.first_instance, args.first_instance + args.instances)
    ]
    solved = [r for r in rows if r["upper"] is not None]
    ratio = np.array([r["upper_over_lower"] for r in solved])
    print(
        f"{scenario.experiment['id']}: {len(rows)} instances, upper bound for {len(solved)}, "
        f"exact for {sum(r['exact'] for r in solved)}; all checks passed"
    )
    if solved:
        print(
            f"UB/LB mean {ratio.mean():.4f} median {np.median(ratio):.4f} max {ratio.max():.4f}; "
            f"slot utilization {np.mean([r['upper']['slot_utilization'] for r in solved]):.3f}; "
            f"MILP max {max(r['upper']['solve_seconds'] for r in solved):.2f} s"
        )
    print(f"{'policy':<12} {'OCR':>7} {'worst-UE':>9} {'E/LB':>7} {'E/UB':>7}")
    for policy in POLICIES if not args.skip_heuristics else ():
        h = [r["heuristics"][policy] for r in rows]
        upper = [x["energy_over_upper"] for x in h if x["energy_over_upper"] is not None]
        print(
            f"{policy:<12} {np.mean([x['ocr'] for x in h]):7.4f} "
            f"{np.mean([x['worst_ue_ocr'] for x in h]):9.4f} "
            f"{np.mean([x['energy_over_lower'] for x in h]):7.3f} "
            f"{np.mean(upper) if upper else float('nan'):7.3f}"
        )
    if args.output is not None:
        args.output.mkdir(parents=True)
        record = {
            "formal": args.formal,
            "provenance": provenance,
            "config": args.config,
            "split": args.split,
            "first_instance": args.first_instance,
            "instances": args.instances,
            "heuristics": not args.skip_heuristics,
            "policy_seed": args.policy_seed,
            "time_limit_s": args.time_limit_s,
            "runs": rows,
        }
        (args.output / "summary.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
