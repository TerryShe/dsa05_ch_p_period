"""Exact centralized minimum-energy reference for N <= K, verified in the environment.

For each instance: solve the per-UE packet plans, replay them as a witness in
PeriodicUplinkEnv (all packets on time and the measured packet energy equal to
the plan), and run the local heuristics on the same instance for comparison.
``--formal`` refuses a dirty Git tree and records the commit.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from dsa_period.baselines import POLICIES
from dsa_period.central import act_central, central_reference
from dsa_period.config import load_scenario
from dsa_period.experiments.scenario_smoke import run_policy
from dsa_period.metrics import window_summary

RTOL = 1e-9


def git_provenance() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(git("status", "--porcelain"))}


def evaluate_instance(scenario, index: int, policy_seed: int) -> dict:
    topology, traffic = scenario.instance(index)
    reference = central_reference(scenario.physical, scenario.menu, topology, traffic)
    env = scenario.make_env(topology, traffic)
    observations = env.reset()
    while not env.truncated:
        observations = env.step(act_central(reference, observations)).observations
    jobs = env.job_records()
    witness = window_summary(
        jobs,
        env.energy_by_slot(),
        warmup_slots=scenario.warmup_slots,
        horizon_slots=scenario.horizon_slots,
        slot_duration_s=scenario.physical.slot_duration_s,
        num_ues=scenario.physical.num_ues,
    )
    due = traffic.packets_due(scenario.warmup_slots, scenario.horizon_slots)
    energy = reference.window_energy_j(due, scenario.objective["max_violation_rate"])
    settled = [job for job in jobs if job["success"] is not None]
    checks = {
        "all_settled_on_time": all(job["success"] for job in settled),
        "window_packet_counts_match": [row["packets"] for row in witness["per_ue"]] == due.tolist(),
        "job_energy_equals_plan": all(
            np.isclose(job["energy_j"], reference.plans[job["ue"]].energy_j, rtol=RTOL, atol=0)
            for job in settled
        ),
        "window_energy_equals_plan": bool(
            np.isclose(witness["packet_energy_j"], energy["strict_j"], rtol=RTOL, atol=0)
        ),
        "continuous_bound_below_plan": all(
            bound is None or bound <= plan.energy_j * (1 + RTOL)
            for bound, plan in zip(reference.continuous_bounds_j, reference.plans, strict=True)
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"instance {index}: witness check failed {checks}")
    heuristics = {
        policy: run_policy(scenario, index, policy, policy_seed)["summary"] for policy in POLICIES
    }
    packets = int(due.sum())
    return {
        "instance": index,
        "topology": topology.to_record(),
        "traffic": traffic.to_record(),
        "plans": [
            {
                "ue": ue,
                "distance_m": float(topology.ue_ap_distances_m[ue]),
                "period_slots": int(traffic.period_slots[ue]),
                "packet_bits": float(traffic.packet_bits[ue]),
                "counts_by_power_id": {
                    str(level): count for level, count in enumerate(plan.counts, start=1)
                },
                "slots": plan.slots,
                "energy_j": plan.energy_j,
                "continuous_bound_j": bound,
            }
            for ue, (plan, bound) in enumerate(
                zip(reference.plans, reference.continuous_bounds_j, strict=True)
            )
        ],
        "window_packets": packets,
        "reference": {
            **energy,
            "strict_energy_per_packet_j": energy["strict_j"] / packets,
            "average_power_per_ue_w": reference.average_power_per_ue_w,
        },
        "witness": {"checks": checks, "summary": witness},
        "heuristics": {
            policy: {
                "ocr": s["ocr"],
                "worst_ue_ocr": s["worst_ue_ocr"],
                "packet_energy_j": s["packet_energy_j"],
                "energy_ratio_to_strict_reference": s["packet_energy_j"] / energy["strict_j"],
                "average_power_per_ue_w": s["average_power_per_ue_w"],
                "per_ue_average_power_w": [row["average_power_w"] for row in s["per_ue"]],
                "per_ue_ocr": [row["ocr"] for row in s["per_ue"]],
            }
            for policy, s in heuristics.items()
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scenario-v1-n4.toml")
    parser.add_argument("--first-instance", type=int, default=0)
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"output {args.output} already exists")
    provenance = git_provenance()
    if args.formal and provenance["dirty"]:
        parser.error("--formal requires a clean, committed working tree")
    scenario = load_scenario(args.config)
    rows = [
        evaluate_instance(scenario, index, args.policy_seed)
        for index in range(args.first_instance, args.first_instance + args.instances)
    ]
    ref = [r["reference"] for r in rows]
    print(f"{scenario.experiment['id']}: {len(rows)} instances, all witness checks passed")
    print(
        f"central reference: {1e6 * np.mean([x['strict_energy_per_packet_j'] for x in ref]):.2f} "
        f"uJ/packet, {1e3 * np.mean([x['average_power_per_ue_w'] for x in ref]):.3f} mW/UE; "
        f"capped/strict = {np.sum([x['capped_j'] for x in ref]) / np.sum([x['strict_j'] for x in ref]):.4f}"
    )
    print(f"{'policy':<12} {'OCR':>7} {'worst-UE':>9} {'mW/UE':>8} {'E/E*':>7}")
    for policy in POLICIES:
        h = [r["heuristics"][policy] for r in rows]
        print(
            f"{policy:<12} {np.mean([x['ocr'] for x in h]):7.4f} "
            f"{np.mean([x['worst_ue_ocr'] for x in h]):9.4f} "
            f"{1e3 * np.mean([x['average_power_per_ue_w'] for x in h]):8.3f} "
            f"{np.mean([x['energy_ratio_to_strict_reference'] for x in h]):7.3f}"
        )
    if args.output is not None:
        args.output.mkdir(parents=True)
        record = {
            "formal": args.formal,
            "provenance": provenance,
            "config": args.config,
            "first_instance": args.first_instance,
            "instances": args.instances,
            "policy_seed": args.policy_seed,
            "runs": rows,
        }
        (args.output / "summary.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
