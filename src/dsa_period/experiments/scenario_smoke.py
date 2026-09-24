"""Run the local heuristic policies on several scenario instances (informal).

Reports fixed-window on-time completion and energy. Use ``--packet-scale`` to
sweep load during calibration. Results are integration checks, not research
evidence; the output directory is never overwritten.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dsa_period.baselines import POLICIES, act
from dsa_period.config import Scenario, load_scenario
from dsa_period.metrics import window_summary


def run_policy(scenario: Scenario, index: int, policy: str, policy_seed: int) -> dict:
    topology, traffic = scenario.instance(index)
    env = scenario.make_env(topology, traffic)
    observations = env.reset()
    seeds = np.random.SeedSequence([policy_seed, index]).spawn(scenario.physical.num_ues)
    rngs = [np.random.Generator(np.random.PCG64(seed)) for seed in seeds]
    while not env.truncated:
        observations = env.step(act(policy, observations, env.config, env.menu, rngs)).observations
    summary = window_summary(
        env.job_records(),
        env.energy_by_slot(),
        warmup_slots=scenario.warmup_slots,
        horizon_slots=scenario.horizon_slots,
        slot_duration_s=scenario.physical.slot_duration_s,
        num_ues=scenario.physical.num_ues,
    )
    return {"instance": index, "policy": policy, "summary": summary}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scenario-v1-n12.toml")
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--packet-scale", type=float, default=1.0)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"output {args.output} already exists")
    scenario = load_scenario(args.config).with_packet_scale(args.packet_scale)
    topology, traffic = scenario.instance(0)
    offered = traffic.offered_bits_per_slot()
    print(
        f"offered load {offered:.0f} bit/slot "
        f"({offered / scenario.physical.slot_duration_s / 1e6:.3f} Mbit/s), "
        f"packet scale {args.packet_scale}"
    )
    rows = [
        run_policy(scenario, index, policy, args.policy_seed)
        for policy in args.policies
        for index in range(args.instances)
    ]
    print(f"{'policy':<12} {'OCR':>7} {'worst-UE':>9} {'mW/UE':>8} {'uJ/pkt':>9} {'lat':>6}")
    for policy in args.policies:
        s = [r["summary"] for r in rows if r["policy"] == policy]
        energy = [x["energy_per_on_time_packet_j"] for x in s if x["energy_per_on_time_packet_j"]]
        latency = [x["latency_slots_mean"] for x in s if x["latency_slots_mean"] is not None]
        print(
            f"{policy:<12} {np.mean([x['ocr'] for x in s]):7.4f} "
            f"{np.mean([x['worst_ue_ocr'] for x in s]):9.4f} "
            f"{1e3 * np.mean([x['average_power_per_ue_w'] for x in s]):8.3f} "
            f"{1e6 * np.mean(energy) if energy else float('nan'):9.2f} "
            f"{np.mean(latency) if latency else float('nan'):6.2f}"
        )
    if args.output is not None:
        args.output.mkdir(parents=True)
        record = {
            "formal": False,
            "config": str(args.config),
            "packet_scale": args.packet_scale,
            "policy_seed": args.policy_seed,
            "topology_0": topology.to_record(),
            "traffic_0": traffic.to_record(),
            "runs": rows,
        }
        (args.output / "summary.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
