"""EXP-DSA05-010 descriptive diagnostics (design §3.2): agreement with N_EDF, collisions.

For each learned method of the EXP-010 methods file (IL, A_mix, M; every seed
layer) at N = 6, 8, 10, runs protocol-v2 episodes on dev instances that earlier
work already used (default 1000-1063) and, in every slot, compares each active
UE's executed action with the action N_EDF would give it in the same state.
Reports the agreement over UE-slots with a packet and the collision share (the
share of transmissions that share their subchannel with another transmission).
Descriptive only; the decision of EXP-010 is analyze_exp010.py.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dsa_period import splits_v2
from dsa_period.config import load_scenario
from dsa_period.marl.features import encode_actions
from dsa_period.rules_v2 import CentralEdfPolicy, make_policy_v2

GRID = (6, 8, 10)


def episode(job: tuple) -> tuple:
    config, group, seed, spec, n, instance = job
    import torch

    torch.set_num_threads(1)
    scenario = load_scenario(config, n)
    policy = make_policy_v2(spec, scenario.physical, scenario.menu)
    teacher = CentralEdfPolicy(scenario.physical, scenario.menu)
    policy.reset(splits_v2.execution_generators("dev", n, instance, 0))
    topology, traffic = scenario.instance(instance)
    env = scenario.make_env(topology, traffic, splits_v2.fading_seeds("dev", n, instance, 0))
    observations = env.reset()
    counts = np.zeros(4, dtype=np.int64)  # active, agree, transmissions, collided
    while not env.truncated:
        actions = policy.act(observations)
        active = np.array([o.remaining_bits > 0 for o in observations])
        mine = encode_actions(actions, scenario.menu)
        edf = encode_actions(teacher.act(observations), scenario.menu)
        counts[0] += active.sum()
        counts[1] += (mine[active] == edf[active]).sum()
        transmitting = actions[:, 1] > 0
        per_channel = np.bincount(
            actions[transmitting, 0], minlength=scenario.physical.num_subchannels
        )
        counts[2] += transmitting.sum()
        counts[3] += per_channel[per_channel > 1].sum()
        observations = env.step(actions).observations
    return group, seed, n, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scenario-v2.toml")
    parser.add_argument("--methods", type=Path, required=True)
    parser.add_argument("--first-instance", type=int, default=1000)
    parser.add_argument("--instances", type=int, default=64)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    chosen = list(range(args.first_instance, args.first_instance + args.instances))
    splits_v2.check_dev_range(chosen)
    methods = json.loads(args.methods.read_text())
    jobs = [
        (args.config, group, s, layer[str(n)], n, i)
        for group in ("IL", "A_mix", "M")
        for s, layer in enumerate(methods[group])
        for n in GRID
        for i in chosen
    ]
    totals: dict = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for group, seed, n, counts in pool.map(episode, jobs, chunksize=1):
            key = f"{group}|{seed}|{n}"
            totals[key] = totals.get(key, np.zeros(4, dtype=np.int64)) + counts
    result = {
        key: {
            "agreement": float(c[1] / max(c[0], 1)),
            "collision_share": float(c[3] / max(c[2], 1)),
            "active_ue_slots": int(c[0]),
        }
        for key, c in sorted(totals.items())
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"instances": [chosen[0], chosen[-1]], "result": result}, indent=2)
    )
    for key, row in result.items():
        print(f"{key:<14} agreement {row['agreement']:.3f} collisions {row['collision_share']:.3f}")


if __name__ == "__main__":
    main()
