"""Behaviour diagnostics of EXP-DSA05-006 (design §7), descriptive only.

Runs protocol-v2 episodes (same instances, fading and execution seeds as the
evaluation) of the given policies on the first validation instances and counts,
per slot and UE:

* collisions: transmissions sharing their (slot, subchannel) with another one;
* subchannel choice: share of transmissions on the UE's own best-gain subchannel;
* switching: share of a packet's consecutive transmissions that change subchannel;
* transmit probability by slack (slots to the deadline minus the slots still
  needed at the interference-free max-power rate of the best subchannel), for
  near (< 50 m) and far (>= 80 m) UEs;
* yielding: transmit probability of near UEs with slack >= 3 when some far
  fast-class UE of the same deployment has slack <= 1, against otherwise;
* power: share of transmissions at max power.
"""

import argparse
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dsa_period import splits_v2
from dsa_period.config import load_scenario
from dsa_period.physics import shannon_bits
from dsa_period.rules_v2 import make_policy_v2

SLACK_BINS = ((-(10**9), 0), (1, 2), (3, 5), (6, 10), (11, 10**9))
NEAR, FAR = 50.0, 80.0


def _bin(slack: int) -> int:
    return next(i for i, (lo, hi) in enumerate(SLACK_BINS) if lo <= slack <= hi)


def episode(job: tuple) -> tuple:
    config, spec, n, instance = job
    if spec.startswith("marl:"):
        import torch

        torch.set_num_threads(1)
    scenario = load_scenario(config, n)
    policy = make_policy_v2(spec, scenario.physical, scenario.menu)
    policy.reset(splits_v2.execution_generators("validation", n, instance, 0))
    topology, traffic = scenario.instance(instance)
    env = scenario.make_env(topology, traffic, splits_v2.fading_seeds("validation", n, instance, 0))
    distance = topology.ue_ap_distances_m
    near, far = distance < NEAR, distance >= FAR
    fast_far = far & (traffic.class_index == 0)
    top = scenario.menu.num_levels - 1
    pmax, noise = scenario.menu.power_w[-1], scenario.physical.noise_power_w
    c = defaultdict(float)
    last_channel = np.full(n, -1)
    observations = env.reset()
    while not env.truncated:
        remaining = np.array([o.remaining_bits for o in observations])
        to_deadline = np.array([o.slots_to_deadline for o in observations])
        gains = np.array([o.own_gain_w for o in observations])
        released = (to_deadline == traffic.period_slots) & (remaining > 0)
        last_channel[released] = -1
        actions = policy.act(observations)
        step = env.step(actions)
        active = step.physical.active
        has = remaining > 0
        rate = shannon_bits(scenario.physical, pmax * gains.max(axis=1) / noise)
        slack = np.where(
            has, to_deadline - np.ceil(np.maximum(remaining, 0) / np.maximum(rate, 1e-9) - 1e-9), 0
        ).astype(int)
        urgent_far = bool(np.any(has & fast_far & (slack <= 1)))
        channel = step.physical.actions[:, 0]
        for ue in np.flatnonzero(has):
            group = "near" if near[ue] else "far" if far[ue] else "mid"
            b = _bin(int(slack[ue]))
            c[f"decide_{group}_{b}"] += 1
            c[f"tx_{group}_{b}"] += float(active[ue])
            if near[ue] and slack[ue] >= 3:
                key = "yield_urgent" if urgent_far else "yield_calm"
                c[f"{key}_decide"] += 1
                c[f"{key}_tx"] += float(active[ue])
        tx = np.flatnonzero(active)
        if tx.size:
            counts = np.bincount(channel[tx], minlength=scenario.physical.num_subchannels)
            c["tx"] += tx.size
            c["collided"] += float(np.sum(counts[channel[tx]] > 1))
            c["co_channel"] += float(np.sum(counts[channel[tx]] - 1))
            c["best_channel"] += float(np.sum(channel[tx] == gains[tx].argmax(axis=1)))
            c["max_power"] += float(np.sum(step.physical.actions[tx, 1] == top))
            again = tx[last_channel[tx] >= 0]
            c["repeat_tx"] += again.size
            c["switched"] += float(np.sum(channel[again] != last_channel[again]))
            last_channel[tx] = channel[tx]
        observations = step.observations
    return spec, n, dict(c)


def summarize(counts: dict) -> dict:
    def ratio(a: str, b: str) -> float | None:
        return counts.get(a, 0.0) / counts[b] if counts.get(b) else None

    out = {
        "tx_slots": counts.get("tx", 0.0),
        "collided_share": ratio("collided", "tx"),
        "co_channel_per_tx": ratio("co_channel", "tx"),
        "best_channel_share": ratio("best_channel", "tx"),
        "max_power_share": ratio("max_power", "tx"),
        "switch_share": ratio("switched", "repeat_tx"),
        "yield": {
            "near_tx_prob_when_far_fast_urgent": ratio("yield_urgent_tx", "yield_urgent_decide"),
            "near_tx_prob_otherwise": ratio("yield_calm_tx", "yield_calm_decide"),
        },
        "tx_prob_by_slack": {},
    }
    for group in ("near", "far"):
        out["tx_prob_by_slack"][group] = {
            f"{lo}..{hi}"
            if hi < 10**9 and lo > -(10**9)
            else (f"<={hi}" if lo < 0 else f">={lo}"): ratio(
                f"tx_{group}_{i}", f"decide_{group}_{i}"
            )
            for i, (lo, hi) in enumerate(SLACK_BINS)
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scenario-v2.toml")
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--grid", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--instances", type=int, default=32, help="first K validation instances")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chosen = list(splits_v2.indices("validation"))[: args.instances]
    jobs = [(args.config, s, n, i) for s in args.policies for n in args.grid for i in chosen]
    totals: dict = defaultdict(lambda: defaultdict(float))
    with ProcessPoolExecutor(args.workers) as pool:
        for spec, n, counts in pool.map(episode, jobs, chunksize=1):
            for key, value in counts.items():
                totals[(spec, n)][key] += value
    result = {
        f"{spec}|N={n}": summarize(dict(counts)) for (spec, n), counts in sorted(totals.items())
    }
    result["_meta"] = {"instances": chosen, "grid": args.grid, "policies": args.policies}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for key, row in result.items():
        if key.startswith("_"):
            continue
        y = row["yield"]
        print(
            f"{key[-60:]:<60} coll {row['collided_share']:.3f} best {row['best_channel_share']:.3f} "
            f"switch {row['switch_share'] if row['switch_share'] is not None else math.nan:.3f} "
            f"maxP {row['max_power_share']:.3f} | near tx urgent {y['near_tx_prob_when_far_fast_urgent'] or math.nan:.3f} "
            f"calm {y['near_tx_prob_otherwise'] or math.nan:.3f}"
        )
        for group, probs in row["tx_prob_by_slack"].items():
            print(
                f"    {group}: "
                + " ".join(
                    f"{k}:{(v if v is not None else math.nan):.2f}" for k, v in probs.items()
                )
            )


if __name__ == "__main__":
    main()
