"""Why do packets miss their deadline? Failure decomposition on scenario v2.0 (dev only).

Exploratory diagnostic, step 0 of the fixed-OBS route
(docs/research/literature-fixed-obs-marl_20261002.md). Runs protocol-v2
episodes (same fading and execution seeds as the evaluation) of the given
policies on dev instances, records the executed actions, and classifies every
violated packet in the evaluation window:

* physics: even max power on the best subchannel in every slot of the window,
  alone, cannot deliver it (``capacity.undeliverable``);
* interference: the UE's own executed transmissions (same slots, subchannels,
  powers) would have delivered it with no other UE on the air;
* choice: not interference, but max power on the best subchannel in the slots
  the UE did transmit would have delivered it alone;
* attempts: not even that; the UE transmitted in too few slots.

"choice" is split into "power" (max power on the executed subchannels would
have delivered it alone) and "channel" (only the best subchannel would have).
Per failing UE the class counts are kept, which gives the share of failing
UEs that would meet the allowance if their interference losses (or their
interference and choice losses) were removed.

The classes are exclusive and checked in this order (physics first). Packets
of UEs that fail the eps = 0.01 allowance are reported apart from all others.
For interference packets of failing UEs it also records the co-channel
transmitters in the UE's transmit slots: distance, class, and slack (slots to
the deadline minus the slots still needed at the interference-free max-power
rate of the best subchannel) at that slot; slack >= 3 is labelled "could_wait".
The label is a convention: the slack ignores interference and does not say
whether yielding would make the interferer itself miss later. Never run this
on the validation or test splits for model selection.
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dsa_period import splits_v2
from dsa_period.capacity import allowance, best_alone_bits, evaluated_packets, undeliverable
from dsa_period.config import load_scenario
from dsa_period.physics import shannon_bits
from dsa_period.rules_v2 import make_policy_v2

CLASSES = ("fast", "medium", "slow")
NEAR, FAR = 50.0, 80.0
WAIT_SLACK = 3


def episode(job: tuple) -> tuple:
    config, spec, n, split, instance = job
    if spec.startswith("marl:"):
        import torch

        torch.set_num_threads(1)
    scenario = load_scenario(config, n)
    policy = make_policy_v2(spec, scenario.physical, scenario.menu)
    policy.reset(splits_v2.execution_generators(split, n, instance, 0))
    topology, traffic = scenario.instance(instance)
    env = scenario.make_env(topology, traffic, splits_v2.fading_seeds(split, n, instance, 0))
    horizon, warmup = scenario.horizon_slots, scenario.warmup_slots
    pmax, noise = scenario.menu.power_w[-1], scenario.physical.noise_power_w
    channel = np.zeros((horizon, n), dtype=np.int64)
    power = np.zeros((horizon, n), dtype=np.int64)
    active = np.zeros((horizon, n), dtype=bool)
    slack = np.zeros((horizon, n), dtype=np.int64)
    observations = env.reset()
    for t in range(horizon):
        remaining = np.array([o.remaining_bits for o in observations])
        to_deadline = np.array([o.slots_to_deadline for o in observations])
        gains = np.array([o.own_gain_w for o in observations])
        rate = shannon_bits(scenario.physical, pmax * gains.max(axis=1) / noise)
        need = np.ceil(np.maximum(remaining, 0) / np.maximum(rate, 1e-9) - 1e-9)
        slack[t] = np.where(remaining > 0, to_deadline - need, 10**6)
        step = env.step(policy.act(observations))
        channel[t], power[t] = step.physical.actions[:, 0], step.physical.actions[:, 1]
        active[t] = step.physical.active
        observations = step.observations
    gains = env.channel.gain_block(0, horizon)  # (T, N, K)
    packets = evaluated_packets(traffic, warmup, horizon)
    hopeless = undeliverable(packets, best_alone_bits(scenario, gains))
    jobs = {
        (j["ue"], j["deadline_slot"]): j
        for j in env.job_records()
        if warmup < j["deadline_slot"] <= horizon
    }
    count = np.bincount(packets["ue"], minlength=n)
    missed = np.zeros(n, dtype=np.int64)
    for (ue, _), job in jobs.items():
        missed[ue] += not job["success"]
    failing = missed > np.array([allowance(0.01, int(m)) for m in count])
    power_w = np.asarray(scenario.menu.power_w)
    distance = topology.ue_ap_distances_m
    out = {"failing_ues": [], "classes": {"failing": defaultdict(int), "other": defaultdict(int)}}
    interferers = defaultdict(int)
    per_ue = defaultdict(lambda: defaultdict(int))
    for p in range(packets["ue"].size):
        ue, a, d = int(packets["ue"][p]), int(packets["arrival"][p]), int(packets["deadline"][p])
        if jobs[(ue, d)]["success"]:
            continue
        slots = np.arange(a, d)
        sent = slots[active[slots, ue]]
        own = gains[sent, ue, channel[sent, ue]]
        alone = shannon_bits(scenario.physical, power_w[power[sent, ue]] * own / noise).sum()
        best = shannon_bits(scenario.physical, pmax * gains[sent, ue].max(axis=1) / noise).sum()
        bits = packets["bits"][p]
        if hopeless[p]:
            kind = "physics"
        elif alone >= bits:
            kind = "interference"
        elif best >= bits:
            executed_max = shannon_bits(scenario.physical, pmax * own / noise).sum()
            kind = "power" if executed_max >= bits else "channel"
        else:
            kind = "attempts"
        group = "failing" if failing[ue] else "other"
        out["classes"][group][kind] += 1
        per_ue[ue][kind] += 1
        out["classes"][group]["tx_slots"] += sent.size
        out["classes"][group]["window_slots"] += slots.size
        if group == "failing" and kind == "interference":
            for t in sent:
                others = np.flatnonzero(active[t] & (channel[t] == channel[t, ue]))
                for j in others[others != ue]:
                    where = "near" if distance[j] < NEAR else "far" if distance[j] >= FAR else "mid"
                    could_wait = "could_wait" if slack[t, j] >= WAIT_SLACK else "urgent"
                    interferers[f"{where}|{CLASSES[traffic.class_index[j]]}|{could_wait}"] += 1
                    interferers["total"] += 1
    for ue in np.flatnonzero(failing):
        out["failing_ues"].append(
            {
                "ue": int(ue),
                "class": CLASSES[traffic.class_index[ue]],
                "distance_m": round(float(distance[ue]), 1),
                "shadowing_db": round(float(topology.shadowing_db[ue]), 1),
                "violations": int(missed[ue]),
                "allowance": allowance(0.01, int(count[ue])),
                "duty": float(active[warmup:, ue].mean()),
                "missed_by_class": dict(per_ue[ue]),
            }
        )
    out["classes"] = {g: dict(c) for g, c in out["classes"].items()}
    out["interferers"] = dict(interferers)
    out["feasible"] = not bool(failing.any())
    return spec, n, instance, out


def merge(results: list[tuple]) -> dict:
    summary: dict = {}
    for spec, n, instance, out in results:
        key = f"{spec}|N={n}"
        row = summary.setdefault(
            key,
            {
                "episodes": 0,
                "failed_episodes": 0,
                "failing_ues": [],
                "classes": {"failing": defaultdict(int), "other": defaultdict(int)},
                "interferers": defaultdict(int),
            },
        )
        row["episodes"] += 1
        row["failed_episodes"] += not out["feasible"]
        for ue in out["failing_ues"]:
            row["failing_ues"].append({"instance": instance, **ue})
        for group, counts in out["classes"].items():
            for k, v in counts.items():
                row["classes"][group][k] += v
        for k, v in out["interferers"].items():
            row["interferers"][k] += v
    for row in summary.values():
        row["classes"] = {g: dict(c) for g, c in row["classes"].items()}
        row["interferers"] = dict(row["interferers"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scenario-v2.toml")
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--grid", type=int, nargs="+", default=[6, 8, 10])
    parser.add_argument("--first-instance", type=int, default=1000)
    parser.add_argument("--instances", type=int, default=400)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    chosen = [
        splits_v2.require("dev", i)
        for i in range(args.first_instance, args.first_instance + args.instances)
    ]
    splits_v2.check_dev_range(chosen)
    jobs = [(args.config, s, n, "dev", i) for s in args.policies for n in args.grid for i in chosen]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(episode, jobs, chunksize=1))
    summary = merge(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"instances": [chosen[0], chosen[-1]], "summary": summary}, indent=2)
    )
    for key, row in summary.items():
        f = row["classes"]["failing"]
        kinds = ("physics", "interference", "power", "channel", "attempts")
        total = sum(f.get(k, 0) for k in kinds)
        shares = " ".join(f"{k} {f.get(k, 0) / total:.2f}" if total else f"{k} -" for k in kinds)
        ues = row["failing_ues"]

        def passes(removed):
            return sum(
                u["violations"] - sum(u["missed_by_class"].get(k, 0) for k in removed)
                <= u["allowance"]
                for u in ues
            )

        rescue = (
            f"pass w/o interference {passes(('interference',))}/{len(ues)}, "
            f"w/o interference+choice {passes(('interference', 'power', 'channel'))}/{len(ues)}"
        )
        tx = f.get("tx_slots", 0) / max(f.get("window_slots", 1), 1)
        inter = row["interferers"]
        wait = sum(v for k, v in inter.items() if k.endswith("could_wait"))
        near = sum(v for k, v in inter.items() if k.startswith("near"))
        print(
            f"{key[-48:]:<48} F {1 - row['failed_episodes'] / row['episodes']:.3f} "
            f"failing UEs {len(row['failing_ues'])} | missed packets of failing UEs {total}: {shares} "
            f"| tx/window {tx:.2f} | interferers {inter.get('total', 0)}: could wait "
            f"{wait / max(inter.get('total', 1), 1):.2f}, near {near / max(inter.get('total', 1), 1):.2f} "
            f"| {rescue}"
        )


if __name__ == "__main__":
    main()
