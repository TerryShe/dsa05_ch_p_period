"""Where do learned policies lose energy relative to the central schedules? (diagnosis)

Replays validation episodes (protocol v1 instances and execution seeds, repeat 0)
of several policies and measures, over the evaluation window [W, H):

* duty: transmissions per UE-slot holding a packet;
* collision share: transmissions sharing their subchannel with another UE;
* interference loss: 1 - sum(actual mutual information) / sum(interference-free
  mutual information of the same transmissions at the same powers);
* overshoot: mutual information delivered beyond the packet's remaining bits in
  completing slots, as a share of all delivered mutual information;
* completion slack: (deadline - completion) / period of on-time packets;
* power-level shares; energy over the per-UE solo optimum (the lower bound), by
  distance band and traffic class.

Validation-only diagnosis (protocol v1 §1); no training and no test data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dsa_period import splits
from dsa_period.central import solo_reference
from dsa_period.central_timeshare import act_cycle, solve_orthogonal_cycle
from dsa_period.config import load_scenario
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.marl.trainer import load_policy_factory
from dsa_period.physics import shannon_bits
from dsa_period.policies import make_policy

FAR_M = 75.0


class CyclePolicy:
    """Central orthogonal time-sharing schedule (uses a clock; reference only)."""

    def __init__(self, schedule):
        self.schedule = schedule

    def reset(self, rngs):
        self.slot = 0

    def act(self, observations):
        actions = act_cycle(self.schedule, self.slot)
        self.slot += 1
        return actions


def episode_metrics(scenario, policy, instance: int) -> dict:
    config, menu = scenario.physical, scenario.menu
    n, levels = config.num_ues, menu.num_levels
    topology, traffic = scenario.instance(instance)
    env = scenario.make_env(topology, traffic)
    policy.reset(splits.execution_generators("validation", n, instance, 0))
    observations = env.reset()
    gains = topology.ue_ap_gain_linear
    power = np.asarray(menu.power_w)
    w, h = scenario.warmup_slots, scenario.horizon_slots
    acc = dict(tx=0, busy=0, collided=0, mi=0.0, mi_alone=0.0, overshoot=0.0)
    level_counts = np.zeros(levels, dtype=np.int64)
    while not env.truncated:
        t = env.elapsed_steps
        holding = np.array([o.remaining_bits > 0 for o in observations])
        before = np.array([o.remaining_bits for o in observations])
        step = env.step(policy.act(observations))
        if t >= w:
            phys = step.physical
            active = phys.active
            acc["busy"] += int(holding.sum())
            acc["tx"] += int(active.sum())
            if active.any():
                channels = phys.actions[active, 0]
                counts = np.bincount(channels, minlength=config.num_subchannels)
                acc["collided"] += int((counts[channels] > 1).sum())
                snr = power[phys.actions[active, 1]] * gains[active] / config.noise_power_w
                acc["mi"] += float(phys.service_bits[active].sum())
                acc["mi_alone"] += float(shannon_bits(config, snr).sum())
                done = step.completed
                acc["overshoot"] += float(
                    np.sum(
                        phys.service_bits[done] - np.minimum(before[done], phys.service_bits[done])
                    )
                )
                level_counts += np.bincount(phys.actions[active, 1], minlength=levels)
        observations = step.observations
    jobs = [j for j in env.job_records() if w < j["deadline_slot"] <= h]
    slack = [
        (j["deadline_slot"] - j["completion_slot"]) / traffic.period_slots[j["ue"]]
        for j in jobs
        if j["success"]
    ]
    lower = solo_reference(config, menu, topology, traffic)
    per_ue = []
    for ue in range(n):
        own = [j for j in jobs if j["ue"] == ue]
        energy = sum(j["energy_j"] for j in own)
        per_ue.append(
            {
                "distance_m": float(topology.ue_ap_distances_m[ue]),
                "period": int(traffic.period_slots[ue]),
                "energy_over_solo": energy / (len(own) * lower.plans[ue].energy_j),
                "violations": sum(not j["success"] for j in own),
                "packets": len(own),
            }
        )
    return {
        "instance": instance,
        "duty": acc["tx"] / max(acc["busy"], 1),
        "collision_share": acc["collided"] / max(acc["tx"], 1),
        "interference_loss": 1.0 - acc["mi"] / max(acc["mi_alone"], 1e-12),
        "overshoot_share": acc["overshoot"] / max(acc["mi"], 1e-12),
        "completion_slack": float(np.mean(slack)) if slack else None,
        "level_counts": level_counts[1:].tolist(),
        "energy_j": float(sum(j["energy_j"] for j in jobs)),
        "lower_j": float(
            sum(len([j for j in jobs if j["ue"] == u]) * lower.plans[u].energy_j for u in range(n))
        ),
        "per_ue": per_ue,
    }


def aggregate(rows: list[dict]) -> dict:
    per_ue = [u for r in rows for u in r["per_ue"]]
    ratio = np.array([u["energy_over_solo"] for u in per_ue])
    dist = np.array([u["distance_m"] for u in per_ue])
    period = np.array([u["period"] for u in per_ue])
    levels = np.sum([r["level_counts"] for r in rows], axis=0)
    out = {
        "episodes": len(rows),
        "energy_over_lower": float(
            np.sum([r["energy_j"] for r in rows]) / np.sum([r["lower_j"] for r in rows])
        ),
        **{
            key: float(np.mean([r[key] for r in rows if r[key] is not None]))
            for key in (
                "duty",
                "collision_share",
                "interference_loss",
                "overshoot_share",
                "completion_slack",
            )
        },
        "power_level_share": (levels / max(levels.sum(), 1)).tolist(),
        "energy_over_solo_by_distance": {
            band: float(np.median(ratio[mask]))
            for band, mask in (
                ("0-50", dist < 50),
                ("50-75", (dist >= 50) & (dist < FAR_M)),
                ("75-100", dist >= FAR_M),
            )
        },
        "energy_over_solo_by_period": {
            str(p): float(np.median(ratio[period == p])) for p in sorted(set(period.tolist()))
        },
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="configs/scenario-v1-n8.toml")
    parser.add_argument("--instances", type=int, default=64, help="first K validation instances")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH[@MU0]")
    parser.add_argument("--baseline", action="append", default=[], metavar="SPEC")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"output {args.output} already exists")
    # Single-threaded, like training: do not compete with concurrent runs for cores.
    torch.set_num_threads(1)
    provenance = git_provenance()
    scenario = load_scenario(args.scenario)
    instances = list(splits.indices("validation"))[: args.instances]
    makers = {}
    for item in args.checkpoint:
        name, _, rest = item.partition("=")
        path, _, mu0 = rest.partition("@")
        makers[name] = load_policy_factory(Path(path), scenario, float(mu0) if mu0 else None)[0]
    for spec in args.baseline:
        makers[spec] = lambda spec=spec: make_policy(spec, scenario.physical, scenario.menu)
    results = {}
    for name, make in makers.items():
        results[name] = aggregate([episode_metrics(scenario, make(), i) for i in instances])
    schedules = []
    for i in instances:
        topology, traffic = scenario.instance(i)
        schedule = solve_orthogonal_cycle(scenario.physical, scenario.menu, topology, traffic)
        schedules.append(episode_metrics(scenario, CyclePolicy(schedule), i))
    results["central-upper-bound"] = aggregate(schedules)
    header = f"{'policy':<24} {'E/LB':>6} {'duty':>6} {'coll':>6} {'I-loss':>7} {'over':>6} {'slack':>6}  levels(5/10/15/19/23)       E/solo by dist        by period"
    print(header)
    for name, r in results.items():
        levels = " ".join(f"{x:.2f}" for x in r["power_level_share"])
        dist = " ".join(f"{v:.2f}" for v in r["energy_over_solo_by_distance"].values())
        per = " ".join(f"{v:.2f}" for v in r["energy_over_solo_by_period"].values())
        print(
            f"{name:<24} {r['energy_over_lower']:6.3f} {r['duty']:6.3f} {r['collision_share']:6.3f} "
            f"{r['interference_loss']:7.3f} {r['overshoot_share']:6.3f} {r['completion_slack']:6.3f}  "
            f"{levels}  {dist}  {per}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "provenance": provenance,
            "scenario": args.scenario,
            "instances": instances,
            "checkpoints": args.checkpoint,
            "baselines": args.baseline,
            "results": results,
        }
        args.output.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
