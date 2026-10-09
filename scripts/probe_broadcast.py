"""Information-boundary probes (descriptive, dev only): AP broadcast plus a yield rule.

Two checks before any design that changes the fixed OBS (discussion of
2026-10-08, after EXP-DSA05-010):

1. Visibility. Does an interferer move onto a failing UE's subchannel because
   that subchannel looked quiet in the public RSSI? For every slot where a UE
   failing the eps = 0.01 allowance transmits together with another UE on the
   same subchannel, the pair is classed as "entry" (the other UE was not on it
   in the previous slot while the failing UE was), "victim entry" or
   "co-residence". For entries the RSSI excess over noise of that subchannel in
   the previous slot (what the entering UE saw), the existence of a free
   subchannel and whether it was the entering UE's best own channel are kept.
   For all entry decisions of all UEs the choice rate per available subchannel
   is tallied by the previous-slot state of the subchannel (free, or occupied
   by RSSI excess bin, and by urgency of its occupants).

2. Broadcast + yield. A wrapper runs an unchanged policy and then applies one
   rule to each UE with a packet that is not urgent (urgent: slack <
   URGENT_SLACK slots, slack = slots to deadline - slots still needed at the
   interference-free max-power rate of the best own subchannel, as in step 0):
   if the sampled subchannel is blocked, the UE defers (OFF this slot) in
   modes B2 and B3, or moves to the unblocked subchannel with its best own gain
   at the same power (OFF if none) in the redirect modes B2r and B3r. A smoke
   run showed that redirecting herds the UEs onto the few free subchannels, so
   deferral is the main rule. Blocked means, from the AP broadcast of the
   previous slot:

   * B2 (occupancy, 2 bits per subchannel: 0, 1 or 2+ transmitters,
     independent of received power): someone other than the UE itself
     transmitted on it;
   * B3 (urgent occupancy, 1 bit per subchannel, plus a 1-bit urgency flag
     with each uplink transmission): an urgent UE transmitted on it, unless
     the UE itself was on it and urgent then (the flag may be its own).

   Urgent UEs act as the policy says. "none" runs the policy unchanged.

   B3 assumes the AP reads every urgency flag, even of collided or weak
   transmissions. Two flag-loss variants (follow-up of 2026-10-09) use the
   same rule as B3 but let the AP hear a flag only if:

   * B3c: its transmitter was alone on its subchannel (any collision loses it);
   * B3s: its transmission reached SINR >= FLAG_SINR_DB (from the AP-reported
     mutual information of that transmission).

Execution and fading seeds are those of protocol v2 (dev); never run this on
validation, test or confirmation instances. Results are not formal.
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dsa_period import splits_v2
from dsa_period.capacity import (
    BOOTSTRAP_SEED,
    EPSILONS,
    allowance,
    best_alone_bits,
    eps_key,
    evaluated_packets,
    undeliverable,
)
from dsa_period.config import load_scenario
from dsa_period.evaluation import _bootstrap, _interval
from dsa_period.physics import shannon_bits
from dsa_period.rules_v2 import make_policy_v2

MODES = ("none", "B2", "B3", "B2r", "B3r", "B3c", "B3s")
DEFAULT_MODES = MODES[:5]  # the first probe run (2026-10-08)
URGENT_SLACK = 3
EXCESS_BINS_DB = (3.0, 10.0, 20.0)  # occupied subchannels: <3, 3-10, 10-20, >=20 dB over noise
FREE = "free"
FLAG_SINR_DB = 0.0  # B3s: SINR the AP needs to read an urgency flag


def slack(scenario, observations) -> np.ndarray:
    remaining = np.array([o.remaining_bits for o in observations])
    to_deadline = np.array([o.slots_to_deadline for o in observations])
    gains = np.array([o.own_gain_w for o in observations])
    pmax, noise = scenario.menu.power_w[-1], scenario.physical.noise_power_w
    rate = shannon_bits(scenario.physical, pmax * gains.max(axis=1) / noise)
    need = np.ceil(np.maximum(remaining, 0) / np.maximum(rate, 1e-9) - 1e-9)
    return np.where(remaining > 0, to_deadline - need, 10**6)


class Broadcast:
    """A policy plus the previous slot's AP broadcast and the yield rule (module doc)."""

    def __init__(self, inner, scenario, mode: str, urgent_slack: int = URGENT_SLACK) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.inner, self.scenario, self.mode, self.urgent_slack = (
            inner,
            scenario,
            mode,
            urgent_slack,
        )
        self.name = f"{getattr(inner, 'name', 'policy')}+{mode}"
        self.k = scenario.physical.num_subchannels
        self.moved = 0  # yield decisions that changed the subchannel
        self.silenced = 0  # yield decisions that went OFF
        self.flag_bits = float(
            shannon_bits(scenario.physical, 10.0 ** (FLAG_SINR_DB / 10.0))
        )  # fmt: skip

    def reset(self, rngs) -> None:
        self.inner.reset(rngs)
        self.prev_urgent = np.zeros(self.scenario.physical.num_ues, dtype=bool)
        self.heard = None  # all flags heard
        self.moved = self.silenced = self.flags_sent = self.flags_heard = 0

    def received(self, prev: np.ndarray, observations) -> np.ndarray:
        """Transmitters of the previous slot whose urgency flag the AP read."""
        sent = prev[:, 1] > 0
        if self.mode == "B3c":
            count = np.bincount(prev[sent, 0], minlength=self.k)
            return sent & (count[prev[:, 0]] == 1)
        if self.mode == "B3s":
            bits = np.array([o.previous_service_bits for o in observations])
            return sent & (bits >= self.flag_bits)
        return sent

    def blocked(self, prev: np.ndarray, ue: int) -> np.ndarray:
        """Which subchannels UE ``ue`` must avoid, from the previous slot's broadcast."""
        sent = prev[:, 1] > 0
        own = prev[ue, 0] if sent[ue] else -1
        mine = np.arange(self.k) == own
        if self.mode.startswith("B2"):
            count = np.minimum(np.bincount(prev[sent, 0], minlength=self.k), 2)
            return count - mine >= 1
        heard = sent if self.heard is None else self.heard
        flag = np.zeros(self.k, dtype=bool)
        flag[prev[sent & self.prev_urgent & heard, 0]] = True
        return flag & ~(mine & self.prev_urgent[ue])

    def act(self, observations) -> np.ndarray:
        urgent_now = slack(self.scenario, observations) < self.urgent_slack
        remaining = np.array([o.remaining_bits for o in observations])
        urgent_now &= remaining > 0
        actions = np.array(self.inner.act(observations), dtype=np.int64, copy=True)
        if self.mode != "none":
            prev = np.array([o.previous_action for o in observations], dtype=np.int64)
            if self.mode.startswith("B3"):
                self.heard = self.received(prev, observations)
                flagged = (prev[:, 1] > 0) & self.prev_urgent
                self.flags_sent += int(flagged.sum())
                self.flags_heard += int((flagged & self.heard).sum())
            for ue in np.flatnonzero((remaining > 0) & ~urgent_now & (actions[:, 1] > 0)):
                blocked = self.blocked(prev, ue)
                if not blocked[actions[ue, 0]]:
                    continue
                gains = np.where(blocked, -np.inf, observations[ue].own_gain_w)
                if self.mode.endswith("r") and np.isfinite(gains).any():
                    actions[ue, 0] = int(np.argmax(gains))
                    self.moved += 1
                else:
                    actions[ue] = 0
                    self.silenced += 1
        self.prev_urgent = urgent_now
        self.urgent = urgent_now
        return actions


def excess_bin(excess_db: float, occupied: bool) -> str:
    if not occupied:
        return FREE
    edges = ("<3", "3-10", "10-20", ">=20")
    return edges[int(np.searchsorted(EXCESS_BINS_DB, excess_db, side="right"))]


def episode(job: tuple) -> tuple:
    config, label, spec, mode, n, instance = job
    if spec.startswith("marl:"):
        import torch

        torch.set_num_threads(1)
    scenario = load_scenario(config, n)
    inner = make_policy_v2(spec, scenario.physical, scenario.menu)
    return label, mode, n, run(scenario, Broadcast(inner, scenario, mode), instance)


def run(scenario, policy: Broadcast, instance: int) -> dict:
    """One protocol-v2 dev episode of a wrapped policy: outcome and visibility statistics."""
    n = scenario.physical.num_ues
    policy.reset(splits_v2.execution_generators("dev", n, instance, 0))
    topology, traffic = scenario.instance(instance)
    env = scenario.make_env(topology, traffic, splits_v2.fading_seeds("dev", n, instance, 0))
    horizon, warmup, k = (
        scenario.horizon_slots,
        scenario.warmup_slots,
        scenario.physical.num_subchannels,
    )
    noise = scenario.physical.noise_power_w
    channel = np.zeros((horizon, n), dtype=np.int64)
    sent = np.zeros((horizon, n), dtype=bool)
    urgent = np.zeros((horizon, n), dtype=bool)
    rssi_seen = np.zeros((horizon, k))  # public RSSI of slot t - 1, seen when deciding slot t
    best_own = np.zeros((horizon, n), dtype=np.int64)
    observations = env.reset()
    for t in range(horizon):
        rssi_seen[t] = observations[0].previous_rssi_w
        best_own[t] = np.argmax([o.own_gain_w for o in observations], axis=1)
        step = env.step(policy.act(observations))
        urgent[t] = policy.urgent
        channel[t], sent[t] = step.physical.actions[:, 0], step.physical.active
        observations = step.observations

    # Outcome exactly as capacity.run_capacity_episode.
    packets = evaluated_packets(traffic, warmup, horizon)
    lost = undeliverable(packets, best_alone_bits(scenario, env.channel.gain_block(0, horizon)))
    count = np.bincount(packets["ue"], minlength=n)
    missed = np.zeros(n, dtype=np.int64)
    for job_record in env.job_records():
        if warmup < job_record["deadline_slot"] <= horizon and not job_record["success"]:
            missed[job_record["ue"]] += 1
    impossible = np.bincount(packets["ue"], weights=lost, minlength=n).astype(np.int64)
    feasible, hopeless = {}, {}
    for eps in EPSILONS:
        allowed = np.array([allowance(eps, int(m)) for m in count])
        feasible[eps_key(eps)] = bool(np.all(missed <= allowed))
        hopeless[eps_key(eps)] = bool(np.any(impossible > allowed))
    failing = missed > np.array([allowance(0.01, int(m)) for m in count])

    # Visibility statistics over the evaluation window.
    excess_db = 10.0 * np.log10(np.maximum(rssi_seen / noise, 1.0))
    occupied = np.zeros((horizon, k), dtype=bool)  # occupied in slot t - 1
    urgent_occupant = np.zeros((horizon, k), dtype=bool)
    for t in range(1, horizon):
        occupied[t, channel[t - 1, sent[t - 1]]] = True
        urgent_occupant[t, channel[t - 1, sent[t - 1] & urgent[t - 1]]] = True
    choice = {"available": defaultdict(int), "chosen": defaultdict(int)}
    pairs = defaultdict(int)
    entries = defaultdict(int)
    for t in range(max(warmup, 1), horizon):
        stayed = sent[t] & sent[t - 1] & (channel[t] == channel[t - 1])
        for j in np.flatnonzero(sent[t] & ~stayed):
            own_prev = channel[t - 1, j] if sent[t - 1, j] else -1
            for c in range(k):
                if c == own_prev:
                    continue
                key = excess_bin(excess_db[t, c], occupied[t, c])
                if occupied[t, c]:
                    key += "|urgent" if urgent_occupant[t, c] else "|calm"
                choice["available"][key] += 1
                choice["chosen"][key] += int(c == channel[t, j])
        before = np.bincount(channel[t - 1, sent[t - 1]], minlength=k)
        for i in np.flatnonzero(failing & sent[t]):
            c = channel[t, i]
            for j in np.flatnonzero(sent[t] & (channel[t] == c)):
                if j == i:
                    continue
                j_before = sent[t - 1, j] and channel[t - 1, j] == c
                i_before = sent[t - 1, i] and channel[t - 1, i] == c
                kind = (
                    "co_residence" if j_before and i_before
                    else "entry" if i_before
                    else "victim_entry" if j_before
                    else "both_new"
                )  # fmt: skip
                pairs[kind] += 1
                if kind != "entry":
                    continue
                entries["total"] += 1
                entries[f"excess|{excess_bin(excess_db[t, c], True)}"] += 1
                others = before - (np.arange(k) == channel[t - 1, j]) * sent[t - 1, j]
                entries["free_subchannel_existed"] += int((others == 0).any())
                entries["best_own_channel"] += int(best_own[t, j] == c)
                entries["entering_urgent"] += int(urgent[t, j])
    collided = 0
    for t in range(warmup, horizon):
        per_channel = np.bincount(channel[t, sent[t]], minlength=k)
        collided += int(per_channel[per_channel > 1].sum())
    return {
        "instance": instance,
        "feasible": feasible,
        "infeasible_for_all": hopeless,
        "violations": missed.tolist(),
        "failing_ues": int(failing.sum()),
        "moved": policy.moved,
        "silenced": policy.silenced,
        "flags_sent": policy.flags_sent,
        "flags_heard": policy.flags_heard,
        "transmissions": int(sent[warmup:].sum()),
        "collided": collided,
        "choice": {key: dict(value) for key, value in choice.items()},
        "pairs": dict(pairs),
        "entries": dict(entries),
    }


def merge(rows: list[dict]) -> dict:
    total: dict = {"choice": {"available": defaultdict(int), "chosen": defaultdict(int)}}
    for name in ("pairs", "entries"):
        total[name] = defaultdict(int)
    for key in ("moved", "silenced", "transmissions", "collided", "failing_ues"):
        total[key] = sum(r[key] for r in rows)
    for key in ("flags_sent", "flags_heard"):  # absent in the first probe run
        total[key] = sum(r.get(key, 0) for r in rows)
    for r in rows:
        for part in ("available", "chosen"):
            for key, v in r["choice"][part].items():
                total["choice"][part][key] += v
        for name in ("pairs", "entries"):
            for key, v in r[name].items():
                total[name][key] += v
    total["choice"] = {p: dict(v) for p, v in total["choice"].items()}
    total["pairs"], total["entries"] = dict(total["pairs"]), dict(total["entries"])
    return total


def summarize(results: list[tuple], resamples: int) -> dict:
    cells: dict = defaultdict(list)
    for label, mode, n, out in results:
        cells[(label, mode, n)].append(out)
    labels = sorted({label for label, _, _ in cells})
    groups = sorted({label.split("|")[0] for label in labels})
    grid = sorted({n for _, _, n in cells})
    summary: dict = {"cells": {}, "groups": {}}
    for (label, mode, n), rows in sorted(cells.items()):
        rows.sort(key=lambda r: r["instance"])
        summary["cells"][f"{label}|{mode}|N={n}"] = {
            "episodes": len(rows),
            "F": {e: float(np.mean([r["feasible"][e] for r in rows])) for e in ("0.01", "0.05")},
            "failed": [r["instance"] for r in rows if not r["feasible"]["0.01"]],
            **merge(rows),
        }
    for group in groups:
        members = [label for label in labels if label.split("|")[0] == group]
        for n in grid:
            mats = {}
            for mode in MODES:
                if all((m, mode, n) in cells for m in members):
                    mats[mode] = np.array(
                        [
                            [float(r["feasible"]["0.01"]) for r in cells[(m, mode, n)]]
                            for m in members
                        ]
                    )
            row = {mode: float(mat.mean()) for mode, mat in mats.items()}
            for mode in MODES[1:]:
                if mode in mats and "none" in mats:
                    diff = mats[mode] - mats["none"]
                    row[f"{mode}-none"] = {
                        "difference": float(diff.mean()),
                        "ci95": _interval(_bootstrap(diff, None, resamples, BOOTSTRAP_SEED)),
                    }
            summary["groups"][f"{group}|N={n}"] = row
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scenario-v2.toml")
    parser.add_argument("--methods", type=Path, default=Path("outputs/exp010/methods-exp010.json"))
    parser.add_argument("--groups", nargs="+", default=["A_mix", "M", "best_local"])
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=MODES)
    parser.add_argument("--grid", type=int, nargs="+", default=[8])
    parser.add_argument("--first-instance", type=int, default=1000)
    parser.add_argument("--instances", type=int, default=400)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    chosen = [
        splits_v2.require("dev", i)
        for i in range(args.first_instance, args.first_instance + args.instances)
    ]
    splits_v2.check_dev_range(chosen)
    methods = json.loads(args.methods.read_text())
    specs: dict[str, dict[int, str]] = {}
    for group in args.groups:
        layers = methods[group] if isinstance(methods[group], list) else [methods[group]]
        for s, layer in enumerate(layers):
            label = f"{group}|{s}" if isinstance(methods[group], list) else group
            specs[label] = {
                n: layer if isinstance(layer, str) else layer[str(n)] for n in args.grid
            }
    jobs = [
        (args.config, label, by_n[n], mode, n, i)
        for n in args.grid
        for label, by_n in specs.items()
        for mode in args.modes
        for i in chosen
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(episode, jobs, chunksize=1))
    summary = summarize(results, args.resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "instances": [chosen[0], chosen[-1]],
                "urgent_slack": URGENT_SLACK,
                "specs": specs,
                "summary": summary,
                "episodes": [
                    {"label": label, "mode": mode, "num_ues": n, **out}
                    for label, mode, n, out in results
                ],
            },
            indent=1,
        )
    )
    for key, row in summary["groups"].items():
        parts = [f"{m} {row[m]:.3f}" for m in MODES if m in row]
        parts += [
            f"{m} {row[m]['difference']:+.3f} [{row[m]['ci95'][0]:+.3f}, {row[m]['ci95'][1]:+.3f}]"
            for m in (f"{mode}-none" for mode in MODES[1:])
            if m in row
        ]
        print(f"{key:<16} " + " | ".join(parts))


if __name__ == "__main__":
    main()
