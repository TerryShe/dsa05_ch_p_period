"""EXP-DSA05-011 mechanism diagnostics (design §5.3 item 7; descriptive).

Re-runs protocol episodes of exp011-eval512 for the given specs and N (same
seeds as ``evaluate_capacity``; each episode's violations are checked against
the formal evaluation when ``--check`` points at its outputs) and records, with
the definitions of the broadcast probe so the numbers are comparable:

* marks: a "shadow" urgency summary at the primary reading threshold (-10 dB)
  is computed for every policy from the executed actions; for U it is exactly
  the broadcast the policy saw, for L0 policies it is never shown to them;
* choice (the probe's entry statistic, restricted to non-urgent UEs): for every
  non-urgent UE with a packet that transmits and does not stay on its previous
  subchannel, each subchannel other than the one it used in the previous slot is
  an option, classed by (marked or not, RSSI excess bin of the previous slot);
  rate = chosen / available, also relative to unmarked options of the same bin;
* deferral: P(OFF) of non-urgent UEs with a packet whose best own subchannel is
  marked, and when it is not;
* yield events per UE (non-urgent, packet, best own subchannel marked, and the UE
  did not transmit there), binned, with the mean violation rate of those UEs;
* collision pairs of failing UEs (eps = 0.01): entry, victim entry,
  co-residence, both new (probe definitions);
* failing share by mean-SNR bin (max power on the mean gain of the best
  subchannel) and traffic class.

Formal use only on exp011-eval512 with a clean committed tree (``--formal``).
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dsa_period import splits_v2
from dsa_period.capacity import allowance, run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.marl.features import stack
from dsa_period.rules_v2 import make_policy_v2
from dsa_period.urgency import SummaryLink, SummarySettings, urgent

SET = "exp011-eval512"
EXCESS_BINS_DB = (3.0, 10.0, 20.0)
SNR_BINS_DB = (0.0, 10.0, 20.0)
YIELD_BINS = (0, 1, 11, 101)  # yield events per UE: 0, 1-10, 11-100, > 100
CONFIG = "configs/scenario-v2.toml"


class Recorder:
    """Wraps a policy: keeps the shadow summary and the per-slot decision records."""

    def __init__(self, inner, scenario) -> None:
        self.inner, self.scenario = inner, scenario
        self.name = getattr(inner, "name", "policy")
        config, menu = scenario.physical, scenario.menu
        self.link = SummaryLink(SummarySettings(), config, menu, config.num_ues)

    def configure(self, *args) -> None:
        configure = getattr(self.inner, "configure", None)
        if configure is not None:
            configure(*args)

    def reset(self, rngs) -> None:
        self.inner.reset(rngs)
        self.link.reset()
        self.slots: list[tuple] = []

    def act(self, observations):
        config, menu = self.scenario.physical, self.scenario.menu
        obs = stack(observations)
        marked = self.link.broadcast(obs)
        urgent_now = urgent(config, menu, obs)
        actions = np.asarray(self.inner.act(observations), dtype=np.int64)
        sending = actions[:, 1] > 0
        self.link.commit(self.link.uplink(obs, sending), sending)
        self.slots.append(
            (marked, urgent_now, obs.remaining > 0, obs.own_gain.argmax(axis=1), obs.rssi.copy(),
             actions[:, 0].copy(), sending.copy())
        )  # fmt: skip
        return actions


def excess_bin(excess_db: float) -> str:
    return ("<3", "3-10", "10-20", ">=20")[int(np.searchsorted(EXCESS_BINS_DB, excess_db, "right"))]


def episode(job: tuple) -> dict:
    spec, n, instance = job
    if "marl:" in spec:
        import torch

        torch.set_num_threads(1)
    scenario = load_scenario(CONFIG, n)
    recorder = Recorder(make_policy_v2(spec, scenario.physical, scenario.menu), scenario)
    outcome = run_capacity_episode(scenario, recorder, "dev", instance)
    return {"spec": spec, "num_ues": n, **outcome, **statistics(scenario, recorder, outcome)}


def statistics(scenario, recorder: Recorder, outcome: dict) -> dict:
    n, k = scenario.physical.num_ues, scenario.physical.num_subchannels
    warmup, noise = scenario.warmup_slots, scenario.physical.noise_power_w
    slots = recorder.slots
    choice = {"available": defaultdict(int), "chosen": defaultdict(int)}
    deferral = defaultdict(int)
    yields = np.zeros(n, dtype=np.int64)
    pairs = defaultdict(int)
    packets = outcome["packets"]
    failing = np.array(outcome["violations"]) > np.array([allowance(0.01, m) for m in packets])
    for t in range(max(warmup, 1), len(slots)):
        marked, urgent_now, pending, best, rssi, channel, sending = slots[t]
        _, _, _, _, _, prev_channel, prev_sending = slots[t - 1]
        excess = 10.0 * np.log10(np.maximum(rssi / noise, 1.0))
        calm = pending & ~urgent_now
        stayed = sending & prev_sending & (channel == prev_channel)
        for ue in np.flatnonzero(calm & sending & ~stayed):
            own_prev = prev_channel[ue] if prev_sending[ue] else -1
            for c in range(k):
                if c == own_prev:
                    continue
                key = f"{'marked' if marked[c] else 'unmarked'}|{excess_bin(excess[c])}"
                choice["available"][key] += 1
                choice["chosen"][key] += int(channel[ue] == c)
        for ue in np.flatnonzero(calm):
            state = "best_marked" if marked[best[ue]] else "best_unmarked"
            deferral[f"{state}|decisions"] += 1
            deferral[f"{state}|off"] += int(not sending[ue])
            if marked[best[ue]] and not (sending[ue] and channel[ue] == best[ue]):
                yields[ue] += 1
        for i in np.flatnonzero(failing & sending):
            c = channel[i]
            for j in np.flatnonzero(sending & (channel == c)):
                if j == i:
                    continue
                j_before = prev_sending[j] and prev_channel[j] == c
                i_before = prev_sending[i] and prev_channel[i] == c
                kind = (
                    "co_residence" if j_before and i_before
                    else "entry" if i_before
                    else "victim_entry" if j_before
                    else "both_new"
                )  # fmt: skip
                pairs[kind] += 1
    return {
        "choice": {part: dict(v) for part, v in choice.items()},
        "deferral": dict(deferral),
        "yield_events": yields.tolist(),
        "failing": failing.tolist(),
        "pairs": dict(pairs),
        "bits_sent": recorder.link.bits_sent,
        "bits_read": recorder.link.bits_read,
    }


def snr_bins(scenario, instance: int) -> list[str]:
    """Mean SNR at max power on each UE's best mean subchannel gain, binned."""
    topology, traffic = scenario.instance(instance)
    env = scenario.make_env(
        topology, traffic, splits_v2.fading_seeds("dev", scenario.physical.num_ues, instance, 0)
    )
    mean = np.asarray(env.channel.mean_gain_linear)
    mean = mean.max(axis=1) if mean.ndim == 2 else mean
    snr_db = 10.0 * np.log10(scenario.menu.power_w[-1] * mean / scenario.physical.noise_power_w)
    labels = ("<0", "0-10", "10-20", ">=20")
    return [labels[int(np.searchsorted(SNR_BINS_DB, s, "right"))] for s in snr_db]


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for (spec, n), group in sorted(_groups(rows).items()):
        choice = {"available": defaultdict(int), "chosen": defaultdict(int)}
        deferral, pairs = defaultdict(int), defaultdict(int)
        by_yield = defaultdict(lambda: [0, 0.0])
        by_group = defaultdict(lambda: [0, 0])
        for row in group:
            for part in choice:
                for key, v in row["choice"][part].items():
                    choice[part][key] += v
            for key, v in row["deferral"].items():
                deferral[key] += v
            for key, v in row["pairs"].items():
                pairs[key] += v
            for count, rate in zip(row["yield_events"], row["violation_rate"]):
                label = str(YIELD_BINS[int(np.searchsorted(YIELD_BINS, count, "right")) - 1])
                by_yield[label][0] += 1
                by_yield[label][1] += rate
            for snr, cls, fail in zip(row["snr_bin"], row["class_index"], row["failing"]):
                by_group[f"{snr}|class{cls}"][0] += 1
                by_group[f"{snr}|class{cls}"][1] += int(fail)
        unmarked = {k.split("|")[1]: k for k in choice["available"] if k.startswith("unmarked")}
        ratio = {}
        for key, avail in choice["available"].items():
            if key.startswith("marked") and avail:
                base = unmarked.get(key.split("|")[1])
                rate = choice["chosen"][key] / avail
                base_rate = choice["chosen"][base] / choice["available"][base] if base else None
                ratio[key.split("|")[1]] = {
                    "options": avail,
                    "rate": rate,
                    "relative_to_unmarked": rate / base_rate if base_rate else None,
                }
        out[f"{spec}|N={n}"] = {
            "episodes": len(group),
            "marked_choice": ratio,
            "deferral": {
                state: deferral[f"{state}|off"] / max(deferral[f"{state}|decisions"], 1)
                for state in ("best_marked", "best_unmarked")
            },
            "deferral_counts": dict(deferral),
            "pairs": dict(pairs),
            "yield_bins": {
                k: {"ues": c, "mean_violation_rate": s / c} for k, (c, s) in by_yield.items()
            },
            "failing_share_by_snr_class": {
                k: {"ues": c, "failing": f / c} for k, (c, f) in by_group.items()
            },
            "bits_sent": sum(r["bits_sent"] for r in group),
            "bits_read": sum(r["bits_read"] for r in group),
        }
    return out


def _groups(rows: list[dict]) -> dict:
    groups: dict = defaultdict(list)
    for row in rows:
        groups[(row["spec"], row["num_ues"])].append(row)
    return groups


def check(rows: list[dict], evaluations: list[Path]) -> int:
    """Each re-run episode must match the formal evaluation of the same (spec, N, instance)."""
    reference = {}
    for path in evaluations:
        for spec, by_n in json.loads((path / "episodes.json").read_text()).items():
            for n, episodes in by_n.items():
                for e in episodes:
                    reference[(spec, int(n), e["instance"])] = e["violations"]
    checked = 0
    for row in rows:
        key = (row["spec"], row["num_ues"], row["instance"])
        if key in reference:
            if reference[key] != row["violations"]:
                raise AssertionError(f"{key}: the diagnostic re-run differs from the evaluation")
            checked += 1
    return checked


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", nargs="+", required=True, help="SPEC@N items")
    parser.add_argument("--instances", type=int, nargs=2, metavar=("FIRST", "COUNT"),
                        help="informal smoke runs on used dev instances only")  # fmt: skip
    parser.add_argument("--check", type=Path, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    provenance = git_provenance()
    if args.instances is None:
        if not args.formal or provenance["dirty"]:
            parser.error(f"{SET} is read only with --formal on a clean committed tree")
        chosen = splits_v2.screening_instances(SET)
    else:
        if args.formal:
            parser.error("--instances is for informal smoke runs")
        first, count = args.instances
        chosen = [splits_v2.require("dev", i) for i in range(first, first + count)]
        splits_v2.check_dev_range(chosen)
    items = [(item.rpartition("@")[0], int(item.rpartition("@")[2])) for item in args.specs]
    jobs = [(spec, n, i) for spec, n in items for i in chosen]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(episode, jobs, chunksize=1))
    bins = {}
    for row in rows:
        key = (row["num_ues"], row["instance"])
        if key not in bins:
            bins[key] = snr_bins(load_scenario(CONFIG, row["num_ues"]), row["instance"])
        row["snr_bin"] = bins[key]
    checked = check(rows, args.check)
    record = {
        "set": SET if args.instances is None else f"dev {chosen[0]}-{chosen[-1]}",
        "formal": args.formal,
        "provenance": provenance,
        "specs": args.specs,
        "checked_against_evaluation": checked,
        "summary": summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))
    print(f"{len(rows)} episodes, {checked} checked against the evaluation")


if __name__ == "__main__":
    main()
