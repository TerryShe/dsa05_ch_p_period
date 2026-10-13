"""EXP-DSA05-011B stage 1: where near-AP starvation occurs and why (design §3.1).

``map`` reads the formal stress-set outputs (``scripts/stress_exp011b.py``) and
the formal exp009-screen256 outputs of the same checkpoints, counts the
starvation cells (``starvation.cells``; on the screening set at the condition's
evaluation N only, as the selection guard) per (condition, seed, checkpoint, N) and
evaluates the gate-A criteria (design §6): whether every (condition, seed) has
a clean checkpoint among 3M-5M, and per condition the seeds whose starvation
grows from 3M to 5M. Conditions come from the run manifests (``--rename``).

``counterfactual`` replays starvation cells of the stress set (per checkpoint,
the first cell at each N; each replay must reproduce the formal stress
episode) and, at every slot in which the starved UE has a packet, evaluates the
actor on the same slot with that UE's own channel gains scaled by each delta
(dB). Every input derived from the own gain changes with it: the per-subchannel
own-gain token, the coverage ratios and the best-gain context entry; RSSI and
the summary do not. The policy itself acts on the real inputs (copies of its
memory and summary link compute the variants), so the episode is unchanged.
Reported per cell: the mean P(OFF) of the starved UE at each delta.

``probes`` (descriptive, approved at gate A) replays one cell per N of a target
checkpoint (its first starvation cell, else its near-AP UE with the highest
violation rate) and the same cell under a control checkpoint of the same run.
At every pending slot it scores the UE's P(OFF) one step ahead with one input
group replaced, the policy again acting on the real inputs:

* ``period=P``: the period input of a class with period P. Every class has the
  same rate (packet bits / period), so this is the class-equivalent state: every
  other actor input is a ratio of bits or of slots and does not change;
* ``hidden=0``: the UE's recurrent state cleared (as at the start of an episode);
* ``gain=D``: the UE's own gains scaled by D dB, as in ``counterfactual``;
* ``broadcast=0`` / ``broadcast=1`` (summary policies): no subchannel / every
  subchannel marked in the broadcast the UE sees.

Per packet (the pending slots after one release) the product of these P(OFF) is
the probability, along the recorded inputs, that the UE stays OFF for the whole
packet; it is reported as a mean over packets per variant (``p_never_sent``),
next to the share of packets the UE in fact never transmitted. For a packet that
was sent the recorded slots stop when it completes, so the product covers fewer
slots than a silent path would and is an upper bound for it.

A second episode is an intervention: the UE's recurrent state is cleared at each
of its packet releases (``HiddenReset``), and its violation rate and the other
UEs' are compared with the replay.
"""

import argparse
import copy
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dsa_period import splits_v2, starvation
from dsa_period.capacity import run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.experiments import select_exp011b
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.marl.features import MAX_PERIOD_SLOTS, legal_mask, stack
from dsa_period.marl.networks import split_logits
from dsa_period.marl.policy import SummaryPolicy
from dsa_period.rules_v2 import make_policy_v2
from dsa_period.urgency import URGENT_SLACK, urgent, visible, with_summary

CONFIG = starvation.CONFIG
EARLY, LATE = 3_000_000, 5_100_000  # the 3M-5M checkpoints of gate A
GRIDS = select_exp011b.GRIDS  # screening N counted per condition (the evaluation N)
NEAR_M = splits_v2.SCREENING_SETS[starvation.STRESS_SET]["near_m"]
PERIOD_INPUT = 3  # context entry period / MAX_PERIOD_SLOTS (LocalMemory.features)
GAIN_DB = (-10.0, -20.0, -30.0)  # own-gain variants of the input probes


# ---------------------------------------------------------------------------- map
def run_of(spec: str, rename: dict[str, str]) -> tuple[str, int, int]:
    """(condition, seed, slots) of a ``marl:`` checkpoint spec from its run manifest."""
    checkpoint = Path(spec.partition(":")[2])
    manifest = json.loads((checkpoint.parent.parent / "manifest.json").read_text())
    experiment = manifest["experiment"]
    condition = experiment.get("condition", experiment["id"])
    return rename.get(condition, condition), int(manifest["seed"]), int(checkpoint.stem[5:])


def read_outputs(paths: list[Path], kind: str) -> dict[str, dict[int, list[dict]]]:
    """spec -> N -> episode records of formal stress (or exp009-screen256) outputs."""
    out: dict = defaultdict(dict)
    for path in paths:
        record = json.loads((path / "summary.json").read_text())
        wanted = starvation.STRESS_SET if kind == "stress" else "exp009-screen256"
        if not record["formal"] or record.get("set", record.get("screen")) != wanted:
            raise ValueError(f"{path}: not a formal {wanted} output")
        for spec, by_n in json.loads((path / "episodes.json").read_text()).items():
            for n, rows in by_n.items():
                out[spec][int(n)] = rows
    return dict(out)


def starvation_map(stress: dict, screen: dict, rename: dict[str, str]) -> dict:
    runs: dict = defaultdict(list)
    for spec, by_n in stress.items():
        condition, seed, slots = run_of(spec, rename)
        stress_cells = {n: starvation.cells(rows, n) for n, rows in sorted(by_n.items())}
        screen_cells = {
            n: starvation.cells(rows, n)
            for n, rows in sorted(screen[spec].items())
            if n in GRIDS[condition]
        }
        total = sum(len(v) for v in (*stress_cells.values(), *screen_cells.values()))
        runs[(condition, seed)].append({
            "spec": spec, "slots": slots, "total": total,
            "stress": {n: len(v) for n, v in stress_cells.items()},
            "screen": {n: len(v) for n, v in screen_cells.items()},
            "stress_cells": stress_cells, "screen_cells": screen_cells,
        })  # fmt: skip
    table: dict = defaultdict(dict)
    clean_window: dict = defaultdict(dict)
    growing: dict = defaultdict(list)
    for (condition, seed), rows in sorted(runs.items()):
        rows.sort(key=lambda r: r["slots"])
        table[condition][str(seed)] = rows
        window = [r for r in rows if EARLY <= r["slots"] < LATE]
        clean_window[condition][str(seed)] = [r["slots"] for r in window if r["total"] == 0]
        if window and window[-1]["total"] > window[0]["total"]:
            growing[condition].append(seed)
    lacking = [
        f"{c} seed {s}" for c, seeds in clean_window.items() for s, ok in seeds.items() if not ok
    ]
    return {
        "definition": {
            "strong_snr_db": starvation.STRONG_SNR_DB,
            "starved_rate": starvation.STARVED_RATE,
        },  # fmt: skip
        "by_run": table,
        "clean_checkpoints_3M_to_5M": clean_window,
        "runs_without_clean_checkpoint": lacking,
        "seeds_with_growing_starvation": dict(growing),
    }


# ------------------------------------------------------------------ counterfactual
def off_probability(actor, tokens, context, hidden, legal) -> float:
    """P(OFF) of one UE's masked action distribution (inputs are that UE's rows)."""
    with torch.no_grad():
        logits, _ = actor.step(torch.from_numpy(tokens), torch.from_numpy(context), hidden)
    logits = split_logits(logits, actor.num_actions)[0][0].double().numpy()
    masked = np.where(legal, logits, -np.inf)
    p = np.exp(masked - masked.max())
    return float(p[0] / p.sum())


def scaled_gain(obs, ue: int, delta_db: float):
    """The observation with UE ``ue``'s own gains scaled by ``delta_db``."""
    gains = obs.own_gain.copy()
    gains[ue] *= 10.0 ** (delta_db / 10.0)
    return replace(obs, own_gain=gains)


class GainProbe:
    """Wraps a MARL policy and scores one UE's P(OFF) with its own gains scaled."""

    def __init__(self, inner, ue: int, deltas_db: list[float]) -> None:
        self.inner, self.ue, self.deltas = inner, ue, deltas_db
        self.name = getattr(inner, "name", "policy")

    def configure(self, *args) -> None:
        configure = getattr(self.inner, "configure", None)
        if configure is not None:
            configure(*args)

    def reset(self, rngs) -> None:
        self.inner.reset(rngs)
        self.p_off: list[list[float]] = []
        self.sends: list[bool] = []  # per pending slot: whether the UE transmitted

    def _p_off(self, obs, memory, link) -> list[float]:
        inner, ue = self.inner, self.ue
        memory.update_duals(memory.observe(obs))
        seen = own = None
        if link is not None:
            s = inner.settings
            urgent_now = urgent(inner.config, inner.menu, obs, s.urgent_slack)
            seen, own = visible(link.broadcast(obs), inner.hide, urgent_now), link.own_previous
        legal = legal_mask(obs, inner.actor.num_actions)[ue]
        out = []
        for delta in self.deltas:
            tokens, context = memory.features(scaled_gain(obs, ue, delta))
            if link is not None:
                tokens, context = with_summary(tokens, context, seen, own)
            row = slice(ue, ue + 1)
            out.append(
                off_probability(inner.actor, tokens[row], context[row], inner._hidden[row], legal)
            )
        return out

    def act(self, observations):
        obs = stack(observations)
        pending = obs.remaining[self.ue] > 0
        if pending:
            link = copy.deepcopy(self.inner.link) if isinstance(self.inner, SummaryPolicy) else None
            self.p_off.append(self._p_off(obs, copy.deepcopy(self.inner._memory), link))
        actions = self.inner.act(observations)
        if pending:
            self.sends.append(bool(actions[self.ue][1] > 0))
        return actions


def _replay(job: tuple) -> dict:
    spec, n, instance, ue, deltas, expected = job
    torch.set_num_threads(1)
    scenario = load_scenario(CONFIG, n)
    probe = GainProbe(make_policy_v2(spec, scenario.physical, scenario.menu), ue, deltas)
    record = run_capacity_episode(scenario, probe, "dev", instance)
    if record["violations"] != expected:
        raise AssertionError(f"{spec} N={n} {instance}: the replay differs from the stress run")
    p = np.array(probe.p_off)
    return {
        "spec": spec, "N": n, "instance": instance, "ue": ue,
        "violation_rate": record["violation_rate"][ue],
        "pending_slots": len(p), "send_share": float(np.mean(probe.sends)) if len(p) else 0.0,
        "mean_p_off": dict(zip(map(str, deltas), p.mean(axis=0).tolist())),
    }  # fmt: skip


def counterfactual(stress: dict, deltas: list[float], workers: int) -> list[dict]:
    jobs = []
    for spec, by_n in sorted(stress.items()):
        for n, rows in sorted(by_n.items()):
            found = starvation.cells(rows, n)
            if found:
                cell = found[0]
                row = next(r for r in rows if r["instance"] == cell["instance"])
                jobs.append((spec, n, cell["instance"], cell["ue"], deltas, row["violations"]))
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_replay, jobs, chunksize=1))


# -------------------------------------------------------------------- input probes
def variants(scenario, summary: bool) -> list[str]:
    """The input variants scored at every pending slot; "actual" first."""
    classes = scenario.traffic_classes
    if len({c.packet_bits / c.period_slots for c in classes}) != 1:
        raise ValueError("the period variant is the class-equivalent state only at equal rates")
    names = ["actual", *(f"period={c.period_slots}" for c in classes), "hidden=0",
             *(f"gain={d:g}" for d in GAIN_DB)]  # fmt: skip
    return [*names, "broadcast=0", "broadcast=1"] if summary else names


def edit(variant: str, context, seen, hidden, ue: int):
    """Copies of the actor inputs with one input group of UE ``ue`` replaced."""
    context, hidden = context.copy(), hidden.clone()
    seen = None if seen is None else seen.copy()
    group, _, value = variant.partition("=")
    if group == "period":
        context[ue, PERIOD_INPUT] = float(value) / MAX_PERIOD_SLOTS
    elif group == "hidden":
        hidden[ue] = 0.0
    elif group == "broadcast":
        if seen is None:
            raise ValueError("the broadcast variants need a summary policy")
        seen[ue] = value == "1"
    elif group != "actual":
        raise ValueError(f"unknown variant {variant!r}")
    return context, seen, hidden


class InputProbe(GainProbe):
    """Scores one UE's P(OFF) under each input variant at its pending slots.

    ``gain=D`` recomputes the features from the scaled gains (``scaled_gain``);
    every other variant edits the actual inputs (``edit``). ``packet`` numbers
    the pending slots by the UE's releases seen so far.
    """

    def __init__(self, inner, ue: int, names: list[str]) -> None:
        super().__init__(inner, ue, [])
        self.variants = names

    def reset(self, rngs) -> None:
        super().reset(rngs)
        self.urgent: list[bool] = []
        self.marked: list[int] = []
        self.packet: list[int] = []
        self.releases = 0

    def _p_off(self, obs, memory, link) -> list[float]:
        inner, ue = self.inner, self.ue
        if obs.to_deadline[ue] == obs.period[ue]:  # a release (LocalMemory.observe)
            self.releases += 1
        self.packet.append(self.releases)
        memory.update_duals(memory.observe(obs))
        tokens, context = memory.features(obs)
        threshold = inner.settings.urgent_slack if link is not None else URGENT_SLACK
        urgent_now = urgent(inner.config, inner.menu, obs, threshold)
        self.urgent.append(bool(urgent_now[ue]))
        seen = own = None
        if link is not None:
            seen, own = visible(link.broadcast(obs), inner.hide, urgent_now), link.own_previous
            self.marked.append(int(seen[ue].sum()))
        legal = legal_mask(obs, inner.actor.num_actions)[ue]
        out = []
        for variant in self.variants:
            group, _, value = variant.partition("=")
            toks, ctx = tokens, context
            if group == "gain":
                (toks, ctx), variant = memory.features(scaled_gain(obs, ue, float(value))), "actual"
            varied, shown, hidden = edit(variant, ctx, seen, inner._hidden, ue)
            inputs = (toks, varied) if shown is None else with_summary(toks, varied, shown, own)
            row = slice(ue, ue + 1)
            out.append(
                off_probability(inner.actor, inputs[0][row], inputs[1][row], hidden[row], legal)
            )
        return out


class HiddenReset:
    """Intervention: clears one UE's recurrent state at each of its packet releases."""

    def __init__(self, inner, ue: int) -> None:
        self.inner, self.ue = inner, ue
        self.name = f"hidden-reset@{getattr(inner, 'name', 'policy')}"

    def configure(self, *args) -> None:
        configure = getattr(self.inner, "configure", None)
        if configure is not None:
            configure(*args)

    def reset(self, rngs) -> None:
        self.inner.reset(rngs)
        self.resets = self.pending = self.sent = 0

    def act(self, observations):
        obs = stack(observations)
        ue = self.ue
        if obs.to_deadline[ue] == obs.period[ue]:  # a release (LocalMemory.observe)
            hidden = self.inner._hidden.clone()
            hidden[ue] = 0.0
            self.inner._hidden = hidden
            self.resets += 1
        actions = self.inner.act(observations)
        if obs.remaining[ue] > 0:
            self.pending += 1
            self.sent += int(actions[ue][1] > 0)
        return actions


def per_packet(p_off, packet, sends):
    """Per packet: the product of each variant's P(OFF) over its slots; whether it was never sent."""
    ids = np.unique(packet)
    silent = np.array([p_off[packet == i].prod(axis=0) for i in ids])
    never = np.array([not sends[packet == i].any() for i in ids])
    return silent, never


def probe_cell(rows: list[dict], n: int) -> dict:
    """The first starvation cell at N, else the near-AP UE with the highest violation rate."""
    found = starvation.cells(rows, n)
    if found:
        return {"instance": found[0]["instance"], "ue": found[0]["ue"], "kind": "starved"}
    near = [
        (r["violation_rate"][ue], -r["instance"], -ue)
        for r in rows
        for ue, distance in enumerate(r["distance_m"])
        if distance < NEAR_M
    ]
    _, instance, ue = max(near)
    return {"instance": -instance, "ue": -ue, "kind": "worst near-AP UE"}


def probe_jobs(pairs: list[tuple[Path, Path]]) -> list[tuple]:
    """Per (target, control) pair and N: the target's probe cell under both checkpoints."""
    jobs = []
    for target_dir, control_dir in pairs:
        ((target, by_n),) = read_outputs([target_dir], "stress").items()
        ((control, control_by_n),) = read_outputs([control_dir], "stress").items()
        if run_of(target, {})[:2] != run_of(control, {})[:2]:
            raise ValueError(f"{control} is not a checkpoint of the run of {target}")
        for n, rows in sorted(by_n.items()):
            cell = probe_cell(rows, n)
            for role, spec, source in (("target", target, rows), ("control", control, control_by_n[n])):  # fmt: skip
                row = next(r for r in source if r["instance"] == cell["instance"])
                jobs.append((role, cell["kind"], spec, n, cell["instance"], cell["ue"],
                             row["violations"]))  # fmt: skip
    return jobs


def _probe(job: tuple) -> dict:
    role, kind, spec, n, instance, ue, expected = job
    torch.set_num_threads(1)
    scenario = load_scenario(CONFIG, n)
    policy = make_policy_v2(spec, scenario.physical, scenario.menu)
    names = variants(scenario, isinstance(policy, SummaryPolicy))
    probe = InputProbe(policy, ue, names)
    replay = run_capacity_episode(scenario, probe, "dev", instance)
    if replay["violations"] != expected:
        raise AssertionError(f"{spec} N={n} {instance}: the replay differs from the stress run")
    reset = HiddenReset(make_policy_v2(spec, scenario.physical, scenario.menu), ue)
    after = run_capacity_episode(scenario, reset, "dev", instance)
    p, pressing = np.array(probe.p_off), np.array(probe.urgent)
    sends = np.array(probe.sends)
    silent, never = per_packet(p, np.array(probe.packet), sends)
    others = [i for i in range(n) if i != ue]

    def other_rates(record: dict) -> dict:
        rates = [record["violation_rate"][i] for i in others]
        return {"mean": float(np.mean(rates)), "max": float(np.max(rates))}

    return {
        "role": role, "kind": kind, "spec": spec, "N": n, "instance": instance, "ue": ue,
        "class_index": replay["class_index"][ue], "distance_m": replay["distance_m"][ue],
        "violation_rate": replay["violation_rate"][ue],
        "pending_slots": len(p), "send_share": float(sends.mean()),
        "urgent_share": float(pressing.mean()),
        "marked_share": float(np.mean(np.array(probe.marked) > 0)) if probe.marked else None,
        "mean_p_off": dict(zip(names, p.mean(axis=0).tolist())),
        "mean_p_off_urgent": dict(zip(names, p[pressing].mean(axis=0).tolist()))
        if pressing.any() else None,
        "packets": len(never), "never_sent_share": float(never.mean()),
        "p_never_sent": dict(zip(names, silent.mean(axis=0).tolist())),
        "hidden_reset": {
            "resets": reset.resets,
            "violation_rate": after["violation_rate"][ue],
            "send_share": reset.sent / max(reset.pending, 1),
            "other_ues": {"replay": other_rates(replay), "reset": other_rates(after)},
        },
    }  # fmt: skip


def probes(pairs: list[tuple[Path, Path]], workers: int) -> list[dict]:
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_probe, probe_jobs(pairs), chunksize=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    mapping = sub.add_parser("map")
    mapping.add_argument("--stress", type=Path, nargs="+", required=True)
    mapping.add_argument("--screen", type=Path, nargs="+", required=True)
    mapping.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW")
    mapping.add_argument("--output", type=Path, required=True)
    probe = sub.add_parser("counterfactual")
    probe.add_argument("--stress", type=Path, nargs="+", required=True)
    probe.add_argument("--deltas", type=float, nargs="+", default=[0.0, -10.0, -20.0, -30.0])
    probe.add_argument("--workers", type=int, default=8)
    probe.add_argument("--formal", action="store_true")
    probe.add_argument("--output", type=Path, required=True)
    inputs = sub.add_parser("probes")
    inputs.add_argument("--pairs", nargs="+", required=True, metavar="TARGET=CONTROL",
                        help="formal stress-set output directories of two checkpoints of a run")  # fmt: skip
    inputs.add_argument("--workers", type=int, default=4)
    inputs.add_argument("--formal", action="store_true")
    inputs.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    if args.command != "map":
        provenance = git_provenance()
        if not args.formal or provenance["dirty"]:
            parser.error("replays of the stress set need --formal on a clean committed tree")
    if args.command == "map":
        stress = read_outputs(args.stress, "stress")
        screen = read_outputs(args.screen, "screen")
        missing = sorted(set(stress) - set(screen))
        if missing:
            parser.error(f"no exp009-screen256 output for {missing[:3]}")
        record = starvation_map(stress, screen, dict(i.split("=", 1) for i in args.rename))
    elif args.command == "counterfactual":
        if args.deltas[0] != 0.0:
            parser.error("the first delta must be 0 (the policy's own inputs)")
        record = {
            "formal": True,
            "provenance": provenance,
            "deltas_db": args.deltas,
            "cells": counterfactual(read_outputs(args.stress, "stress"), args.deltas, args.workers),
        }
    else:
        pairs = [tuple(Path(p) for p in item.split("=", 1)) for item in args.pairs]
        record = {
            "formal": True,
            "provenance": provenance,
            "near_m": NEAR_M,
            "cells": probes(pairs, args.workers),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
