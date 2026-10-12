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

from dsa_period import starvation
from dsa_period.capacity import run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.experiments import select_exp011b
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.marl.features import legal_mask, stack
from dsa_period.marl.networks import split_logits
from dsa_period.marl.policy import SummaryPolicy
from dsa_period.rules_v2 import make_policy_v2
from dsa_period.urgency import urgent, visible, with_summary

CONFIG = starvation.CONFIG
EARLY, LATE = 3_000_000, 5_100_000  # the 3M-5M checkpoints of gate A
GRIDS = select_exp011b.GRIDS  # screening N counted per condition (the evaluation N)


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
        self.sent = 0

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
            gains = obs.own_gain.copy()
            gains[ue] *= 10.0 ** (delta / 10.0)
            tokens, context = memory.features(replace(obs, own_gain=gains))
            if link is not None:
                tokens, context = with_summary(tokens, context, seen, own)
            with torch.no_grad():
                logits, _ = inner.actor.step(
                    torch.from_numpy(tokens[ue : ue + 1]),
                    torch.from_numpy(context[ue : ue + 1]),
                    inner._hidden[ue : ue + 1],
                )
            logits = split_logits(logits, inner.actor.num_actions)[0][0].double().numpy()
            masked = np.where(legal, logits, -np.inf)
            p = np.exp(masked - masked.max())
            out.append(float(p[0] / p.sum()))
        return out

    def act(self, observations):
        obs = stack(observations)
        if obs.remaining[self.ue] > 0:
            link = copy.deepcopy(self.inner.link) if isinstance(self.inner, SummaryPolicy) else None
            self.p_off.append(self._p_off(obs, copy.deepcopy(self.inner._memory), link))
        actions = self.inner.act(observations)
        self.sent += int(obs.remaining[self.ue] > 0 and actions[self.ue][1] > 0)
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
        "pending_slots": len(p), "send_share": probe.sent / max(len(p), 1),
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
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    stress = read_outputs(args.stress, "stress")
    if args.command == "map":
        screen = read_outputs(args.screen, "screen")
        missing = sorted(set(stress) - set(screen))
        if missing:
            parser.error(f"no exp009-screen256 output for {missing[:3]}")
        record = starvation_map(stress, screen, dict(i.split("=", 1) for i in args.rename))
    else:
        provenance = git_provenance()
        if not args.formal or provenance["dirty"]:
            parser.error("replays of the stress set need --formal on a clean committed tree")
        if args.deltas[0] != 0.0:
            parser.error("the first delta must be 0 (the policy's own inputs)")
        record = {
            "formal": True,
            "provenance": provenance,
            "deltas_db": args.deltas,
            "cells": counterfactual(stress, args.deltas, args.workers),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
