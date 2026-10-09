"""Archive the information-boundary probe (descriptive): summaries and derived tables.

Reads outputs/probe-broadcast/<run>.json for every run template in --runs
(default "n{n}", the first probe; scripts/probe_broadcast.py), merges the
episodes of the runs per N (same specs and instances, disjoint modes; the
summary is then recomputed with the probe's own summarize) and writes into
--output:

* summary-n{N}.json: the probe's own summary (cells and groups) and specs,
  without the per-episode records;
* analysis.json: per N and group, F at eps = 0.01 per mode with a two-level
  bootstrap interval; per layer and mode, F, deployments repaired / broken
  against "none", yield decisions and transmissions per episode, collision
  share; the deployments infeasible for every policy; per group (no broadcast),
  choice rates relative to a free subchannel and the entry statistics around
  failing UEs; for the B3 variants, urgency flags sent and heard per episode.
"""

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dsa_period.capacity import BOOTSTRAP_SEED
from dsa_period.evaluation import _bootstrap, _interval

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "outputs" / "probe-broadcast"
GRID = (6, 8, 10)
_spec = importlib.util.spec_from_file_location("probe", ROOT / "scripts" / "probe_broadcast.py")
probe = importlib.util.module_from_spec(_spec)
sys.modules["probe"] = probe
_spec.loader.exec_module(probe)
MODES = probe.MODES
OCCUPIED = [
    f"{b}|{u}" for u in ("calm", "urgent") for b in ("<3", "3-10", "10-20", ">=20")
]  # fmt: skip
PAIRS = ("entry", "victim_entry", "co_residence", "both_new")
ENTRY_KEYS = (
    "excess|<3",
    "excess|3-10",
    "excess|10-20",
    "excess|>=20",
    "free_subchannel_existed",
    "best_own_channel",
    "entering_urgent",
)


def analyse(data: dict, n: int, resamples: int) -> dict:
    cells = data["summary"]["cells"]
    by_cell = defaultdict(dict)
    for e in data["episodes"]:
        by_cell[(e["label"], e["mode"])][e["instance"]] = e
    labels = sorted({label for label, _ in by_cell})
    modes = [m for m in MODES if any((label, m) in by_cell for label in labels)]
    instances = sorted(by_cell[(labels[0], "none")])
    out: dict = {
        "infeasible_for_all": sorted(
            i
            for i in instances
            if any(by_cell[(lab, "none")][i]["infeasible_for_all"]["0.01"] for lab in labels)
        ),
        "groups": {},
        "layers": {},
        "visibility": {},
    }
    for group in sorted({label.split("|")[0] for label in labels}):
        members = [label for label in labels if label.split("|")[0] == group]
        row = {}
        for m in modes:
            mat = np.array(
                [
                    [float(by_cell[(lab, m)][i]["feasible"]["0.01"]) for i in instances]
                    for lab in members
                ]
            )
            row[m] = {
                "F": float(mat.mean()),
                "ci95": _interval(_bootstrap(mat, None, resamples, BOOTSTRAP_SEED)),
            }
            if m != "none":
                row[m]["minus_none"] = data["summary"]["groups"][f"{group}|N={n}"][f"{m}-none"]
        out["groups"][group] = row
        available, chosen, pairs, entries = Counter(), Counter(), Counter(), Counter()
        for lab in members:
            cell = cells[f"{lab}|none|N={n}"]
            available.update(cell["choice"]["available"])
            chosen.update(cell["choice"]["chosen"])
            pairs.update(cell["pairs"])
            entries.update(cell["entries"])
        free = chosen["free"] / available["free"]
        total_available = sum(available.values())
        out["visibility"][group] = {
            "choice_rate_free": free,
            "relative_choice_rate": {k: chosen[k] / available[k] / free for k in OCCUPIED},
            "available_share": {k: available[k] / total_available for k in ("free", *OCCUPIED)},
            "pair_share": {k: pairs[k] / sum(pairs.values()) for k in PAIRS},
            "entries": entries["total"],
            "entry_share": {k: entries[k] / entries["total"] for k in ENTRY_KEYS},
        }
    for lab in labels:
        base = by_cell[(lab, "none")]
        rows = {}
        for m in modes:
            cell = cells[f"{lab}|{m}|N={n}"]
            eps = by_cell[(lab, m)]
            rows[m] = {
                "F": cell["F"]["0.01"],
                "failed": len(cell["failed"]),
                "repaired": sum(
                    (not base[i]["feasible"]["0.01"]) and eps[i]["feasible"]["0.01"]
                    for i in instances
                ),
                "broken": sum(
                    base[i]["feasible"]["0.01"] and not eps[i]["feasible"]["0.01"]
                    for i in instances
                ),
                "silenced_per_episode": cell["silenced"] / cell["episodes"],
                "moved_per_episode": cell["moved"] / cell["episodes"],
                "transmissions_per_episode": cell["transmissions"] / cell["episodes"],
                "collision_share": cell["collided"] / cell["transmissions"],
                "failing_ues": cell["failing_ues"],
                "pairs": {k: cell["pairs"].get(k, 0) for k in PAIRS},
            }
            if cell.get("flags_sent"):
                rows[m]["flags_sent_per_episode"] = cell["flags_sent"] / cell["episodes"]
                rows[m]["flags_heard_share"] = cell["flags_heard"] / cell["flags_sent"]
        out["layers"][lab] = rows
    return out


def load(runs: list[str], n: int, resamples: int) -> dict:
    parts = [json.loads((SOURCE / f"{run.format(n=n)}.json").read_text()) for run in runs]
    data = parts[0]
    if len(parts) == 1:
        return data
    seen = set()
    for part in parts:
        if part["specs"] != data["specs"] or part["instances"] != data["instances"]:
            raise ValueError("runs differ in policies or instances")
        modes = {e["mode"] for e in part["episodes"]}
        if modes & seen:
            raise ValueError(f"modes {sorted(modes & seen)} appear in more than one run")
        seen |= modes
    episodes = [e for part in parts for e in part["episodes"]]
    results = [
        (
            e["label"],
            e["mode"],
            e["num_ues"],
            {k: v for k, v in e.items() if k not in ("label", "mode", "num_ues")},
        )
        for e in episodes
    ]
    return {**data, "summary": probe.summarize(results, resamples), "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--runs", nargs="+", default=["n{n}"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    analysis = {}
    for n in GRID:
        data = load(args.runs, n, args.resamples)
        archived = {k: data[k] for k in ("instances", "urgent_slack", "specs", "summary")}
        (args.output / f"summary-n{n}.json").write_text(json.dumps(archived, indent=1))
        analysis[str(n)] = analyse(data, n, args.resamples)
    (args.output / "analysis.json").write_text(json.dumps(analysis, indent=1))


if __name__ == "__main__":
    main()
