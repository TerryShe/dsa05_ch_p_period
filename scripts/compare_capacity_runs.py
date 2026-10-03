"""Bit-for-bit comparison of two capacity-MAPPO runs (EXP-DSA05-008 design §4.4).

R0 of EXP-008 and variant A of EXP-006 phase 1 share every resolved setting and
seed, so their checkpoints should be identical. ``checkpoints`` compares every
tensor and random state of the checkpoints both runs have (networks,
optimizers, minibatch shuffling, numpy/sampler/torch generators, slots,
iteration, seed); metadata such as the settings dicts is not compared.
``log`` compares training.jsonl row by row over the reference's keys, without
the timing fields; a candidate that is still running is compared over the rows
it has. ``validation`` compares the rows of two validation summaries at the N
both evaluated, for the checkpoints with the same name in paired runs (``--runs
fixed-seed101=r0-seed101``). Each command prints the first difference and exits
with status 1 when there is one.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

TIMING = {"collect_s", "update_s", "slots_per_s"}
SECTIONS = ("slots", "iteration", "seed", "ppo", "rng")


def differences(reference, candidate, path: str = "") -> list[str]:
    """Paths at which ``candidate`` differs from ``reference`` (exact equality)."""
    if isinstance(reference, torch.Tensor) or isinstance(candidate, torch.Tensor):
        same = (
            isinstance(reference, torch.Tensor)
            and isinstance(candidate, torch.Tensor)
            and reference.dtype == candidate.dtype
            and reference.shape == candidate.shape
            and torch.equal(reference, candidate)
        )
        return [] if same else [path]
    if isinstance(reference, np.ndarray) or isinstance(candidate, np.ndarray):
        return [] if np.array_equal(reference, candidate) else [path]
    if isinstance(reference, dict) and isinstance(candidate, dict):
        if set(reference) != set(candidate):
            return [f"{path} keys {sorted(map(str, set(reference) ^ set(candidate)))}"]
        found = []
        for key in reference:
            found += differences(reference[key], candidate[key], f"{path}/{key}")
        return found
    if isinstance(reference, (list, tuple)) and isinstance(candidate, (list, tuple)):
        if len(reference) != len(candidate):
            return [f"{path} length"]
        found = []
        for i, (a, b) in enumerate(zip(reference, candidate)):
            found += differences(a, b, f"{path}[{i}]")
        return found
    return [] if reference == candidate else [path]


def compare_checkpoint(reference: Path, candidate: Path) -> list[str]:
    a = torch.load(reference, map_location="cpu", weights_only=False)
    b = torch.load(candidate, map_location="cpu", weights_only=False)
    found = []
    for section in SECTIONS:
        found += differences(a[section], b[section], section)
    return found


def compare_log(reference: Path, candidate: Path) -> tuple[int, str | None]:
    """Rows compared, and the first difference as 'iteration N: key'."""
    rows_a = [json.loads(line) for line in reference.read_text().splitlines()]
    compared = 0
    for line in candidate.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # a line still being written
            break
        if compared >= len(rows_a):
            return compared, f"row {compared + 1}: the reference has no such row"
        ref = rows_a[compared]
        for key in ref:
            if key not in TIMING and differences(ref[key], row.get(key)):
                return compared, f"iteration {ref['iteration']}: {key}"
        compared += 1
    return compared, None


def _by_run(record: dict, runs: dict[str, str]) -> dict[tuple[str, str], str]:
    """(paired run name, checkpoint name) -> spec, for the runs in ``runs``."""
    out = {}
    for spec in record["summaries"]:
        path = Path(spec.partition(":")[2])
        run = path.parent.parent.name
        if run in runs:
            out[(runs[run], path.name)] = spec
    return out


def compare_validation(reference: Path, candidate: Path, pairs: dict[str, str]) -> list[str]:
    a, b = (json.loads(p.read_text()) for p in (reference, candidate))
    if a["split"] != "validation" or b["split"] != "validation" or a["instances"] != b["instances"]:
        return ["the summaries are not validation evaluations of the same instances"]
    specs_a = _by_run(a, pairs)
    specs_b = _by_run(b, {run: run for run in pairs.values()})
    common = sorted(set(specs_a) & set(specs_b))
    if not common:
        return ["no checkpoint of the paired runs is in both summaries"]
    found = []
    for key in common:
        name = "/".join(key)
        sa, sb = a["summaries"][specs_a[key]], b["summaries"][specs_b[key]]
        for key in sa["per_eps"]:
            rows_a = {r["num_ues"]: r for r in sa["per_eps"][key]["rows"]}
            rows_b = {r["num_ues"]: r for r in sb["per_eps"][key]["rows"]}
            for n in sorted(set(rows_a) & set(rows_b)):
                found += differences(rows_a[n], rows_b[n], f"{name}/eps {key}/N {n}")
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("checkpoints")
    c.add_argument("--reference", type=Path, required=True, help="run directory")
    c.add_argument("--candidate", type=Path, required=True, help="run directory")
    lg = sub.add_parser("log")
    lg.add_argument("--reference", type=Path, required=True, help="run directory")
    lg.add_argument("--candidate", type=Path, required=True, help="run directory")
    v = sub.add_parser("validation")
    v.add_argument("--reference", type=Path, required=True, help="summary.json")
    v.add_argument("--candidate", type=Path, required=True, help="summary.json")
    v.add_argument("--runs", nargs="+", required=True, metavar="REF_RUN=CAND_RUN")
    args = parser.parse_args(argv)
    if args.command == "checkpoints":
        names = sorted(
            {p.name for p in (args.reference / "checkpoints").glob("*.pt")}
            & {p.name for p in (args.candidate / "checkpoints").glob("*.pt")}
        )
        if not names:
            sys.exit("no checkpoint is in both runs")
        bad = False
        for name in names:
            found = compare_checkpoint(
                args.reference / "checkpoints" / name, args.candidate / "checkpoints" / name
            )
            print(f"{name}: {'identical' if not found else 'DIFFERENT ' + ', '.join(found[:5])}")
            bad = bad or bool(found)
        sys.exit(1 if bad else 0)
    if args.command == "log":
        compared, first = compare_log(
            args.reference / "training.jsonl", args.candidate / "training.jsonl"
        )
        print(f"{compared} rows compared: {'identical' if first is None else first}")
        sys.exit(1 if first else 0)
    pairs = dict(item.split("=", 1) for item in args.runs)
    found = compare_validation(args.reference, args.candidate, pairs)
    print("identical" if not found else "DIFFERENT " + ", ".join(found[:5]))
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
