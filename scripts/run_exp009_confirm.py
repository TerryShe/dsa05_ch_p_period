"""EXP-DSA05-009 confirmation driver (design §6): read v2-confirm2 once, then analyze.

Reads the committed, frozen methods file (``select_exp009 methods``) and
evaluates on v2-confirm2 (``--confirm-c2 --formal``), one chunk per (spec, N)
of 500 episodes:

* seed layers (M, A_mix): each model at the N it serves (N >= 24 with the N = 20
  model, as listed in the layer);
* single specs (M_last, best_local, the six rules): every N of the protocol grid;
* both N_EDF versions: the grid and 36, 40.

A (spec, N) pair used by several groups is evaluated once. Then
``analyze_exp008_confirm.py --set v2-confirm2`` with the same methods file.

Refuses to run unless the methods file is tracked and unchanged in a clean
tree; finished chunks are skipped on a restart. ``--dry-run`` prints the plan.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs" / "exp009-confirm"
METHODS = ROOT / "docs" / "results" / "exp009-confirm_methods" / "methods-v2-confirm2.json"
GRID = (2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32)
EDF_EXTRA = (36, 40)
CORES, MAX_WORKERS, MIN_WORKERS = 16, 8, 2
PY = str(ROOT / ".venv" / "bin" / "python")


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def pairs(methods: dict) -> list[tuple[str, int]]:
    """Distinct (spec, N) pairs to evaluate, in a stable order."""
    found: dict[tuple[str, int], None] = {}
    for group, members in methods.items():
        if group.startswith("_") or group in ("primary", "baseline", "best_local", "rules", "edf"):
            continue
        for member in members:
            if isinstance(member, str):
                found.update(dict.fromkeys((member, n) for n in GRID))
            else:
                found.update(dict.fromkeys((spec, int(n)) for n, spec in member.items()))
    for spec in (methods["best_local"], *methods["rules"]):
        found.update(dict.fromkeys((spec, n) for n in GRID))
    for spec in methods["edf"]:
        found.update(dict.fromkeys((spec, n) for n in (*GRID, *EDF_EXTRA)))
    return list(found)


def label(spec: str, n: int) -> str:
    name = spec.replace("marl:", "").replace("outputs/", "").replace("/checkpoints/", "-")
    return f"{name.replace('/', '-').replace(':', '-').replace('=', '-')}-n{n}"


def main() -> None:
    methods = json.loads(METHODS.read_text())
    todo = pairs(methods)
    if "--dry-run" in sys.argv[1:]:
        print(f"{len(todo)} (spec, N) chunks x 500 episodes = {500 * len(todo)} episodes")
        return
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(METHODS)], cwd=ROOT, capture_output=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    if tracked.returncode != 0 or status.stdout.strip():
        sys.exit("the methods file must be committed and the tree clean before reading v2-confirm2")
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    queue = list(todo)
    evaluating: dict[str, tuple[subprocess.Popen, int]] = {}
    while queue or evaluating:
        for name, (process, _) in list(evaluating.items()):
            if process.poll() is None:
                continue
            del evaluating[name]
            if process.returncode != 0:
                for other, _ in evaluating.values():
                    other.kill()
                sys.exit(f"evaluation {name} failed with status {process.returncode}")
        busy = sum(w for _, w in evaluating.values())
        while queue and CORES - busy >= MIN_WORKERS:
            spec, n = queue.pop(0)
            name = label(spec, n)
            output = OUT / "eval" / name
            if (output / "summary.json").exists():
                continue
            workers = min(MAX_WORKERS, CORES - busy)
            command = [
                PY, "-m", "dsa_period.experiments.evaluate_capacity",
                "--config", "configs/scenario-v2.toml", "--split", "v2-confirm2",
                "--confirm-c2", "--formal", "--grid", str(n), "--policies", spec,
                "--workers", str(workers), "--output", str(output),
            ]  # fmt: skip
            with open(OUT / "eval" / f"{name}.log", "w") as handle:
                evaluating[name] = (
                    subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=handle),
                    workers,
                )
            busy += workers
        time.sleep(20)
    log("evaluation done")
    outputs = sorted(str(p.parent) for p in (OUT / "eval").glob("*/summary.json"))
    if len(outputs) != len(todo):
        sys.exit(f"{len(outputs)} evaluations, expected {len(todo)}")
    subprocess.run(
        [PY, "scripts/analyze_exp008_confirm.py", "--set", "v2-confirm2", "--methods",
         str(METHODS), "--evaluations", *outputs, "--output", str(OUT / "analysis.json")],
        cwd=ROOT, check=True,
    )  # fmt: skip
    log("confirmation done")


if __name__ == "__main__":
    main()
