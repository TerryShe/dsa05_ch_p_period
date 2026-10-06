"""EXP-DSA05-009 step 2 driver (design §5): samplers on B = M0, screening, selection, rules.

* Trains S1, P1, P2 (configs/exp009-step2-*.toml, mixed N like R0) with seeds
  101 and 202: 6 formal runs, one core each.
* Screens every checkpoint at every N of exp009-screen256 as soon as it is
  written (a checkpoint counts as written 60 s after its last modification),
  one chunk per (checkpoint, N) of 256 episodes, with as many workers as are
  free under CORES (at most MAX_WORKERS).
* Then ``select_exp009 select`` of the step-2 runs and ``select_exp009 step2``
  against U = M0 (outputs/exp009-step1/selection.json, from step 1).

Needs a clean committed tree (every run is --formal). Finished screening chunks
are skipped on a restart; a training run is never restarted. ``--dry-run``
prints the plan and starts nothing.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs" / "exp009-step2"
STEP1 = ROOT / "outputs" / "exp009-step1" / "selection.json"
GRID = (2, 4, 6, 8, 10, 12, 16, 20)
SEEDS = (101, 202)
CONDITIONS = ("s1", "p1", "p2")
CHECKPOINTS = 5
CORES, MAX_WORKERS, MIN_WORKERS, SETTLE_S = 18, 8, 2, 60
PY = str(ROOT / ".venv" / "bin" / "python")


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def screen_command(spec: str, n: int, workers: int, output: Path) -> list[str]:
    return [
        PY, "-m", "dsa_period.experiments.evaluate_capacity",
        "--config", "configs/scenario-v2.toml", "--split", "dev",
        "--screen", "exp009-screen256", "--grid", str(n), "--policies", spec,
        "--workers", str(workers), "--formal", "--output", str(output),
    ]  # fmt: skip


def kill_all(processes) -> None:
    for process in processes:
        process.kill()


def main() -> None:
    if not STEP1.exists():
        sys.exit(f"{STEP1} is missing: step 1 must be finished")
    runs = [f"{c}-seed{s}" for c in CONDITIONS for s in SEEDS]
    if "--dry-run" in sys.argv[1:]:
        print(f"train {len(runs)} runs: {' '.join(runs)}")
        print(f"chunks: {len(runs) * CHECKPOINTS * len(GRID)}; budget {CORES} cores")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen").mkdir(exist_ok=True)
    training: dict[str, subprocess.Popen] = {}
    for name in runs:
        condition, seed = name.split("-seed")
        if (OUT / name).exists():
            sys.exit(f"{OUT / name} exists: training runs are never restarted")
        command = [
            PY, "-m", "dsa_period.experiments.train_capacity_mappo",
            "--config", f"configs/exp009-step2-{condition}.toml", "--variant", "fixed",
            "--seed", seed, "--formal", "--output", str(OUT / name),
        ]  # fmt: skip
        with open(OUT / f"{name}.log", "w") as handle:
            training[name] = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=handle)
        log(f"training {name}")
    seen: set[Path] = set()
    queue: list[tuple[str, str, int]] = []
    evaluating: dict[str, tuple[subprocess.Popen, int]] = {}
    while True:
        for name, process in list(training.items()):
            code = process.poll()
            if code is None:
                continue
            del training[name]
            if code != 0:
                kill_all([*training.values(), *(p for p, _ in evaluating.values())])
                sys.exit(f"training {name} failed with status {code}")
            log(f"trained {name}")
        now = time.time()
        for name in runs:
            for ck in sorted((OUT / name / "checkpoints").glob("*.pt")):
                if ck in seen or now - ck.stat().st_mtime < SETTLE_S:
                    continue
                seen.add(ck)
                spec = f"marl:{ck.relative_to(ROOT)}"
                queue.extend((f"{name}-{ck.stem}-n{n}", spec, n) for n in GRID)
                log(f"queued {name}/{ck.name}")
        for label, (process, _) in list(evaluating.items()):
            code = process.poll()
            if code is None:
                continue
            del evaluating[label]
            if code != 0:
                kill_all([*training.values(), *(p for p, _ in evaluating.values())])
                sys.exit(f"screening {label} failed with status {code}")
        busy = len(training) + sum(w for _, w in evaluating.values())
        while queue and CORES - busy >= MIN_WORKERS:
            label, spec, n = queue.pop(0)
            output = OUT / "screen" / label
            if (output / "summary.json").exists():
                continue
            workers = min(MAX_WORKERS, CORES - busy)
            with open(OUT / "screen" / f"{label}.log", "w") as handle:
                process = subprocess.Popen(
                    screen_command(spec, n, workers, output), cwd=ROOT, stdout=handle, stderr=handle
                )
            evaluating[label] = (process, workers)
            busy += workers
        if not training and not evaluating and not queue and len(seen) == len(runs) * CHECKPOINTS:
            break
        time.sleep(20)
    log("screening done")
    chunks = sorted(str(p.parent) for p in (OUT / "screen").glob("*/summary.json"))
    if len(chunks) != len(runs) * CHECKPOINTS * len(GRID):
        sys.exit(f"{len(chunks)} screening outputs, expected {len(runs) * CHECKPOINTS * len(GRID)}")
    select = [PY, "-m", "dsa_period.experiments.select_exp009"]
    subprocess.run(
        [*select, "select", "--screen", *chunks, "--output", str(OUT / "selection.json")],
        cwd=ROOT, check=True,
    )  # fmt: skip
    subprocess.run(
        [*select, "step2", "--selection", str(STEP1), str(OUT / "selection.json"),
         "--baseline", "M0", "--output", str(OUT / "step2.json")], cwd=ROOT, check=True,
    )  # fmt: skip
    log("step 2 done")


if __name__ == "__main__":
    main()
