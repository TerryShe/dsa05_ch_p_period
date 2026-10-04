"""EXP-DSA05-009 step 1 driver (design §4): P0 per-N training, screening, selection, rules.

* Trains P0 (configs/exp009-step1-p0.toml) at every N of the design and seeds
  101, 202: 16 formal runs, one core each.
* Screens on exp009-screen256, one chunk per (checkpoint, N) of 256 episodes:
  every M0 checkpoint (the EXP-008 round-1 R0 runs, mixed N) at every N, and
  every P0 checkpoint at its own N once its run has finished.
* Schedules everything under a budget of CORES processes: training first, then
  screening chunks with as many workers as are free (at most MAX_WORKERS).
* Then ``select_exp009 select`` (R0 renamed M0) and ``select_exp009 step1``.

Needs a clean committed tree (every run is --formal). Finished screening chunks
are skipped on a restart; a training run is never restarted. ``--dry-run`` prints
the plan and starts nothing.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs" / "exp009-step1"
M0_RUNS = [ROOT / "outputs" / "exp008-round1" / f"r0-seed{s}" for s in (101, 202)]
GRID = (2, 4, 6, 8, 10, 12, 16, 20)
SEEDS = (101, 202)
CORES, MAX_WORKERS, MIN_WORKERS = 18, 8, 2
PY = str(ROOT / ".venv" / "bin" / "python")


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def checkpoints(run: Path) -> list[Path]:
    return sorted((run / "checkpoints").glob("*.pt"))


def screen_command(spec: str, n: int, workers: int, output: Path) -> list[str]:
    return [
        PY, "-m", "dsa_period.experiments.evaluate_capacity",
        "--config", "configs/scenario-v2.toml", "--split", "dev",
        "--screen", "exp009-screen256", "--grid", str(n), "--policies", spec,
        "--workers", str(workers), "--formal", "--output", str(output),
    ]  # fmt: skip


def main() -> None:
    for run in M0_RUNS:
        if len(checkpoints(run)) != 5 or not (run / "result.json").exists():
            sys.exit(f"{run} is not a finished 5-checkpoint run")
    if "--dry-run" in sys.argv[1:]:
        runs = [f"p0-n{n}-seed{s}" for n in GRID for s in SEEDS]
        print(f"train {len(runs)} runs: {' '.join(runs)}")
        print(f"M0 chunks: {2 * 5 * len(GRID)}; P0 chunks: {len(runs) * 5}; budget {CORES} cores")
        print("example:", " ".join(screen_command("marl:<checkpoint>", 8, MAX_WORKERS, OUT / "x")))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen").mkdir(exist_ok=True)
    training: dict[str, subprocess.Popen] = {}
    for n in GRID:
        for seed in SEEDS:
            name = f"p0-n{n}-seed{seed}"
            if (OUT / name).exists():
                sys.exit(f"{OUT / name} exists: training runs are never restarted")
            command = [
                PY, "-m", "dsa_period.experiments.train_capacity_mappo",
                "--config", "configs/exp009-step1-p0.toml", "--variant", "fixed",
                "--seed", str(seed), "--num-ues", str(n), "--formal", "--output", str(OUT / name),
            ]  # fmt: skip
            with open(OUT / f"{name}.log", "w") as handle:
                training[name] = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=handle)
            log(f"training {name}")
    # Screening chunks: (label, spec, N, output); P0 chunks join when their run ends.
    queue = [
        (f"m0-{run.name}-{ck.stem}-n{n}", f"marl:{ck.relative_to(ROOT)}", n)
        for run in M0_RUNS
        for ck in checkpoints(run)
        for n in GRID
    ]
    evaluating: dict[str, tuple[subprocess.Popen, int]] = {}
    while True:
        for name, process in list(training.items()):
            code = process.poll()
            if code is None:
                continue
            del training[name]
            if code != 0:
                for other in [*training.values(), *(p for p, _ in evaluating.values())]:
                    other.kill()
                sys.exit(f"training {name} failed with status {code}")
            log(f"trained {name}")
            n = int(name.split("-n")[1].split("-")[0])
            for ck in checkpoints(OUT / name):
                queue.append((f"{name}-{ck.stem}", f"marl:{ck.relative_to(ROOT)}", n))
        for label, (process, _) in list(evaluating.items()):
            code = process.poll()
            if code is None:
                continue
            del evaluating[label]
            if code != 0:
                for other in [*training.values(), *(p for p, _ in evaluating.values())]:
                    other.kill()
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
        if not training and not evaluating and not queue:
            break
        time.sleep(20)
    log("screening done")
    chunks = sorted(str(p.parent) for p in (OUT / "screen").glob("*/summary.json"))
    expected = 2 * 5 * len(GRID) + len(GRID) * len(SEEDS) * 5
    if len(chunks) != expected:
        sys.exit(f"{len(chunks)} screening outputs, expected {expected}")
    select = [PY, "-m", "dsa_period.experiments.select_exp009"]
    subprocess.run(
        [*select, "select", "--screen", *chunks, "--rename", "R0=M0", "--output",
         str(OUT / "selection.json")], cwd=ROOT, check=True,
    )  # fmt: skip
    subprocess.run(
        [*select, "step1", "--selection", str(OUT / "selection.json"), "--output",
         str(OUT / "step1.json")], cwd=ROOT, check=True,
    )  # fmt: skip
    log("step 1 done")


if __name__ == "__main__":
    main()
