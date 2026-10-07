"""EXP-DSA05-009 finalization driver (design §6): M's third seed, A-mix, methods file.

* Trains seed 303 of M = S1 (configs/exp009-final-s1.toml): 1 formal run.
* Screens on exp009-screen256, one chunk per (checkpoint, N) of 256 episodes:
  the last 5 checkpoints of each EXP-006 long run (A-mix, mixed N) at every N,
  and every checkpoint of the new S1 run at every N as soon as it is written.
* ``select_exp009 select`` of these chunks (the EXP-006 runs renamed A_mix), then
  ``select_exp009 methods`` over the step-2 and the new selection: the methods
  file of the confirmation analysis, written to outputs/exp009-final/ for review.
  It is frozen by committing it before v2-confirm2 is read (run_exp009_confirm.py).

Needs a clean committed tree (every run is --formal). ``--dry-run`` prints the plan.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs" / "exp009-final"
STEP2 = ROOT / "outputs" / "exp009-step2" / "selection.json"
A_MIX = [ROOT / "outputs" / "exp006-phase2" / f"fixed-seed{s}" for s in (101, 202, 303)]
RUN = "s1-seed303"
GRID = (2, 4, 6, 8, 10, 12, 16, 20)
CHECKPOINTS, A_MIX_LAST = 5, 5
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


def main() -> None:
    if not STEP2.exists():
        sys.exit(f"{STEP2} is missing: step 2 must be finished")
    a_mix = []
    for run in A_MIX:
        checkpoints = sorted((run / "checkpoints").glob("*.pt"))
        if len(checkpoints) != 10:
            sys.exit(f"{run} is not a finished 10-checkpoint EXP-006 long run")
        a_mix += checkpoints[-A_MIX_LAST:]
    queue = [
        (f"amix-{ck.parent.parent.name}-{ck.stem}-n{n}", f"marl:{ck.relative_to(ROOT)}", n)
        for ck in a_mix
        for n in GRID
    ]
    if "--dry-run" in sys.argv[1:]:
        print(f"train {RUN}; A-mix checkpoints {[c.stem for c in a_mix]}")
        print(f"chunks: A-mix {len(queue)}, S1 seed 303 {CHECKPOINTS * len(GRID)}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen").mkdir(exist_ok=True)
    if (OUT / RUN).exists():
        sys.exit(f"{OUT / RUN} exists: training runs are never restarted")
    command = [
        PY, "-m", "dsa_period.experiments.train_capacity_mappo",
        "--config", "configs/exp009-final-s1.toml", "--variant", "fixed",
        "--seed", "303", "--formal", "--output", str(OUT / RUN),
    ]  # fmt: skip
    with open(OUT / f"{RUN}.log", "w") as handle:
        training = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=handle)
    log(f"training {RUN}")
    seen: set[Path] = set()
    evaluating: dict[str, tuple[subprocess.Popen, int]] = {}
    done_training = False
    while True:
        if not done_training and training.poll() is not None:
            if training.returncode != 0:
                for process, _ in evaluating.values():
                    process.kill()
                sys.exit(f"training {RUN} failed with status {training.returncode}")
            done_training = True
            log(f"trained {RUN}")
        now = time.time()
        for ck in sorted((OUT / RUN / "checkpoints").glob("*.pt")):
            if ck in seen or now - ck.stat().st_mtime < SETTLE_S:
                continue
            seen.add(ck)
            queue.extend((f"{RUN}-{ck.stem}-n{n}", f"marl:{ck.relative_to(ROOT)}", n) for n in GRID)
            log(f"queued {RUN}/{ck.name}")
        for label, (process, _) in list(evaluating.items()):
            if process.poll() is None:
                continue
            del evaluating[label]
            if process.returncode != 0:
                if not done_training:
                    training.kill()
                for other, _ in evaluating.values():
                    other.kill()
                sys.exit(f"screening {label} failed with status {process.returncode}")
        busy = (0 if done_training else 1) + sum(w for _, w in evaluating.values())
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
        if done_training and not evaluating and not queue and len(seen) == CHECKPOINTS:
            break
        time.sleep(20)
    log("screening done")
    chunks = sorted(str(p.parent) for p in (OUT / "screen").glob("*/summary.json"))
    expected = (len(a_mix) + CHECKPOINTS) * len(GRID)
    if len(chunks) != expected:
        sys.exit(f"{len(chunks)} screening outputs, expected {expected}")
    select = [PY, "-m", "dsa_period.experiments.select_exp009"]
    subprocess.run(
        [*select, "select", "--screen", *chunks, "--rename", "exp006-capacity-mappo-long=A_mix",
         "--output", str(OUT / "selection.json")], cwd=ROOT, check=True,
    )  # fmt: skip
    subprocess.run(
        [*select, "methods", "--selection", str(STEP2), str(OUT / "selection.json"),
         "--primary", "S1", "--baseline", "A_mix",
         "--output", str(OUT / "methods-v2-confirm2.json")], cwd=ROOT, check=True,
    )  # fmt: skip
    log("finalization done")


if __name__ == "__main__":
    main()
