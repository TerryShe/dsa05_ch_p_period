"""EXP-DSA05-010 driver (design §2-§3): DAgger runs, selection, evaluation, decision.

* Trains IL (configs/exp010-il.toml) at N = 6, 8, 10 with seeds 101, 202: 6
  formal runs, one core each.
* Screens every checkpoint of a finished run at its own N on exp009-screen256
  (256 deployments) and selects per (seed, N) with ``select_exp009 select
  --grid 6 8 10`` (F_0.01, then F_0.05, then the later iteration).
* Builds the evaluation methods file: the IL seed layers, the A_mix and M seed
  layers of the frozen EXP-009 methods file at N = 6, 8, 10, best_local and the
  canonical N_EDF; evaluates every (spec, N) once on exp010-eval512 (512
  deployments); then ``analyze_exp010.py`` applies the decision rule.

Needs a clean committed tree (every run is --formal). Finished chunks are
skipped on a restart; a training run is never restarted. ``--dry-run`` prints
the plan.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs" / "exp010"
EXP009_METHODS = ROOT / "docs" / "results" / "exp009-confirm_methods" / "methods-v2-confirm2.json"
GRID = (6, 8, 10)
SEEDS = (101, 202)
CORES, MAX_WORKERS, MIN_WORKERS = 16, 8, 2
PY = str(ROOT / ".venv" / "bin" / "python")


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def evaluate_command(screen: str, spec: str, n: int, workers: int, output: Path) -> list[str]:
    return [
        PY, "-m", "dsa_period.experiments.evaluate_capacity",
        "--config", "configs/scenario-v2.toml", "--split", "dev", "--screen", screen,
        "--grid", str(n), "--policies", spec, "--workers", str(workers),
        "--formal", "--output", str(output),
    ]  # fmt: skip


def run_queue(queue: list[tuple[str, str, str, int]], directory: Path, busy_of=lambda: 0) -> None:
    """Run (label, screen, spec, N) chunks under the core budget, failing fast."""
    directory.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple[subprocess.Popen, int]] = {}
    while queue or running:
        for label, (process, _) in list(running.items()):
            if process.poll() is None:
                continue
            del running[label]
            if process.returncode != 0:
                for other, _ in running.values():
                    other.kill()
                sys.exit(f"evaluation {label} failed with status {process.returncode}")
        busy = busy_of() + sum(w for _, w in running.values())
        while queue and CORES - busy >= MIN_WORKERS:
            label, screen, spec, n = queue.pop(0)
            output = directory / label
            if (output / "summary.json").exists():
                continue
            workers = min(MAX_WORKERS, CORES - busy)
            with open(directory / f"{label}.log", "w") as handle:
                running[label] = (
                    subprocess.Popen(
                        evaluate_command(screen, spec, n, workers, output),
                        cwd=ROOT,
                        stdout=handle,
                        stderr=handle,
                    ),
                    workers,
                )
            busy += workers
        time.sleep(20)


def methods_file(selection: dict) -> dict:
    frozen = json.loads(EXP009_METHODS.read_text())

    def cut(layers: list[dict]) -> list[dict]:
        return [{str(n): layer[str(n)] for n in GRID} for layer in layers]

    return {
        "IL": [
            {n: cell["selected"] for n, cell in cells.items()}
            for _, cells in sorted(selection["IL"].items())
        ],
        "A_mix": cut(frozen["A_mix"]),
        "M": cut(frozen["M"]),
        "best_local": "best_local",
        "edf": "central_edf",
    }


def main() -> None:
    runs = [f"il-n{n}-seed{s}" for n in GRID for s in SEEDS]
    if "--dry-run" in sys.argv[1:]:
        print(f"train {len(runs)} runs: {' '.join(runs)}")
        print(f"screen {len(runs) * 6} checkpoints; evaluate 30 (spec, N) chunks of 512")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    training: dict[str, subprocess.Popen] = {}
    for name in runs:
        n, seed = name.split("-n")[1].split("-seed")
        if (OUT / name).exists():
            sys.exit(f"{OUT / name} exists: training runs are never restarted")
        command = [
            PY, "-m", "dsa_period.experiments.train_imitation", "--config",
            "configs/exp010-il.toml", "--seed", seed, "--num-ues", n, "--formal",
            "--output", str(OUT / name),
        ]  # fmt: skip
        with open(OUT / f"{name}.log", "w") as handle:
            training[name] = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=handle)
        log(f"training {name}")
    for name, process in training.items():
        if process.wait() != 0:
            for other in training.values():
                other.kill()
            sys.exit(f"training {name} failed with status {process.returncode}")
        log(f"trained {name}")
    screen_queue = [
        (
            f"{name}-{ck.stem}",
            "exp009-screen256",
            f"marl:{ck.relative_to(ROOT)}",
            int(name.split("-n")[1].split("-")[0]),
        )
        for name in runs
        for ck in sorted((OUT / name / "checkpoints").glob("*.pt"))
    ]
    run_queue(screen_queue, OUT / "screen")
    log("screening done")
    chunks = sorted(str(p.parent) for p in (OUT / "screen").glob("*/summary.json"))
    select = [PY, "-m", "dsa_period.experiments.select_exp009"]
    subprocess.run(
        [*select, "select", "--screen", *chunks, "--grid", *map(str, GRID),
         "--output", str(OUT / "selection.json")], cwd=ROOT, check=True,
    )  # fmt: skip
    selection = json.loads((OUT / "selection.json").read_text())["selection"]
    methods = methods_file(selection)
    (OUT / "methods-exp010.json").write_text(json.dumps(methods, indent=2))
    pairs: dict[tuple[str, int], None] = {}
    for group in ("IL", "A_mix", "M"):
        for layer in methods[group]:
            pairs.update(dict.fromkeys((spec, int(n)) for n, spec in layer.items()))
    for spec in (methods["best_local"], methods["edf"]):
        pairs.update(dict.fromkeys((spec, n) for n in GRID))
    eval_queue = [
        (
            f"{spec.replace('marl:outputs/', '').replace('/', '-').replace(':', '-')}-n{n}",
            "exp010-eval512",
            spec,
            n,
        )
        for spec, n in pairs
    ]
    run_queue(eval_queue, OUT / "eval")
    outputs = sorted(str(p.parent) for p in (OUT / "eval").glob("*/summary.json"))
    if len(outputs) != len(pairs):
        sys.exit(f"{len(outputs)} evaluations, expected {len(pairs)}")
    subprocess.run(
        [PY, "scripts/analyze_exp010.py", "--evaluations", *outputs, "--methods",
         str(OUT / "methods-exp010.json"), "--output", str(OUT / "analysis.json")],
        cwd=ROOT, check=True,
    )  # fmt: skip
    log("exp010 done")


if __name__ == "__main__":
    main()
