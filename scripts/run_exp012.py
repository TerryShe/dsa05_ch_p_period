"""EXP-DSA05-012 driver: one stage per invocation, gated (design §4).

    run_exp012.py stage1   pilot: SC, SC-B, M-S and M, seeds 101 and 202, 5M slots
                           each on scenario v2.1; the last checkpoints and the rules
                           M+S, M+Y and R17-like (on the pilot's M) on exp012-pilot256
                           at N = 8, 10, 12; the gate-A analysis

A stage runs only if ``outputs/exp012/gates.json`` records the user's approval of
the gate before it (stage 1 needs gate 0). Gate records are written by hand after
the user agrees. Everything is --formal on a clean committed tree; a training run
is never restarted and finished evaluation chunks are skipped on a restart.
``--dry-run`` prints the stage's work; any other argument is refused.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs" / "exp012"
PY = str(ROOT / ".venv" / "bin" / "python")
SCENARIO = "configs/scenario-v2.1.toml"
PILOT_SET, PILOT_GRID = "exp012-pilot256", (8, 10, 12)
PILOT = {"sc": "SC", "scb": "SC-B", "ms": "M-S", "m": "M"}
SEEDS = (101, 202)
RULES = {
    "M+S": "yield[theta=-10,content=survival]@{m}",
    "M+Y": "yield[theta=-10]@{m}",
    "R17-like": "boost[]@{m}",
}
CORES, WORKERS = 28, 7
GATES = {"stage1": "0"}


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def check_gate(stage: str) -> None:
    gates_file = OUT / "gates.json"
    gates = json.loads(gates_file.read_text()) if gates_file.exists() else {}
    if GATES[stage] not in gates:
        sys.exit(f"{stage} needs the user's approval of gate {GATES[stage]} in {gates_file}")


def run_parallel(jobs: list[tuple[str, list[str], Path]], cores: int, per_job: int) -> None:
    """Run (name, command, output) jobs, ``cores // per_job`` at a time, failing fast."""
    pending = [job for job in jobs if not job[2].exists()]
    running: dict[str, subprocess.Popen] = {}
    while pending or running:
        for name, process in list(running.items()):
            if process.poll() is None:
                continue
            del running[name]
            if process.returncode != 0:
                for other in running.values():
                    other.kill()
                sys.exit(f"{name} failed with status {process.returncode}")
            log(f"{name} done")
        while pending and len(running) < max(1, cores // per_job):
            name, command, output = pending.pop(0)
            output.parent.mkdir(parents=True, exist_ok=True)
            handle = open(output.parent / f"{output.name}.log", "w")
            running[name] = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=handle)
            log(f"started {name}")
        time.sleep(20)


def last_checkpoint(run: Path) -> Path:
    found = sorted((run / "checkpoints").glob("step-*.pt"))
    if not found:
        sys.exit(f"{run} has no checkpoint")
    return found[-1]


def stage1(dry: bool) -> None:
    runs = [(key, seed) for key in PILOT for seed in SEEDS]
    train = [
        (
            f"{key}-seed{seed}",
            [
                PY,
                "-m",
                "dsa_period.experiments.train_capacity_mappo",
                "--config",
                f"configs/exp012-pilot-{key}.toml",
                "--scenario",
                SCENARIO,
                "--variant",
                "fixed",
                "--seed",
                str(seed),
                "--formal",
                "--output",
                str(OUT / "pilot" / f"{key}-seed{seed}"),
            ],
            OUT / "pilot" / f"{key}-seed{seed}",
        )  # fmt: skip
        for key, seed in runs
    ]
    if dry:
        print(f"{len(train)} pilot runs, then {len(PILOT) + len(RULES)} methods x {len(SEEDS)} "
              f"seeds on {PILOT_SET} at N = {PILOT_GRID}, then the gate-A analysis")  # fmt: skip
        return
    OUT.mkdir(parents=True, exist_ok=True)
    run_parallel(train, len(train), 1)  # one core per training run
    log("pilot training done")
    groups: dict[str, list[dict]] = {name: [] for name in (*PILOT.values(), *RULES)}
    chunks = []
    for seed in SEEDS:
        specs = {
            PILOT[
                key
            ]: f"marl:{last_checkpoint(OUT / 'pilot' / f'{key}-seed{seed}').relative_to(ROOT)}"
            for key in PILOT
        }
        specs |= {name: rule.format(m=specs["M"]) for name, rule in RULES.items()}
        for name, spec in specs.items():
            groups[name].append({str(n): spec for n in PILOT_GRID})
            label = f"{name.replace('+', 'p')}-seed{seed}"
            output = OUT / "pilot-eval" / label
            command = [PY, "-m", "dsa_period.experiments.evaluate_capacity", "--config", SCENARIO,
                       "--split", "dev", "--screen", PILOT_SET, "--grid", *map(str, PILOT_GRID),
                       "--policies", spec, "--formal", "--workers", str(WORKERS),
                       "--output", str(output)]  # fmt: skip
            chunks.append((label, command, output))
    plan = {"grid": list(PILOT_GRID), "groups": groups}
    (OUT / "pilot-plan.json").write_text(json.dumps(plan, indent=2))
    run_parallel(chunks, CORES, WORKERS)
    log("pilot evaluation done")
    analysis = OUT / "pilot-analysis.json"
    if not analysis.exists():
        command = [PY, "scripts/analyze_exp012.py", "pilot", "--plan", str(OUT / "pilot-plan.json"),
                   "--evaluations", *[str(c[2]) for c in chunks], "--output", str(analysis)]  # fmt: skip
        subprocess.run(command, cwd=ROOT, check=True)
    log("stage 1 done")


def main() -> None:
    args = sys.argv[1:]
    stages = {"stage1": stage1}
    if not args or args[0] not in stages:
        sys.exit(f"usage: run_exp012.py {{{','.join(stages)}}} [--dry-run]")
    if any(flag != "--dry-run" for flag in args[1:]):
        sys.exit(f"unknown arguments {args[1:]}; the only option is --dry-run")
    dry = "--dry-run" in args[1:]
    if not dry:
        check_gate(args[0])
    stages[args[0]](dry)


if __name__ == "__main__":
    main()
