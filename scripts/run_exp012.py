"""EXP-DSA05-012 driver: one stage per invocation, gated (design §4).

    run_exp012.py stage1   pilot: SC, SC-B, M-S and M, seeds 101 and 202, 5M slots
                           each on scenario v2.1; the last checkpoints and the rules
                           M+S, M+Y and R17-like (on the pilot's M) on exp012-pilot256
                           at N = 8, 10, 12; the gate-A analysis
    run_exp012.py stage1-extra
                           after stage 1: M-S+S (the M+S yield rule on the pilot's M-S,
                           design §10.4) on exp012-pilot256, and the analysis with it
                           (descriptive; gate A is the stage-1 analysis)
    run_exp012.py stage1b  gate A = adjust (design §10.5): SC-B with lambda_d = 0.1 and 0.3,
                           seeds 101 and 202, 5M slots each; the last checkpoints on
                           exp012-pilot256 at N = 8, 10, 12; the stage-1b analysis, which
                           picks the lambda_d to carry forward
    run_exp012.py stage1c  design revision 4 (§11.4): SP and SP-B, seeds 101, 202, 303, 404,
                           5M slots each on the frozen pilot M (seeds 101/303 on M seed 101,
                           202/404 on M seed 202); the last checkpoints, M, M+S, M+Y,
                           R17-like and SC-B on exp012-pilot2-256 at N = 8, 10, 12; gate A2.
                           SP0 (design §11.10): the untrained SP on M seeds 101 and 202,
                           evaluated alongside

A stage runs only if ``outputs/exp012/gates.json`` records the user's approval of
the gate before it (stage 1 needs gate 0, stage 1b gate A, stage 1c gate 0'). Gate records are written by hand after
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
STAGE1B = {"scb-ld01": "SC-B[0.1]", "scb-ld03": "SC-B[0.3]"}
PILOT2_SET = "exp012-pilot2-256"
STAGE1C = {"sp": "SP", "spb": "SP-B"}
SEEDS_1C = (101, 202, 303, 404)
BASE_SEED = {101: 101, 202: 202, 303: 101, 404: 202}  # the pilot M under each SP seed
CORES, WORKERS = 28, 7
GATES = {"stage1": "0", "stage1-extra": "0", "stage1b": "A", "stage1c": "0'"}


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


def train_job(
    config: str, seed: int, output: Path, extra: tuple[str, ...] = ()
) -> tuple[str, list[str], Path]:
    command = [PY, "-m", "dsa_period.experiments.train_capacity_mappo", "--config", config,
               "--scenario", SCENARIO, "--variant", "fixed", "--seed", str(seed), "--formal",
               "--output", str(output), *extra]  # fmt: skip
    return output.name, command, output


def eval_job(
    label: str, spec: str, output: Path, screen: str = PILOT_SET
) -> tuple[str, list[str], Path]:
    command = [PY, "-m", "dsa_period.experiments.evaluate_capacity", "--config", SCENARIO,
               "--split", "dev", "--screen", screen, "--grid", *map(str, PILOT_GRID),
               "--policies", spec, "--formal", "--workers", str(WORKERS),
               "--output", str(output)]  # fmt: skip
    return label, command, output


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


def stage1_extra(dry: bool) -> None:
    """M-S+S on the pilot slice after stage 1 (descriptive, design §10.4)."""
    plan_path, base = OUT / "pilot-plan.json", OUT / "pilot-analysis.json"
    if not dry and not base.exists():
        sys.exit("stage 1 has not finished (no pilot-analysis.json)")
    if dry:
        print(f"M-S+S x {len(SEEDS)} seeds on {PILOT_SET} at N = {PILOT_GRID}, then the analysis")
        return
    plan = json.loads(plan_path.read_text())
    chunks, layers = [], []
    for seed in SEEDS:
        ms = last_checkpoint(OUT / "pilot" / f"ms-seed{seed}").relative_to(ROOT)
        spec = RULES["M+S"].format(m=f"marl:{ms}")
        layers.append({str(n): spec for n in PILOT_GRID})
        label = f"M-SpS-seed{seed}"
        output = OUT / "pilot-eval" / label
        command = [PY, "-m", "dsa_period.experiments.evaluate_capacity", "--config", SCENARIO,
                   "--split", "dev", "--screen", PILOT_SET, "--grid", *map(str, PILOT_GRID),
                   "--policies", spec, "--formal", "--workers", str(WORKERS),
                   "--output", str(output)]  # fmt: skip
        chunks.append((label, command, output))
    run_parallel(chunks, CORES, WORKERS)
    extra_plan = {**plan, "groups": {**plan["groups"], "M-S+S": layers}}
    (OUT / "pilot-plan-extra.json").write_text(json.dumps(extra_plan, indent=2))
    evaluations = sorted(p.parent for p in (OUT / "pilot-eval").glob("*/summary.json"))
    analysis = OUT / "pilot-analysis-extra.json"
    if not analysis.exists():
        command = [PY, "scripts/analyze_exp012.py", "pilot",
                   "--plan", str(OUT / "pilot-plan-extra.json"),
                   "--evaluations", *map(str, evaluations), "--output", str(analysis)]  # fmt: skip
        subprocess.run(command, cwd=ROOT, check=True)
    log("stage 1 extra (M-S+S) done")


def stage1b(dry: bool) -> None:
    """SC-B at lambda_d = 0.1 and 0.3 on the pilot after gate A = adjust (design §10.5)."""
    base_plan = OUT / "pilot-plan-extra.json"
    if not dry and not base_plan.exists():
        sys.exit("stage 1 and its M-S+S extra have not finished (no pilot-plan-extra.json)")
    if dry:
        print(f"{len(STAGE1B) * len(SEEDS)} runs ({', '.join(STAGE1B.values())} x seeds {SEEDS}),"
              f" then {PILOT_SET} at N = {PILOT_GRID}, then the stage-1b analysis")  # fmt: skip
        return
    train = [
        train_job(f"configs/exp012-pilot1b-{key}.toml", seed, OUT / "pilot1b" / f"{key}-seed{seed}")
        for key in STAGE1B
        for seed in SEEDS
    ]
    run_parallel(train, len(train), 1)  # one core per training run
    log("stage 1b training done")
    plan = json.loads(base_plan.read_text())
    chunks = []
    for key, name in STAGE1B.items():
        layers = []
        for seed in SEEDS:
            checkpoint = last_checkpoint(OUT / "pilot1b" / f"{key}-seed{seed}").relative_to(ROOT)
            spec = f"marl:{checkpoint}"
            layers.append({str(n): spec for n in PILOT_GRID})
            label = f"{key}-seed{seed}"
            chunks.append(eval_job(label, spec, OUT / "pilot1b-eval" / label))
        plan["groups"][name] = layers
    (OUT / "pilot1b-plan.json").write_text(json.dumps(plan, indent=2))
    run_parallel(chunks, CORES, WORKERS)
    log("stage 1b evaluation done")
    evaluations = sorted(
        p.parent for d in ("pilot-eval", "pilot1b-eval") for p in (OUT / d).glob("*/summary.json")
    )
    analysis = OUT / "pilot1b-analysis.json"
    if not analysis.exists():
        command = [PY, "scripts/analyze_exp012.py", "pilot1b",
                   "--plan", str(OUT / "pilot1b-plan.json"),
                   "--evaluations", *map(str, evaluations), "--output", str(analysis)]  # fmt: skip
        subprocess.run(command, cwd=ROOT, check=True)
    log("stage 1b done")


def stage1c(dry: bool) -> None:
    """SP and SP-B on the frozen pilot M, evaluated on exp012-pilot2-256 (design §11.4)."""
    train = [
        train_job(f"configs/exp012-pilot1c-{key}.toml", seed, OUT / "pilot1c" / f"{key}-seed{seed}")
        for key in STAGE1C
        for seed in SEEDS_1C
    ]
    # SP0: the SP checkpoint before any update; it depends only on the base M (§11.10)
    train += [
        train_job(
            "configs/exp012-pilot1c-sp.toml",
            seed,
            OUT / "pilot1c" / f"sp0-seed{seed}",
            ("--initial-checkpoint",),
        )  # fmt: skip
        for seed in SEEDS
    ]
    if dry:
        print(f"{len(train) - len(SEEDS)} runs ({', '.join(STAGE1C.values())} x seeds "
              f"{SEEDS_1C}) and SP0 on seeds {SEEDS}, then {PILOT2_SET} at N = {PILOT_GRID}, "
              "then gate A2")  # fmt: skip
        return
    run_parallel(train, len(train), 1)  # one core per training run
    log("stage 1c training done")
    groups: dict[str, list[dict]] = {}
    chunks = []

    def add(name: str, label: str, spec: str) -> None:
        groups.setdefault(name, []).append({str(n): spec for n in PILOT_GRID})
        chunks.append(eval_job(label, spec, OUT / "pilot1c-eval" / label, PILOT2_SET))

    for key, name in STAGE1C.items():
        for seed in SEEDS_1C:
            checkpoint = last_checkpoint(OUT / "pilot1c" / f"{key}-seed{seed}").relative_to(ROOT)
            add(name, f"{key}-seed{seed}", f"marl:{checkpoint}")
    for seed in SEEDS:
        checkpoint = last_checkpoint(OUT / "pilot1c" / f"sp0-seed{seed}").relative_to(ROOT)
        add("SP0", f"sp0-seed{seed}", f"marl:{checkpoint}")
    for seed in SEEDS:
        m = f"marl:{last_checkpoint(OUT / 'pilot' / f'm-seed{seed}').relative_to(ROOT)}"
        add("M", f"M-seed{seed}", m)
        for name, rule in RULES.items():
            add(name, f"{name.replace('+', 'p')}-seed{seed}", rule.format(m=m))
        scb = last_checkpoint(OUT / "pilot" / f"scb-seed{seed}").relative_to(ROOT)
        add("SC-B", f"SC-B-seed{seed}", f"marl:{scb}")
    plan = {"grid": list(PILOT_GRID), "groups": groups, "base_seed": BASE_SEED}
    (OUT / "pilot1c-plan.json").write_text(json.dumps(plan, indent=2))
    run_parallel(chunks, CORES, WORKERS)
    log("stage 1c evaluation done")
    analysis = OUT / "pilot1c-analysis.json"
    if not analysis.exists():
        command = [PY, "scripts/analyze_exp012.py", "pilot1c",
                   "--plan", str(OUT / "pilot1c-plan.json"),
                   "--evaluations", *[str(c[2]) for c in chunks], "--output", str(analysis)]  # fmt: skip
        subprocess.run(command, cwd=ROOT, check=True)
    log("stage 1c done")


def main() -> None:
    args = sys.argv[1:]
    stages = {"stage1": stage1, "stage1-extra": stage1_extra, "stage1b": stage1b,
              "stage1c": stage1c}  # fmt: skip
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
