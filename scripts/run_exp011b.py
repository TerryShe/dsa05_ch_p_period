"""EXP-DSA05-011B driver: one stage per invocation, gated (design §3, §6).

    run_exp011b.py stage1   starvation scan of the 75 existing checkpoints on the
                            stress set, the starvation map and the counterfactual
    run_exp011b.py stage2   training: the 15 existing runs resume from 5M to 10M;
                            seeds 404 and 505 of every condition run 0 -> 5M
                            (configs/exp011b-*-base.toml), then resume to 10M
    run_exp011b.py stage3   the new checkpoints on exp009-screen256 and the stress
                            set, the guarded selection, the methods file and the plan
    run_exp011b.py stage4   evaluation on exp011b-eval512, diagnostics, analysis

A stage runs only if ``outputs/exp011b/gates.json`` records the user's approval
of the gate before it (0, A, B, C; gate A also records branch "G") and the
previous stage's outputs exist; stages never chain. Gate records are written by
hand after the user agrees. Everything is --formal on a clean committed tree;
finished chunks are skipped on a restart and a training run is never restarted.
``--dry-run`` prints the stage's work.
"""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs" / "exp011b"
PY = str(ROOT / ".venv" / "bin" / "python")
GRID, CORE = (6, 8, 10, 12, 16), (8, 10, 12)
KEYS = {"m": "M", "u": "U", "o": "O", "l": "L", "u0": "U0"}
GRIDS = {"m": GRID, "u": GRID, "o": GRID, "l": GRID, "u0": CORE}
OLD_SEEDS, NEW_SEEDS = (101, 202, 303), (404, 505)
FIVE_M = "step-005001472.pt"
CORES, MAX_WORKERS, MIN_WORKERS = 28, 14, 2
GATES = {"stage1": "0", "stage2": "A", "stage3": "B", "stage4": "C"}
M_SCREENS = (
    "outputs/exp009-step2/screen/s1-seed*-step-*",
    "outputs/exp009-final/screen/s1-seed303-step-*",
)


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp011 = _module("run_exp011", "scripts/run_exp011.py")
label_of, pairs_of = exp011.label_of, exp011.pairs_of


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def source_run(key: str, seed: int) -> Path:
    """The 0 -> 5M run of a (condition, seed)."""
    if seed in NEW_SEEDS:
        return OUT / f"{key}-seed{seed}-base"
    if key == "m":
        return (
            ROOT
            / ("outputs/exp009-final" if seed == 303 else "outputs/exp009-step2")
            / (f"s1-seed{seed}")
        )
    return ROOT / "outputs" / "exp011" / f"{key}-seed{seed}"


def checkpoints(key: str, seed: int, new_only: bool = False) -> list[Path]:
    """1M-5M of the source run (unless ``new_only`` for old seeds) and 6M-10M of the resume."""
    early = sorted((source_run(key, seed) / "checkpoints").glob("*.pt"))
    late = sorted((OUT / f"{key}-seed{seed}" / "checkpoints").glob("*.pt"))
    return late if new_only and seed in OLD_SEEDS else early + late


def spec_of(path: Path) -> str:
    return f"marl:{path.relative_to(ROOT)}"


def label(path: Path) -> str:
    run = path.parent.parent.name.replace("s1-seed", "m-seed")
    return f"{run}-{path.stem}"


def run_queue(queue: list[tuple[str, list[str], Path, int]], directory: Path) -> None:
    """Run (label, command, output, workers) chunks under the core budget, failing fast."""
    if len({item[2] for item in queue}) != len(queue):
        sys.exit("two chunks share an output directory")
    directory.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple[subprocess.Popen, int]] = {}
    queue = [item for item in queue if not (item[2] / "summary.json").exists()]
    while queue or running:
        for name, (process, _) in list(running.items()):
            if process.poll() is None:
                continue
            del running[name]
            if process.returncode != 0:
                for other, _ in running.values():
                    other.kill()
                sys.exit(f"{name} failed with status {process.returncode}")
        busy = sum(w for _, w in running.values())
        while queue and CORES - busy >= MIN_WORKERS:
            name, command, _, wanted = queue.pop(0)
            workers = min(wanted, CORES - busy)
            command = [*command, "--workers", str(workers)]
            with open(directory / f"{name}.log", "w") as handle:
                process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=handle)
            running[name] = (process, workers)
            busy += workers
        time.sleep(20)


def stress_chunk(path: Path) -> tuple[str, list[str], Path, int]:
    output = OUT / "stress" / label(path)
    command = [PY, "scripts/stress_exp011b.py", "--specs", spec_of(path), "--formal",
               "--output", str(output)]  # fmt: skip
    return label(path), command, output, MAX_WORKERS


def screen_chunk(key: str, path: Path) -> tuple[str, list[str], Path, int]:
    output = OUT / "screen" / label(path)
    command = [
        PY, "-m", "dsa_period.experiments.evaluate_capacity", "--config",
        "configs/scenario-v2.toml", "--split", "dev", "--screen", "exp009-screen256",
        "--grid", *map(str, GRIDS[key]), "--policies", spec_of(path), "--formal",
        "--output", str(output),
    ]  # fmt: skip
    return label(path), command, output, MAX_WORKERS


def eval_chunk(spec: str, n: int) -> tuple[str, list[str], Path, int]:
    name = label_of(spec, n)
    output = OUT / "eval" / name
    command = [
        PY, "-m", "dsa_period.experiments.evaluate_capacity", "--config",
        "configs/scenario-v2.toml", "--split", "dev", "--screen", "exp011b-eval512",
        "--grid", str(n), "--policies", spec, "--formal", "--output", str(output),
    ]  # fmt: skip
    return name, command, output, MAX_WORKERS


def screen_dirs_existing() -> list[str]:
    """exp009-screen256 outputs of the 1M-5M checkpoints of seeds 101-303 (EXP-009, EXP-011)."""
    found = [str(p.parent) for g in M_SCREENS for p in ROOT.glob(f"{g}/summary.json")]
    found += [str(p.parent) for p in (ROOT / "outputs/exp011/screen").glob("*/summary.json")]
    if len(found) != 120 + 60:
        sys.exit(f"found {len(found)} existing screening outputs, expected 180")
    return sorted(found)


def plan(methods: dict) -> dict:
    """analyze_exp011b groups (design §3.4) from the methods file."""

    def wrap(layers: list[dict], template: str, grid) -> list[dict]:
        return [{str(n): template.format(layer[str(n)]) for n in grid} for layer in layers]

    groups = {
        "U": wrap(methods["U"], "{}", GRID),
        "M": wrap(methods["M"], "{}", GRID),
        "M+Y": wrap(methods["M"], "yield[theta=-10]@{}", GRID),
        "O": wrap(methods["O"], "{}", GRID),
        "L": wrap(methods["L"], "{}", GRID),
        "U-hide-all": wrap(methods["U"], "summary[hide=all]@{}", GRID),
        "U-hide-nonurgent": wrap(methods["U"], "summary[hide=nonurgent]@{}", GRID),
        "best_local": methods["best_local"],
        "CG-opt": methods["cg_opt"],
        "N_EDF": methods["edf"],
        "U0": wrap(methods["U0"], "{}", CORE),
        "U@0": wrap(methods["U"], "summary[theta=0]@{}", CORE),
    }
    grids = {name: list(CORE if name in ("U0", "U@0") else GRID) for name in groups}
    return {"groups": groups, "grids": grids}


def check_gate(stage: str) -> None:
    gates_file = OUT / "gates.json"
    gates = json.loads(gates_file.read_text()) if gates_file.exists() else {}
    gate = GATES[stage]
    if gate not in gates:
        sys.exit(f"{stage} needs the user's approval of gate {gate} in {gates_file}")
    if stage == "stage2" and gates["A"].get("branch") != "G":
        sys.exit("gate A did not choose branch G; the design must be amended first (§6)")


def stage1(dry: bool) -> None:
    paths = [p for key in KEYS for s in OLD_SEEDS for p in checkpoints(key, s)]
    if len(paths) != 75:
        sys.exit(f"found {len(paths)} existing checkpoints, expected 75")
    if dry:
        print(f"stress runs of {len(paths)} checkpoints, then map and counterfactual")
        return
    run_queue([stress_chunk(p) for p in paths], OUT / "stress")
    log("stress runs done")
    stress = [str(OUT / "stress" / label(p)) for p in paths]
    out = OUT / "stage1"
    tool = [PY, "scripts/diagnose_exp011b.py"]
    if not (out / "map.json").exists():
        subprocess.run([*tool, "map", "--stress", *stress, "--screen", *screen_dirs_existing(),
                        "--rename", "S1=M", "--output", str(out / "map.json")],
                       cwd=ROOT, check=True)  # fmt: skip
    if not (out / "counterfactual.json").exists():
        subprocess.run([*tool, "counterfactual", "--stress", *stress, "--formal", "--workers",
                        str(CORES), "--output", str(out / "counterfactual.json")],
                       cwd=ROOT, check=True)  # fmt: skip
    log("stage 1 done")


def stage2(dry: bool) -> None:
    if not (OUT / "stage1" / "map.json").exists():
        sys.exit("stage 1 has not finished")
    runs = [(k, s) for k in KEYS for s in (*OLD_SEEDS, *NEW_SEEDS)]
    if dry:
        print(f"{len(runs)} runs to 10M; {len(KEYS) * len(NEW_SEEDS)} of them from scratch")
        return

    def command(config: str, seed: int, output: Path, resume: Path | None) -> list[str]:
        extra = [] if resume is None else ["--resume", str(resume)]
        return [PY, "-m", "dsa_period.experiments.train_capacity_mappo", "--config", config,
                "--variant", "fixed", "--seed", str(seed), "--formal", *extra,
                "--output", str(output)]  # fmt: skip

    def launch(cmd: list[str], output: Path) -> subprocess.Popen:
        handle = open(output.parent / f"{output.name}.log", "w")
        return subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=handle)

    OUT.mkdir(parents=True, exist_ok=True)
    active: dict[str, tuple[subprocess.Popen, tuple]] = {}
    for key, seed in runs:
        base, final = source_run(key, seed), OUT / f"{key}-seed{seed}"
        if final.exists():
            continue  # never restarted
        if seed in NEW_SEEDS and not base.exists():
            cmd = command(f"configs/exp011b-{key}-base.toml", seed, base, None)
            active[base.name] = (launch(cmd, base), (key, seed))
            log(f"training {base.name}")
            continue
        cmd = command(f"configs/exp011b-{key}.toml", seed, final, base / "checkpoints" / FIVE_M)
        active[final.name] = (launch(cmd, final), (key, seed))
        log(f"resuming {final.name}")
    while active:
        for name, (process, (key, seed)) in list(active.items()):
            if process.poll() is None:
                continue
            del active[name]
            if process.returncode != 0:
                for other, _ in active.values():
                    other.kill()
                sys.exit(f"{name} failed with status {process.returncode}")
            log(f"finished {name}")
            if name.endswith("-base"):
                final = OUT / f"{key}-seed{seed}"
                cmd = command(f"configs/exp011b-{key}.toml", seed, final,
                              source_run(key, seed) / "checkpoints" / FIVE_M)  # fmt: skip
                active[final.name] = (launch(cmd, final), (key, seed))
                log(f"resuming {final.name}")
        time.sleep(60)
    missing = [
        f"{k}-seed{s}" for k, s in runs if not (OUT / f"{k}-seed{s}" / "result.json").exists()
    ]
    if missing:
        sys.exit(f"runs without result.json: {missing}")
    log("stage 2 done")


def stage3(dry: bool) -> None:
    runs = [(k, s) for k in KEYS for s in (*OLD_SEEDS, *NEW_SEEDS)]
    missing = [
        f"{k}-seed{s}" for k, s in runs if not (OUT / f"{k}-seed{s}" / "result.json").exists()
    ]
    if missing:
        sys.exit(f"stage 2 has not finished: {missing}")
    new = [(k, p) for k, s in runs for p in checkpoints(k, s, new_only=True)]
    if dry:
        print(f"{len(new)} new checkpoints on exp009-screen256 and the stress set; selection")
        return
    run_queue([screen_chunk(k, p) for k, p in new] + [stress_chunk(p) for _, p in new],
              OUT / "chunks")  # fmt: skip
    log("screening and stress runs done")
    every = [p for k, s in runs for p in checkpoints(k, s)]
    stress = sorted(str(OUT / "stress" / label(p)) for p in every)
    screen = screen_dirs_existing() + sorted(str(OUT / "screen" / label(p)) for _, p in new)
    select = [PY, "-m", "dsa_period.experiments.select_exp011b"]
    if not (OUT / "selection.json").exists():
        subprocess.run([*select, "select", "--screen", *screen, "--stress", *stress,
                        "--rename", "S1=M", "--output", str(OUT / "selection.json")],
                       cwd=ROOT, check=True)  # fmt: skip
    if not (OUT / "methods-exp011b.json").exists():
        subprocess.run([*select, "methods", "--selection", str(OUT / "selection.json"),
                        "--output", str(OUT / "methods-exp011b.json")], cwd=ROOT, check=True)  # fmt: skip
    evaluation_plan = plan(json.loads((OUT / "methods-exp011b.json").read_text()))
    (OUT / "plan-exp011b.json").write_text(json.dumps(evaluation_plan, indent=2))
    log("stage 3 done")


def stage4(dry: bool) -> None:
    if not (OUT / "plan-exp011b.json").exists():
        sys.exit("stage 3 has not finished")
    evaluation_plan = json.loads((OUT / "plan-exp011b.json").read_text())
    pairs = pairs_of(evaluation_plan)
    labels = [label_of(spec, n) for spec, n in pairs]
    if len(set(labels)) != len(labels):
        sys.exit("two evaluation chunks share a label")
    if dry:
        print(f"{len(pairs)} evaluation chunks on exp011b-eval512, diagnostics, analysis")
        return
    run_queue([eval_chunk(spec, n) for spec, n in pairs], OUT / "eval")
    outputs = [str(OUT / "eval" / name) for name in labels]
    if any(not (Path(o) / "summary.json").exists() for o in outputs):
        sys.exit("evaluations missing")
    log("evaluation done")
    if not (OUT / "diagnostics.json").exists():
        items = [f"{layer[str(n)]}@{n}" for name in ("U", "M", "M+Y")
                 for layer in evaluation_plan["groups"][name] for n in CORE]  # fmt: skip
        subprocess.run([PY, "scripts/diagnose_exp011.py", "--set", "exp011b-eval512", "--specs",
                        *items, "--formal", "--workers", str(CORES), "--check", *outputs,
                        "--output", str(OUT / "diagnostics.json")], cwd=ROOT, check=True)  # fmt: skip
        log("diagnostics done")
    subprocess.run([PY, "scripts/analyze_exp011b.py", "--evaluations", *outputs, "--plan",
                    str(OUT / "plan-exp011b.json"), "--output", str(OUT / "analysis.json")],
                   cwd=ROOT, check=True)  # fmt: skip
    log("stage 4 done")


def main() -> None:
    args = sys.argv[1:]
    stages = {"stage1": stage1, "stage2": stage2, "stage3": stage3, "stage4": stage4}
    if not args or args[0] not in stages:
        sys.exit(f"usage: run_exp011b.py {{{','.join(stages)}}} [--dry-run]")
    dry = "--dry-run" in args[1:]
    if not dry:
        check_gate(args[0])
    stages[args[0]](dry)


if __name__ == "__main__":
    main()
