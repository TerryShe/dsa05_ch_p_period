"""EXP-DSA05-011 driver (design §3-§5): training, selection, evaluation, diagnostics, analysis.

* Trains U, O, L and U0 (configs/exp011-{u,o,l,u0}.toml) with seeds 101, 202,
  303: 12 formal mixed-N runs of 5M slots, one core each.
* Screens every checkpoint on exp009-screen256 (U, O, L at N = 6, 8, 10, 12, 16;
  U0 at N = 8, 10, 12) and selects per (condition, seed, N) by P2 with
  ``select_exp011 select``; M (EXP-009's S1) is re-selected from its existing
  formal screening outputs (``--rename S1=M``).
* Writes the methods file and the evaluation plan (``analyze_exp011`` groups),
  evaluates every (spec, N) once on exp011-eval512 (512 deployments), runs the
  mechanism diagnostics for U, M and M+Y at N = 8, 10, 12 (checked against the
  evaluation) and the pre-registered analysis.

Needs a clean committed tree (everything is --formal). Finished chunks are
skipped on a restart; a training run is never restarted. ``--dry-run`` prints
the plan.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs" / "exp011"
GRID = (6, 8, 10, 12, 16)
CORE = (8, 10, 12)
CONDITIONS = {"u": ("U", GRID), "o": ("O", GRID), "l": ("L", GRID), "u0": ("U0", CORE)}
SEEDS = (101, 202, 303)
THETAS = ("all", "-15", "-5", "0")  # besides the primary -10 dB (design §3.5)
HIDES = ("all", "nonurgent", "urgent")
M_SCREENS = (
    "outputs/exp009-step2/screen/s1-seed*-step-*",
    "outputs/exp009-final/screen/s1-seed303-step-*",
)
CORES, MAX_WORKERS, MIN_WORKERS = 28, 14, 2
PY = str(ROOT / ".venv" / "bin" / "python")


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def evaluate_command(screen: str, specs: list[str], grid, workers: int, output: Path) -> list[str]:
    return [
        PY, "-m", "dsa_period.experiments.evaluate_capacity",
        "--config", "configs/scenario-v2.toml", "--split", "dev", "--screen", screen,
        "--grid", *map(str, grid), "--policies", *specs, "--workers", str(workers),
        "--formal", "--output", str(output),
    ]  # fmt: skip


def run_queue(queue: list[tuple], directory: Path) -> None:
    """Run (label, screen, specs, grid) chunks under the core budget, failing fast."""
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
        busy = sum(w for _, w in running.values())
        while queue and CORES - busy >= MIN_WORKERS:
            label, screen, specs, grid = queue.pop(0)
            output = directory / label
            if (output / "summary.json").exists():
                continue
            workers = min(MAX_WORKERS, CORES - busy)
            with open(directory / f"{label}.log", "w") as handle:
                running[label] = (
                    subprocess.Popen(
                        evaluate_command(screen, specs, grid, workers, output),
                        cwd=ROOT,
                        stdout=handle,
                        stderr=handle,
                    ),
                    workers,
                )
            busy += workers
        time.sleep(20)


def label_of(spec: str, n: int) -> str:
    """A directory name for one (spec, N) evaluation chunk."""
    text = spec.replace("marl:outputs/", "").replace("checkpoints/", "")
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", text).strip("_") + f"-n{n}"


def plan(methods: dict) -> dict:
    """analyze_exp011 groups (design §3.1, §3.3, §3.5) from the methods file."""

    def wrap(layers: list[dict], template: str, grid) -> list[dict]:
        return [{str(n): template.format(layer[str(n)]) for n in grid} for layer in layers]

    groups: dict = {
        "U": wrap(methods["U"], "{}", GRID),
        "M": wrap(methods["M"], "{}", GRID),
        "M+Y": wrap(methods["M"], "yield[theta=-10]@{}", GRID),
        "O": wrap(methods["O"], "{}", GRID),
        "L": wrap(methods["L"], "{}", GRID),
        **{
            f"U-hide-{hide}": wrap(methods["U"], f"summary[hide={hide}]@{{}}", GRID)
            for hide in HIDES
        },
        "best_local": methods["best_local"],
        "best_local+Y": f"yield[theta=-10]@{methods['best_local']}",
        "CG-opt": methods["cg_opt"],
        "N_EDF": methods["edf"],
        "M_exp009": methods["M_exp009"],
        "U0": wrap(methods["U0"], "{}", CORE),
    }
    for theta in THETAS:
        groups[f"U@{theta}"] = wrap(methods["U"], f"summary[theta={theta}]@{{}}", CORE)
        groups[f"M+Y@{theta}"] = wrap(methods["M"], f"yield[theta={theta}]@{{}}", CORE)
    grids = {name: list(CORE if name == "U0" or "@" in name else GRID) for name in groups}
    return {"groups": groups, "grids": grids}


def pairs_of(evaluation_plan: dict) -> list[tuple[str, int]]:
    pairs: dict[tuple[str, int], None] = {}
    for name, group in evaluation_plan["groups"].items():
        for n in evaluation_plan["grids"][name]:
            for member in group if isinstance(group, list) else [group]:
                pairs[(member if isinstance(member, str) else member[str(n)], n)] = None
    return list(pairs)


def main() -> None:
    runs = [(key, seed) for key in CONDITIONS for seed in SEEDS]
    if "--dry-run" in sys.argv[1:]:
        print(f"train {len(runs)} runs; screen {len(runs) * 5} checkpoints on exp009-screen256")
        print("then select by P2, evaluate on exp011-eval512, diagnose and analyze")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    training: dict[str, subprocess.Popen] = {}
    for key, seed in runs:
        name = f"{key}-seed{seed}"
        if (OUT / name).exists():
            continue  # finished earlier; never restarted (a broken run is handled by hand)
        command = [
            PY, "-m", "dsa_period.experiments.train_capacity_mappo",
            "--config", f"configs/exp011-{key}.toml", "--variant", "fixed",
            "--seed", str(seed), "--formal", "--output", str(OUT / name),
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
    for key, seed in runs:
        if not (OUT / f"{key}-seed{seed}" / "result.json").exists():
            sys.exit(f"{key}-seed{seed} has no result.json")
    screen_queue = [
        (
            f"{key}-seed{seed}-{ck.stem}",
            "exp009-screen256",
            [f"marl:{ck.relative_to(ROOT)}"],
            CONDITIONS[key][1],
        )
        for key, seed in runs
        for ck in sorted((OUT / f"{key}-seed{seed}" / "checkpoints").glob("*.pt"))
    ]
    run_queue(screen_queue, OUT / "screen")
    log("screening done")
    chunks = sorted(str(p.parent) for p in (OUT / "screen").glob("*/summary.json"))
    m_chunks = sorted(str(p.parent) for g in M_SCREENS for p in ROOT.glob(f"{g}/summary.json"))
    if len(m_chunks) != 120:
        sys.exit(f"found {len(m_chunks)} screening outputs of M, expected 120")
    select = [PY, "-m", "dsa_period.experiments.select_exp011"]
    if not (OUT / "selection.json").exists():
        subprocess.run(
            [*select, "select", "--screen", *chunks, *m_chunks, "--rename", "S1=M",
             "--output", str(OUT / "selection.json")], cwd=ROOT, check=True,
        )  # fmt: skip
    if not (OUT / "methods-exp011.json").exists():
        subprocess.run(
            [*select, "methods", "--selection", str(OUT / "selection.json"),
             "--output", str(OUT / "methods-exp011.json")], cwd=ROOT, check=True,
        )  # fmt: skip
    evaluation_plan = plan(json.loads((OUT / "methods-exp011.json").read_text()))
    (OUT / "plan-exp011.json").write_text(json.dumps(evaluation_plan, indent=2))
    pairs = pairs_of(evaluation_plan)
    labels = [label_of(spec, n) for spec, n in pairs]
    if len(set(labels)) != len(labels):
        sys.exit("two evaluation chunks share a label")
    run_queue(
        [(label, "exp011-eval512", [spec], [n]) for label, (spec, n) in zip(labels, pairs)],
        OUT / "eval",
    )
    outputs = [str(OUT / "eval" / label) for label in labels]
    missing = [o for o in outputs if not (Path(o) / "summary.json").exists()]
    if missing:
        sys.exit(f"{len(missing)} evaluations missing")
    log("evaluation done")
    if not (OUT / "diagnostics.json").exists():
        items = [
            f"{layer[str(n)]}@{n}"
            for name in ("U", "M", "M+Y")
            for layer in evaluation_plan["groups"][name]
            for n in CORE
        ]
        subprocess.run(
            [PY, "scripts/diagnose_exp011.py", "--specs", *items, "--formal", "--workers",
             str(CORES), "--check", *outputs, "--output", str(OUT / "diagnostics.json")],
            cwd=ROOT, check=True,
        )  # fmt: skip
        log("diagnostics done")
    subprocess.run(
        [PY, "scripts/analyze_exp011.py", "--evaluations", *outputs, "--plan",
         str(OUT / "plan-exp011.json"), "--output", str(OUT / "analysis.json")],
        cwd=ROOT, check=True,
    )  # fmt: skip
    log("exp011 done")


if __name__ == "__main__":
    main()
