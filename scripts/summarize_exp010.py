"""Archive EXP-DSA05-010 (descriptive): chunk summaries, training logs, failure overlap.

Reads outputs/exp010 (the driver's outputs) and writes into --output:

* screen.json, evaluations.json: summary.json of every screening and
  evaluation chunk, keyed by chunk label;
* training.json: training.jsonl, result.json and provenance of each DAgger run;
* overlap.json: on exp010-eval512, per N, the deployments each seed layer fails
  at eps = 0.01 and how far the failure sets of IL, A_mix and M coincide.
"""

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs" / "exp010"
GRID = (6, 8, 10)


def summaries(directory: Path) -> dict:
    return {
        p.parent.name: json.loads(p.read_text()) for p in sorted(directory.glob("*/summary.json"))
    }


def overlap(episodes: dict, methods: dict) -> dict:
    result = {}
    for n in GRID:
        sets = {
            f"{group}|{s}": layer[str(n)]
            for group in ("IL", "A_mix", "M")
            for s, layer in enumerate(methods[group])
        }
        sets.update(best_local=methods["best_local"], edf=methods["edf"])
        failed = {
            name: sorted(e["instance"] for e in episodes[spec][n] if not e["feasible"]["0.01"])
            for name, spec in sets.items()
        }
        physics = sorted(
            e["instance"] for e in episodes[methods["edf"]][n] if e["infeasible_for_all"]["0.01"]
        )
        il = [set(failed[k]) for k in failed if k.startswith("IL|")]
        rl = [set(failed[k]) for k in failed if k.startswith(("A_mix|", "M|"))]
        amix = [set(failed[k]) for k in failed if k.startswith("A_mix|")]
        il_all, il_any = set.intersection(*il), set.union(*il)
        amix_all = set.intersection(*amix)
        result[str(n)] = {
            "failed": failed,
            "infeasible_for_all": physics,
            "IL_all_seeds": len(il_all),
            "IL_all_seeds_failed_by_some_RL_layer": len(il_all & set.union(*rl)),
            "A_mix_all_seeds": len(amix_all),
            "A_mix_all_seeds_failed_by_some_IL_seed": len(amix_all & il_any),
            "pairs": {
                f"IL|{s} vs A_mix|{s}": {
                    "IL_only": len(set(failed[f"IL|{s}"]) - set(failed[f"A_mix|{s}"])),
                    "A_mix_only": len(set(failed[f"A_mix|{s}"]) - set(failed[f"IL|{s}"])),
                    "both": len(set(failed[f"IL|{s}"]) & set(failed[f"A_mix|{s}"])),
                }
                for s in range(len(il))
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("analyze", ROOT / "scripts" / "analyze_exp010.py")
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("screen", "eval"):
        target = "evaluations.json" if name == "eval" else "screen.json"
        (args.output / target).write_text(json.dumps(summaries(OUT / name), indent=1))
    training = {}
    for run in sorted(OUT.glob("il-n*-seed*/")):
        manifest = json.loads((run / "manifest.json").read_text())
        training[run.name] = {
            "provenance": manifest["provenance"],
            "config_sha256": manifest["config_sha256"],
            "result": json.loads((run / "result.json").read_text()),
            "iterations": [
                json.loads(x) for x in (run / "training.jsonl").read_text().splitlines()
            ],
        }
    (args.output / "training.json").write_text(json.dumps(training, indent=1))
    paths = sorted(p.parent for p in (OUT / "eval").glob("*/summary.json"))
    episodes, _ = analyze.load(paths)
    methods = json.loads((OUT / "methods-exp010.json").read_text())
    (args.output / "overlap.json").write_text(json.dumps(overlap(episodes, methods), indent=1))


if __name__ == "__main__":
    main()
