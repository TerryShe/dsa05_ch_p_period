"""Evaluate frozen MARL runs and local baselines on a protocol v1 split.

For every formal training run, the protocol-selected candidate and/or the final
checkpoint (each with its selected mu0) is replayed on the split with the frozen
instances and execution seeds. Methods are grouped by constraint profile and
checkpoint choice across training seeds and summarized with the protocol's
two-level bootstrap; each is also paired instance-by-instance with a reference
baseline. The test split requires --confirm-test and --formal (protocol §7).
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dsa_period import splits
from dsa_period.config import load_scenario
from dsa_period.evaluation import aggregate, evaluate_policy, load_references, paired_difference
from dsa_period.experiments.central_reference import git_provenance


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_jobs(run: Path, scenario_path: str, choice: str) -> list[dict]:
    """Checkpoint jobs of one formal training run: 'selected', 'final' or both."""
    manifest = json.loads((run / "manifest.json").read_text())
    if not manifest["formal"] or manifest["provenance"]["dirty"]:
        raise SystemExit(f"{run} is not a formal clean run")
    if Path(manifest["scenario"]).name != Path(scenario_path).name:
        raise SystemExit(f"{run} was trained on {manifest['scenario']}, not {scenario_path}")
    result = json.loads((run / "result.json").read_text())
    picks = {
        "selected": (result["selected_checkpoint"], result["selected_mu0"]),
        "final": (result["final_checkpoint"], result["selected_mu0"]),
    }
    jobs = []
    for name in ("selected", "final") if choice == "both" else (choice,):
        checkpoint, mu0 = picks[name]
        path = run / "checkpoints" / f"{checkpoint}.pt"
        jobs.append(
            {
                "method": f"{manifest['profile']}:{name}",
                "seed": str(manifest["seed"]),
                "checkpoint": str(path),
                "checkpoint_sha256": _sha256(path),
                "candidate": f"{checkpoint}@mu0={mu0:g}"
                if manifest["variant"] == "local"
                else checkpoint,
                "mu0": mu0,
                "run": str(run),
                "commit": manifest["provenance"]["commit"],
            }
        )
    return jobs


def _evaluate(job: dict) -> tuple[dict, list[dict]]:
    """Worker: replay one checkpoint or baseline on the split (single-threaded)."""
    import torch

    from dsa_period.marl.trainer import load_policy_factory
    from dsa_period.policies import make_policy

    torch.set_num_threads(1)
    scenario = load_scenario(job["scenario"])
    references = load_references(job["split"], scenario.physical.num_ues)
    if "baseline" in job:
        make = lambda: make_policy(job["baseline"], scenario.physical, scenario.menu)  # noqa: E731
    else:
        make = load_policy_factory(Path(job["checkpoint"]), scenario, job["mu0"])[0]
    episodes = evaluate_policy(scenario, make, job["split"], job["instances"], references)
    return job, episodes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--baselines", nargs="*", default=[])
    parser.add_argument("--reference", help="baseline spec that methods are paired against")
    parser.add_argument("--checkpoints", choices=("selected", "final", "both"), default="both")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--instances", type=int, help="non-formal: first K instances only")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--confirm-test", action="store_true")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"output {args.output} already exists")
    provenance = git_provenance()
    if args.formal and (provenance["dirty"] or args.instances is not None):
        parser.error("--formal requires a clean tree and the whole split")
    if args.split == "test" and not (args.confirm_test and args.formal):
        parser.error("test evaluation requires --confirm-test and --formal")
    if args.reference is not None and args.reference not in args.baselines:
        parser.error("--reference must be one of --baselines")
    instances = list(splits.indices(args.split))
    if args.instances is not None:
        instances = instances[: args.instances]
    common = {"scenario": args.scenario, "split": args.split, "instances": instances}
    jobs = [
        job | common for run in args.runs for job in run_jobs(run, args.scenario, args.checkpoints)
    ]
    jobs += [{"method": spec, "seed": "-", "baseline": spec} | common for spec in args.baselines]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        finished = list(pool.map(_evaluate, jobs))
    by_method: dict[str, dict[str, list[dict]]] = {}
    sources: dict[str, list[dict]] = {}
    for job, episodes in finished:
        seeds = by_method.setdefault(job["method"], {})
        if job["seed"] in seeds:
            raise SystemExit(f"duplicate training seed {job['seed']} for {job['method']}")
        seeds[job["seed"]] = episodes
        sources.setdefault(job["method"], []).append(
            {
                k: job[k]
                for k in ("seed", "run", "commit", "checkpoint", "checkpoint_sha256", "candidate")
                if k in job
            }
        )
    results = {}
    for method, episodes_by_seed in by_method.items():
        results[method] = {
            "summary": aggregate(episodes_by_seed),
            "sources": sources[method],
            "episodes": episodes_by_seed,
        }
        if args.reference is not None and method != args.reference:
            reference = {"-": by_method[args.reference]["-"]}
            # Pair every training seed against the same baseline episodes.
            baseline = {seed: reference["-"] for seed in episodes_by_seed}
            results[method]["paired_vs_reference"] = {
                metric: paired_difference(episodes_by_seed, baseline, metric)
                for metric in ("e_over_ub", "feasible")
            }
    scenario = load_scenario(args.scenario)
    print(f"{scenario.experiment['id']} {args.split}: {len(instances)} instances")
    print(
        f"{'method':<28} {'seeds':>5} {'F':>6} {'F 95% CI':>15} {'E/UB':>7} {'E/UB 95% CI':>15} {'vs ref E/UB':>22}"
    )
    for method, result in results.items():
        m = result["summary"]["metrics"]
        pair = result.get("paired_vs_reference", {}).get("e_over_ub")
        pair_text = (
            f"{pair['mean']:+.3f} [{pair['ci95'][0]:+.3f},{pair['ci95'][1]:+.3f}]" if pair else "-"
        )
        print(
            f"{method:<28} {len(result['summary']['training_seeds']):>5} "
            f"{m['feasible']['mean']:6.3f} [{m['feasible']['ci95'][0]:.3f},{m['feasible']['ci95'][1]:.3f}] "
            f"{m['e_over_ub']['mean']:7.3f} [{m['e_over_ub']['ci95'][0]:.3f},{m['e_over_ub']['ci95'][1]:.3f}] "
            f"{pair_text:>22}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "protocol_id": splits.PROTOCOL_ID,
            "formal": args.formal,
            "provenance": provenance,
            "scenario": args.scenario,
            "scenario_sha256": _sha256(Path(args.scenario)),
            "split": args.split,
            "instances": instances,
            "checkpoints": args.checkpoints,
            "reference": args.reference,
            "results": results,
        }
        args.output.write_text(json.dumps(record))


if __name__ == "__main__":
    main()
