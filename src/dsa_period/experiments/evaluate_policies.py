"""Evaluate local policies under MARL protocol v1 on a frozen split.

Validation and test use the frozen instances, execution seeds and archived
central references; ``dev`` takes an explicit index range and has no
references. Test evaluation additionally needs ``--confirm-test`` and
``--formal``: the protocol reads the test set only for frozen methods.
"""

import argparse
import json
from pathlib import Path

from dsa_period import splits
from dsa_period.config import load_scenario
from dsa_period.evaluation import aggregate, evaluate_policy, load_references, select
from dsa_period.experiments.central_reference import git_provenance
from dsa_period.policies import make_policy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("dev", "validation", "test"), required=True)
    parser.add_argument("--first-instance", type=int, help="dev split only")
    parser.add_argument("--instances", type=int, help="dev split only")
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--confirm-test", action="store_true")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"output {args.output} already exists")
    provenance = git_provenance()
    if args.formal and provenance["dirty"]:
        parser.error("--formal requires a clean, committed working tree")
    if args.split == "test" and not (args.confirm_test and args.formal):
        parser.error("test evaluation requires --confirm-test and --formal")
    if args.split == "dev":
        if args.first_instance is None or args.instances is None:
            parser.error("dev evaluation needs --first-instance and --instances")
        stop = args.first_instance + args.instances
        chosen = [splits.require("dev", i) for i in range(args.first_instance, stop)]
    else:
        if args.first_instance is not None or args.instances is not None:
            parser.error("validation/test always use the whole frozen split")
        chosen = list(splits.indices(args.split))
    scenario = load_scenario(args.config)
    references = (
        None if args.split == "dev" else load_references(args.split, scenario.physical.num_ues)
    )
    results = {}
    for spec in args.policies:
        make_policy(spec, scenario.physical, scenario.menu)  # validate before running
    for spec in args.policies:
        episodes = evaluate_policy(
            scenario,
            lambda spec=spec: make_policy(spec, scenario.physical, scenario.menu),
            args.split,
            chosen,
            references,
        )
        results[spec] = {"summary": aggregate({"-": episodes}), "episodes": episodes}
    print(f"{scenario.experiment['id']} {args.split}: {len(chosen)} instances")
    print(
        f"{'policy':<30} {'F':>6} {'F 95% CI':>16} {'E/UB':>7} {'E/UB 95% CI':>16} {'worst':>7} {'mW':>7}"
    )
    for spec, result in results.items():
        m = result["summary"]["metrics"]
        ub = m.get("e_over_ub")
        print(
            f"{spec:<30} {m['feasible']['mean']:6.3f} "
            f"[{m['feasible']['ci95'][0]:.3f},{m['feasible']['ci95'][1]:.3f}] "
            + (
                f"{ub['mean']:7.3f} [{ub['ci95'][0]:.3f},{ub['ci95'][1]:.3f}] "
                if ub
                else f"{'-':>7} {'-':>16} "
            )
            + f"{m['worst_ue_ocr']['mean']:7.4f} {m['mw_per_ue']['mean']:7.2f}"
        )
    selected = select({k: v["summary"] for k, v in results.items()}) if references else None
    if selected is not None:
        print(f"protocol selection rule picks: {selected}")
    if args.output is not None:
        args.output.mkdir(parents=True)
        record = {
            "protocol_id": splits.PROTOCOL_ID,
            "formal": args.formal,
            "provenance": provenance,
            "config": args.config,
            "split": args.split,
            "instances": chosen,
            "policies": args.policies,
            "selected_by_protocol_rule": selected,
            "results": results,
        }
        (args.output / "summary.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
