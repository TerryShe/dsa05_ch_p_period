"""Single-UE ceiling of the configured-grant reference (descriptive, not pre-registered).

For every UE of a formal CG evaluation, the predicted outage it would have with
the whole grid to itself (P_i occasions placed by the same rules, the same
fading bank). A deployment holding a UE whose lone outage exceeds eps cannot
be made feasible by any sharing of the grid under these rules, so the share of
deployments without such a UE is an approximate ceiling of F_eps(N; CG)
(approximate: the realized violation count has the protocol's finite-sample
allowance).

With ``--exp010`` (exp010-eval512 runs only) the deployments that CG fails and
each EXP-010 seed layer passes are split into those holding such a UE and the
rest (lost to the sharing of the grid), together with the deployments CG
passes and the layer fails.

    python scripts/cg_single_ue_ceiling.py --run outputs/cg-baseline/eval512 \
        --exp010 outputs/exp010/eval/... \
        --output docs/results/cg-baseline_20261010/ceiling-eval512.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from dsa_period.config import load_scenario
from dsa_period.rules_v2 import allocate_grants, fading_bank

ROOT = Path(__file__).resolve().parents[1]
SPEC = "configured_grant"
EXP010_METHODS = ROOT / "docs" / "results" / "exp010_20261008" / "methods-exp010.json"
GROUPS = ("A_mix", "M", "best_local", "edf")


def _exp010():
    spec = importlib.util.spec_from_file_location(
        "analyze_exp010", ROOT / "scripts" / "analyze_exp010.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_against(cg: dict, capped: set, other: dict) -> dict:
    """Deployments one other method passes and CG fails, by cause, and the reverse."""
    gained = [i for i in cg if other[i] and not cg[i]]
    return {
        "other_only_lone_capped": sum(i in capped for i in gained),
        "other_only_sharing": sum(i not in capped for i in gained),
        "cg_only": sum(cg[i] and not other[i] for i in cg),
        "other_fails": sum(not v for v in other.values()),
        "other_fails_lone_capped": sum(not other[i] and i in capped for i in other),
    }


def lone_outage(physical, bank, snr_db: float, period: int, packet_bits: int) -> float:
    allocation = allocate_grants(
        physical,
        np.array([10 ** (snr_db / 10)]),
        np.array([packet_bits]),
        np.array([period]),
        np.array([0]),  # the bank is stationary: the phase only relabels the window
        bank,
    )
    return allocation.outage[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="formal CG evaluation directory")
    parser.add_argument(
        "--exp010", type=Path, nargs="+", help="EXP-010 evaluation directories (eval512 only)"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    record = json.loads((args.run / "summary.json").read_text())
    if not record["formal"]:
        parser.error(f"{args.run} is not a formal evaluation")
    if args.exp010:
        exp010 = _exp010()
        if record.get("screen") != exp010.SET:
            parser.error("--exp010 pairs with an exp010-eval512 run only")
        others, _ = exp010.load(args.exp010)
        methods = json.loads(EXP010_METHODS.read_text())
    episodes = json.loads((args.run / "episodes.json").read_text())[SPEC]
    scenario = load_scenario(ROOT / record["config"], 4)
    physical, fading, classes = scenario.physical, scenario.fading, scenario.traffic_classes
    bank = fading_bank(
        physical.num_subchannels, fading.doppler_hz, fading.sinusoids, physical.slot_duration_s,
        max(c.period_slots for c in classes),
    )  # fmt: skip
    result = {}
    for n, rows in episodes.items():
        capped = {"0.01": 0, "0.05": 0}
        failing = failing_capped = ues_capped = 0
        lone_capped, feasible = set(), {}
        for e in rows:
            snr_db = e["policy_diagnostics"]["mean_snr_db"]
            lone = [
                lone_outage(
                    physical, bank, snr_db[ue], classes[c].period_slots, classes[c].packet_bits
                )
                for ue, c in enumerate(e["class_index"])
            ]
            for key in capped:
                capped[key] += max(lone) > float(key)
            if max(lone) > 0.01:
                lone_capped.add(e["instance"])
            feasible[e["instance"]] = e["feasible"]["0.01"]
            ues_capped += sum(q > 0.01 for q in lone)
            if not e["feasible"]["0.01"]:
                failing += 1
                failing_capped += max(lone) > 0.01
        result[n] = {
            "deployments": len(rows),
            "F_0.01": 1 - failing / len(rows),
            "ceiling": {key: 1 - c / len(rows) for key, c in capped.items()},
            "failing_0.01": failing,
            "failing_with_lone_outage_above_0.01": failing_capped,
            "ues_with_lone_outage_above_0.01": ues_capped,
            "ues": sum(len(e["class_index"]) for e in rows),
        }
        if args.exp010:
            result[n]["against_exp010"] = {
                group: [
                    split_against(
                        feasible,
                        lone_capped,
                        {
                            o["instance"]: o["feasible"]["0.01"]
                            for o in others[m if isinstance(m, str) else m[n]][int(n)]
                        },
                    )
                    for m in (
                        methods[group] if isinstance(methods[group], list) else [methods[group]]
                    )
                ]
                for group in GROUPS
            }
        r = result[n]
        print(
            f"N={n:>2} F={r['F_0.01']:.3f} ceiling(0.01)={r['ceiling']['0.01']:.3f} "
            f"ceiling(0.05)={r['ceiling']['0.05']:.3f} failing {failing} of which lone-capped "
            f"{failing_capped}; UEs {ues_capped}/{r['ues']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"run": str(args.run), "by_n": result}, indent=1))


if __name__ == "__main__":
    main()
