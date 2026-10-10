"""Improved single-UE ceiling of the static configured grant (CG-opt design §5, descriptive).

For every UE of a formal CG-opt evaluation, three predicted outages alone on
the whole grid, all on the CG bank B (the CG report §5 used the first):

* ``rule``: P_i occasions placed by the CG rule (``cg_single_ue_ceiling``);
* ``best_known``: for UEs with rule > 0.005, the CG-opt local search of the
  lone UE on bank A from the rule placement (with every residue held, it moves
  occasions between subchannels), evaluated on bank B; the smaller of the two;
* ``genie``: the best subchannel in every slot of the window, which no static
  allocation can beat (a loose bound).

A deployment holding a UE whose outage exceeds eps alone cannot be made
feasible by any static allocation of this structure, up to the protocol's
finite-sample allowance and, for ``best_known``, up to better placements than
the search finds. The shares of deployments without such a UE are the improved
approximate ceiling and the loose bound.

With ``--cg-run`` the share of the CG-to-ceiling gap that CG-opt closes; with
``--exp010`` (exp010-eval512 runs only) the deployments each EXP-010 layer
passes and CG-opt fails, split by the improved weak-UE criterion (as CG §5).

    python scripts/cg_opt_ceiling.py --run outputs/cg-opt/eval512 \
        --cg-run outputs/cg-baseline/eval512 --exp010 outputs/exp010/eval/... \
        --output docs/results/cg-opt_2026xxxx/ceiling-eval512.json
"""

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

from dsa_period import cg_opt
from dsa_period.config import load_scenario
from dsa_period.rules_v2 import CG_BANK_SAMPLES, CG_BANK_SEED, allocate_grants, fading_bank

ROOT = Path(__file__).resolve().parents[1]
SPEC = "configured_grant_opt"
CG_SPEC = "configured_grant"
SEARCH_ABOVE = 0.005
EPS = ("0.01", "0.05")


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ceiling = _script("cg_single_ue_ceiling")
mechanism = _script("cg_mechanism")


@cache
def _setting(config: str):
    scenario = load_scenario(ROOT / config, 4)
    physical, fading, classes = scenario.physical, scenario.fading, scenario.traffic_classes
    window = max(c.period_slots for c in classes)
    banks = {
        name: fading_bank(
            physical.num_subchannels, fading.doppler_hz, fading.sinusoids,
            physical.slot_duration_s, window, samples, seed,
        )
        for name, samples, seed in (
            ("A", cg_opt.CG_OPT_BANK_SAMPLES, cg_opt.CG_OPT_BANK_SEED),
            ("B", CG_BANK_SAMPLES, CG_BANK_SEED),
        )
    }  # fmt: skip
    return physical, classes, banks


def lone(config: str, snr_db: float, cls: int) -> dict:
    physical, classes, banks = _setting(config)
    period, bits = classes[cls].period_slots, classes[cls].packet_bits
    snr = 10 ** (snr_db / 10)
    rule = ceiling.lone_outage(physical, banks["B"], snr_db, period, bits)
    best = rule
    if rule > SEARCH_ABOVE:
        args = (np.array([snr]), np.array([bits]), np.array([period]), np.array([0]))
        start = allocate_grants(physical, *args, banks["A"]).occasions
        on_a = cg_opt.deployment(physical, *args, banks["A"])
        occasions, _, _ = cg_opt.local_search(on_a, start, max_evaluations=None)
        on_b = cg_opt.deployment(physical, *args, banks["B"])
        best = min(rule, cg_opt.predicted_outage(on_b, occasions)[0])
    genie = mechanism.outage(physical, banks["B"], "best_per_slot", snr, period, bits, period)
    return {"rule": rule, "best_known": best, "genie": genie}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="formal CG-opt evaluation")
    parser.add_argument("--cg-run", type=Path, help="formal CG evaluation of the same set")
    parser.add_argument(
        "--exp010", type=Path, nargs="+", help="EXP-010 evaluation directories (eval512 only)"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    record = json.loads((args.run / "summary.json").read_text())
    if not record["formal"]:
        parser.error(f"{args.run} is not a formal evaluation")
    episodes = json.loads((args.run / "episodes.json").read_text())[SPEC]
    cg = None
    if args.cg_run:
        cg_record = json.loads((args.cg_run / "summary.json").read_text())
        same = all(cg_record[k] == record[k] for k in ("split", "instances", "config"))
        if not (cg_record["formal"] and same and cg_record.get("screen") == record.get("screen")):
            parser.error("--cg-run must be the formal CG evaluation of the same set")
        cg = json.loads((args.cg_run / "episodes.json").read_text())[CG_SPEC]
    if args.exp010:
        exp010 = ceiling._exp010()
        if record.get("screen") != exp010.SET:
            parser.error("--exp010 pairs with an exp010-eval512 run only")
        others, _ = exp010.load(args.exp010)
        methods = json.loads(ceiling.EXP010_METHODS.read_text())
    ues = sorted(
        {
            (snr, cls)
            for rows in episodes.values()
            for e in rows
            for snr, cls in zip(e["policy_diagnostics"]["mean_snr_db"], e["class_index"])
        }
    )
    with ProcessPoolExecutor(args.workers) as pool:
        alone = dict(
            zip(ues, pool.map(lone, [record["config"]] * len(ues), *zip(*ues), chunksize=16))
        )
    result = {}
    for n, rows in episodes.items():
        worst = {
            e["instance"]: {
                kind: max(
                    alone[snr, cls][kind]
                    for snr, cls in zip(e["policy_diagnostics"]["mean_snr_db"], e["class_index"])
                )
                for kind in ("rule", "best_known", "genie")
            }
            for e in rows
        }
        feasible = {e["instance"]: e["feasible"]["0.01"] for e in rows}
        share = lambda kind, eps: float(np.mean([w[kind] <= float(eps) for w in worst.values()]))  # noqa: E731
        r = result[n] = {
            "deployments": len(rows),
            "F_0.01": float(np.mean(list(feasible.values()))),
            "ceiling": {
                kind: {eps: share(kind, eps) for eps in EPS} for kind in worst[rows[0]["instance"]]
            },
            "ues_searched": sum(
                alone[snr, cls]["rule"] > SEARCH_ABOVE
                for e in rows
                for snr, cls in zip(e["policy_diagnostics"]["mean_snr_db"], e["class_index"])
            ),
            "deployments_rule_capped_not_best_known": sum(
                w["rule"] > 0.01 >= w["best_known"] for w in worst.values()
            ),
        }
        capped = {i for i, w in worst.items() if w["best_known"] > 0.01}
        r["failing_with_best_known_above_0.01"] = sum(
            not v and i in capped for i, v in feasible.items()
        )
        if cg is not None:
            f_cg = float(np.mean([e["feasible"]["0.01"] for e in cg[n]]))
            room = r["ceiling"]["best_known"]["0.01"] - f_cg
            r["F_0.01_CG"] = f_cg
            r["gap_closed"] = (r["F_0.01"] - f_cg) / room if room > 0 else None
        if args.exp010:
            r["against_exp010"] = {
                group: [
                    ceiling.split_against(
                        feasible,
                        capped,
                        {
                            o["instance"]: o["feasible"]["0.01"]
                            for o in others[m if isinstance(m, str) else m[n]][int(n)]
                        },
                    )
                    for m in (
                        methods[group] if isinstance(methods[group], list) else [methods[group]]
                    )
                ]
                for group in ceiling.GROUPS
            }
        c = r["ceiling"]
        print(
            f"N={n:>2} F={r['F_0.01']:.3f} ceiling(0.01) rule {c['rule']['0.01']:.3f} best known "
            f"{c['best_known']['0.01']:.3f} genie {c['genie']['0.01']:.3f}"
            + (f" gap closed {r['gap_closed']}" if cg is not None else "")
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "run": str(args.run),
                "cg_run": str(args.cg_run) if args.cg_run else None,
                "by_n": result,
                "ues": [[snr, cls, v] for (snr, cls), v in alone.items()],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
