"""Why the configured-grant reference trails the learned policies (descriptive).

Three parts, none of them pre-registered and none running new episodes:

1. The fading bank: autocorrelation of |h|^2 within a window and the
   correlation across subchannels.
2. One UE alone: the mean SNR at which its outage reaches 0.01 when it sends on
   its first n CG occasions (placed by the CG rules), on the best subchannel of
   each of the first n slots, or on the subchannel that is best at the window
   start (no choice of slots in either).
3. exp010-eval512, paired by UE: violation rate, share of failing UEs and duty
   by the UE's mean SNR, for CG and the EXP-010 groups (seed layers pooled).

    python scripts/cg_mechanism.py --eval512 outputs/cg-baseline/eval512 \
        --exp010 outputs/exp010/eval/... \
        --output docs/results/cg-baseline_20261010/mechanism.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from dsa_period.capacity import allowance
from dsa_period.config import load_scenario
from dsa_period.physics import shannon_bits
from dsa_period.rules_v2 import allocate_grants, fading_bank

ROOT = Path(__file__).resolve().parents[1]
EXP010_METHODS = ROOT / "docs" / "results" / "exp010_20261008" / "methods-exp010.json"
LAGS = (1, 5, 9, 19, 39)
SNR_EDGES_DB = (3, 6, 10, 20)
SNR_LABELS = ("<3", "3-6", "6-10", "10-20", ">=20")
TARGET = 0.01


def _exp010():
    spec = importlib.util.spec_from_file_location(
        "analyze_exp010", ROOT / "scripts" / "analyze_exp010.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correlations(bank) -> dict:
    x = bank - bank.mean(axis=0)
    var = np.mean(x[:, 0, :] ** 2)
    return {
        "lag_slots": {lag: float(np.mean(x[:, 0, :] * x[:, lag, :]) / var) for lag in LAGS},
        "across_subchannels": float(np.corrcoef(bank[:, 0, 0], bank[:, 0, 1])[0, 1]),
    }


def outage(physical, bank, scheme: str, snr: float, period: int, bits: int, n: int) -> float:
    window = bank[:, :period, :]
    if scheme == "cg":
        allocation = allocate_grants(
            physical, np.array([snr]), np.array([bits]), np.array([period]), np.array([0]), bank
        )
        gains = np.stack([bank[:, r, k] for r, k in allocation.occasions[0][:n]], axis=1)
    elif scheme == "best_per_slot":
        gains = window.max(axis=2)[:, :n]
    elif scheme == "best_at_start":
        k = window[:, 0, :].argmax(axis=1)
        gains = np.take_along_axis(window, k[:, None, None], axis=2)[:, :n, 0]
    else:
        raise ValueError(scheme)
    return float(np.mean(shannon_bits(physical, snr * gains).sum(axis=1) < bits))


def snr_needed(physical, bank, scheme, period, bits, n, lo=-10.0, hi=40.0) -> float:
    """Mean SNR (dB) at which the outage reaches TARGET, by bisection to ~0.01 dB."""
    for _ in range(40):
        mid = (lo + hi) / 2
        q = outage(physical, bank, scheme, 10 ** (mid / 10), period, bits, n)
        lo, hi = (mid, hi) if q > TARGET else (lo, mid)
    return hi


def by_snr(episodes: dict, groups: dict, cg_spec: str, n: int) -> dict:
    cg = {e["instance"]: e for e in episodes[cg_spec][n]}
    out = {}
    for name, members in groups.items():
        cells = {
            label: {"ues": 0, "v": 0, "p": 0, "failing": 0, "duty": 0.0} for label in SNR_LABELS
        }
        duty_total = []
        for member in members:
            spec = member if isinstance(member, str) else member[str(n)]
            for e in episodes[spec][n]:
                ref = cg[e["instance"]]
                if e["class_index"] != ref["class_index"]:
                    raise ValueError("unpaired deployments")
                duty_total.append(sum(e["duty"]))
                for ue, snr_db in enumerate(ref["policy_diagnostics"]["mean_snr_db"]):
                    c = cells[SNR_LABELS[int(np.searchsorted(SNR_EDGES_DB, snr_db, "right"))]]
                    v, p = e["violations"][ue], e["packets"][ue]
                    c["ues"] += 1
                    c["v"] += v
                    c["p"] += p
                    c["failing"] += v > allowance(TARGET, p)
                    c["duty"] += e["duty"][ue]
        out[name] = {
            "layers": len(members),
            "transmissions_per_slot": float(np.mean(duty_total)),
            "by_mean_snr_db": {
                label: {
                    "ues_per_layer": c["ues"] // len(members),
                    "violation_rate": c["v"] / c["p"],
                    "failing_share": c["failing"] / c["ues"],
                    "duty": c["duty"] / c["ues"],
                }
                for label, c in cells.items()
                if c["ues"]
            },
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval512", type=Path, required=True, help="formal CG eval512 run")
    parser.add_argument("--exp010", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    exp010 = _exp010()
    episodes, _ = exp010.load([args.eval512, *args.exp010])  # formal exp010-eval512 only
    methods = json.loads(EXP010_METHODS.read_text())
    record = json.loads((args.eval512 / "summary.json").read_text())
    scenario = load_scenario(ROOT / record["config"], 4)
    physical, fading, classes = scenario.physical, scenario.fading, scenario.traffic_classes
    bank = fading_bank(
        physical.num_subchannels, fading.doppler_hz, fading.sinusoids, physical.slot_duration_s,
        max(c.period_slots for c in classes),
    )  # fmt: skip
    lone = []
    for c in classes:
        for n in (c.period_slots, c.period_slots // 2):
            row = {"period": c.period_slots, "packet_bits": c.packet_bits, "slots": n}
            for scheme in ("cg", "best_per_slot", "best_at_start"):
                row[scheme] = snr_needed(physical, bank, scheme, c.period_slots, c.packet_bits, n)
            lone.append(row)
            print(
                f"P={row['period']:2d} slots {n:2d}: CG {row['cg']:5.1f} dB, best per slot "
                f"{row['best_per_slot']:5.1f} dB, best at start {row['best_at_start']:5.1f} dB"
            )
    groups = {
        "CG": ["configured_grant"],
        "A_mix": methods["A_mix"],
        "M": methods["M"],
        "best_local": [methods["best_local"]],
        "N_EDF": [methods["edf"]],
    }
    paired = {n: by_snr(episodes, groups, "configured_grant", n) for n in exp010.GRID}
    for n, table in paired.items():
        print(f"N = {n}: failing share by mean SNR " + " ".join(SNR_LABELS))
        for name, row in table.items():
            cells = " ".join(f"{c['failing_share']:.3f}" for c in row["by_mean_snr_db"].values())
            print(f"  {name:<10} {cells}  transmissions/slot {row['transmissions_per_slot']:.2f}")
    result = {
        "bank": correlations(bank),
        "lone_ue_snr_db_for_outage_0.01": lone,
        "paired_by_mean_snr": paired,
    }
    args.output.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
