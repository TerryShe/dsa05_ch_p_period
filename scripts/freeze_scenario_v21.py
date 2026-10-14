"""Write the freeze manifest of scenario v2.1 (2026-10-14).

docs/research/scenario-v2.1-freeze.json: config hash, parameters, derived values,
the decisions of docs/research/scenario-v2.1.md §8 and, because v2.1 changed the
topology generator, digests of its instances: dev 0-63 (the phase-0 instances)
and the training-pool probe of protocol v2, at every N of the grid and of N_EDF.
The protocol revision registers the validation, test and confirmation sets.

Run once at the freeze; tests/test_frozen_scenario_v21.py checks the manifest
against the code and the configuration from then on.
"""

import hashlib
import importlib.util
import json
import math
import tomllib
from pathlib import Path

from dsa_period import splits_v2
from dsa_period.capacity import EPSILONS, allowance
from dsa_period.config import load_scenario

ROOT = Path(__file__).parents[1]
CONFIG = Path("configs/scenario-v2.1.toml")
PREVIOUS = json.loads((ROOT / "docs/research/scenario-v2-freeze.json").read_text())
FROZEN_ON = "2026-10-14"
DEV_INSTANCES = range(0, 64)


def _v20_freeze():
    spec = importlib.util.spec_from_file_location(
        "freeze_v2", ROOT / "scripts/freeze_scenario_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digests(num_ues: int) -> dict:
    """Instance digests of scenario v2.1 at N: dev 0-63 and the training probe."""
    scenario = load_scenario(ROOT / CONFIG, num_ues)
    return {
        "dev_0_63": splits_v2.instance_digest(scenario, DEV_INSTANCES, "dev"),
        "train_probe": _v20_freeze().train_probe_digest(scenario),
    }


def main() -> None:
    raw = tomllib.loads((ROOT / CONFIG).read_text())
    base = load_scenario(ROOT / CONFIG, 2)
    every_n = sorted({*base.capacity["grid"], *base.capacity["reference_extra"]})
    packets = {
        c.name: (base.horizon_slots - base.warmup_slots) // c.period_slots
        for c in base.traffic_classes
    }
    manifest = {
        "schema_version": 1,
        "scenario_id": raw["experiment"]["scenario_id"],
        "scenario_version": raw["experiment"]["scenario_version"],
        "status": "frozen",
        "frozen_on": FROZEN_ON,
        "confirmed_by": "user",
        "specification": "docs/research/scenario-v2.1.md",
        "phase0_report": "docs/results/v21-phase0_20261014/report.md",
        "previous": {
            "scenario_version": PREVIOUS["scenario_version"],
            "config_path": PREVIOUS["config_path"],
            "config_sha256": PREVIOUS["config_sha256"],
        },
        "config_path": str(CONFIG),
        "config_sha256": hashlib.sha256((ROOT / CONFIG).read_bytes()).hexdigest(),
        "parameters": raw,
        "derived": {
            "noise_power_dbm": round(10 * math.log10(base.physical.noise_power_w) + 30, 6),
            "num_joint_actions": 1 + base.physical.num_subchannels * (base.menu.num_levels - 1),
            "per_ue_offered_bits_per_slot": 614.4,
            "evaluation_slots": base.horizon_slots - base.warmup_slots,
            "packets_per_ue": packets,
            "allowance": {
                f"{eps:g}": {name: allowance(eps, m) for name, m in packets.items()}
                for eps in EPSILONS
            },
            "min_distance_m": base.min_distance_m,
        },
        "decisions": [
            "only change from v2.0: minimum 2D UE-AP distance 10 m",
            "a UE closer than 10 m is redrawn in the same topology stream, like the coverage rule; "
            "seed bases unchanged, so deployments with every UE accepted at >= 10 m equal v2.0",
            "justification: validity range of the 3GPP TR 38.901 UMa/UMi path loss (d2D >= 10 m), "
            "not InF",
            "main scenario keeps periodic traffic; Bernoulli arrivals or jitter only as a later, "
            "separately registered robustness scenario",
            "CG-opt calibration (optimizer search budget) not redone; best_local recalibrated "
            "per N on the v2.1 validation split later",
        ],
        "dev_instances": [DEV_INSTANCES.start, DEV_INSTANCES.stop],
        "train_probe_indices": _v20_freeze().TRAIN_PROBES,
        "digests": {str(n): digests(n) for n in every_n},
    }
    (ROOT / "docs/research/scenario-v2.1-freeze.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"wrote the v2.1 manifest; digests for N = {every_n}")


if __name__ == "__main__":
    main()
