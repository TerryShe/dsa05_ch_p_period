"""Write the freeze manifests of scenario v2.0 and MARL protocol v2 (2026-10-01).

* docs/research/scenario-v2-freeze.json: config hash, parameters, derived values
  and the decisions of draft 6;
* docs/research/marl-protocol-v2.json: splits, seeds, grid, calibration table
  and, for every N of the grid and of N_EDF, the digests of the validation and
  test instances (topology, traffic, fading) plus a probe of the training pool.

Run once at the freeze; tests/test_frozen_scenario_v2.py checks the manifests
against the code and the configuration from then on.
"""

import hashlib
import json
import math
import tomllib
from pathlib import Path

from dsa_period import splits_v2
from dsa_period.capacity import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, EPSILONS, allowance
from dsa_period.config import load_scenario
from dsa_period.rules_v2 import CALIBRATION_TABLE

ROOT = Path(__file__).parents[1]
CONFIG = Path("configs/scenario-v2.toml")
FROZEN_ON = "2026-10-01"
TRAIN_PROBES = [*range(1_000_000, 1_000_008), *range(1_999_992, 2_000_000)]


def train_probe_digest(scenario) -> str:
    """Positions and shadowing of the probe instances of the training pool."""
    probe = {}
    for index in TRAIN_PROBES:
        topology, traffic = scenario.instance(index)
        probe[str(index)] = [
            topology.path.to_record()["array_sha256"]["ue_positions_m"],
            topology.to_record()["shadowing_sha256"],
            traffic.phase_slots.tolist(),
        ]
    return hashlib.sha256(json.dumps(probe, sort_keys=True).encode()).hexdigest()


def main() -> None:
    raw = tomllib.loads((ROOT / CONFIG).read_text())
    base = load_scenario(ROOT / CONFIG, 2)
    grid = base.capacity["grid"]
    every_n = sorted({*grid, *base.capacity["reference_extra"]})
    packets = {
        c.name: (base.horizon_slots - base.warmup_slots) // c.period_slots
        for c in base.traffic_classes
    }
    scenario = {
        "schema_version": 1,
        "scenario_id": raw["experiment"]["scenario_id"],
        "scenario_version": raw["experiment"]["scenario_version"],
        "status": "frozen",
        "frozen_on": FROZEN_ON,
        "confirmed_by": "user",
        "specification": "docs/research/scenario-v2.md (draft 6)",
        "phase0_report": "docs/results/v2-phase0_20260930/report.md",
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
        },
        "decisions": [
            "capacity N*: grid point before the first N with F(N) < 0.95, F at eps = 0.01 (0.05 as comparison)",
            "energy is not an objective, reward, reference or metric",
            "grid {2,4,6,8,10,12,16,20,24,28,32}; N_EDF also at 36, 40 (draft 6, local rules fall below N = 4)",
            "deployments nested across N; fading and execution seeds per UE, without N (draft 6)",
            "main central reference N_EDF (slack order); LP upper bounds optional offline analysis (draft 6)",
            "packet sizes of v1.0 kept: phase-0 magnitude check met the section-11 criteria (N_EDF = 20)",
            "no arrival jitter; own current CSI per subchannel; shadowing 6 dB with coverage rule; Jakes SoS fD = 10 Hz",
        ],
    }
    (ROOT / "docs/research/scenario-v2-freeze.json").write_text(
        json.dumps(scenario, indent=2) + "\n"
    )

    digests = {}
    for n in every_n:
        sc = load_scenario(ROOT / CONFIG, n)
        digests[str(n)] = {
            split: splits_v2.instance_digest(sc, splits_v2.indices(split), split)
            for split in ("validation", "test")
        }
        digests[str(n)]["train_probe"] = train_probe_digest(sc)
    protocol = {
        "schema_version": 1,
        "protocol_id": splits_v2.PROTOCOL_ID,
        "status": "frozen",
        "frozen_on": FROZEN_ON,
        "scenario_id": scenario["scenario_id"],
        "scenario_version": scenario["scenario_version"],
        "document": "docs/research/marl-protocol-v2.md",
        "instance_seeds": "topology_seed = spatial.topology_seed + i; phase_seed = traffic.phase_seed + i; "
        "UEs drawn in order (nested across N)",
        "splits": {
            k: {"start": a, "stop": b, "count": b - a} for k, (a, b) in splits_v2.SPLITS.items()
        },
        "evaluation_repeats": splits_v2.EVALUATION_REPEATS,
        "split_codes": splits_v2.SPLIT_CODES,
        "execution_seeds": {
            "base": splits_v2.EXECUTION_SEED_BASE,
            "scheme": "SeedSequence([base, split_code, instance, repeat]).spawn(N)[j] -> PCG64 of UE j",
        },
        "fading_seeds": {
            "base": splits_v2.FADING_SEED_BASE,
            "scheme": "SeedSequence([base, split_code, instance, repeat]).spawn(N)[j] -> SoS parameters of UE j",
        },
        "epsilons": list(EPSILONS),
        "feasible_episode_share": base.objective["feasible_episode_share"],
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "grid": grid,
        "reference_extra": base.capacity["reference_extra"],
        "calibration_table": {family: list(specs) for family, specs in CALIBRATION_TABLE.items()},
        "train_probe_indices": TRAIN_PROBES,
        "digests": digests,
    }
    (ROOT / "docs/research/marl-protocol-v2.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"wrote manifests; digests for N = {every_n}")


if __name__ == "__main__":
    main()
