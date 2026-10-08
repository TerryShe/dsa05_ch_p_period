"""Write the set manifests of EXP-DSA05-008, -009 and -010 (protocol v2).

EXP-008 (docs/research/exp008-sets.json): the screening set exp008-screen256
(dev 2000-2255, N = 6 and 8) and the confirmation set v2-confirm1 (reserve
30000-30499). EXP-009 (docs/research/exp009-sets.json): exp009-screen256 (dev
3000-3255, N in {2, ..., 20}) and v2-confirm2 (reserve 30500-30999). EXP-010
(docs/research/exp010-sets.json): the evaluation slice exp010-eval512 (dev
4000-4511, N = 6, 8, 10). Each set
carries its instance and fading digests (``splits_v2.instance_digest``:
topology, shadowing, traffic and the fading of slots 0-7) at every N it is read
at; confirmation sets at every N of the protocol-v2 grid and of N_EDF. No policy
runs. tests/test_exp008.py and tests/test_exp009.py check the manifests against
the code; rerunning this script must reproduce them byte for byte.
"""

import hashlib
import json
from pathlib import Path

from dsa_period import splits_v2
from dsa_period.config import load_scenario

ROOT = Path(__file__).parents[1]
CONFIG = "configs/scenario-v2.toml"
REGISTRY = {
    "EXP-DSA05-008": {
        "output": "docs/research/exp008-sets.json",
        "design": "docs/experiment-designs/008-training-signal_20261002.md",
        "document": "docs/research/exp008-registration.md",
        "registered_on": "2026-10-03",
        "screening": ["exp008-screen256"],
        "confirmation": ["v2-confirm1"],
        "score": "S_near = (F_0.01(6) + F_0.01(8)) / 2",
        "selection": "per run: max (S_near, checkpoint slots); design §4.1",
        "dev_used": [[0, 64], [1000, 1400]],
    },
    "EXP-DSA05-009": {
        "output": "docs/research/exp009-sets.json",
        "design": "docs/experiment-designs/009-per-n-training-hard-deployments_20261004.md",
        "document": "docs/research/exp009-registration.md",
        "registered_on": "2026-10-04",
        "screening": ["exp009-screen256"],
        "confirmation": ["v2-confirm2"],
        "score": "per N: F_0.01(N) of that N's model; S_near = (F_0.01(6) + F_0.01(8)) / 2",
        "selection": "per (condition, seed, N): max (F_0.01(N), F_0.05(N), slots); design §3",
        "dev_used": [[0, 64], [1000, 1400], [2000, 2256]],
    },
    "EXP-DSA05-010": {
        "output": "docs/research/exp010-sets.json",
        "design": "docs/experiment-designs/010-edf-imitation-diagnostic_20261008.md",
        "document": "docs/research/exp010-registration.md",
        "registered_on": "2026-10-08",
        "screening": ["exp010-eval512"],
        "confirmation": [],
        "score": "F_0.01(N) of each method; D8 = F_0.01(8; IL) - F_0.01(8; A_mix)",
        "selection": "none: evaluation only (checkpoints are selected on exp009-screen256)",
        "dev_used": [[0, 64], [1000, 1400], [2000, 2256], [3000, 3256]],
    },
}


def build(experiment: str = "EXP-DSA05-008") -> dict:
    spec = REGISTRY[experiment]
    base = load_scenario(ROOT / CONFIG, 2)
    every_n = sorted({*base.capacity["grid"], *base.capacity["reference_extra"]})
    screening = {}
    for name in spec["screening"]:
        entry = splits_v2.SCREENING_SETS[name]
        start, stop = entry["range"]
        screening[name] = {
            "split": "dev",
            "start": start,
            "stop": stop,
            "count": stop - start,
            "grid": list(entry["grid"]),
            "repeats": splits_v2.evaluation_repeats("dev"),
            "seed_code": splits_v2.SPLIT_CODES["dev"],
            "score": spec["score"],
            "selection": spec["selection"],
            "entry_point": "evaluate_capacity --split dev --screen " + name + " --formal",
            "digests": {
                str(n): splits_v2.instance_digest(
                    load_scenario(ROOT / CONFIG, n), range(start, stop), "dev"
                )
                for n in entry["grid"]
            },
        }
    confirmation = {}
    for name in spec["confirmation"]:
        start, stop = splits_v2.CONFIRMATION_SETS[name]
        confirmation[name] = {
            "pool": "reserve",
            "start": start,
            "stop": stop,
            "count": stop - start,
            "repeats": splits_v2.evaluation_repeats(name),
            "seed_code": splits_v2.SPLIT_CODES["reserve"],
            "grid": base.capacity["grid"],
            "reference_extra": base.capacity["reference_extra"],
            "read_once": True,
            "entry_point": "evaluate_capacity --split "
            + name
            + (" --confirm-c1" if name == "v2-confirm1" else " --confirm-c2")
            + " --formal",
            "digests": {
                str(n): splits_v2.instance_digest(
                    load_scenario(ROOT / CONFIG, n), range(start, stop), name
                )
                for n in every_n
            },
        }
    return {
        "schema_version": 1,
        "experiment": experiment,
        "design": spec["design"],
        "document": spec["document"],
        "protocol_id": splits_v2.PROTOCOL_ID,
        "registered_on": spec["registered_on"],
        "scenario_config": CONFIG,
        "scenario_config_sha256": hashlib.sha256((ROOT / CONFIG).read_bytes()).hexdigest(),
        "horizon_slots": base.horizon_slots,
        "warmup_slots": base.warmup_slots,
        "execution_seeds": {
            "base": splits_v2.EXECUTION_SEED_BASE,
            "scheme": "SeedSequence([base, seed_code, instance, repeat]).spawn(N)[j]",
        },
        "fading_seeds": {
            "base": splits_v2.FADING_SEED_BASE,
            "scheme": "SeedSequence([base, seed_code, instance, repeat]).spawn(N)[j]",
        },
        "screening": screening,
        "confirmation": confirmation,
        "usage_checked_on_registration": {
            "v2_dev_ranges_used": spec["dev_used"],
            "v2_reserve_ranges_read": [],
        },
    }


def main() -> None:
    for experiment, spec in REGISTRY.items():
        output = ROOT / spec["output"]
        output.write_text(json.dumps(build(experiment), indent=2) + "\n")
        print(f"wrote {spec['output']}")


if __name__ == "__main__":
    main()
