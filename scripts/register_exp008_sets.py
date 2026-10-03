"""Write the EXP-DSA05-008 set manifest docs/research/exp008-sets.json.

Registers the checkpoint screening set exp008-screen256 (dev 2000-2255, N = 6
and 8) and the confirmation set v2-confirm1 (reserve 30000-30499, every N of the
protocol-v2 grid and of N_EDF) with their instance and fading digests
(``splits_v2.instance_digest``: topology, shadowing, traffic and the fading of
slots 0-7). No policy runs. tests/test_exp008.py checks the manifest against
the code; rerunning this script must reproduce it byte for byte.
"""

import hashlib
import json
from pathlib import Path

from dsa_period import splits_v2
from dsa_period.config import load_scenario

ROOT = Path(__file__).parents[1]
CONFIG = "configs/scenario-v2.toml"
OUTPUT = ROOT / "docs" / "research" / "exp008-sets.json"
DESIGN = "docs/experiment-designs/008-training-signal_20261002.md"


def build() -> dict:
    base = load_scenario(ROOT / CONFIG, 2)
    every_n = sorted({*base.capacity["grid"], *base.capacity["reference_extra"]})
    screening = {}
    for name, entry in splits_v2.SCREENING_SETS.items():
        start, stop = entry["range"]
        screening[name] = {
            "split": "dev",
            "start": start,
            "stop": stop,
            "count": stop - start,
            "grid": list(entry["grid"]),
            "repeats": splits_v2.evaluation_repeats("dev"),
            "seed_code": splits_v2.SPLIT_CODES["dev"],
            "score": "S_near = (F_0.01(6) + F_0.01(8)) / 2",
            "selection": "per run: max (S_near, checkpoint slots); design §4.1",
            "entry_point": "evaluate_capacity --split dev --screen " + name + " --formal",
            "digests": {
                str(n): splits_v2.instance_digest(
                    load_scenario(ROOT / CONFIG, n), range(start, stop), "dev"
                )
                for n in entry["grid"]
            },
        }
    confirmation = {}
    for name, (start, stop) in splits_v2.CONFIRMATION_SETS.items():
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
            "entry_point": "evaluate_capacity --split " + name + " --confirm-c1 --formal",
            "digests": {
                str(n): splits_v2.instance_digest(
                    load_scenario(ROOT / CONFIG, n), range(start, stop), name
                )
                for n in every_n
            },
        }
    return {
        "schema_version": 1,
        "experiment": "EXP-DSA05-008",
        "design": DESIGN,
        "document": "docs/research/exp008-registration.md",
        "protocol_id": splits_v2.PROTOCOL_ID,
        "registered_on": "2026-10-03",
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
            "v2_dev_ranges_used": [[0, 64], [1000, 1400]],
            "v2_reserve_ranges_read": [],
        },
    }


def main() -> None:
    OUTPUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
