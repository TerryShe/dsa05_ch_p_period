"""Thresholds tau_N of the EXP-DSA05-009 hard stratum (design §5.1), from the train pool.

The hardness of a deployment is ``level_sampler.deployment_hardness`` (smallest
large-scale service margin over its UEs). tau_N is its 10% quantile at each
training N over SAMPLE distinct training-pool instances drawn with SEED.
Deployments are nested across N, so each instance is generated once at the
largest N and its prefixes give the smaller N (checked on a few instances).
Writes configs/exp009-tau.toml; tests/test_exp009.py checks it on a subsample.
"""

import argparse
from pathlib import Path

import numpy as np

from dsa_period import splits
from dsa_period.config import load_scenario
from dsa_period.marl.level_sampler import deployment_hardness
from dsa_period.physics import shannon_bits

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "scenario-v2.toml"
OUTPUT = ROOT / "configs" / "exp009-tau.toml"
GRID = (2, 4, 6, 8, 10, 12, 16, 20)
SAMPLE, SEED, QUANTILE = 20_000, 20_261_004, 0.10


def sample_indices(size: int = SAMPLE, seed: int = SEED) -> np.ndarray:
    start, stop = splits.SPLITS["train"]
    rng = np.random.default_rng(seed)
    return start + rng.choice(stop - start, size=size, replace=False)


def prefix_hardness(indices) -> dict[int, np.ndarray]:
    """Hardness of every index at every N in GRID, from the largest-N instance."""
    top = load_scenario(CONFIG, max(GRID))
    power, noise = top.menu.power_w[-1], top.physical.noise_power_w
    out = {n: np.empty(len(indices)) for n in GRID}
    for k, index in enumerate(indices):
        topology, traffic = top.instance(int(index))
        margin = shannon_bits(top.physical, power * topology.ue_ap_gain_linear / noise) / (
            traffic.packet_bits / traffic.period_slots
        )
        for n in GRID:
            out[n][k] = margin[:n].min()
    for index in indices[:5]:  # nesting: the prefix equals the deployment generated at N
        for n in GRID:
            scenario = load_scenario(CONFIG, n)
            direct = deployment_hardness(scenario, *scenario.instance(int(index)))
            assert np.isclose(direct, out[n][list(indices).index(index)], rtol=1e-12)
    return out


def thresholds(indices) -> dict[str, float]:
    return {str(n): float(np.quantile(h, QUANTILE)) for n, h in prefix_hardness(indices).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=SAMPLE)
    args = parser.parse_args()
    tau = thresholds(sample_indices(args.sample))
    lines = [
        "# EXP-DSA05-009 hard stratum (design §5.1), written by scripts/compute_exp009_tau.py.",
        f"# tau_N = {QUANTILE:.0%} quantile of the deployment hardness over {args.sample} distinct",
        f"# train-pool instances drawn with seed {SEED}. Frozen with the design; do not edit.",
        "[tau]",
        *(f'"{n}" = {value!r}' for n, value in tau.items()),
        "",
    ]
    OUTPUT.write_text("\n".join(lines))
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {tau}")


if __name__ == "__main__":
    main()
