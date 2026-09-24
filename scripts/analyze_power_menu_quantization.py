"""Energy quantization loss of discrete power menus (idealized, interference-free).

Used to choose the v1.0 power menu (2026-09-24). Each UE, drawn uniformly by area,
transmits one packet per period alone on a channel and may use at most m_max
slots of its period:

    m_max = P                  if N <= K  (a channel of its own)
    m_max = floor(P * K / N)   if N >  K  (orthogonal fair time share)

A menu level p needs m = ceil(L / (B tau log2(1 + p g / sigma^2))) slots and
costs p * m * tau; the menu's energy is the cheapest level with m <= m_max. The
reference is continuous power over exactly m_max slots, capped at 23 dBm. The
reported ratio is total menu energy / total continuous energy over all sampled
UEs and the three traffic classes. Co-channel interference is ignored, so this
guides menu design only and is not a performance bound.
"""

import argparse
import math

import numpy as np

TAU, BANDWIDTH = 1e-3, 1e6
NOISE_W = 10 ** ((-174 + 10 * math.log10(BANDWIDTH) + 7 - 30) / 10)
CLASSES = ((10, 6144.0), (20, 12288.0), (40, 24576.0))
MENUS = {
    "[0,5,10,15,20,23]": (0, 5, 10, 15, 20, 23),
    "[5,10,15,23]": (5, 10, 15, 23),
    "[5,10,15,19,23]": (5, 10, 15, 19, 23),
    "[6,12,18,23]": (6, 12, 18, 23),
}


def _watts(dbm):
    return 10 ** ((np.asarray(dbm, dtype=float) - 30) / 10)


def ratio(menu_dbm, gains, num_ues, num_subchannels=4):
    menu_energy = reference_energy = 0.0
    power = _watts(menu_dbm)
    for period, packet in CLASSES:
        m_max = period if num_ues <= num_subchannels else period * num_subchannels // num_ues
        slots = np.ceil(packet / (BANDWIDTH * TAU * np.log2(1 + np.outer(gains, power) / NOISE_W)))
        energy = np.where(slots <= m_max, power * slots * TAU, np.inf).min(axis=1)
        needed = (2 ** (packet / (m_max * BANDWIDTH * TAU)) - 1) * NOISE_W / gains
        continuous = np.where(needed <= _watts(23), needed * m_max * TAU, np.inf)
        both = np.isfinite(energy) & np.isfinite(continuous)
        menu_energy += energy[both].sum()
        reference_energy += continuous[both].sum()
    return menu_energy / reference_energy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    distance = np.maximum(100 * np.sqrt(rng.random(args.samples)), 1.0)
    gains = 10 ** (-(40 + 40 * np.log10(distance)) / 10)
    sizes = (4, 6, 8, 12)
    print(f"{'menu (dBm)':<20}" + "".join(f"{'N=' + str(n):>8}" for n in sizes))
    for name, menu in MENUS.items():
        print(f"{name:<20}" + "".join(f"{ratio(menu, gains, n):8.3f}" for n in sizes))


if __name__ == "__main__":
    main()
