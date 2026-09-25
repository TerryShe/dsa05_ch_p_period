"""Exploratory: a MAC sum-rate LP lower bound, compared with the solo lower bound.

Not used by the reports' bounds; kept to reproduce the negative result they cite.

For any TIN schedule, the bits of the UEs sharing one subchannel in one slot lie
in the multiple-access capacity region, so their sum is at most
tau*B*log2(1 + sum of received SNRs); summed over the K subchannels, Jensen gives
at most tau*B*K*log2(1 + X/K), X = sum over all transmitters' received SNR. The
LP relaxes the power levels to fractions, keeps per-UE solo-rate caps, imposes
this concave per-slot cap through tangent lines, and adds the valid cut that
every packet costs at least its UE's solo optimum. Averaging any feasible
schedule over the hyperperiods yields a feasible point, so the LP value is a
lower bound on the cyclic energy. Quantities are in microjoules and kilobits.
"""

import argparse
import math

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_array, csr_array

from dsa_period.central import solo_reference
from dsa_period.central_timeshare import cyclic_windows, solve_orthogonal_cycle
from dsa_period.config import load_scenario
from dsa_period.physics import shannon_bits


def mac_lower_bound_j(scenario, topology, traffic, tangents: int = 40) -> float:
    config, menu = scenario.physical, scenario.menu
    cycle, windows = cyclic_windows(traffic)
    n, k, levels = config.num_ues, config.num_subchannels, menu.num_levels - 1
    power = np.asarray(menu.power_w[1:])
    energy = config.slot_duration_s * power * 1e6
    snr = np.outer(topology.ue_ap_gain_linear, power) / config.noise_power_w
    kbits = shannon_bits(config, snr) / 1e3
    uses = config.channel_uses_per_slot / 1e3
    solo = solo_reference(config, menu, topology, traffic)
    num_x = n * cycle * levels

    def x(i, t, level):
        return (i * cycle + t) * levels + level

    def r(i, t):
        return num_x + i * cycle + t

    rows, rhs = [], []
    for i in range(n):
        for t in range(cycle):
            rows.append([(x(i, t, level), 1.0) for level in range(levels)])
            rhs.append(1.0)
            rows.append([(r(i, t), 1.0)] + [(x(i, t, lv), -kbits[i, lv]) for lv in range(levels)])
            rhs.append(0.0)
    for w in windows:
        rows.append([(r(w.ue, int(t)), -1.0) for t in w.slots])
        rhs.append(-traffic.packet_bits[w.ue] / 1e3)
        rows.append([(x(w.ue, int(t), lv), -energy[lv]) for t in w.slots for lv in range(levels)])
        rhs.append(-solo.plans[w.ue].energy_j * 1e6)
    largest = float(snr.max(axis=1).sum())
    for x0 in np.concatenate(([0.0], np.geomspace(1e-3, largest, tangents))):
        value = k * math.log2(1 + x0 / k)
        slope = 1 / ((1 + x0 / k) * math.log(2))
        for t in range(cycle):
            rows.append(
                [(r(i, t), 1.0) for i in range(n)]
                + [
                    (x(i, t, lv), -uses * slope * snr[i, lv])
                    for i in range(n)
                    for lv in range(levels)
                ]
            )
            rhs.append(uses * (value - slope * x0))
    entries = [(row, col, val) for row, cells in enumerate(rows) for col, val in cells]
    row_idx, col_idx, values = zip(*entries, strict=True)
    matrix = csr_array(
        coo_array((values, (row_idx, col_idx)), shape=(len(rows), num_x + n * cycle))
    )
    cost = np.concatenate((np.tile(energy, n * cycle), np.zeros(n * cycle)))
    result = linprog(cost, A_ub=matrix, b_ub=np.asarray(rhs), bounds=(0, None), method="highs")
    if result.status != 0:
        raise RuntimeError(result.message)
    return result.fun * 1e-6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/scenario-v1-n8.toml", "configs/scenario-v1-n12.toml"],
    )
    parser.add_argument("--instances", type=int, default=5)
    args = parser.parse_args()
    for path in args.configs:
        scenario = load_scenario(path)
        for index in range(args.instances):
            topology, traffic = scenario.instance(index)
            solo = solo_reference(scenario.physical, scenario.menu, topology, traffic)
            cycle = 40
            lower = sum(
                plan.energy_j * (cycle // int(traffic.period_slots[ue]))
                for ue, plan in enumerate(solo.plans)
            )
            mac = mac_lower_bound_j(scenario, topology, traffic)
            upper = solve_orthogonal_cycle(
                scenario.physical, scenario.menu, topology, traffic
            ).cycle_energy_j
            print(
                f"{scenario.experiment['id']} instance {index}: solo LB {1e3 * lower:.4f}, "
                f"MAC LB {1e3 * mac:.4f}, orthogonal UB {1e3 * upper:.4f} mJ/cycle; "
                f"MAC/solo {mac / lower:.4f}, UB/MAC {upper / mac:.4f}"
            )


if __name__ == "__main__":
    main()
