"""Orthogonal time-sharing upper bound on the minimum energy, for any N.

Restrict schedules to at most K transmitters per slot, each on a subchannel of its
own, hence interference-free. On static, frequency-flat channels a UE sees the
same gain on every subchannel, so only the number of transmitters per slot
matters. Arrivals repeat with the hyperperiod C = lcm(P_i). The bound optimizes
one cyclic C-slot schedule and repeats it: every packet window of UE i is one of
its C / P_i cyclic windows, so the repeated schedule serves every packet. Any
such schedule is feasible, so the energy it spends in the environment is an
upper bound on the optimum; together with ``central.solo_reference`` (a lower
bound) it brackets the true optimum, which may also use co-channel sharing.

MILP (HiGHS): integer n[w, l] = slots at power level l in cyclic window w of its
UE; continuous y[i, t] in [0, 1] = UE i transmits in cycle slot t.

    sum_{t in w} y[i, t] = sum_l n[w, l]          (window slot count)
    sum_i y[i, t]        <= K                      (per-slot capacity)
    sum_l b[i, l] n[w, l] >= L_i                   (mutual information)
    minimize sum_{w, l} e_l n[w, l]

For integral n the y-rows form a bipartite window/slot transportation system
whose matrix is totally unimodular, so an integral y exists; max-flow recovers
it exactly. Levels are then placed highest first within each window.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import maximum_flow

from dsa_period.physics import PhysicalConfig, PowerMenu, shannon_bits
from dsa_period.topology import Topology
from dsa_period.traffic import TrafficPattern

# Relative margin on the mutual-information rows so HiGHS feasibility tolerances
# can never admit a plan that falls short when replayed in exact float order.
_MI_MARGIN = 1e-9
# The model is solved in microjoules and kilobits. In joules a cycle costs ~1e-3,
# so HiGHS's absolute MIP gap (1e-6) would stop at ~0.1% relative suboptimality.
_ENERGY_UNIT_J = 1e-6
_BITS_UNIT = 1e3


@dataclass(frozen=True)
class CyclicWindow:
    ue: int
    index: int
    slots: NDArray[np.int64]  # cycle slots in time order from the window start


def cyclic_windows(traffic: TrafficPattern) -> tuple[int, tuple[CyclicWindow, ...]]:
    """Hyperperiod C and every UE's packet windows on the C-slot cycle."""
    cycle = math.lcm(*(int(p) for p in traffic.period_slots))
    windows = []
    for ue in range(traffic.num_ues):
        period, phase = int(traffic.period_slots[ue]), int(traffic.phase_slots[ue])
        for index in range(cycle // period):
            start = phase + index * period
            windows.append(CyclicWindow(ue, index, (start + np.arange(period)) % cycle))
    return cycle, tuple(windows)


@dataclass(frozen=True)
class CycleSchedule:
    """Power ID per (cycle slot, UE); 0 is OFF. At most K nonzero entries per slot."""

    cycle_slots: int
    power_ids: NDArray[np.int64]
    window_counts: NDArray[np.int64]  # (windows, active levels)
    window_energy_j: NDArray[np.float64]
    windows: tuple[CyclicWindow, ...]
    cycle_energy_j: float
    mip_status: int
    mip_gap: float | None
    mip_dual_bound_j: float | None
    solve_seconds: float

    def planned_energy_j(self, ue: int, arrival_slot: int, period: int, phase: int) -> float:
        """Planned energy of the packet of ``ue`` that arrives at ``arrival_slot``."""
        index = ((arrival_slot - phase) % self.cycle_slots) // period
        for window, energy in zip(self.windows, self.window_energy_j, strict=True):
            if window.ue == ue and window.index == index:
                return float(energy)
        raise KeyError((ue, arrival_slot))


def _assign_slots(
    cycle: int, capacity: int, windows: tuple[CyclicWindow, ...], demand: NDArray[np.int64]
) -> NDArray[np.bool_]:
    """Integral slot assignment meeting every window's slot demand, by max-flow."""
    count = len(windows)
    source, sink = 0, count + cycle + 1
    rows, cols, caps = [], [], []
    for w, window in enumerate(windows):
        if demand[w]:
            rows.append(source), cols.append(1 + w), caps.append(int(demand[w]))
            for t in window.slots:
                rows.append(1 + w), cols.append(1 + count + int(t)), caps.append(1)
    for t in range(cycle):
        rows.append(1 + count + t), cols.append(sink), caps.append(capacity)
    graph = csr_array(
        coo_array(
            (np.asarray(caps, np.int32), (np.asarray(rows), np.asarray(cols))),
            shape=(sink + 1, sink + 1),
        )
    )
    result = maximum_flow(graph, source, sink)
    if result.flow_value != int(demand.sum()):
        raise RuntimeError("slot counts from the MILP admit no capacity-feasible assignment")
    flow = result.flow.tocoo()
    assigned = np.zeros((cycle, len({w.ue for w in windows})), dtype=np.bool_)
    for r, c, value in zip(flow.row, flow.col, flow.data, strict=True):
        if value > 0 and 1 <= r <= count and 1 + count <= c < sink:
            assigned[c - 1 - count, windows[r - 1].ue] = True
    return assigned


def solve_orthogonal_cycle(
    config: PhysicalConfig,
    menu: PowerMenu,
    topology: Topology,
    traffic: TrafficPattern,
    *,
    time_limit_s: float = 120.0,
    mip_rel_gap: float = 1e-6,
) -> CycleSchedule | None:
    """Minimum-energy orthogonal cyclic schedule, or None if even that is infeasible."""
    if topology.num_ues != config.num_ues or traffic.num_ues != config.num_ues:
        raise ValueError("topology and traffic must match num_ues")
    cycle, windows = cyclic_windows(traffic)
    num_ues, levels = config.num_ues, menu.num_levels - 1
    power = np.asarray(menu.power_w[1:])
    energy = config.slot_duration_s * power
    gains = topology.ue_ap_gain_linear
    bits = np.array([shannon_bits(config, power * g / config.noise_power_w) for g in gains])
    num_n = len(windows) * levels
    num_y = num_ues * cycle

    def n_index(w: int, level: int) -> int:
        return w * levels + level

    def y_index(ue: int, t: int) -> int:
        return num_n + ue * cycle + t

    count_rows, capacity_rows, mi_rows = [], [], []
    for w, window in enumerate(windows):
        count_rows += [(w, y_index(window.ue, int(t)), 1.0) for t in window.slots]
        count_rows += [(w, n_index(w, level), -1.0) for level in range(levels)]
        mi_rows += [
            (w, n_index(w, level), bits[window.ue, level] / _BITS_UNIT) for level in range(levels)
        ]
    for t in range(cycle):
        capacity_rows += [(t, y_index(ue, t), 1.0) for ue in range(num_ues)]

    def matrix(entries, rows):
        r, c, v = zip(*entries, strict=True)
        return csr_array(coo_array((v, (r, c)), shape=(rows, num_n + num_y)))

    packets = np.array([traffic.packet_bits[w.ue] for w in windows])
    upper_n = np.repeat([len(w.slots) for w in windows], levels)
    started = time.perf_counter()
    result = milp(
        c=np.concatenate((np.tile(energy / _ENERGY_UNIT_J, len(windows)), np.zeros(num_y))),
        integrality=np.concatenate((np.ones(num_n), np.zeros(num_y))),
        bounds=Bounds(np.zeros(num_n + num_y), np.concatenate((upper_n, np.ones(num_y)))),
        constraints=[
            LinearConstraint(matrix(count_rows, len(windows)), 0.0, 0.0),
            LinearConstraint(matrix(capacity_rows, cycle), -np.inf, config.num_subchannels),
            LinearConstraint(
                matrix(mi_rows, len(windows)), packets / _BITS_UNIT * (1 + _MI_MARGIN), np.inf
            ),
        ],
        options={"time_limit": time_limit_s, "mip_rel_gap": mip_rel_gap, "disp": False},
    )
    seconds = time.perf_counter() - started
    if result.status == 2:
        return None
    if result.x is None:
        raise RuntimeError(f"MILP returned no solution: {result.message}")
    raw = result.x[:num_n].reshape(len(windows), levels)
    counts = np.rint(raw).astype(np.int64)
    if np.max(np.abs(raw - counts)) > 1e-6:
        raise RuntimeError("MILP slot counts are not integral")
    for w, window in enumerate(windows):
        if counts[w] @ bits[window.ue] < packets[w]:
            raise RuntimeError(f"window {w} falls short of its packet after rounding")
    assigned = _assign_slots(cycle, config.num_subchannels, windows, counts.sum(axis=1))
    power_ids = np.zeros((cycle, num_ues), dtype=np.int64)
    for w, window in enumerate(windows):
        used = [int(t) for t in window.slots if assigned[t, window.ue]]
        order = [level + 1 for level in range(levels - 1, -1, -1) for _ in range(counts[w, level])]
        power_ids[used, window.ue] = order
    window_energy = counts @ energy
    dual_bound = getattr(result, "mip_dual_bound", None)
    return CycleSchedule(
        cycle_slots=cycle,
        power_ids=power_ids,
        window_counts=counts,
        window_energy_j=window_energy,
        windows=windows,
        cycle_energy_j=float(window_energy.sum()),
        mip_status=int(result.status),
        mip_gap=getattr(result, "mip_gap", None),
        mip_dual_bound_j=None if dual_bound is None else dual_bound * _ENERGY_UNIT_J,
        solve_seconds=seconds,
    )


def act_cycle(schedule: CycleSchedule, slot: int) -> NDArray[np.int64]:
    """Centralized witness: this slot's scheduled UEs on distinct subchannels."""
    ids = schedule.power_ids[slot % schedule.cycle_slots]
    actions = np.zeros((ids.size, 2), dtype=np.int64)
    for channel, ue in enumerate(np.flatnonzero(ids)):
        actions[ue] = (channel, ids[ue])
    return actions
