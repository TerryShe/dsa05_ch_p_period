"""LP capacity bounds of scenario v2.0 (docs/research/scenario-v2.md §7).

For one episode (instance, fading realization) both LPs find s*, the smallest
worst-UE fractional violation rate any schedule in their class can reach:

    min s  s.t.  sum_{t in window(p), k} b_itk x_itk / L_p + z_p >= 1   (packet p of UE i)
                 sum_k x_itk <= 1                                      (UE i, slot t)
                 sum_{p of i} z_p <= s M_i                             (UE i)
                 0 <= x, z <= 1, plus the per-(t, k) constraint below,

where b_itk is UE i's mutual information alone at max power on subchannel k in
slot t and x_itk the fraction of that slot it uses. Only evaluated packets
(deadline in (W, H]) carry constraints; other slots get no variables, which only
relaxes the problem.

* ``orth`` (N_orth): sum_i x_itk <= 1 for each (slot, subchannel), and the
  whole-slot cut sum_{t in window(p), k} x_itk >= m_p (1 - z_p), where m_p is
  the fewest slots that deliver packet p alone on its best subchannel at max
  power (its best slots first). Every orthogonal schedule, causal or not,
  satisfies both: a delivered packet occupies at least m_p whole (slot,
  subchannel) pairs, which no other UE can use. Without the cut the LP lets a
  packet that needs 1.2 slots of airtime leave 0.8 of its second slot to others.
* ``orth_agg``: ``orth`` with the K subchannels merged into one resource of K
  units per slot; each UE counts its best subchannel's rate. Every orthogonal
  schedule maps into it (at most K UEs per slot, each at most at its best
  rate), so it is a looser but cheaper bound on N_orth; it drops only the
  condition that two UEs cannot both use the same best subchannel.
* ``mac`` (N_UB): sum_i b_itk x_itk <= tau B log2(1 + sum_i SNR_itk), the sum
  capacity of all N UEs at max power; no receiver, SIC included, exceeds it,
  and treating interference as noise stays inside the capacity region. Several
  UEs may share a (slot, subchannel) here, so the whole-slot cut does not apply.

z_p is a relaxed violation indicator. An actual execution gives x (accumulated
mutual information over b, at most 1 on one subchannel), z_p = 1 for a missed
packet and 0 otherwise, so V_i = sum z, and every constraint above holds.
Undeliverable packets (m_p infinite) get z_p = 1. Hence s* <= min_i floor(eps
M_i) / M_i whenever some schedule meets eps; with eps * M_i integral (H =
20,200) the episode is infeasible for the class iff s* > eps. Nesting across N
makes s* of ``orth`` and ``orth_agg`` nondecreasing in N; s*_mac is not monotone, because the sum
capacity grows with N.
"""

import time

import highspy
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from dsa_period.capacity import evaluated_packets
from dsa_period.config import Scenario
from dsa_period.physics import shannon_bits
from dsa_period.traffic import TrafficPattern

KINDS = ("orth", "orth_agg", "mac")
MONOTONE_KINDS = ("orth", "orth_agg")
FEASIBILITY_TOLERANCE = 1e-6


def alone_bits(scenario: Scenario, gains: NDArray[np.float64]) -> NDArray[np.float64]:
    """(T, N, K) mutual information of each UE alone at max power."""
    snr = scenario.menu.power_w[-1] * gains / scenario.physical.noise_power_w
    return shannon_bits(scenario.physical, snr)


def sum_capacity_bits(scenario: Scenario, gains: NDArray[np.float64]) -> NDArray[np.float64]:
    """(T, K) sum capacity of all UEs at max power on each (slot, subchannel)."""
    snr = scenario.menu.power_w[-1] * gains / scenario.physical.noise_power_w
    return shannon_bits(scenario.physical, snr.sum(axis=1))


def minimum_slots(
    best_bits: NDArray[np.float64],
    packet_of_row: NDArray[np.int64],
    packet_bits: NDArray[np.float64],
    lengths: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Per packet, the fewest window slots whose best-subchannel bits reach its size.

    ``best_bits`` lists, row by row, the window slots of packet 0, then packet 1,
    ...; inf marks a packet its whole window cannot deliver.
    """
    order = np.lexsort((-best_bits, packet_of_row))  # packets stay in place
    cumulative = np.cumsum(best_bits[order])
    ends = np.cumsum(lengths)
    before = np.concatenate([[0.0], cumulative[ends[:-1] - 1]])
    within = cumulative - np.repeat(before, lengths)
    short = np.bincount(
        packet_of_row, weights=within < packet_bits[packet_of_row], minlength=lengths.size
    )
    needed = short + 1.0
    needed[within[ends - 1] < packet_bits] = np.inf
    return needed


def build_lp(
    scenario: Scenario, traffic: TrafficPattern, gains: NDArray[np.float64], kind: str
) -> dict:
    """Sparse LP of one episode; ``gains`` covers slots [0, H)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    packets = evaluated_packets(traffic, scenario.warmup_slots, scenario.horizon_slots)
    bits = alone_bits(scenario, gains)
    subchannels = bits.shape[2]
    if kind == "orth_agg":
        bits = bits.max(axis=2, keepdims=True)  # each UE on its best subchannel
    horizon, n, k = bits.shape
    lengths = packets["deadline"] - packets["arrival"]
    num_packets = lengths.size
    # One row r per (packet, slot of its window); K x-variables per row.
    packet_of_row = np.repeat(np.arange(num_packets), lengths)
    starts = np.repeat(packets["arrival"] - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)
    slot_of_row = starts + np.arange(lengths.sum())
    ue_of_row = packets["ue"][packet_of_row]
    rows = slot_of_row.size
    num_x = rows * k
    z0, s_index = num_x, num_x + num_packets
    num_vars = s_index + 1
    b = bits[slot_of_row, ue_of_row, :]  # (rows, K)
    x_index = np.arange(num_x).reshape(rows, k)

    blocks, rhs = [], []
    # Packets: -sum b x / L - z <= -1.
    coef = -(b / packets["bits"][packet_of_row][:, None]).ravel()
    blocks.append(
        sp.coo_matrix(
            (
                np.concatenate([coef, -np.ones(num_packets)]),
                (
                    np.concatenate([np.repeat(packet_of_row, k), np.arange(num_packets)]),
                    np.concatenate([x_index.ravel(), z0 + np.arange(num_packets)]),
                ),
            ),
            shape=(num_packets, num_vars),
        )
    )
    rhs.append(-np.ones(num_packets))
    if k > 1:
        # One subchannel per (UE, slot): sum_k x <= 1 (a bound when k = 1).
        blocks.append(
            sp.coo_matrix(
                (np.ones(num_x), (np.repeat(np.arange(rows), k), x_index.ravel())),
                shape=(rows, num_vars),
            )
        )
        rhs.append(np.ones(rows))
    # Per (slot, subchannel), rows only for slots that carry variables.
    slots, slot_row = np.unique(slot_of_row, return_inverse=True)
    tk_row = (slot_row[:, None] * k + np.arange(k)[None, :]).ravel()
    if kind in MONOTONE_KINDS:
        values = np.ones(num_x)
    else:
        capacity = sum_capacity_bits(scenario, gains)[slots]  # (slots, K)
        values = (b / capacity[slot_row]).ravel()
    blocks.append(
        sp.coo_matrix((values, (tk_row, x_index.ravel())), shape=(slots.size * k, num_vars))
    )
    # orth_agg: at most K UEs per slot (one merged resource of K subchannels).
    rhs.append(np.full(slots.size * k, subchannels if kind == "orth_agg" else 1.0))
    # Per UE: sum z - s M_i <= 0.
    count = np.bincount(packets["ue"], minlength=n)
    blocks.append(
        sp.coo_matrix(
            (
                np.concatenate([np.ones(num_packets), -count.astype(float)]),
                (
                    np.concatenate([packets["ue"], np.arange(n)]),
                    np.concatenate([z0 + np.arange(num_packets), np.full(n, s_index)]),
                ),
            ),
            shape=(n, num_vars),
        )
    )
    rhs.append(np.zeros(n))
    lower = np.zeros(num_vars)
    if kind in MONOTONE_KINDS:
        slots_needed = minimum_slots(b.max(axis=1), packet_of_row, packets["bits"], lengths)
        deliverable = np.isfinite(slots_needed)
        # Whole-slot cut: -sum x - m_p z_p <= -m_p for deliverable packets.
        cut = np.flatnonzero(deliverable)
        in_cut = deliverable[packet_of_row]
        cut_row = np.cumsum(deliverable) - 1  # row of packet p among the cut rows
        cut_x = x_index[in_cut].ravel()
        blocks.append(
            sp.coo_matrix(
                (
                    np.concatenate([-np.ones(cut_x.size), -slots_needed[cut]]),
                    (
                        np.concatenate(
                            [np.repeat(cut_row[packet_of_row[in_cut]], k), np.arange(cut.size)]
                        ),
                        np.concatenate([cut_x, z0 + cut]),
                    ),
                ),
                shape=(cut.size, num_vars),
            )
        )
        rhs.append(-slots_needed[cut])
        lower[z0 + np.flatnonzero(~deliverable)] = 1.0
    cost = np.zeros(num_vars)
    cost[s_index] = 1.0
    upper = np.ones(num_vars)
    upper[s_index] = np.inf
    return {
        "A_ub": sp.vstack(blocks).tocsr(),
        "b_ub": np.concatenate(rhs),
        "c": cost,
        "bounds": np.column_stack([lower, upper]),
        "packets": count,
        "num_x": num_x,
    }


def solve_bound(
    scenario: Scenario,
    traffic: TrafficPattern,
    gains: NDArray[np.float64],
    kind: str,
    time_limit_s: float = 3600.0,
) -> dict:
    """s* of one episode by HiGHS interior point without crossover.

    Crossover (recovering a basic solution) is not needed for s* and took hours
    on tight instances; without it the objective is exact up to the IPM
    tolerances, which the record keeps (``objective_error``).
    """
    lp = build_lp(scenario, traffic, gains, kind)
    matrix = lp["A_ub"].tocsc()
    model = highspy.HighsLp()
    model.num_row_, model.num_col_ = matrix.shape
    model.col_cost_ = lp["c"]
    model.col_lower_ = lp["bounds"][:, 0]
    model.col_upper_ = np.where(np.isinf(lp["bounds"][:, 1]), highspy.kHighsInf, lp["bounds"][:, 1])
    model.row_lower_ = np.full(matrix.shape[0], -highspy.kHighsInf)
    model.row_upper_ = lp["b_ub"]
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.start_ = matrix.indptr
    model.a_matrix_.index_ = matrix.indices
    model.a_matrix_.value_ = matrix.data
    solver = highspy.Highs()
    for option, value in (
        ("output_flag", False),
        ("solver", "ipm"),
        ("run_crossover", "off"),
        ("threads", 1),
        ("time_limit", float(time_limit_s)),
    ):
        solver.setOptionValue(option, value)
    solver.passModel(model)
    start = time.perf_counter()
    solver.run()
    elapsed = time.perf_counter() - start
    status = solver.getModelStatus()
    info = solver.getInfo()
    optimal = status == highspy.HighsModelStatus.kOptimal
    return {
        "kind": kind,
        "status": solver.modelStatusToString(status),
        "s_star": float(info.objective_function_value) if optimal else None,
        "objective_error": float(info.primal_dual_objective_error) if optimal else None,
        "solve_s": elapsed,
        "variables": int(lp["c"].size),
        "constraints": int(lp["b_ub"].size),
    }


def feasible_at(s_star: float | None, eps: float) -> bool | None:
    """Whether some schedule of the class can meet eps (None if the LP failed)."""
    if s_star is None:
        return None
    return s_star <= eps + FEASIBILITY_TOLERANCE
