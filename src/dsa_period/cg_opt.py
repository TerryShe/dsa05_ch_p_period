"""Optimized static configured grant CG-opt (design: configured-grant-optimized_20261010.md).

Same knowledge, configuration form and transmission rule as the configured-grant
reference N_CG (``rules_v2.ConfiguredGrantPolicy``); only the allocation of
exclusive periodic occasions changes.

Objective (design §3): leximin of the predicted outage on the optimization bank
A (seed base 20,261,010), i.e. the per-UE miss counts on the bank sorted from
the largest down and compared lexicographically. Counts are integers, so every
comparison is exact.

Two deterministic stages under a fixed budget (§4):

1. ``solve_counts``: a count/packing MILP (HiGHS). Each UE's misses as a
   function of its number of occasions (``outage_curve``: its lone placement
   order on bank A), exact packing, subchannel balance and time spread; first
   min max, then min sum under that max.
2. ``local_search``: first improvement on the true misses on bank A (add,
   relocate, transfer) from the better of the greedy on bank A and stage 1.

``optimize_grants`` runs both and returns a ``GrantAllocation`` whose
``outage`` is the predicted outage on bank A.
"""

import math
import multiprocessing
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dsa_period.physics import PhysicalConfig, shannon_bits
from dsa_period.rules_v2 import GrantAllocation, allocate_grants

CG_OPT_BANK_SEED = 20_261_010
CG_OPT_BANK_SAMPLES = 20_000
SPREAD_BLOCK_SLOTS = 10
# Budget (design §4.4), frozen 2026-10-10: calibrated in phase 0 on dev instances
# 0-63 at N in {6, 8, 10, 16, 24, 32}, allocation metrics only
# (docs/results/cg-opt-calibration_20261010): 99% of the stage-1 solves need at
# most 3,824 nodes, every search reaches its local optimum within 4,510.
MIP_MAX_NODES: int | None = 10_000
SEARCH_EVALUATIONS: int | None = 10_000
TIME_LIMIT_S = 120.0
# HiGHS can overrun its own time limit at the root of symmetric models (phase 0:
# 3,120 s against 120 s), so every solve runs in a child killed at the limit
# plus this grace (no solution then).
KILL_GRACE_S = 5.0

Occasion = tuple[int, int]  # (slot residue r mod P_i, subchannel k)


def leximin_key(misses: Sequence[int]) -> tuple[int, ...]:
    """Misses sorted from the largest down; smaller is better (design §3)."""
    return tuple(sorted((int(m) for m in misses), reverse=True))


@dataclass(frozen=True)
class Deployment:
    """What the AP knows (CG design §2) plus the bank it optimizes on."""

    config: PhysicalConfig
    snr: NDArray[np.float64]
    packet_bits: NDArray[np.float64]
    period: tuple[int, ...]
    phase: tuple[int, ...]
    bank: NDArray[np.float64]

    @property
    def num_ues(self) -> int:
        return len(self.period)

    @property
    def num_subchannels(self) -> int:
        return self.bank.shape[2]

    @property
    def samples(self) -> int:
        return self.bank.shape[0]

    @property
    def hyperperiod(self) -> int:
        return math.lcm(*self.period)

    def bits(self, ue: int, occasion: Occasion) -> NDArray[np.float64]:
        """Mutual information of one occasion on every sample (window position)."""
        r, k = occasion
        position = (r - self.phase[ue]) % self.period[ue]
        return shannon_bits(self.config, self.snr[ue] * self.bank[:, position, k])

    def accumulated(self, ue: int, occasions) -> NDArray[np.float64]:
        """Bits per sample with ``occasions``, summed in the canonical (sorted) order."""
        total = np.zeros(self.samples)
        for occasion in sorted(occasions):
            total += self.bits(ue, occasion)
        return total

    def misses(self, ue: int, bits: NDArray[np.float64]) -> int:
        return int(np.count_nonzero(bits < self.packet_bits[ue]))

    def all_misses(self, occasions: Sequence) -> list[int]:
        return [self.misses(ue, self.accumulated(ue, o)) for ue, o in enumerate(occasions)]


def deployment(
    config: PhysicalConfig,
    snr: NDArray[np.float64],
    packet_bits: NDArray[np.float64],
    period: NDArray[np.int64],
    phase: NDArray[np.int64],
    bank: NDArray[np.float64],
) -> Deployment:
    period = tuple(int(p) for p in period)
    if max(period) > bank.shape[1] or bank.shape[2] != config.num_subchannels:
        raise ValueError("the fading bank must cover the longest window and every subchannel")
    return Deployment(
        config,
        np.asarray(snr, dtype=np.float64),
        np.asarray(packet_bits, dtype=np.float64),
        period,
        tuple(int(p) for p in phase),
        bank,
    )


def predicted_outage(d: Deployment, occasions: Sequence) -> list[float]:
    """Each UE's predicted outage with ``occasions`` on the deployment's bank."""
    return [m / d.samples for m in d.all_misses(occasions)]


# ------------------------------------------------------------------ stage 1


def outage_curve(d: Deployment, ue: int) -> NDArray[np.int64]:
    """Misses of UE ``ue`` with the first n occasions of its lone placement, n = 0..P_i.

    The order is the CG placement rule for the UE alone at phase 0 (positions
    in the window); the running minimum keeps the curve non-increasing.
    """
    period = d.period[ue]
    lone = allocate_grants(
        d.config,
        d.snr[ue : ue + 1],
        d.packet_bits[ue : ue + 1],
        np.array([period]),
        np.array([0]),
        d.bank,
    ).occasions[0]
    bits = np.zeros(d.samples)
    misses = [d.misses(ue, bits)]
    for r, k in lone:
        bits += shannon_bits(d.config, d.snr[ue] * d.bank[:, r, k])
        misses.append(d.misses(ue, bits))
    return np.minimum.accumulate(np.array(misses, dtype=np.int64))


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass
class _Milp:
    """Sparse rows of a HiGHS model, built once and solved with two objectives."""

    lower: list[float]
    upper: list[float]
    integer: list[bool]
    row_lower: list[float]
    row_upper: list[float]
    entries: list[tuple[int, int, float]]

    def column(self, lower: float, upper: float, integer: bool) -> int:
        self.lower.append(lower)
        self.upper.append(upper)
        self.integer.append(integer)
        return len(self.lower) - 1

    def row(self, lower: float, upper: float, terms) -> None:
        index = len(self.row_lower)
        self.row_lower.append(lower)
        self.row_upper.append(upper)
        self.entries.extend((index, col, float(v)) for col, v in terms)

    def solve(
        self,
        cost: NDArray[np.float64],
        upper: NDArray[np.float64],
        start: NDArray[np.float64] | None,
        max_nodes: int | None,
        time_limit_s: float,
    ) -> tuple[NDArray[np.float64] | None, dict]:
        """One HiGHS solve in a forked child with a hard wall-clock limit.

        A solve that reaches the time limit gives no solution, whether HiGHS
        stops at it or the child is killed at the limit plus ``KILL_GRACE_S``
        (HiGHS's own stop comes seconds late under load, so keeping its
        incumbent would make the result depend on those seconds). The allocation
        then depends on the machine only through whether the limit is hit, and
        every hit is reported (design §4.4, phase-0 amendment). Reaching the
        node limit keeps the incumbent (deterministic).
        """
        context = multiprocessing.get_context("fork")
        receiver, sender = context.Pipe(duplex=False)
        args = (cost, upper, start, max_nodes, time_limit_s)
        child = context.Process(target=self._child, args=(sender, args))
        began = time.perf_counter()
        child.start()
        sender.close()
        if receiver.poll(time_limit_s + KILL_GRACE_S):
            values, record = receiver.recv()
        else:
            values, record = (
                None,
                {
                    "status": "Killed at the time limit",
                    "nodes": None,
                    "gap": None,
                    "objective": None,
                    "time_limit_hit": True,
                    "node_limit_hit": False,
                },
            )
            child.kill()
        child.join()
        receiver.close()
        record["seconds"] = time.perf_counter() - began
        if record["time_limit_hit"]:
            values = None
        return values, record

    def _child(self, sender, args) -> None:
        sender.send(self._run(*args))
        sender.close()

    def _run(
        self,
        cost: NDArray[np.float64],
        upper: NDArray[np.float64],
        start: NDArray[np.float64] | None,
        max_nodes: int | None,
        time_limit_s: float,
    ) -> tuple[NDArray[np.float64] | None, dict]:
        import highspy
        from scipy import sparse

        rows, cols, values = (np.array(v) for v in zip(*self.entries))
        matrix = sparse.csc_matrix(
            (values, (rows, cols)), shape=(len(self.row_lower), len(self.lower))
        )
        model = highspy.HighsLp()
        model.num_row_, model.num_col_ = matrix.shape
        model.col_cost_ = np.asarray(cost, dtype=np.float64)
        model.col_lower_ = np.array(self.lower, dtype=np.float64)
        model.col_upper_ = np.asarray(upper, dtype=np.float64)
        model.row_lower_ = np.where(
            np.isinf(self.row_lower), -highspy.kHighsInf, np.array(self.row_lower)
        )
        model.row_upper_ = np.where(
            np.isinf(self.row_upper), highspy.kHighsInf, np.array(self.row_upper)
        )
        model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        model.a_matrix_.start_ = matrix.indptr
        model.a_matrix_.index_ = matrix.indices
        model.a_matrix_.value_ = matrix.data
        model.integrality_ = [
            highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous
            for b in self.integer
        ]
        solver = highspy.Highs()
        options = [
            ("output_flag", False),
            ("threads", 1),
            ("random_seed", 0),
            ("time_limit", float(time_limit_s)),
        ]
        if max_nodes is not None:
            options.append(("mip_max_nodes", int(max_nodes)))
        for option, value in options:
            solver.setOptionValue(option, value)
        solver.passModel(model)
        if start is not None:
            hint = highspy.HighsSolution()
            hint.col_value = [float(v) for v in start]
            hint.value_valid = True
            solver.setSolution(hint)
        solver.run()
        status = solver.getModelStatus()
        info = solver.getInfo()
        found = info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible
        record = {
            "status": solver.modelStatusToString(status),
            "nodes": int(info.mip_node_count),
            "gap": float(info.mip_gap),
            "objective": float(info.objective_function_value) if found else None,
            "time_limit_hit": status == highspy.HighsModelStatus.kTimeLimit,
            "node_limit_hit": status == highspy.HighsModelStatus.kSolutionLimit,
        }
        return (np.array(solver.getSolution().col_value) if found else None), record


def solve_counts(
    d: Deployment,
    curves: Sequence[NDArray[np.int64]],
    max_nodes: int | None = MIP_MAX_NODES,
    time_limit_s: float = TIME_LIMIT_S,
    block_slots: int = SPREAD_BLOCK_SLOTS,
) -> tuple[list[list[Occasion]] | None, list[dict]]:
    """Stage 1 (design §4.2): occasions of the count/packing MILP and both solve records.

    x_{i,r,k}: UE i holds occasion (r, k); y_{i,n}: UE i holds n occasions.
    Rows: every resource unit (tau, k) of the hyperperiod at most once; one
    occasion per slot residue and UE; sum_n y = 1 and sum x = sum n y;
    floor(n/K) <= occasions on k <= ceil(n/K); with B_i = P_i / block_slots > 1
    blocks of the window (relative position), at most ceil(ceil(n/K) / B_i)
    occasions per (k, block); t >= c_i(n_i). First min t, then min sum_i c_i(n_i)
    with t <= t*, started from the first solution. None if no solution was found.
    """
    num_subchannels, hyperperiod = d.num_subchannels, d.hyperperiod
    milp = _Milp([], [], [], [], [], [])
    x, y = {}, {}
    for i, period in enumerate(d.period):
        for r in range(period):
            for k in range(num_subchannels):
                x[i, r, k] = milp.column(0, 1, True)
    for i, period in enumerate(d.period):
        for n in range(period + 1):
            y[i, n] = milp.column(0, 1, True)
    t = milp.column(0, 1, False)
    share = [np.asarray(c, dtype=np.float64) / d.samples for c in curves]
    for k in range(num_subchannels):
        for tau in range(hyperperiod):
            milp.row(-np.inf, 1, [(x[i, tau % p, k], 1) for i, p in enumerate(d.period)])
    for i, period in enumerate(d.period):
        counts = range(period + 1)
        for r in range(period):
            milp.row(-np.inf, 1, [(x[i, r, k], 1) for k in range(num_subchannels)])
        milp.row(1, 1, [(y[i, n], 1) for n in counts])
        milp.row(
            0,
            0,
            [(x[i, r, k], 1) for r in range(period) for k in range(num_subchannels)]
            + [(y[i, n], -n) for n in counts],
        )
        blocks = max(1, period // block_slots)
        for k in range(num_subchannels):
            held = [(x[i, r, k], 1) for r in range(period)]
            milp.row(0, np.inf, held + [(y[i, n], -(n // num_subchannels)) for n in counts])
            milp.row(-np.inf, 0, held + [(y[i, n], -_ceil_div(n, num_subchannels)) for n in counts])
            if blocks > 1:
                for b in range(blocks):
                    inside = [
                        (x[i, r, k], 1)
                        for r in range(period)
                        if min(((r - d.phase[i]) % period) // block_slots, blocks - 1) == b
                    ]
                    cap = [
                        (y[i, n], -_ceil_div(_ceil_div(n, num_subchannels), blocks)) for n in counts
                    ]
                    milp.row(-np.inf, 0, inside + cap)
        milp.row(0, np.inf, [(t, 1)] + [(y[i, n], -share[i][n]) for n in counts])

    def occasions_of(values) -> list[list[Occasion]]:
        return [
            sorted(
                (r, k)
                for r in range(period)
                for k in range(num_subchannels)
                if values[x[i, r, k]] > 0.5
            )
            for i, period in enumerate(d.period)
        ]

    upper = np.array(milp.upper, dtype=np.float64)
    cost = np.zeros(len(milp.lower))
    cost[t] = 1.0
    first, record_first = milp.solve(cost, upper, None, max_nodes, time_limit_s)
    if first is None:
        return None, [record_first]
    occasions = occasions_of(first)
    worst = max(int(curves[i][len(o)]) for i, o in enumerate(occasions))
    record_first["max_misses"] = worst
    upper[t] = (worst + 0.5) / d.samples
    cost[t] = 0.0
    for (i, n), col in y.items():
        cost[col] = share[i][n]
    start = np.round(first)
    start[t] = worst / d.samples
    second, record_second = milp.solve(cost, upper, start, max_nodes, time_limit_s)
    if second is not None:
        occasions = occasions_of(second)
        record_second["sum_misses"] = sum(int(curves[i][len(o)]) for i, o in enumerate(occasions))
    return occasions, [record_first, record_second]


# ------------------------------------------------------------------ stage 2


class _Grid:
    """Occupancy of the resource grid and each UE's bits on the bank."""

    def __init__(self, d: Deployment, occasions: Sequence) -> None:
        self.d = d
        self.owner = np.full((d.num_subchannels, d.hyperperiod), -1, dtype=np.int64)
        self.occasions: list[set[Occasion]] = [set() for _ in range(d.num_ues)]
        for ue, grants in enumerate(occasions):
            for occasion in grants:
                if not self.fits(ue, occasion):
                    raise ValueError(f"UE {ue}: occasion {occasion} is not free")
                self.take(ue, occasion)
        self.bits = [d.accumulated(ue, o) for ue, o in enumerate(self.occasions)]
        self.misses = [d.misses(ue, b) for ue, b in enumerate(self.bits)]

    def fits(self, ue: int, occasion: Occasion) -> bool:
        r, k = occasion
        if any(rr == r for rr, _ in self.occasions[ue]):
            return False
        return bool(np.all(self.owner[k, r :: self.d.period[ue]] == -1))

    def take(self, ue: int, occasion: Occasion) -> None:
        r, k = occasion
        self.owner[k, r :: self.d.period[ue]] = ue
        self.occasions[ue].add(occasion)

    def release(self, ue: int, occasion: Occasion) -> None:
        r, k = occasion
        self.owner[k, r :: self.d.period[ue]] = -1
        self.occasions[ue].remove(occasion)

    def refresh(self, ue: int) -> None:
        self.bits[ue] = self.d.accumulated(ue, self.occasions[ue])
        self.misses[ue] = self.d.misses(ue, self.bits[ue])

    def free_positions(self, ue: int) -> list[Occasion]:
        return [
            (r, k)
            for k in range(self.d.num_subchannels)
            for r in range(self.d.period[ue])
            if self.fits(ue, (r, k))
        ]


@dataclass(frozen=True)
class _Move:
    kind: str  # "add", "relocate" or "transfer"
    ue: int
    target: Occasion
    donor: int | None = None  # the UE giving up ``released`` (UE itself when relocating)
    released: Occasion | None = None


def _moves(grid: _Grid, ue: int) -> Iterator[_Move]:
    """Candidate moves for ``ue`` in the design order (§4.3), dominated ones skipped.

    Positions are enumerated in (k, r) order. Skipped without evaluation, since
    they can never be a strict improvement once the add moves of the same state
    were rejected (more occasions never raise a UE's misses): relocations to a
    position that was already free for the UE, and transfers whose target does
    not use a unit the donor released. Every candidate is yielded with the
    grid in its current state.
    """
    d = grid.d
    period = d.period[ue]
    for target in grid.free_positions(ue):
        yield _Move("add", ue, target)
    for released in sorted(grid.occasions[ue], key=lambda o: (o[1], o[0])):
        r0, k0 = released
        for k in range(d.num_subchannels):
            if k != k0 and np.all(grid.owner[k, r0::period] == -1):
                yield _Move("relocate", ue, (r0, k), ue, released)
    donors = sorted((j for j in range(d.num_ues) if j != ue), key=lambda j: (grid.misses[j], j))
    for donor in donors:
        for released in sorted(grid.occasions[donor], key=lambda o: (o[1], o[0])):
            r0, k0 = released
            freed = range(r0, d.hyperperiod, d.period[donor])
            grid.release(donor, released)
            targets = [(r, k0) for r in sorted({tau % period for tau in freed})]
            targets = [o for o in targets if grid.fits(ue, o)]
            grid.take(donor, released)
            for target in targets:
                yield _Move("transfer", ue, target, donor, released)


def local_search(
    d: Deployment, occasions: Sequence, max_evaluations: int | None = SEARCH_EVALUATIONS
) -> tuple[list[list[Occasion]], list[int], dict]:
    """Stage 2 (design §4.3): first-improvement leximin search on the true misses.

    UEs are taken from the most misses down (ties: lower index); UEs without
    misses are skipped (no move can lower them). For each: add, relocate,
    transfer (donors from the fewest misses up). The first candidate with a
    strictly smaller leximin key is applied and the pass restarts at the worst
    UE. Stops at a pass without improvement (local optimum) or after
    ``max_evaluations`` candidate evaluations.
    """
    grid = _Grid(d, occasions)
    evaluations = 0
    accepted = {"add": 0, "relocate": 0, "transfer": 0}
    began = time.perf_counter()
    local_optimum = False
    exhausted = False
    while not exhausted:
        current = leximin_key(grid.misses)
        order = sorted(range(d.num_ues), key=lambda u: (-grid.misses[u], u))
        improved = False
        for ue in order:
            if grid.misses[ue] == 0:
                break
            removal = (None, None, None)
            for move in _moves(grid, ue):
                if max_evaluations is not None and evaluations >= max_evaluations:
                    exhausted = True
                    break
                evaluations += 1
                misses = list(grid.misses)
                gain = grid.bits[ue] + d.bits(ue, move.target)
                if move.released is not None:
                    if removal[:2] != (move.donor, move.released):
                        removal = (move.donor, move.released, d.bits(move.donor, move.released))
                    if move.donor == ue:
                        gain = gain - removal[2]
                    else:
                        misses[move.donor] = d.misses(
                            move.donor, grid.bits[move.donor] - removal[2]
                        )
                misses[ue] = d.misses(ue, gain)
                if leximin_key(misses) < current:
                    if move.released is not None:
                        grid.release(move.donor, move.released)
                        grid.refresh(move.donor)
                    grid.take(ue, move.target)
                    grid.refresh(ue)
                    accepted[move.kind] += 1
                    improved = True
                    break
            if improved or exhausted:
                break
        if not improved and not exhausted:
            local_optimum = True
            break
    record = {
        "evaluations": evaluations,
        "accepted": accepted,
        "local_optimum": local_optimum,
        "seconds": time.perf_counter() - began,
    }
    return [sorted(o) for o in grid.occasions], list(grid.misses), record


# ------------------------------------------------------------------ both stages


def blocked_ues(d: Deployment, occasions: Sequence) -> int:
    """UEs short of P_i occasions with none placeable while P_i's share of units is free."""
    grid = _Grid(d, occasions)
    free = int(np.count_nonzero(grid.owner == -1))
    return sum(
        len(o) < p and free >= d.hyperperiod // p and not grid.free_positions(ue)
        for ue, (o, p) in enumerate(zip(grid.occasions, d.period))
    )


@dataclass(frozen=True)
class OptimizedGrants:
    allocation: GrantAllocation  # outage: predicted outage on bank A
    record: dict


def optimize_grants(
    config: PhysicalConfig,
    snr: NDArray[np.float64],
    packet_bits: NDArray[np.float64],
    period: NDArray[np.int64],
    phase: NDArray[np.int64],
    bank: NDArray[np.float64],
    *,
    max_nodes: int | None = MIP_MAX_NODES,
    max_evaluations: int | None = SEARCH_EVALUATIONS,
    time_limit_s: float = TIME_LIMIT_S,
) -> OptimizedGrants:
    """CG-opt allocation on ``bank`` (bank A in the policy), design §4.

    Candidate 0 is the CG greedy on the same bank; stage 1 the count/packing
    MILP; stage 2 starts from the one with the smaller leximin key (ties:
    candidate 0, which leaves fewer units idle) and never makes it worse.
    Units left unassigned stay idle.
    """
    began = time.perf_counter()
    d = deployment(config, snr, packet_bits, period, phase, bank)
    greedy = allocate_grants(config, snr, packet_bits, period, phase, bank)
    candidates = {"greedy": [sorted(o) for o in greedy.occasions]}
    curves = [outage_curve(d, ue) for ue in range(d.num_ues)]
    milp, solves = solve_counts(d, curves, max_nodes, time_limit_s)
    if milp is not None:
        candidates["milp"] = milp
    misses = {name: d.all_misses(o) for name, o in candidates.items()}
    start = min(candidates, key=lambda name: (leximin_key(misses[name]), name != "greedy"))
    occasions, final, search = local_search(d, candidates[start], max_evaluations)
    grid = _Grid(d, occasions)
    allocation = GrantAllocation(
        occasions=tuple(tuple(o) for o in occasions),
        outage=tuple(m / d.samples for m in final),
        need=greedy.need,
        order=(),
        hyperperiod=d.hyperperiod,
        assigned_share=float(np.mean(grid.owner != -1)),
        blocked=blocked_ues(d, occasions),
    )
    record = {
        "misses": {name: m for name, m in misses.items()} | {"final": final},
        "start": start,
        "stage1": solves,
        "stage2": search,
        "budget": {
            "max_nodes": max_nodes,
            "max_evaluations": max_evaluations,
            "time_limit_s": time_limit_s,
        },
        "seconds": time.perf_counter() - began,
    }
    return OptimizedGrants(allocation, record)
