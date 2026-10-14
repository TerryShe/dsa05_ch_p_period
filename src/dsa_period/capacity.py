"""Reliability and capacity evaluation of scenario v2.0 under protocol v2.

Definitions (docs/research/scenario-v2.md §5):

* evaluated packets: deadline in (W, H]; UE i has M_i of them (fixed by traffic);
* allowance A_i = floor(eps * M_i); V_i counts evaluated packets not completed
  on time, undeliverable ones included;
* an episode is feasible at eps iff V_i <= A_i for every UE;
* a packet is undeliverable iff max power on the best subchannel in every slot
  of its window, without interference, cannot deliver it; an episode in which
  some UE has more undeliverable packets than its allowance is infeasible for
  every policy;
* F(N): share of feasible episodes; capacity N*: the grid point before the first
  one with F < 0.95 (``capacity``), which preserves the order of F curves.

Energy is not evaluated; transmit power and duty are kept as diagnostics only.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from dsa_period import splits_v2
from dsa_period.config import Scenario
from dsa_period.evaluation import _bootstrap, _interval
from dsa_period.physics import shannon_bits
from dsa_period.policies import Policy
from dsa_period.survival import TOLERANCES as SURVIVAL_TOLERANCES
from dsa_period.survival import service_record
from dsa_period.traffic import TrafficPattern

EPSILONS = (0.01, 0.05)
FEASIBLE_EPISODE_SHARE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_930


def eps_key(eps: float) -> str:
    return f"{eps:g}"


def allowance(eps: float, packets: int) -> int:
    # Integer test avoids float noise: floor(eps * M) with eps * M integral here.
    return math.floor(eps * packets + 1e-9)


def evaluated_packets(traffic: TrafficPattern, warmup: int, horizon: int) -> dict:
    """Arrays (ue, arrival, deadline, bits) of all packets with warmup < deadline <= horizon."""
    rows = []
    for ue in range(traffic.num_ues):
        period, phase = int(traffic.period_slots[ue]), int(traffic.phase_slots[ue])
        first = max((warmup - phase) // period, 0)  # smallest n with phase + (n+1)P > warmup
        arrival = np.arange(phase + first * period, horizon, period)
        arrival = arrival[(arrival + period > warmup) & (arrival + period <= horizon)]
        rows.append((np.full(arrival.size, ue), arrival))
    ue = np.concatenate([r[0] for r in rows]).astype(np.int64)
    arrival = np.concatenate([r[1] for r in rows]).astype(np.int64)
    deadline = arrival + traffic.period_slots[ue]
    return {"ue": ue, "arrival": arrival, "deadline": deadline, "bits": traffic.packet_bits[ue]}


def best_alone_bits(scenario: Scenario, gains: NDArray[np.float64]) -> NDArray[np.float64]:
    """(T, N) mutual information per slot at max power on the best subchannel, alone."""
    top = scenario.menu.power_w[-1]
    return shannon_bits(
        scenario.physical, top * gains.max(axis=2) / scenario.physical.noise_power_w
    )


def undeliverable(packets: Mapping, best_bits: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Per packet: even the best subchannel at max power in every window slot falls short.

    ``best_bits`` covers slots [0, T) with T >= the latest deadline.
    """
    cumulative = np.vstack([np.zeros(best_bits.shape[1]), np.cumsum(best_bits, axis=0)])
    ue, arrival, deadline = packets["ue"], packets["arrival"], packets["deadline"]
    capacity = cumulative[deadline, ue] - cumulative[arrival, ue]
    return capacity < packets["bits"]


def run_capacity_episode(
    scenario: Scenario, policy: Policy, split: str, instance: int, repeat: int = 0
) -> dict:
    """One protocol-v2 episode: nested instance, per-UE execution and fading seeds."""
    n = scenario.physical.num_ues
    rngs = splits_v2.execution_generators(split, n, instance, repeat)
    topology, traffic = scenario.instance(instance)
    env = scenario.make_env(topology, traffic, splits_v2.fading_seeds(split, n, instance, repeat))
    configure = getattr(policy, "configure", None)
    if configure is not None:  # semi-static references (N_CG): mean gains, TSCAI, statistics
        configure(env.channel.mean_gain_linear, traffic, scenario.fading)
    policy.reset(rngs)
    observations = env.reset()
    while not env.truncated:
        observations = env.step(policy.act(observations)).observations
    warmup, horizon = scenario.warmup_slots, scenario.horizon_slots
    packets = evaluated_packets(traffic, warmup, horizon)
    best = best_alone_bits(scenario, env.channel.gain_block(0, horizon))
    lost = undeliverable(packets, best)
    count = np.bincount(packets["ue"], minlength=n)
    missed = np.zeros(n, dtype=np.int64)
    scored: list[list[tuple[int, bool]]] = [[] for _ in range(n)]
    for job in env.job_records():
        if warmup < job["deadline_slot"] <= horizon:
            scored[job["ue"]].append((job["deadline_slot"], bool(job["success"])))
            if not job["success"]:
                missed[job["ue"]] += 1
    # EXP-DSA05-012: per UE, the indices (in deadline order) of its missed scored packets,
    # and outages / unavailable slots at survival times of 1 and 2 intervals (design §2).
    missed_packets = [[k for k, (_, ok) in enumerate(sorted(rows)) if not ok] for rows in scored]
    jobs = env.job_records()
    service = {str(s): service_record(jobs, n, warmup, horizon, s) for s in SURVIVAL_TOLERANCES}
    impossible = np.bincount(packets["ue"], weights=lost, minlength=n).astype(np.int64)
    energy = env.energy_by_slot()[warmup:horizon]
    window_s = (horizon - warmup) * scenario.physical.slot_duration_s
    rate = missed / count
    feasible, hopeless = {}, {}
    for eps in EPSILONS:
        allowed = np.array([allowance(eps, int(m)) for m in count])
        feasible[eps_key(eps)] = bool(np.all(missed <= allowed))
        hopeless[eps_key(eps)] = bool(np.any(impossible > allowed))
    result = {
        "num_ues": n,
        "instance": instance,
        "repeat": repeat,
        "packets": count.tolist(),
        "violations": missed.tolist(),
        "missed_packets": missed_packets,
        "survival": service,
        "survival_window": horizon - warmup,
        "undeliverable": impossible.tolist(),
        "violation_rate": rate.tolist(),
        "worst_violation_rate": float(rate.max()),
        "feasible": feasible,
        "infeasible_for_all": hopeless,
        "class_index": traffic.class_index.tolist(),
        "distance_m": topology.ue_ap_distances_m.tolist(),
        "shadowing_db": topology.shadowing_db.tolist(),
        # Diagnostics only (energy is not an objective in v2.0).
        "mean_power_mw": (1e3 * energy.sum(axis=0) / window_s).tolist(),
        "duty": (energy > 0).mean(axis=0).tolist(),
    }
    diagnostics = getattr(policy, "episode_diagnostics", None)
    if diagnostics is not None:
        result["policy_diagnostics"] = diagnostics()
    return result


# --------------------------------------------------------------- aggregation
def capacity(
    grid: Sequence[int], shares: Sequence[float], threshold: float = FEASIBLE_EPISODE_SHARE
) -> dict:
    """First-drop rule: the grid point before the first N with F(N) < threshold.

    ``n_star`` is 0 when the smallest grid point already fails ("below the
    grid"); ``at_least`` is True when no grid point fails.
    """
    if list(grid) != sorted(set(grid)) or len(grid) != len(shares) or not grid:
        raise ValueError("grid must be strictly increasing and match shares")
    for position, (n, share) in enumerate(zip(grid, shares, strict=True)):
        if share < threshold:
            return {"n_star": grid[position - 1] if position else 0, "at_least": False}
    return {"n_star": grid[-1], "at_least": True}


def capacity_label(point: dict, grid: Sequence[int]) -> str:
    """'<2' below the grid, '>=32' at its top, otherwise N* itself."""
    if point["at_least"]:
        return f">={point['n_star']}"
    return f"<{min(grid)}" if point["n_star"] == 0 else str(point["n_star"])


def _feasible_matrix(
    episodes_by_seed: Mapping[str, Sequence[dict]], key: str, field: str = "feasible"
) -> tuple[list[int], NDArray]:
    seeds = sorted(episodes_by_seed)
    instances = sorted({e["instance"] for e in episodes_by_seed[seeds[0]]})
    values = np.empty((len(seeds), len(instances)))
    for s, seed in enumerate(seeds):
        cells: dict[int, list[float]] = {}
        for episode in episodes_by_seed[seed]:
            cells.setdefault(episode["instance"], []).append(float(episode[field][key]))
        if sorted(cells) != instances:
            raise ValueError("every training seed must cover the same instances")
        values[s] = [np.mean(cells[i]) for i in instances]
    return instances, values


def summarize(
    results: Mapping[int, Mapping[str, Sequence[dict]]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """F(N), per-UE violation statistics and N* of one method at every eps.

    ``results[N][training_seed]`` lists that seed's episodes at N (a rule or a
    reference has one pseudo-seed). Instances must be identical across N and
    seeds: the capacity bootstrap resamples instances and seeds jointly over N.
    """
    grid = sorted(results)
    seeds = sorted(results[grid[0]])
    if any(sorted(results[n]) != seeds for n in grid):
        raise ValueError("every N needs the same training seeds")
    summary: dict = {"grid": grid, "training_seeds": seeds, "per_eps": {}}
    for eps in EPSILONS:
        key = eps_key(eps)
        matrices, instances = {}, None
        rows = []
        for n in grid:
            found, values = _feasible_matrix(results[n], key)
            if instances is None:
                instances = found
            elif found != instances:
                raise ValueError("every N needs the same instances")
            matrices[n] = values
            _, hopeless = _feasible_matrix(results[n], key, "infeasible_for_all")
            row = {
                "num_ues": n,
                "feasible_share": float(values.mean()),
                "ci95": _interval(_bootstrap(values, None, resamples, seed)),
                "infeasible_for_all_share": float(hopeless.mean()),
            }
            episodes = [e for s in seeds for e in results[n][s]]
            if "violation_rate" in episodes[0]:  # methods, not LP references
                rates = np.concatenate([e["violation_rate"] for e in episodes])
                row["per_ue_violation_rate"] = {
                    "p50": float(np.percentile(rates, 50)),
                    "p95": float(np.percentile(rates, 95)),
                    "max": float(rates.max()),
                }
                row["worst_ue_violation_rate_mean"] = float(
                    np.mean([e["worst_violation_rate"] for e in episodes])
                )
            rows.append(row)
        point = capacity(grid, [r["feasible_share"] for r in rows])
        summary["per_eps"][key] = {
            "rows": rows,
            "capacity": point,
            "capacity_bootstrap": _capacity_bootstrap(grid, matrices, resamples, seed),
        }
    summary["instances"] = len(instances)
    return summary


def _capacity_bootstrap(
    grid: list[int], matrices: Mapping[int, NDArray], resamples: int, seed: int
) -> dict:
    """Distribution of N* when seeds and instances are resampled jointly over N."""
    rng = np.random.default_rng(seed)
    stacked = np.stack([matrices[n] for n in grid])  # (grid, seeds, instances)
    _, num_seeds, num_instances = stacked.shape
    counts: dict[str, int] = {}
    for _ in range(resamples):
        seeds = rng.integers(num_seeds, size=num_seeds)
        inst = rng.integers(num_instances, size=num_instances)
        shares = stacked[:, seeds][:, :, inst].mean(axis=(1, 2))
        point = capacity(grid, shares.tolist())
        label = f">={point['n_star']}" if point["at_least"] else str(point["n_star"])
        counts[label] = counts.get(label, 0) + 1
    return {label: count / resamples for label, count in sorted(counts.items())}


def select_per_n(summaries: Mapping[str, dict], specs: Sequence[str]) -> dict[int, dict]:
    """Per-N choice among ``specs`` (protocol v2 §5, scenario v2.0 §8).

    Highest F at eps = 0.01; ties by higher F at 0.05, then by the lower mean
    worst-UE violation rate per episode, then by the order of ``specs``.
    ``summaries[spec]`` is the output of ``summarize`` for that spec.
    """
    if not specs:
        raise ValueError("nothing to select from")

    def rows(spec: str, key: str) -> dict[int, dict]:
        return {r["num_ues"]: r for r in summaries[spec]["per_eps"][key]["rows"]}

    grids = {spec: sorted(rows(spec, eps_key(EPSILONS[0]))) for spec in specs}
    grid = grids[specs[0]]
    if any(g != grid for g in grids.values()):
        raise ValueError("every spec needs the same grid")
    chosen = {}
    for n in grid:

        def key(position: int) -> tuple:
            spec = specs[position]
            main, loose = rows(spec, eps_key(0.01))[n], rows(spec, eps_key(0.05))[n]
            return (
                -main["feasible_share"],
                -loose["feasible_share"],
                main["worst_ue_violation_rate_mean"],
                position,
            )

        best = min(range(len(specs)), key=key)
        main = rows(specs[best], eps_key(0.01))[n]
        chosen[n] = {
            "spec": specs[best],
            "feasible_share": {
                eps_key(eps): rows(specs[best], eps_key(eps))[n]["feasible_share"]
                for eps in EPSILONS
            },
            "worst_ue_violation_rate_mean": main["worst_ue_violation_rate_mean"],
        }
    return chosen
