"""Optimized static configured grant CG-opt (design: configured-grant-optimized_20261010.md)."""

import dataclasses
import itertools
import math
import os
import signal
from pathlib import Path

import numpy as np
import pytest

from dsa_period import cg_opt, splits_v2
from dsa_period.capacity import run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.rules_v2 import (
    ConfiguredGrantOptPolicy,
    ConfiguredGrantPolicy,
    allocate_grants,
    fading_bank,
    make_policy_v2,
)

CONFIG = Path(__file__).parents[1] / "configs" / "scenario-v2.toml"


def short(n: int, horizon: int = 300):
    scenario = load_scenario(CONFIG, n)
    return dataclasses.replace(scenario, horizon_slots=horizon, warmup_slots=40)


def inputs(scenario, instance):
    topology, traffic = scenario.instance(instance)
    snr = scenario.menu.power_w[-1] * topology.ue_ap_gain_linear / scenario.physical.noise_power_w
    return (
        snr,
        np.asarray(traffic.packet_bits, dtype=np.float64),
        np.asarray(traffic.period_slots),
        np.asarray(traffic.phase_slots),
    )


def small_bank(scenario, samples=2000):
    fading, physical = scenario.fading, scenario.physical
    return fading_bank(
        physical.num_subchannels, fading.doppler_hz, fading.sinusoids, physical.slot_duration_s,
        40, samples, cg_opt.CG_OPT_BANK_SEED,
    )  # fmt: skip


def synthetic(periods, phases, num_subchannels, samples=400, seed=7, snr=3.0, bits=2500.0):
    """A deployment on an i.i.d. Rayleigh bank (no time correlation)."""
    physical = load_scenario(CONFIG, len(periods)).physical
    bank = np.random.default_rng(seed).exponential(size=(samples, max(periods), num_subchannels))
    return cg_opt.Deployment(
        physical,
        np.full(len(periods), snr),
        np.full(len(periods), bits),
        tuple(periods),
        tuple(phases),
        bank,
    )


def check_grid(d, occasions):
    owner = np.full((d.num_subchannels, d.hyperperiod), -1)
    for ue, grants in enumerate(occasions):
        residues = [r for r, _ in grants]
        assert len(set(residues)) == len(residues) <= d.period[ue]
        for r, k in grants:
            slots = np.arange(r, d.hyperperiod, d.period[ue])
            assert np.all(owner[k, slots] == -1), "a resource unit is granted twice"
            owner[k, slots] = ue
    return owner


def test_leximin_compares_the_worst_ue_first():
    key = cg_opt.leximin_key
    assert key([3, 9, 1]) == (9, 3, 1)
    assert key([9, 2, 2]) < key([1, 9, 3])  # same worst, better second
    assert key([8, 8, 8]) < key([9, 0, 0])
    assert not key([5, 1]) < key([1, 5])  # a permutation is a tie


# ------------------------------------------------------------------ stage 1


def brute_force(d, curves, block_slots):
    """min (max_i c_i(n_i), sum_i c_i(n_i)) over every allocation the MILP allows."""
    num_subchannels = d.num_subchannels

    def options(ue):
        period = d.period[ue]
        blocks = max(1, period // block_slots)
        for choice in itertools.product(range(-1, num_subchannels), repeat=period):
            grants = [(r, k) for r, k in enumerate(choice) if k >= 0]
            n = len(grants)
            per = np.bincount([k for _, k in grants], minlength=num_subchannels)
            if per.min() < n // num_subchannels or per.max() > -(-n // num_subchannels):
                continue
            cap = -(-(-(-n // num_subchannels)) // blocks)
            cells = [
                (k, min(((r - d.phase[ue]) % period) // block_slots, blocks - 1)) for r, k in grants
            ]
            if blocks > 1 and any(cells.count(c) > cap for c in set(cells)):
                continue
            yield grants

    best = None
    for combination in itertools.product(*(list(options(ue)) for ue in range(d.num_ues))):
        owner = np.zeros((num_subchannels, d.hyperperiod), dtype=int)
        for ue, grants in enumerate(combination):
            for r, k in grants:
                owner[k, r :: d.period[ue]] += 1
        if owner.max() > 1:
            continue
        values = [int(curves[ue][len(g)]) for ue, g in enumerate(combination)]
        key = (max(values), sum(values))
        best = key if best is None or key < best else best
    return best


@pytest.mark.parametrize(
    "periods, phases, num_subchannels",
    [((4, 4), (0, 1), 2), ((2, 4, 4), (1, 0, 3), 2), ((4, 2, 2), (2, 0, 1), 1)],
)
def test_stage1_finds_the_brute_force_optimum(periods, phases, num_subchannels):
    d = synthetic(periods, phases, num_subchannels)
    rng = np.random.default_rng(sum(periods) + num_subchannels)
    curves = []
    for p in periods:  # arbitrary non-increasing miss curves starting at S
        steps = np.sort(rng.integers(0, d.samples, size=p))[::-1]
        curves.append(np.concatenate([[d.samples], steps]))
    occasions, records = cg_opt.solve_counts(d, curves, block_slots=2)
    assert [r["status"] for r in records] == ["Optimal", "Optimal"]
    check_grid(d, occasions)
    values = [int(curves[ue][len(o)]) for ue, o in enumerate(occasions)]
    assert (max(values), sum(values)) == brute_force(d, curves, block_slots=2)
    assert records[0]["max_misses"] == max(values)
    assert records[1]["sum_misses"] == sum(values)


def test_stage1_balances_subchannels_and_spreads_in_time():
    scenario = short(8)
    snr, bits, period, phase = inputs(scenario, 3)
    d = cg_opt.deployment(scenario.physical, snr, bits, period, phase, small_bank(scenario))
    curves = [cg_opt.outage_curve(d, ue) for ue in range(d.num_ues)]
    for curve, p in zip(curves, d.period):
        assert len(curve) == p + 1 and curve[0] == d.samples
        assert np.all(np.diff(curve) <= 0)
    occasions, _ = cg_opt.solve_counts(d, curves)
    check_grid(d, occasions)
    for ue, grants in enumerate(occasions):
        n, p = len(grants), d.period[ue]
        per = np.bincount([k for _, k in grants], minlength=4)
        assert n // 4 <= per.min() and per.max() <= -(-n // 4)
        blocks = p // 10
        for k in range(4):
            cells = [((r - d.phase[ue]) % p) // 10 for r, kk in grants if kk == k]
            if blocks > 1:
                assert max(np.bincount(cells, minlength=blocks)) <= -(-(-(-n // 4)) // blocks)


# ------------------------------------------------------------------ stage 2


def improving_moves(d, occasions):
    """Every add, relocate and transfer, unpruned, that lowers the leximin key."""
    misses = d.all_misses(occasions)
    current = cg_opt.leximin_key(misses)
    found = []

    def fits(grants, ue, r, k, owner):
        return all(rr != r for rr, _ in grants[ue]) and np.all(owner[k, r :: d.period[ue]] == -1)

    def owner_of(grants):
        owner = np.full((d.num_subchannels, d.hyperperiod), -1)
        for ue, g in enumerate(grants):
            for r, k in g:
                owner[k, r :: d.period[ue]] = ue
        return owner

    positions = [[(r, k) for k in range(d.num_subchannels) for r in range(p)] for p in d.period]
    for ue in range(d.num_ues):
        for donor in range(d.num_ues):
            for released in [None, *occasions[donor]]:
                if released is None and donor != ue:
                    continue  # plain adds are listed once, with donor == ue
                grants = [list(g) for g in occasions]
                if released is not None:
                    grants[donor].remove(released)
                owner = owner_of(grants)
                for r, k in positions[ue]:
                    if (r, k) == released and donor == ue or not fits(grants, ue, r, k, owner):
                        continue
                    trial = [list(g) for g in grants]
                    trial[ue].append((r, k))
                    if cg_opt.leximin_key(d.all_misses(trial)) < current:
                        found.append((ue, donor, released, (r, k)))
    return found


def test_local_search_ends_at_a_local_optimum_of_the_full_neighborhood():
    d = synthetic((2, 4, 4), (1, 0, 3), 2, samples=300)
    # Start: UEs 0 and 1 fill the grid, UE 2 holds nothing.
    start = [[(0, 0), (1, 1)], [(1, 0), (3, 0), (0, 1), (2, 1)], []]
    occasions, misses, record = cg_opt.local_search(d, start)
    check_grid(d, occasions)
    assert record["local_optimum"] and sum(record["accepted"].values()) > 0
    assert misses == d.all_misses(occasions)
    assert cg_opt.leximin_key(misses) < cg_opt.leximin_key(d.all_misses(start))
    assert occasions[2]  # the starved UE was given occasions
    assert improving_moves(d, occasions) == []


def test_local_search_never_worsens_and_stops_at_the_budget():
    d = synthetic((2, 4, 4), (1, 0, 3), 2, samples=300)
    start = [[(0, 0), (1, 1)], [(1, 0), (3, 0), (0, 1), (2, 1)], []]
    _, _, full = cg_opt.local_search(d, start)
    assert full["evaluations"] > 3
    keys = [cg_opt.leximin_key(d.all_misses(start))]
    for budget in range(full["evaluations"] + 1):
        occasions, misses, record = cg_opt.local_search(d, start, max_evaluations=budget)
        assert record["evaluations"] == budget or record["local_optimum"]
        assert record["local_optimum"] == (budget >= full["evaluations"])
        keys.append(cg_opt.leximin_key(misses))
        assert misses == d.all_misses(occasions)
    assert all(b <= a for a, b in zip(keys, keys[1:]))  # more budget is never worse


# ------------------------------------------------------------------ both stages


def test_optimize_is_feasible_deterministic_and_no_worse_than_its_candidates():
    scenario = short(10)
    bank = small_bank(scenario)
    snr, bits, period, phase = inputs(scenario, 5)
    first = cg_opt.optimize_grants(scenario.physical, snr, bits, period, phase, bank)
    second = cg_opt.optimize_grants(scenario.physical, snr, bits, period, phase, bank)
    assert first.allocation == second.allocation
    d = cg_opt.deployment(scenario.physical, snr, bits, period, phase, bank)
    owner = check_grid(d, first.allocation.occasions)
    record = first.record
    final = d.all_misses(first.allocation.occasions)
    assert final == record["misses"]["final"]
    assert first.allocation.outage == tuple(m / d.samples for m in final)
    for name in ("greedy", "milp"):
        assert cg_opt.leximin_key(final) <= cg_opt.leximin_key(record["misses"][name])
    greedy = allocate_grants(scenario.physical, snr, bits, period, phase, bank)
    assert record["misses"]["greedy"] == d.all_misses(greedy.occasions)
    assert first.allocation.assigned_share == pytest.approx(np.mean(owner != -1))
    assert record["stage2"]["local_optimum"]
    assert record["budget"] == {
        "max_nodes": cg_opt.MIP_MAX_NODES,
        "max_evaluations": cg_opt.SEARCH_EVALUATIONS,
        "time_limit_s": cg_opt.TIME_LIMIT_S,
    }


def test_limits_are_recorded_and_fall_back_to_the_greedy():
    scenario = short(8)
    bank = small_bank(scenario)
    snr, bits, period, phase = inputs(scenario, 29)
    args = (scenario.physical, snr, bits, period, phase, bank)
    capped = cg_opt.optimize_grants(*args, max_nodes=1, max_evaluations=2)
    assert capped.record["stage2"]["evaluations"] == 2
    assert not capped.record["stage2"]["local_optimum"]
    assert all(s["nodes"] <= 1 for s in capped.record["stage1"])
    # A solve at the time limit gives no stage-1 candidate: start from the greedy.
    timed_out = cg_opt.optimize_grants(*args, time_limit_s=0.0, max_evaluations=0)
    assert [s["time_limit_hit"] for s in timed_out.record["stage1"]] == [True]
    assert "milp" not in timed_out.record["misses"]
    assert timed_out.record["start"] == "greedy"
    greedy = allocate_grants(*args)
    assert timed_out.allocation.occasions == tuple(tuple(sorted(o)) for o in greedy.occasions)
    check_grid(cg_opt.deployment(*args), timed_out.allocation.occasions)


def test_a_solve_overrunning_the_limit_is_killed(monkeypatch):
    scenario = short(8)
    snr, bits, period, phase = inputs(scenario, 29)
    d = cg_opt.deployment(scenario.physical, snr, bits, period, phase, small_bank(scenario))
    curves = [cg_opt.outage_curve(d, ue) for ue in range(d.num_ues)]
    monkeypatch.setattr(cg_opt, "KILL_GRACE_S", 0.0)
    occasions, records = cg_opt.solve_counts(d, curves, time_limit_s=0.0)
    assert occasions is None
    assert records[0]["status"] == "Killed at the time limit" and records[0]["nodes"] is None
    assert records[0]["time_limit_hit"] and records[0]["seconds"] < 5


def test_a_crashed_solve_is_retried_once_without_restarts(monkeypatch):
    scenario = short(8)
    snr, bits, period, phase = inputs(scenario, 29)
    args = (scenario.physical, snr, bits, period, phase, small_bank(scenario))
    d = cg_opt.deployment(*args)
    curves = [cg_opt.outage_curve(d, ue) for ue in range(d.num_ues)]
    _, intact = cg_opt.solve_counts(d, curves)
    run = cg_opt._Milp._run

    def crash_with_restarts(self, *solve_args):
        if solve_args[-1]:  # allow_restart
            os.kill(os.getpid(), signal.SIGKILL)
        return run(self, *solve_args)

    monkeypatch.setattr(cg_opt._Milp, "_run", crash_with_restarts)
    occasions, records = cg_opt.solve_counts(d, curves)
    assert occasions is not None and len(records) == 2
    for record in records:
        assert record["crashed_attempt"]["status"] == "Crashed"
        assert record["crashed_attempt"]["exit_code"] == -signal.SIGKILL
        assert not record["crashed"] and not record["allow_restart"]
    assert records[0]["status"] == "Optimal"
    assert records[0]["max_misses"] == intact[0]["max_misses"]

    def always_crash(self, *solve_args):
        os.kill(os.getpid(), signal.SIGKILL)

    monkeypatch.setattr(cg_opt._Milp, "_run", always_crash)
    occasions, records = cg_opt.solve_counts(d, curves)
    assert occasions is None and len(records) == 1
    assert records[0]["status"] == "Crashed" and records[0]["crashed_attempt"]["crashed"]
    fallback = cg_opt.optimize_grants(*args, max_evaluations=0)
    assert fallback.record["start"] == "greedy" and "milp" not in fallback.record["misses"]


def test_an_error_in_the_solve_child_is_raised_not_taken_for_a_crash(monkeypatch):
    scenario = short(8)
    snr, bits, period, phase = inputs(scenario, 29)
    d = cg_opt.deployment(scenario.physical, snr, bits, period, phase, small_bank(scenario))
    curves = [cg_opt.outage_curve(d, ue) for ue in range(d.num_ues)]

    def broken(self, *solve_args):
        raise ValueError("broken model")

    monkeypatch.setattr(cg_opt._Milp, "_run", broken)
    with pytest.raises(ValueError, match="broken model"):
        cg_opt.solve_counts(d, curves)


def test_opt_policy_transmits_like_cg_and_reports_both_banks():
    scenario = short(10)
    policy = make_policy_v2("configured_grant_opt", scenario.physical, scenario.menu)
    assert isinstance(policy, ConfiguredGrantOptPolicy) and policy.name == "configured_grant_opt"
    topology, traffic = scenario.instance(0)
    policy.configure(topology.ue_ap_gain_linear, traffic, scenario.fading)
    env = scenario.make_env(topology, traffic, splits_v2.fading_seeds("dev", 10, 0, 0))
    grants = [dict(o) for o in policy.allocation.occasions]
    policy.reset(splits_v2.execution_generators("dev", 10, 0, 0))
    observations = env.reset()
    top = scenario.menu.num_levels - 1
    sent = 0
    while not env.truncated:
        t = env.elapsed_steps
        actions = policy.act(observations)
        for ue, obs in enumerate(observations):
            channel = grants[ue].get(t % traffic.period_slots[ue])
            on = channel is not None and obs.remaining_bits > 0
            assert actions[ue].tolist() == ([channel, top] if on else [0, 0])
            sent += on
        busy = actions[actions[:, 1] > 0, 0]
        assert len(busy) == len(set(busy.tolist()))
        observations = env.step(actions).observations
    assert sent > 0
    diagnostics = policy.episode_diagnostics()
    assert diagnostics["predicted_outage"] == list(policy.allocation.outage)
    cg = ConfiguredGrantPolicy(scenario.physical, scenario.menu)
    cg.configure(topology.ue_ap_gain_linear, traffic, scenario.fading)
    b = cg_opt.deployment(
        scenario.physical, cg._snr, np.asarray(traffic.packet_bits, dtype=np.float64),
        np.asarray(traffic.period_slots), np.asarray(traffic.phase_slots),
        fading_bank(4, scenario.fading.doppler_hz, scenario.fading.sinusoids,
                    scenario.physical.slot_duration_s, max(traffic.period_slots)),
    )  # fmt: skip
    assert diagnostics["predicted_outage_bank_b"] == cg_opt.predicted_outage(
        b, policy.allocation.occasions
    )
    # The same definition as CG's own prediction (summation order aside).
    np.testing.assert_allclose(
        cg.allocation.outage, cg_opt.predicted_outage(b, cg.allocation.occasions), atol=1e-4
    )
    assert set(diagnostics["solver"]) >= {"misses", "start", "stage1", "stage2", "budget"}
    result = run_capacity_episode(scenario, policy, "dev", 0)
    assert result["policy_diagnostics"]["solver"]["start"] in ("greedy", "milp")
    assert math.isclose(sum(result["policy_diagnostics"]["occasions"]), sum(map(len, grants)))
