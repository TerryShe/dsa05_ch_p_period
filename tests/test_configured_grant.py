"""Configured-grant reference N_CG (design: configured-grant-baseline_20261009.md)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from dsa_period import splits_v2
from dsa_period.capacity import run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.physics import shannon_bits
from dsa_period.rules_v2 import (
    ConfiguredGrantPolicy,
    allocate_grants,
    fading_bank,
    make_policy_v2,
    place_occasion,
)

CONFIG = Path(__file__).parents[1] / "configs" / "scenario-v2.toml"


def short(n: int, horizon: int = 300):
    scenario = load_scenario(CONFIG, n)
    return dataclasses.replace(scenario, horizon_slots=horizon, warmup_slots=40)


def bank_of(scenario):
    fading, physical = scenario.fading, scenario.physical
    return fading_bank(
        physical.num_subchannels, fading.doppler_hz, fading.sinusoids, physical.slot_duration_s, 40
    )


def configured(scenario, instance=0, **params):
    topology, traffic = scenario.instance(instance)
    policy = ConfiguredGrantPolicy(scenario.physical, scenario.menu, **params)
    policy.configure(topology.ue_ap_gain_linear, traffic, scenario.fading)
    return policy, topology, traffic


def grid_of(occasions, period, hyperperiod=40, num_subchannels=4):
    owner = np.full((num_subchannels, hyperperiod), -1)
    for ue, grants in enumerate(occasions):
        for r, k in grants:
            slots = np.arange(r, hyperperiod, period[ue])
            assert np.all(owner[k, slots] == -1), "a resource unit is granted twice"
            owner[k, slots] = ue
    return owner


def test_bank_gives_the_rayleigh_outage_of_one_occasion():
    scenario = short(4)
    bank = bank_of(scenario)
    assert bank.shape == (20_000, 40, 4)
    for snr_db, bits in ((15, 6144), (20, 6144), (25, 6144), (30, 24576)):
        snr = 10 ** (snr_db / 10)
        outage = np.mean(shannon_bits(scenario.physical, snr * bank[:, 7, 2]) < bits)
        threshold = (2 ** (bits / scenario.physical.channel_uses_per_slot) - 1) / snr
        assert outage == pytest.approx(1 - np.exp(-threshold), abs=0.02)


def test_greedy_serves_the_largest_outage_then_the_smallest_share_of_need():
    scenario = short(4)
    config = scenario.physical
    bits = float(shannon_bits(config, 1.0))  # no fading: every occasion carries this
    allocation = allocate_grants(
        config,
        snr=np.array([1.0, 1.0]),
        packet_bits=np.array([2.5 * bits, 0.5 * bits]),  # needs 2.5 and 0.5 occasions
        period=np.array([10, 10]),
        phase=np.array([0, 3]),
        bank=np.ones((4, 10, 4)),
    )
    assert allocation.need == pytest.approx((2.5, 0.5))
    # Both start at outage 1 (the index breaks the tie); B is served after one
    # occasion, A after three; then fewer occasions per unit of need goes first
    # (A: 3 / 2.5 and 4 / 2.5 against B: 1 / 0.5).
    assert allocation.order[:6] == (0, 1, 0, 0, 0, 0)
    assert [len(o) for o in allocation.occasions] == [10, 10]  # closed at P_i
    assert allocation.outage == (0.0, 0.0)
    assert allocation.assigned_share == 0.5 and allocation.blocked == 0
    # Identical UEs tie on both keys: the lower index goes first, alternately.
    twins = allocate_grants(
        config, np.array([1.0, 1.0]), np.array([2.5 * bits] * 2), np.array([10, 10]),
        np.array([0, 0]), np.ones((4, 10, 4)),
    )  # fmt: skip
    assert twins.order[:6] == (0, 1, 0, 1, 0, 1)


@pytest.mark.parametrize("single_channel", [False, True])
def test_allocation_is_exclusive_complete_and_predicts_its_own_outage(single_channel):
    scenario = short(16)
    policy, _, traffic = configured(scenario, single_channel=single_channel)
    allocation = policy.allocation
    period, phase = traffic.period_slots, traffic.phase_slots
    owner = grid_of(allocation.occasions, period)
    bank = bank_of(scenario)
    for ue, grants in enumerate(allocation.occasions):
        residues = [r for r, _ in grants]
        assert len(set(residues)) == len(residues) <= period[ue]
        if single_channel:
            assert len({k for _, k in grants}) == 1
        # Closed: either P_i occasions or nowhere left to place one.
        if len(grants) < period[ue]:
            spot = place_occasion(
                ue, owner, grants, int(period[ue]), int(phase[ue]), 10, single_channel
            )
            assert spot is None
        bits = sum(
            shannon_bits(
                scenario.physical, policy._snr[ue] * bank[:, (r - phase[ue]) % period[ue], k]
            )
            for r, k in grants
        )
        assert allocation.outage[ue] == pytest.approx(np.mean(bits < traffic.packet_bits[ue]))
    assert allocation.assigned_share == pytest.approx(np.mean(owner != -1))


def test_placement_prefers_new_subchannels_shared_columns_then_spread():
    free = np.full((4, 40), -1)
    # Frequency diversity: the subchannels the UE uses least, then the lowest.
    owner = free.copy()
    owner[0, 0] = owner[1, 1] = 0
    assert place_occasion(0, owner, [(0, 0), (1, 1)], 40, 0, 10) == (2, 2)
    # Best fit: a column already partly used by another UE.
    owner = free.copy()
    owner[1, 3] = 5
    assert place_occasion(0, owner, [], 40, 0, 10) == (13, 1)
    # Time spread within the window (phase 0) on the same subchannel.
    owner = free.copy()
    grants = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for r, k in grants:
        owner[k, r] = 0
    assert place_occasion(0, owner, grants, 40, 0, 10) == (39, 0)
    # The window starts at the phase: positions are (r - phase) mod P.
    assert place_occasion(0, owner, grants, 40, 30, 10) == (29, 0)
    # A fast occasion needs its whole column free.
    owner = free.copy()
    owner[0, 25] = 7
    assert place_occasion(1, owner, [], 10, 0, 10) == (0, 0)
    assert place_occasion(1, owner, [(0, 0)], 10, 0, 10) == (1, 1)


def test_single_channel_starts_on_the_freest_subchannel_and_stays():
    owner = np.full((4, 40), -1)
    owner[0, :20] = 3
    assert place_occasion(0, owner, [], 40, 0, 10, single_channel=True) == (0, 1)
    owner[1, 0] = 0
    assert place_occasion(0, owner, [(0, 1)], 40, 0, 10, single_channel=True)[1] == 1
    owner[1, :] = 4
    owner[1, 0] = 0
    assert place_occasion(0, owner, [(0, 1)], 40, 0, 10, single_channel=True) is None


def test_ues_transmit_at_max_power_exactly_in_their_grants():
    scenario = short(10)
    policy, topology, traffic = configured(scenario)
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
            expected = [channel, top] if on else [0, 0]
            assert actions[ue].tolist() == expected
            sent += on
        busy = actions[actions[:, 1] > 0, 0]
        assert len(busy) == len(set(busy.tolist()))  # exclusive in every slot
        observations = env.step(actions).observations
    assert sent > 0
    with pytest.raises(ValueError, match="out of step"):
        policy.reset(splits_v2.execution_generators("dev", 10, 0, 0))
        policy.act(observations)  # observations of the last slot, policy at slot 0


def test_run_capacity_episode_configures_before_reset_and_keeps_diagnostics():
    scenario = short(4)
    calls = []

    class Stub:
        name = "stub"

        def configure(self, mean_gain, traffic, fading):
            calls.append(("configure", np.array(mean_gain), traffic, fading))

        def reset(self, rngs):
            calls.append(("reset",))

        def act(self, observations):
            return np.zeros((len(observations), 2), dtype=np.int64)

        def episode_diagnostics(self):
            return {"seen": len(calls)}

    result = run_capacity_episode(scenario, Stub(), "dev", 0)
    topology, traffic = scenario.instance(0)
    assert [c[0] for c in calls] == ["configure", "reset"]
    np.testing.assert_array_equal(calls[0][1], topology.ue_ap_gain_linear)
    np.testing.assert_array_equal(calls[0][2].phase_slots, traffic.phase_slots)
    assert calls[0][3] == scenario.fading
    assert result["policy_diagnostics"] == {"seen": 2}
    # Policies without the hooks are untouched (no diagnostics key).
    plain = run_capacity_episode(
        scenario, make_policy_v2("central_edf", scenario.physical, scenario.menu), "dev", 0
    )
    assert "policy_diagnostics" not in plain


def test_registered_specs_build_both_variants():
    scenario = short(4)
    main = make_policy_v2("configured_grant", scenario.physical, scenario.menu)
    single = make_policy_v2("configured_grant:single_channel=1", scenario.physical, scenario.menu)
    assert (main.single_channel, single.single_channel) == (False, True)
    with pytest.raises(ValueError, match="configure"):
        main.reset(splits_v2.execution_generators("dev", 4, 0, 0))
