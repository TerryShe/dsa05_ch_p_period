import numpy as np
import pytest
from conftest import MENU, TOP, build_env, solo_bits

from dsa_period.env import Outcome
from dsa_period.metrics import window_summary

POS = [[50.0, 0.0]]
TX = [[0, TOP]]


def test_completion_in_the_last_slot_before_the_deadline_is_on_time():
    s = solo_bits(POS[0])
    env = build_env(POS, [3], [2.5 * s], [0])
    for _ in range(2):
        step = env.step(TX)
        assert not step.completed[0] and step.on_time[0] == 0
    step = env.step(TX)
    assert step.completed[0] and step.on_time[0] == 1 and step.violations[0] == 0
    assert step.released[0], "the next packet arrives at the deadline boundary"
    assert step.remaining_bits[0] == pytest.approx(2.5 * s)
    job = env.job_records()[0]
    assert (job["completion_slot"], job["deadline_slot"], job["success"]) == (3, 3, True)
    assert job["delivered_bits"] == pytest.approx(2.5 * s) and job["dropped_bits"] == 0.0
    assert job["tx_slots"] == 3
    assert job["energy_j"] == pytest.approx(3 * 0.001 * MENU.power_w[TOP])


def test_missed_deadline_drops_the_remainder_and_releases_a_fresh_packet():
    s = solo_bits(POS[0])
    env = build_env(POS, [2], [2.5 * s], [0])
    env.step(TX)
    step = env.step(TX)
    assert step.violations[0] == 1 and step.on_time[0] == 0
    assert step.dropped_bits[0] == pytest.approx(0.5 * s)
    assert step.remaining_bits[0] == pytest.approx(2.5 * s)
    job = env.job_records()[0]
    assert job["success"] is False and job["completion_slot"] is None
    assert job["delivered_bits"] + job["dropped_bits"] == pytest.approx(job["payload_bits"])


def test_ues_without_a_packet_are_forced_off_before_arrival_and_after_completion():
    s = solo_bits(POS[0])
    env = build_env(POS, [5], [0.5 * s], [2])
    assert env.observe()[0].slots_to_deadline == 2
    for _ in range(2):
        step = env.step(TX)
        assert not step.physical.active[0] and step.physical.tx_energy_j[0] == 0.0
    assert step.released[0] and step.observations[0].slots_to_deadline == 5
    step = env.step(TX)
    assert step.completed[0] and step.observations[0].previous_outcome == Outcome.DECODED
    step = env.step(TX)
    assert not step.physical.active[0]
    assert step.observations[0].previous_outcome == Outcome.NO_TX
    assert step.observations[0].slots_to_deadline == 3


def test_observation_reports_own_service_and_public_rssi_only():
    s = solo_bits(POS[0])
    env = build_env(POS + [[0.0, 80.0]], [4, 4], [10 * s, 10 * s], [0, 0], num_subchannels=2)
    first = env.observe()[0]
    assert first.previous_outcome == Outcome.NO_FEEDBACK
    assert first.previous_rssi_w.tolist() == [env.config.noise_power_w] * 2
    step = env.step([[1, TOP], [0, 0]])
    obs = step.observations
    assert obs[0].previous_action == (1, TOP) and obs[0].previous_outcome == Outcome.TX
    assert obs[0].previous_service_bits == pytest.approx(s)
    assert obs[1].previous_outcome == Outcome.NO_TX and obs[1].previous_service_bits == 0.0
    assert obs[0].previous_rssi_w.tolist() == obs[1].previous_rssi_w.tolist()
    assert obs[0].remaining_bits == pytest.approx(9 * s)


def test_horizon_truncates_and_reset_replays_the_same_instance():
    s = solo_bits(POS[0])
    env = build_env(POS, [3], [1.5 * s], [1], horizon=7)
    for _ in range(7):
        env.step(TX)
    assert env.truncated
    with pytest.raises(RuntimeError):
        env.step(TX)
    first = env.job_records()
    env.reset()
    assert env.elapsed_steps == 0 and env.job_records() == []  # first arrival at slot 1
    for _ in range(7):
        env.step(TX)
    assert env.job_records() == first


def test_invalid_raw_action_of_an_idle_ue_is_still_rejected():
    env = build_env(POS, [5], [1.0], [3])
    with pytest.raises(ValueError):
        env.step([[0, 9]])
    assert env.elapsed_steps == 0


def test_window_counts_packets_by_deadline_and_excludes_censored_ones():
    s = solo_bits(POS[0])
    env = build_env(POS, [3], [2.5 * s], [0], horizon=10)
    for t in range(10):
        env.step(TX if t < 6 else [[0, 0]])
    summary = window_summary(
        env.job_records(),
        env.energy_by_slot(),
        warmup_slots=3,
        horizon_slots=10,
        slot_duration_s=0.001,
        num_ues=1,
    )
    # Deadlines 6 and 9 fall in (3, 10]; the packet due at 12 is still open.
    assert (summary["packets"], summary["on_time"], summary["violations"]) == (2, 1, 1)
    assert summary["worst_ue_ocr"] == pytest.approx(0.5)
    assert summary["window_energy_j"] == pytest.approx(3 * 0.001 * MENU.power_w[TOP])
    assert summary["latency_slots_mean"] == pytest.approx(3.0)
    assert np.isclose(summary["dropped_bits"], 2.5 * s)
