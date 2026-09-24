import math

import numpy as np
import pytest
from conftest import MENU, TOP, physical

from dsa_period.physics import PowerMenu, compute_step


def test_off_is_exactly_zero_watts_and_zero_dbm_is_active():
    assert MENU.power_w[0] == 0.0
    assert MENU.power_w[1] == pytest.approx(1e-3)


@pytest.mark.parametrize(
    "levels",
    [(0.0, 10.0), ("OFF",), ("OFF", 10.0, 10.0), ("OFF", 10.0, 5.0), ("OFF", float("nan"))],
)
def test_power_menu_rejects_malformed_levels(levels):
    with pytest.raises(ValueError):
        PowerMenu(levels)


def test_power_menu_must_respect_device_maximum():
    with pytest.raises(ValueError, match="max_tx_power_dbm"):
        compute_step(physical(1), PowerMenu(("OFF", 30.0)), np.ones((1, 1)), [[0, 1]])


def test_solo_transmission_carries_shannon_mutual_information():
    config = physical(1, 2)
    gains = np.full((1, 2), 1e-10)
    result = compute_step(config, MENU, gains, [[1, TOP]])
    power = MENU.power_w[TOP]
    snr = power * 1e-10 / config.noise_power_w
    assert result.sinr_linear[0] == pytest.approx(snr)
    assert result.service_bits[0] == pytest.approx(1000.0 * math.log2(1.0 + snr))
    assert result.tx_energy_j[0] == pytest.approx(0.001 * power)
    assert result.rssi_w[0] == pytest.approx(config.noise_power_w)
    assert result.rssi_w[1] == pytest.approx(config.noise_power_w + power * 1e-10)


def test_cochannel_users_interfere_and_subchannels_are_orthogonal():
    config = physical(3, 2)
    gains = np.array([[1e-9] * 2, [1e-10] * 2, [1e-11] * 2])
    result = compute_step(config, MENU, gains, [[0, TOP], [0, 2], [1, 1]])
    p = MENU.power_w
    noise = config.noise_power_w
    assert result.sinr_linear[0] == pytest.approx(p[TOP] * 1e-9 / (noise + p[2] * 1e-10))
    assert result.sinr_linear[1] == pytest.approx(p[2] * 1e-10 / (noise + p[TOP] * 1e-9))
    assert result.sinr_linear[2] == pytest.approx(p[1] * 1e-11 / noise)


def test_each_transmitter_uses_its_gain_on_the_selected_subchannel():
    config = physical(1, 2)
    gains = np.array([[1e-12, 1e-9]])
    low = compute_step(config, MENU, gains, [[0, TOP]]).sinr_linear[0]
    high = compute_step(config, MENU, gains, [[1, TOP]]).sinr_linear[0]
    assert high == pytest.approx(1000.0 * low)


def test_off_is_canonical_and_silent():
    result = compute_step(physical(1, 3), MENU, np.ones((1, 3)) * 1e-9, [[2, 0]])
    assert result.actions.tolist() == [[0, 0]]
    assert not result.active[0]
    assert math.isnan(result.sinr_linear[0])
    assert result.service_bits[0] == 0.0
    assert result.tx_energy_j[0] == 0.0


@pytest.mark.parametrize(
    "actions",
    [[[0, 1]], [[0, 1], [0, 1], [0, 1]], [[2, 1], [0, 1]], [[0, 4], [0, 1]], [[0, True], [0, 1]]],
)
def test_invalid_actions_are_rejected(actions):
    with pytest.raises(ValueError):
        compute_step(physical(2, 2), MENU, np.ones((2, 2)) * 1e-9, actions)


@pytest.mark.parametrize("gains", [np.ones((2, 1)), np.zeros((2, 2)), np.full((2, 2), np.inf)])
def test_invalid_gains_are_rejected(gains):
    with pytest.raises(ValueError):
        compute_step(physical(2, 2), MENU, gains, [[0, 1], [1, 1]])


def test_results_are_read_only():
    result = compute_step(physical(1), MENU, np.ones((1, 1)) * 1e-9, [[0, 1]])
    with pytest.raises(ValueError):
        result.service_bits[0] = 1.0
