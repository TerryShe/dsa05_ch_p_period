import numpy as np
import pytest

from dsa_period.channel import StaticChannel
from dsa_period.env import PeriodicUplinkEnv
from dsa_period.physics import PhysicalConfig, PowerMenu, compute_step
from dsa_period.topology import SpatialConfig, topology_from_positions
from dsa_period.traffic import TrafficPattern

SPATIAL = SpatialConfig(
    radius_m=100.0,
    reference_distance_m=1.0,
    reference_path_loss_db=40.0,
    path_loss_exponent=4.0,
    topology_seed=0,
)
MENU = PowerMenu(("OFF", 0.0, 10.0, 23.0))
TOP = MENU.num_levels - 1


def physical(num_ues: int, num_subchannels: int = 1) -> PhysicalConfig:
    return PhysicalConfig(
        num_ues=num_ues,
        num_subchannels=num_subchannels,
        slot_duration_s=0.001,
        subchannel_bandwidth_hz=1e6,
    )


def build_env(positions, periods, bits, phases, *, num_subchannels=1, horizon=20):
    n = len(positions)
    config = physical(n, num_subchannels)
    topology = topology_from_positions(SPATIAL, positions)
    traffic = TrafficPattern(
        ("c",),
        np.zeros(n, np.int64),
        np.asarray(periods),
        np.asarray(bits, dtype=float),
        np.asarray(phases),
    )
    channel = StaticChannel(topology, num_subchannels)
    return PeriodicUplinkEnv(config, MENU, channel, traffic, horizon)


def solo_bits(position, power_id=TOP) -> float:
    """Mutual information of one interference-free slot for a UE at ``position``."""
    config = physical(1)
    topology = topology_from_positions(SPATIAL, [position])
    gains = StaticChannel(topology, 1).gains(0)
    return float(compute_step(config, MENU, gains, [[0, power_id]]).service_bits[0])


@pytest.fixture
def menu():
    return MENU
