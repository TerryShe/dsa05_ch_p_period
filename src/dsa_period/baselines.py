"""Local heuristic access policies that use only each UE's own observation.

Each UE owns a private RNG. UEs without a pending packet choose OFF and draw no
random numbers. The link estimate below is derivable locally: the UE knows its
own power, the AP-reported mutual information of its last transmission and the
public per-subchannel RSSI, hence its own SINR, received power and path gain.
"""

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from dsa_period.env import Outcome, UEObservation
from dsa_period.physics import LN2, PhysicalConfig, PowerMenu, shannon_bits

POLICIES = ("random_max", "sticky_max", "lazy_pacing")


def estimate_link(
    obs: UEObservation, config: PhysicalConfig, menu: PowerMenu
) -> tuple[float | None, NDArray[np.float64]]:
    """Own path gain (None if unknown) and last-slot interference-plus-noise per subchannel.

    On the UE's own last subchannel its received power is removed from the RSSI:
    SINR = 2^(bits / (B tau)) - 1 and interference = RSSI / (1 + SINR).
    """
    interference = np.array(obs.previous_rssi_w, dtype=np.float64)
    channel, power_id = obs.previous_action
    if obs.previous_outcome not in (Outcome.TX, Outcome.DECODED) or power_id == 0:
        return None, interference
    sinr = math.expm1(obs.previous_service_bits / config.channel_uses_per_slot * LN2)
    own_received = interference[channel] * sinr / (1.0 + sinr)
    interference[channel] = interference[channel] / (1.0 + sinr)
    return own_received / menu.power_w[power_id], interference


def _min_power_meeting(
    required_bits: float,
    gain: float,
    interference: float,
    config: PhysicalConfig,
    menu: PowerMenu,
) -> int | None:
    for power_id in range(1, menu.num_levels):
        bits = shannon_bits(config, menu.power_w[power_id] * gain / interference)
        if bits >= required_bits:
            return power_id
    return None


def _act_one(
    name: str,
    obs: UEObservation,
    config: PhysicalConfig,
    menu: PowerMenu,
    rng: np.random.Generator,
) -> tuple[int, int]:
    top = menu.num_levels - 1
    previous_channel = obs.previous_action[0]
    transmitted = obs.previous_outcome in (Outcome.TX, Outcome.DECODED)
    # Bits per remaining slot needed to finish by the deadline.
    required = obs.remaining_bits / obs.slots_to_deadline
    if name == "random_max":
        return int(rng.integers(config.num_subchannels)), top
    if name == "sticky_max":
        keep = transmitted and obs.previous_service_bits >= required
        channel = previous_channel if keep else int(rng.integers(config.num_subchannels))
        return channel, top
    # lazy_pacing: stay on the last subchannel if some power keeps pace there;
    # otherwise move to the least-interfered subchannel. Pick the lowest pacing power.
    gain, interference = estimate_link(obs, config, menu)
    if gain is None:
        return int(rng.integers(config.num_subchannels)), top
    power = _min_power_meeting(required, gain, interference[previous_channel], config, menu)
    if power is not None:
        return previous_channel, power
    best = np.flatnonzero(interference == interference.min())
    channel = int(best[rng.integers(best.size)]) if best.size > 1 else int(best[0])
    power = _min_power_meeting(required, gain, interference[channel], config, menu)
    return channel, top if power is None else power


def act(
    name: str,
    observations: Sequence[UEObservation],
    config: PhysicalConfig,
    menu: PowerMenu,
    rngs: Sequence[np.random.Generator],
) -> NDArray[np.int64]:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {POLICIES}")
    if len(observations) != config.num_ues or len(rngs) != config.num_ues:
        raise ValueError("need exactly one observation and one RNG per UE")
    if len({id(rng) for rng in rngs}) != len(rngs):
        raise ValueError("each UE must own a distinct random generator")
    actions = np.zeros((config.num_ues, 2), dtype=np.int64)
    for ue, (obs, rng) in enumerate(zip(observations, rngs, strict=True)):
        if obs.remaining_bits > 0:
            actions[ue] = _act_one(name, obs, config, menu, rng)
    return actions
