"""Near-AP starvation and the stress set of EXP-DSA05-011B (design §2.3).

A starvation cell is one (policy, N, deployment) in which some UE whose mean SNR
alone at maximum power is at least ``STRONG_SNR_DB`` has a violation rate of at
least ``STARVED_RATE``. The mean SNR uses path loss and shadowing only (every
subchannel has the same mean gain in scenario v2.0). In EXP-DSA05-011 every such
UE was the one nearest the AP, left almost silent by one checkpoint.

The stress set ``exp011b-stress`` reserves a dev block; at each N it reads the
deployments of the block whose first N UEs include one within ``near_m`` of the
AP. Topologies are nested in N, so a deployment kept at N stays at every larger N.
"""

from functools import cache

import numpy as np
from numpy.typing import NDArray

from dsa_period import splits_v2
from dsa_period.config import Scenario, load_scenario

STRONG_SNR_DB = 20.0
STARVED_RATE = 0.5
STRESS_SET = "exp011b-stress"
CONFIG = "configs/scenario-v2.toml"


def mean_snr_db(scenario: Scenario, instance: int) -> NDArray[np.float64]:
    """Each UE's mean SNR (dB) alone at maximum power on deployment ``instance``."""
    topology, _ = scenario.instance(instance)
    gain = np.asarray(topology.ue_ap_gain_linear, dtype=np.float64)
    return 10.0 * np.log10(scenario.menu.power_w[-1] * gain / scenario.physical.noise_power_w)


def starved(violation_rate, snr_db) -> NDArray[np.bool_]:
    """Per UE: strong (mean SNR >= 20 dB) and violating at least half of its deadlines."""
    return (np.asarray(snr_db) >= STRONG_SNR_DB) & (np.asarray(violation_rate) >= STARVED_RATE)


def is_starvation_cell(violation_rate, snr_db) -> bool:
    return bool(starved(violation_rate, snr_db).any())


@cache
def _scenario(n: int) -> Scenario:
    return load_scenario(CONFIG, n)


@cache
def dev_snr_db(n: int, instance: int) -> tuple[float, ...]:
    """``mean_snr_db`` of a dev deployment at N (scenario v2.0), cached."""
    return tuple(mean_snr_db(_scenario(n), splits_v2.require("dev", instance)).tolist())


def cells(episodes: list[dict], n: int) -> list[dict]:
    """The starvation cells among dev episode records at N (``run_capacity_episode``)."""
    out = []
    for e in episodes:
        snr = np.array(dev_snr_db(n, e["instance"]))
        rate = np.array(e["violation_rate"])
        for ue in np.flatnonzero(starved(rate, snr)):
            out.append({
                "instance": e["instance"], "ue": int(ue), "violation_rate": float(rate[ue]),
                "snr_db": float(snr[ue]), "distance_m": float(e["distance_m"][ue]),
            })  # fmt: skip
    return out


def stress_instances(scenario: Scenario, name: str = STRESS_SET) -> list[int]:
    """The stress-set deployments read at the scenario's N, in index order."""
    entry = splits_v2.SCREENING_SETS[name]
    n = scenario.physical.num_ues
    if n not in entry["grid"]:
        raise ValueError(f"{name} is registered at N = {list(entry['grid'])}, not {n}")
    start, stop = entry["range"]
    return [
        splits_v2.require("dev", i)
        for i in range(start, stop)
        if scenario.instance(i)[0].ue_ap_distances_m.min() < entry["near_m"]
    ]
