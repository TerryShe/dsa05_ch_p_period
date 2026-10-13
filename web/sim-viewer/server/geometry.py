"""Deployment geometry regenerated from the scenario file (no policy run; README §2.1).

``Scenario.instance`` is deterministic in (config, N, instance), and positions are
nested across N. The caller cross-checks the result against the formal record.
"""

from functools import lru_cache
from pathlib import Path

from dsa_period.config import Scenario, load_scenario


@lru_cache(maxsize=16)
def _scenario(config: Path, n: int) -> Scenario:
    return load_scenario(config, n)


def deployment_geometry(config: Path, n: int, instance: int) -> dict:
    scenario = _scenario(config, n)
    topology, traffic = scenario.instance(instance)
    return {
        "radius_m": scenario.spatial.radius_m,
        "positions_m": topology.path.ue_positions_m.tolist(),
        "distance_m": topology.path.ue_ap_distances_m.tolist(),
        "shadowing_db": topology.shadowing_db.tolist(),
        "class_names": list(traffic.class_names),
        "class_index": traffic.class_index.tolist(),
        "period_slots": traffic.period_slots.tolist(),
        "packet_bits": traffic.packet_bits.tolist(),
    }
