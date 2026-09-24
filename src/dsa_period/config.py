"""Load a scenario TOML and build reproducible (topology, traffic) instances.

Instance ``i`` uses topology seed ``spatial.topology_seed + i`` and phase seed
``traffic.phase_seed + i``; positions and phases are sampled independently.
"""

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from dsa_period.channel import StaticChannel
from dsa_period.env import PeriodicUplinkEnv
from dsa_period.physics import PhysicalConfig, PowerMenu
from dsa_period.topology import SpatialConfig, Topology, generate_topology
from dsa_period.traffic import TrafficClass, TrafficPattern, build_pattern


@dataclass(frozen=True)
class Scenario:
    experiment: dict
    physical: PhysicalConfig
    spatial: SpatialConfig
    menu: PowerMenu
    traffic_classes: tuple[TrafficClass, ...]
    phase_seed: int
    horizon_slots: int
    warmup_slots: int
    objective: dict

    def __post_init__(self) -> None:
        self.menu.validate_for(self.physical)
        if sum(c.num_ues for c in self.traffic_classes) != self.physical.num_ues:
            raise ValueError("traffic class UE counts must sum to physical.num_ues")
        longest = max(c.period_slots for c in self.traffic_classes)
        if self.warmup_slots < longest:
            raise ValueError("warmup_slots must cover the longest period")
        if self.horizon_slots <= self.warmup_slots:
            raise ValueError("horizon_slots must exceed warmup_slots")

    def instance(self, index: int) -> tuple[Topology, TrafficPattern]:
        if index < 0:
            raise ValueError("instance index must be nonnegative")
        spatial = replace(self.spatial, topology_seed=self.spatial.topology_seed + index)
        topology = generate_topology(spatial, self.physical.num_ues)
        return topology, build_pattern(self.traffic_classes, self.phase_seed + index)

    def make_env(self, topology: Topology, traffic: TrafficPattern) -> PeriodicUplinkEnv:
        channel = StaticChannel(topology, self.physical.num_subchannels)
        return PeriodicUplinkEnv(self.physical, self.menu, channel, traffic, self.horizon_slots)

    def with_packet_scale(self, scale: float) -> "Scenario":
        """Scale every class's packet size; a load knob for calibration sweeps."""
        classes = tuple(replace(c, packet_bits=c.packet_bits * scale) for c in self.traffic_classes)
        return replace(self, traffic_classes=classes)


def load_scenario(path: str | Path) -> Scenario:
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    traffic = raw["traffic"]
    run = raw["run"]
    return Scenario(
        experiment=dict(raw.get("experiment", {})),
        physical=PhysicalConfig(**raw["physical"]),
        spatial=SpatialConfig(**raw["spatial"]),
        menu=PowerMenu(tuple(raw["power"]["power_dbm"])),
        traffic_classes=tuple(TrafficClass(**entry) for entry in traffic["classes"]),
        phase_seed=int(traffic["phase_seed"]),
        horizon_slots=int(run["horizon_slots"]),
        warmup_slots=int(run["warmup_slots"]),
        objective=dict(raw.get("objective", {})),
    )
