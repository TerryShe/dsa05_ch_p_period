"""Load a scenario TOML and build reproducible (topology, traffic) instances.

Instance ``i`` uses topology seed ``spatial.topology_seed + i`` and phase seed
``traffic.phase_seed + i``; positions and phases are sampled independently.

Scenario v2.0 (docs/research/scenario-v2.md) adds ``[shadowing]`` and ``[fading]``
sections and round-robin class assignment for any N: its TOML omits
``physical.num_ues`` and the class counts, and ``load_scenario(path, num_ues)``
fixes N. Its topology applies the coverage rule, and ``make_env`` needs the
per-UE fading seeds of the episode. v1.0 files load and behave exactly as before.

Scenario v2.1 (docs/research/scenario-v2.1.md) adds ``[spatial] min_distance_m``:
the topology also redraws UEs closer than that to the AP. The key is kept out of
``SpatialConfig``, so v2.0 topologies and their records are unchanged.
"""

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from dsa_period.channel import ShadowedJakesChannel, StaticChannel
from dsa_period.env import PeriodicUplinkEnv
from dsa_period.physics import PhysicalConfig, PowerMenu, shannon_bits
from dsa_period.topology import (
    ShadowedTopology,
    ShadowingConfig,
    SpatialConfig,
    Topology,
    generate_shadowed_topology,
    generate_topology,
)
from dsa_period.traffic import (
    TrafficClass,
    TrafficPattern,
    build_pattern,
    build_round_robin_pattern,
    round_robin_counts,
)

ASSIGNMENTS = ("grouped", "round-robin")


@dataclass(frozen=True)
class FadingConfig:
    """Per-subchannel Jakes fading by sum of sinusoids (scenario v2.0 §2.2)."""

    model: str
    doppler_hz: float
    sinusoids: int = 32
    carrier_hz: float | None = None  # documentation only (Doppler <-> speed)

    def __post_init__(self) -> None:
        if self.model != "jakes-sos":
            raise ValueError("the only fading model is 'jakes-sos'")
        if not self.doppler_hz >= 0 or int(self.sinusoids) != self.sinusoids or self.sinusoids < 1:
            raise ValueError("doppler_hz must be >= 0 and sinusoids a positive integer")


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
    shadowing: ShadowingConfig | None = None
    fading: FadingConfig | None = None
    class_assignment: str = "grouped"
    capacity: dict = field(default_factory=dict)
    min_distance_m: float = 0.0

    def __post_init__(self) -> None:
        self.menu.validate_for(self.physical)
        if not 0 <= self.min_distance_m < self.spatial.radius_m:
            raise ValueError("min_distance_m must be in [0, radius_m)")
        if self.min_distance_m > 0 and self.fading is None:
            raise ValueError("min_distance_m needs a v2 scenario (shadowed topology)")
        if self.class_assignment not in ASSIGNMENTS:
            raise ValueError(f"class_assignment must be one of {ASSIGNMENTS}")
        if (self.shadowing is None) != (self.fading is None):
            raise ValueError("shadowing and fading come together (scenario v2.0)")
        if self.fading is not None and self.class_assignment != "round-robin":
            raise ValueError("scenario v2.0 assigns classes round-robin")
        if sum(c.num_ues for c in self.traffic_classes) != self.physical.num_ues:
            raise ValueError("traffic class UE counts must sum to physical.num_ues")
        if self.class_assignment == "round-robin" and [
            c.num_ues for c in self.traffic_classes
        ] != round_robin_counts(len(self.traffic_classes), self.physical.num_ues):
            raise ValueError("class counts must follow the round-robin assignment")
        longest = max(c.period_slots for c in self.traffic_classes)
        if self.warmup_slots < longest:
            raise ValueError("warmup_slots must cover the longest period")
        if self.horizon_slots <= self.warmup_slots:
            raise ValueError("horizon_slots must exceed warmup_slots")

    @property
    def is_v2(self) -> bool:
        return self.fading is not None

    def instance(self, index: int) -> tuple[Topology | ShadowedTopology, TrafficPattern]:
        if index < 0:
            raise ValueError("instance index must be nonnegative")
        spatial = replace(self.spatial, topology_seed=self.spatial.topology_seed + index)
        n = self.physical.num_ues
        if not self.is_v2:
            topology = generate_topology(spatial, n)
            return topology, build_pattern(self.traffic_classes, self.phase_seed + index)
        traffic = build_round_robin_pattern(self.traffic_classes, n, self.phase_seed + index)
        topology = generate_shadowed_topology(
            spatial, self.shadowing, n, self.covered(traffic), min_distance_m=self.min_distance_m
        )
        return topology, traffic

    def covered(self, traffic: TrafficPattern):
        """Coverage rule (v2.0 §2.1): one packet alone at max power on the mean
        channel needs at most ``coverage_window_fraction`` of its window."""
        top = self.menu.power_w[-1]
        noise = self.physical.noise_power_w
        fraction = self.shadowing.coverage_window_fraction

        def rule(ue: int, mean_gain: float) -> bool:
            bits = float(shannon_bits(self.physical, top * mean_gain / noise))
            return traffic.packet_bits[ue] <= fraction * traffic.period_slots[ue] * bits

        return rule

    def make_env(
        self,
        topology: Topology | ShadowedTopology,
        traffic: TrafficPattern,
        fading_seeds: list[np.random.SeedSequence] | None = None,
        horizon_slots: int | None = None,
    ) -> PeriodicUplinkEnv:
        """Environment of one episode; v2.0 needs that episode's per-UE fading seeds."""
        horizon = self.horizon_slots if horizon_slots is None else horizon_slots
        if not self.is_v2:
            if fading_seeds is not None:
                raise ValueError("scenario v1.0 has no fading")
            channel = StaticChannel(topology, self.physical.num_subchannels)
            return PeriodicUplinkEnv(self.physical, self.menu, channel, traffic, horizon)
        if fading_seeds is None:
            raise ValueError("scenario v2.0 needs the episode's per-UE fading seeds")
        channel = self.make_channel(topology, fading_seeds)
        return PeriodicUplinkEnv(self.physical, self.menu, channel, traffic, horizon, own_csi=True)

    def make_channel(
        self, topology: ShadowedTopology, fading_seeds: list[np.random.SeedSequence]
    ) -> ShadowedJakesChannel:
        return ShadowedJakesChannel(
            topology.ue_ap_gain_linear,
            self.physical.num_subchannels,
            doppler_hz=self.fading.doppler_hz,
            slot_duration_s=self.physical.slot_duration_s,
            ue_seeds=list(fading_seeds),
            sinusoids=self.fading.sinusoids,
        )

    def with_num_ues(self, num_ues: int) -> "Scenario":
        """The same round-robin scenario with another number of UEs."""
        if self.class_assignment != "round-robin":
            raise ValueError("only round-robin scenarios can change N")
        counts = round_robin_counts(len(self.traffic_classes), num_ues)
        classes = tuple(
            replace(c, num_ues=k) for c, k in zip(self.traffic_classes, counts, strict=True)
        )
        physical = replace(self.physical, num_ues=num_ues)
        return replace(self, physical=physical, traffic_classes=classes)

    def with_doppler(self, doppler_hz: float) -> "Scenario":
        """Doppler sensitivity (v2.0 §13): same instances, other fading speed."""
        if not self.is_v2:
            raise ValueError("scenario v1.0 has no fading")
        return replace(self, fading=replace(self.fading, doppler_hz=doppler_hz))

    def with_packet_scale(self, scale: float) -> "Scenario":
        """Scale every class's packet size; a load knob for calibration sweeps."""
        classes = tuple(replace(c, packet_bits=c.packet_bits * scale) for c in self.traffic_classes)
        return replace(self, traffic_classes=classes)


def load_scenario(path: str | Path, num_ues: int | None = None) -> Scenario:
    """Load a scenario; a round-robin (v2.0) file needs ``num_ues``, a v1.0 file must not."""
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    traffic = raw["traffic"]
    run = raw["run"]
    assignment = traffic.get("assignment", "grouped")
    physical = dict(raw["physical"])
    entries = [dict(entry) for entry in traffic["classes"]]
    if assignment == "round-robin":
        if "num_ues" in physical or any("num_ues" in entry for entry in entries):
            raise ValueError("a round-robin scenario takes N from load_scenario, not the file")
        if num_ues is None:
            raise ValueError("a round-robin scenario needs num_ues")
        physical["num_ues"] = num_ues
        for entry, count in zip(entries, round_robin_counts(len(entries), num_ues), strict=True):
            entry["num_ues"] = count
    elif num_ues is not None:
        raise ValueError("this scenario fixes N in its file")
    spatial = dict(raw["spatial"])
    min_distance_m = float(spatial.pop("min_distance_m", 0.0))
    return Scenario(
        experiment=dict(raw.get("experiment", {})),
        physical=PhysicalConfig(**physical),
        spatial=SpatialConfig(**spatial),
        menu=PowerMenu(tuple(raw["power"]["power_dbm"])),
        traffic_classes=tuple(TrafficClass(**entry) for entry in entries),
        phase_seed=int(traffic["phase_seed"]),
        horizon_slots=int(run["horizon_slots"]),
        warmup_slots=int(run["warmup_slots"]),
        objective=dict(raw.get("objective", {})),
        shadowing=ShadowingConfig(**raw["shadowing"]) if "shadowing" in raw else None,
        fading=FadingConfig(**raw["fading"]) if "fading" in raw else None,
        class_assignment=assignment,
        capacity=dict(raw.get("capacity", {})),
        min_distance_m=min_distance_m,
    )
