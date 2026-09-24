"""Centralized minimum-energy reference for N <= K on static, frequency-flat channels.

Orthogonality is optimal there. Take any feasible schedule and move every UE to a
subchannel of its own, keeping its powers and transmission slots: its SINR can
only rise, so every packet completes no later, and energy does not depend on the
subchannel. The optimum therefore decomposes into independent per-packet
problems, identical for all packets of one UE:

    min  sum_l n_l * e_l   s.t.  sum_l n_l * b_l >= L,  sum_l n_l <= P,  n_l in N,

where level l has slot energy e_l = tau * p_l and interference-free slot mutual
information b_l. Slots are interchangeable, so only the counts matter. The
solver enumerates every count vector of the upper levels and fills the lowest
level minimally, which is exact. N > K requires time sharing and is not handled.
"""

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from dsa_period.env import UEObservation
from dsa_period.physics import PhysicalConfig, PowerMenu, shannon_bits
from dsa_period.topology import Topology
from dsa_period.traffic import TrafficPattern


@lru_cache(maxsize=32)
def _compositions(total: int, parts: int) -> NDArray[np.int64]:
    """All nonnegative integer vectors of length ``parts`` with sum <= total."""
    if parts == 0:
        return np.zeros((1, 0), dtype=np.int64)
    rows = []
    # Stars and bars with one slack part: choose ``parts`` bar positions.
    for bars in itertools.combinations(range(total + parts), parts):
        previous, counts = -1, []
        for bar in bars:
            counts.append(bar - previous - 1)
            previous = bar
        rows.append(counts)
    result = np.asarray(rows, dtype=np.int64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PacketPlan:
    """Optimal per-packet power-level counts; ``counts[l-1]`` is for power ID l."""

    counts: tuple[int, ...]
    energy_j: float
    slots: int
    mutual_information_bits: float

    def levels(self) -> tuple[int, ...]:
        """Power IDs of the planned transmissions, highest power first."""
        return tuple(
            level for level in range(len(self.counts), 0, -1) for _ in range(self.counts[level - 1])
        )


def optimal_packet_plan(
    slot_bits: NDArray[np.float64],
    slot_energy_j: NDArray[np.float64],
    packet_bits: float,
    max_slots: int,
) -> PacketPlan | None:
    """Exact minimum-energy level counts, or None if even max power cannot finish.

    ``slot_bits``/``slot_energy_j`` list the active levels in increasing power.
    Ties in energy prefer fewer slots, then the lexicographically first counts.
    """
    bits = np.asarray(slot_bits, dtype=np.float64)
    energy = np.asarray(slot_energy_j, dtype=np.float64)
    if bits.ndim != 1 or bits.shape != energy.shape or bits.size == 0:
        raise ValueError("slot_bits and slot_energy_j must be matching nonempty vectors")
    if np.any(np.diff(bits) <= 0) or np.any(np.diff(energy) <= 0) or bits[0] <= 0:
        raise ValueError("levels must strictly increase in both bits and energy")
    upper = _compositions(max_slots, bits.size - 1)
    upper_bits = upper @ bits[1:]
    need = np.maximum(packet_bits - upper_bits, 0.0)
    lowest = np.ceil(need / bits[0]).astype(np.int64)
    slots = upper.sum(axis=1) + lowest
    feasible = slots <= max_slots
    if not feasible.any():
        return None
    total_energy = upper @ energy[1:] + lowest * energy[0]
    candidates = np.flatnonzero(feasible)
    best = candidates[np.lexsort((slots[candidates], total_energy[candidates]))[0]]
    counts = (int(lowest[best]), *(int(x) for x in upper[best]))
    return PacketPlan(
        counts=counts,
        energy_j=float(total_energy[best]),
        slots=int(slots[best]),
        mutual_information_bits=float(upper_bits[best] + lowest[best] * bits[0]),
    )


def continuous_lower_bound_j(
    config: PhysicalConfig, gain: float, packet_bits: float, max_slots: int, max_power_w: float
) -> float | None:
    """Energy with any power in [0, p_max] per slot: spread evenly (log2 is concave)."""
    power = math.expm1(packet_bits / (max_slots * config.channel_uses_per_slot) * math.log(2))
    power *= config.noise_power_w / gain
    return None if power > max_power_w else power * max_slots * config.slot_duration_s


@dataclass(frozen=True)
class CentralReference:
    """Per-UE optimal packet plans on dedicated subchannels (UE i -> subchannel i)."""

    plans: tuple[PacketPlan, ...]
    continuous_bounds_j: tuple[float | None, ...]
    period_slots: tuple[int, ...]
    slot_duration_s: float

    @property
    def average_power_per_ue_w(self) -> float:
        """Long-run mean transmit power per UE when every packet is sent on time."""
        per_ue = [
            plan.energy_j / (period * self.slot_duration_s)
            for plan, period in zip(self.plans, self.period_slots, strict=True)
        ]
        return float(np.mean(per_ue))

    def window_energy_j(self, packets_per_ue: NDArray[np.int64], violation_cap: float) -> dict:
        """Optimal energy of a fixed packet set, strictly on time and under the cap.

        With per-UE violation rate <= cap, the optimum skips floor(cap * n_i)
        packets of UE i entirely (all of its packets cost the same).
        """
        counts = np.asarray(packets_per_ue, dtype=np.int64)
        energy = np.array([plan.energy_j for plan in self.plans])
        skipped = np.floor(violation_cap * counts + 1e-9).astype(np.int64)
        return {
            "strict_j": float(counts @ energy),
            "capped_j": float((counts - skipped) @ energy),
            "skipped_packets": skipped.tolist(),
        }


def solo_reference(
    config: PhysicalConfig, menu: PowerMenu, topology: Topology, traffic: TrafficPattern
) -> CentralReference:
    """Per-UE interference-free optimum over each full packet window, for any N.

    Exact when N <= K. For N > K it is a lower bound on every schedule: a UE's
    SINR never exceeds its SNR alone, so each of its packets costs at least the
    solo optimum in any feasible schedule, with or without co-channel sharing.
    """
    if topology.num_ues != config.num_ues or traffic.num_ues != config.num_ues:
        raise ValueError("topology and traffic must match num_ues")
    power = np.asarray(menu.power_w[1:])
    energy = config.slot_duration_s * power
    plans, bounds = [], []
    for ue, gain in enumerate(topology.ue_ap_gain_linear):
        bits = shannon_bits(config, power * gain / config.noise_power_w)
        period = int(traffic.period_slots[ue])
        packet = float(traffic.packet_bits[ue])
        plan = optimal_packet_plan(bits, energy, packet, period)
        if plan is None:
            raise ValueError(f"UE {ue} cannot finish a packet even at maximum power")
        plans.append(plan)
        bounds.append(continuous_lower_bound_j(config, float(gain), packet, period, power[-1]))
    return CentralReference(
        tuple(plans),
        tuple(bounds),
        tuple(int(p) for p in traffic.period_slots),
        config.slot_duration_s,
    )


def central_reference(
    config: PhysicalConfig, menu: PowerMenu, topology: Topology, traffic: TrafficPattern
) -> CentralReference:
    """Exact optimum for N <= K (dedicated subchannels); see the module docstring."""
    if config.num_ues > config.num_subchannels:
        raise NotImplementedError(
            "N > K needs time sharing; the orthogonal reference is exact only for N <= K"
        )
    return solo_reference(config, menu, topology, traffic)


def act_central(
    reference: CentralReference, observations: tuple[UEObservation, ...]
) -> NDArray[np.int64]:
    """Replay the reference: UE i uses subchannel i in the first slots of each packet.

    Uses only each UE's slot index within its current packet, derived from its
    own period and slots_to_deadline, so the witness runs in the real environment.
    """
    actions = np.zeros((len(observations), 2), dtype=np.int64)
    for ue, obs in enumerate(observations):
        if obs.remaining_bits <= 0:
            continue
        levels = reference.plans[ue].levels()
        index = obs.period_slots - obs.slots_to_deadline
        if index < len(levels):
            actions[ue] = (ue, levels[index])
    return actions
