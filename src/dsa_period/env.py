"""Continuing uplink environment with asynchronous periodic, deadline-bound packets.

Slot t occupies [t, t+1). At boundary t+1 the environment, in this order:

1. settles slot t's service; a packet whose accumulated mutual information
   reaches its size completes at C = t+1;
2. settles every packet with deadline d = t+1: on time iff it completed
   (C <= d); otherwise it is one violation and its remaining bits are dropped;
3. releases every packet arriving at t+1 (for a UE, deadline == next arrival);
4. builds the next causal observations.

Each UE therefore holds at most one unfinished packet. A UE without one is
forced OFF and waits for its next arrival. Network-wide idleness does not end
the run; only the technical horizon does. The environment returns per-UE
energy and deadline outcomes but no reward: the learner defines the objective.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from dsa_period.channel import ChannelModel
from dsa_period.physics import (
    PhysicalConfig,
    PowerMenu,
    StepResult,
    compute_step,
    validate_actions,
)
from dsa_period.traffic import TrafficPattern


def _frozen(values: object, dtype: object = np.float64) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class Outcome(IntEnum):
    """Own-slot feedback from the AP."""

    NO_FEEDBACK = 0  # before the first slot
    NO_TX = 1
    TX = 2  # transmitted; the packet is not yet decoded
    DECODED = 3  # the packet completed in this slot


@dataclass(frozen=True)
class UEObservation:
    """Local, causal information of one UE; no identifier, clock or others' state.

    ``slots_to_deadline`` counts slots to the UE's own next arrival, which is the
    current packet's deadline when one is pending and the idle wait otherwise.
    ``previous_service_bits`` is the AP-reported mutual information of the UE's
    own last transmission (0 when OFF). ``previous_rssi_w`` is the public
    per-subchannel received power (noise included) of the last slot.
    """

    remaining_bits: float
    packet_bits: float
    period_slots: int
    slots_to_deadline: int
    previous_action: tuple[int, int]
    previous_outcome: Outcome
    previous_service_bits: float
    previous_rssi_w: NDArray[np.float64]


@dataclass
class _Job:
    ue: int
    index: int
    arrival_slot: int
    deadline_slot: int
    payload_bits: float
    completion_slot: int | None = None
    delivered_bits: float = 0.0
    dropped_bits: float = 0.0
    energy_j: float = 0.0
    tx_slots: int = 0
    success: bool | None = None  # set when the deadline is settled


@dataclass(frozen=True)
class EpisodeStep:
    """Simulator diagnostics of one slot; actors may only see ``observations[i]``."""

    observations: tuple[UEObservation, ...]
    physical: StepResult
    delivered_bits: NDArray[np.float64]
    remaining_bits: NDArray[np.float64]
    completed: NDArray[np.bool_]
    on_time: NDArray[np.int64]
    violations: NDArray[np.int64]
    dropped_bits: NDArray[np.float64]
    released: NDArray[np.bool_]
    elapsed_steps: int
    truncated: bool


class PeriodicUplinkEnv:
    """Fixed topology/channel and traffic pattern; ``reset`` replays the same instance."""

    def __init__(
        self,
        config: PhysicalConfig,
        menu: PowerMenu,
        channel: ChannelModel,
        traffic: TrafficPattern,
        horizon_slots: int,
    ) -> None:
        if not isinstance(config, PhysicalConfig) or not isinstance(menu, PowerMenu):
            raise ValueError("config and menu must be PhysicalConfig and PowerMenu")
        menu.validate_for(config)
        if (channel.num_ues, channel.num_subchannels) != (config.num_ues, config.num_subchannels):
            raise ValueError("channel dimensions must match the physical config")
        if not isinstance(traffic, TrafficPattern) or traffic.num_ues != config.num_ues:
            raise ValueError("traffic must be a TrafficPattern with num_ues entries")
        if isinstance(horizon_slots, bool) or int(horizon_slots) != horizon_slots:
            raise ValueError("horizon_slots must be an integer")
        if horizon_slots < 1:
            raise ValueError("horizon_slots must be positive")
        self.config = config
        self.menu = menu
        self.channel = channel
        self.traffic = traffic
        self.horizon_slots = int(horizon_slots)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> tuple[UEObservation, ...]:
        n, k = self.config.num_ues, self.config.num_subchannels
        self.channel.reset()
        self._t = 0
        self._jobs: list[_Job] = []
        self._job_count = np.zeros(n, dtype=np.int64)
        self._current: list[_Job | None] = [None] * n
        self._next_arrival = np.array(self.traffic.phase_slots, dtype=np.int64)
        self._remaining = np.zeros(n)
        self._previous_action = np.zeros((n, 2), dtype=np.int64)
        self._previous_outcome = np.full(n, Outcome.NO_FEEDBACK, dtype=np.int8)
        self._previous_service = np.zeros(n)
        self._previous_rssi = np.full(k, self.config.noise_power_w)
        self._energy = np.zeros((self.horizon_slots, n))
        self._release(0)
        return self.observe()

    def _release(self, boundary: int) -> NDArray[np.bool_]:
        released = self._next_arrival == boundary
        for ue in np.flatnonzero(released):
            period = int(self.traffic.period_slots[ue])
            job = _Job(
                ue=int(ue),
                index=int(self._job_count[ue]),
                arrival_slot=boundary,
                deadline_slot=boundary + period,
                payload_bits=float(self.traffic.packet_bits[ue]),
            )
            self._jobs.append(job)
            self._current[ue] = job
            self._job_count[ue] += 1
            self._remaining[ue] = job.payload_bits
            self._next_arrival[ue] = boundary + period
        return released

    @property
    def elapsed_steps(self) -> int:
        return self._t

    @property
    def truncated(self) -> bool:
        return self._t >= self.horizon_slots

    def observe(self) -> tuple[UEObservation, ...]:
        rssi = _frozen(self._previous_rssi)
        return tuple(
            UEObservation(
                remaining_bits=float(self._remaining[ue]),
                packet_bits=float(self.traffic.packet_bits[ue]),
                period_slots=int(self.traffic.period_slots[ue]),
                slots_to_deadline=int(self._next_arrival[ue] - self._t),
                previous_action=(
                    int(self._previous_action[ue, 0]),
                    int(self._previous_action[ue, 1]),
                ),
                previous_outcome=Outcome(int(self._previous_outcome[ue])),
                previous_service_bits=float(self._previous_service[ue]),
                previous_rssi_w=rssi,
            )
            for ue in range(self.config.num_ues)
        )

    # ------------------------------------------------------------------- step
    def step(self, joint_actions: object) -> EpisodeStep:
        if self.truncated:
            raise RuntimeError("horizon reached; call reset before step")
        t = self._t
        n = self.config.num_ues
        # Validate the raw request first, then force UEs without a packet OFF.
        actions = validate_actions(self.config, self.menu, joint_actions)
        actions[self._remaining == 0] = 0
        physical = compute_step(self.config, self.menu, self.channel.gains(t), actions)

        delivered = np.zeros(n)
        completed = np.zeros(n, dtype=np.bool_)
        outcome = np.where(physical.active, Outcome.TX, Outcome.NO_TX).astype(np.int8)
        boundary = t + 1
        for ue in np.flatnonzero(physical.active):
            job = self._current[ue]
            service = float(physical.service_bits[ue])
            left = float(self._remaining[ue])
            if service >= left:
                delivered[ue], self._remaining[ue] = left, 0.0
                completed[ue] = True
                job.completion_slot = boundary
                outcome[ue] = Outcome.DECODED
            else:
                delivered[ue] = service
                self._remaining[ue] = left - service
            job.delivered_bits += delivered[ue]
            job.energy_j += float(physical.tx_energy_j[ue])
            job.tx_slots += 1
        self._energy[t] = physical.tx_energy_j

        on_time = np.zeros(n, dtype=np.int64)
        violations = np.zeros(n, dtype=np.int64)
        dropped = np.zeros(n)
        for ue in range(n):
            job = self._current[ue]
            if job is None or job.deadline_slot != boundary:
                continue
            job.success = job.completion_slot is not None
            if job.success:
                on_time[ue] = 1
            else:
                violations[ue] = 1
                dropped[ue] = job.dropped_bits = float(self._remaining[ue])
                self._remaining[ue] = 0.0
            self._current[ue] = None
        released = self._release(boundary)

        self._t = boundary
        self._previous_action = np.array(physical.actions, copy=True)
        self._previous_outcome = outcome
        self._previous_service = np.array(physical.service_bits, copy=True)
        self._previous_rssi = np.array(physical.rssi_w, copy=True)
        return EpisodeStep(
            observations=self.observe(),
            physical=physical,
            delivered_bits=_frozen(delivered),
            remaining_bits=_frozen(self._remaining),
            completed=_frozen(completed, np.bool_),
            on_time=_frozen(on_time, np.int64),
            violations=_frozen(violations, np.int64),
            dropped_bits=_frozen(dropped),
            released=_frozen(released, np.bool_),
            elapsed_steps=boundary,
            truncated=self.truncated,
        )

    # ---------------------------------------------------------------- records
    def energy_by_slot(self) -> NDArray[np.float64]:
        """(elapsed_slots, N) transmit energy in joules of every executed slot."""
        return _frozen(self._energy[: self._t])

    def job_records(self) -> list[dict]:
        """All released packets. ``success`` is None while the deadline is unsettled."""
        return [
            {
                "ue": job.ue,
                "job": job.index,
                "traffic_class": self.traffic.class_names[int(self.traffic.class_index[job.ue])],
                "arrival_slot": job.arrival_slot,
                "deadline_slot": job.deadline_slot,
                "payload_bits": job.payload_bits,
                "completion_slot": job.completion_slot,
                "delivered_bits": job.delivered_bits,
                "dropped_bits": job.dropped_bits,
                "energy_j": job.energy_j,
                "tx_slots": job.tx_slots,
                "success": job.success,
            }
            for job in self._jobs
        ]
