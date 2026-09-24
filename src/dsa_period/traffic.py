"""Asynchronous periodic packet arrivals with heterogeneous traffic classes.

UE i releases packet n of size L_i at slot boundary a_{i,n} = phi_i + n * P_i.
Its deadline is the UE's next arrival, d_{i,n} = a_{i,n} + P_i. Physical slots
are synchronous; only arrival phases differ ("asynchronous"). Phases are drawn
once per instance, independently of UE positions, and stay fixed in a run.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

PHASE_SAMPLING = "phi_i ~ Uniform{0,...,P_i-1} independently, PCG64(phase_seed)"


def _int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class TrafficClass:
    """``num_ues`` devices share one reporting period (slots) and packet size (bits)."""

    name: str
    period_slots: int
    packet_bits: float
    num_ues: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("traffic class name must be a nonempty string")
        object.__setattr__(self, "period_slots", _int(self.period_slots, "period_slots", 1))
        object.__setattr__(self, "num_ues", _int(self.num_ues, "num_ues", 0))
        bits = self.packet_bits
        if isinstance(bits, (bool, np.bool_)) or not isinstance(bits, Real):
            raise ValueError("packet_bits must be a positive finite number")
        if not np.isfinite(bits) or bits <= 0:
            raise ValueError("packet_bits must be a positive finite number")
        object.__setattr__(self, "packet_bits", float(bits))


def _frozen(values: object, dtype: object) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class TrafficPattern:
    """Per-UE class, period, packet size and phase of one traffic instance."""

    class_names: tuple[str, ...]
    class_index: NDArray[np.int64]
    period_slots: NDArray[np.int64]
    packet_bits: NDArray[np.float64]
    phase_slots: NDArray[np.int64]
    phase_seed: int | None = None

    def __post_init__(self) -> None:
        count = len(self.class_index)
        arrays = {
            "class_index": _frozen(self.class_index, np.int64),
            "period_slots": _frozen(self.period_slots, np.int64),
            "packet_bits": _frozen(self.packet_bits, np.float64),
            "phase_slots": _frozen(self.phase_slots, np.int64),
        }
        if count < 1 or any(a.shape != (count,) for a in arrays.values()):
            raise ValueError("traffic arrays must be nonempty vectors of one length")
        if np.any(arrays["period_slots"] < 1) or np.any(arrays["packet_bits"] <= 0):
            raise ValueError("periods and packet sizes must be positive")
        if not np.all(np.isfinite(arrays["packet_bits"])):
            raise ValueError("packet sizes must be finite")
        phases = arrays["phase_slots"]
        if np.any(phases < 0) or np.any(phases >= arrays["period_slots"]):
            raise ValueError("each phase must lie in [0, period)")
        index = arrays["class_index"]
        if np.any(index < 0) or np.any(index >= len(self.class_names)):
            raise ValueError("class_index is out of range")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "class_names", tuple(self.class_names))

    @property
    def num_ues(self) -> int:
        return len(self.class_index)

    @property
    def max_period_slots(self) -> int:
        return int(self.period_slots.max())

    def offered_bits_per_slot(self) -> float:
        return float(np.sum(self.packet_bits / self.period_slots))

    def to_record(self) -> dict:
        return {
            "schema_version": 1,
            "class_names": list(self.class_names),
            "class_index": self.class_index.tolist(),
            "period_slots": self.period_slots.tolist(),
            "packet_bits": self.packet_bits.tolist(),
            "phase_slots": self.phase_slots.tolist(),
            "phase_seed": self.phase_seed,
            "phase_sampling": PHASE_SAMPLING if self.phase_seed is not None else "explicit",
        }


def build_pattern(classes: Sequence[TrafficClass], phase_seed: int) -> TrafficPattern:
    """Assign UEs to classes in listed order and sample each UE's phase.

    UE positions are i.i.d., so the fixed class order does not tie classes to
    locations. Phases may coincide; they are not forced to be distinct.
    """
    classes = tuple(classes)
    if not classes or any(not isinstance(c, TrafficClass) for c in classes):
        raise ValueError("classes must be a nonempty sequence of TrafficClass")
    seed = _int(phase_seed, "phase_seed", 0)
    index = np.concatenate([np.full(c.num_ues, k, np.int64) for k, c in enumerate(classes)])
    if index.size == 0:
        raise ValueError("traffic classes contain no UE")
    periods = np.array([classes[k].period_slots for k in index], dtype=np.int64)
    bits = np.array([classes[k].packet_bits for k in index], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    phases = np.array([rng.integers(period) for period in periods], dtype=np.int64)
    return TrafficPattern(tuple(c.name for c in classes), index, periods, bits, phases, seed)


def pattern_from_record(record: Mapping) -> TrafficPattern:
    """Replay saved per-UE arrays; never re-sample from the seed."""
    return TrafficPattern(
        tuple(record["class_names"]),
        np.asarray(record["class_index"]),
        np.asarray(record["period_slots"]),
        np.asarray(record["packet_bits"]),
        np.asarray(record["phase_slots"]),
        record.get("phase_seed"),
    )
