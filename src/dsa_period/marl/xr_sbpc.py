"""XR-SBPC rewards (EXP-DSA05-005) and per-packet shaping diagnostics.

XR-SBPC (Bagherinejad et al., WCNC 2025, eq. 7) rewards each UE holding a packet
in every slot. With S the remaining bits and D the slots to the deadline at the
start of the slot (D = 1 in the last slot), R the mutual information the UE
received in the slot and phi = S / D its target throughput:

* v1 (the paper's reward): +success if the packet completes (R >= S); otherwise
  -failure if D = 1 (it expires); otherwise +progress * R / phi if R >= phi and
  -lag if not;
* v1s (the paper's objective without shaping): +success on completion, -failure
  on expiry, 0 otherwise.

Idle UEs (no packet) receive 0. ``PacketTracker`` accumulates, per packet, the
v1 intermediate (shaping) reward and the R / phi of transmitting slots that did
not complete the packet, so any policy's behaviour can be described in the terms
of that shaping.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

REWARD_KINDS = ("v1", "v1s", "v2")


@dataclass(frozen=True)
class RewardSettings:
    """``kind`` v1/v1s use the XR-SBPC terms; v2 is the EXP-DSA05-001 team reward."""

    kind: str
    success: float = 50.0
    failure: float = 100.0
    progress: float = 5.0
    lag: float = 10.0
    weight: float = 10.0  # v2: violation penalty in units of E0 = tau * p_max

    def __post_init__(self) -> None:
        if self.kind not in REWARD_KINDS:
            raise ValueError(f"reward kind must be one of {REWARD_KINDS}")
        if min(self.success, self.failure, self.progress, self.lag, self.weight) < 0:
            raise ValueError("reward magnitudes are non-negative; signs are fixed by the terms")


def intermediate(
    remaining: NDArray, to_deadline: NDArray, service: NDArray, progress: float, lag: float
) -> NDArray[np.float64]:
    """v1 shaping term of each UE; meaningful only where the packet neither completes
    nor expires (callers mask the other cases)."""
    remaining = np.asarray(remaining, dtype=np.float64)
    target = remaining / np.maximum(np.asarray(to_deadline, dtype=np.float64), 1.0)
    ratio = np.divide(service, target, out=np.zeros_like(target), where=target > 0)
    return np.where(ratio >= 1.0, progress * ratio, -lag)


def xr_rewards(
    settings: RewardSettings,
    remaining: NDArray,
    to_deadline: NDArray,
    service: NDArray,
    completed: NDArray,
    expired: NDArray,
) -> NDArray[np.float64]:
    """Per-UE v1 or v1s reward of one slot, from the state at its start and its outcome."""
    if settings.kind not in ("v1", "v1s"):
        raise ValueError("xr_rewards covers v1 and v1s only")
    holding = np.asarray(remaining) > 0
    completed = np.asarray(completed, dtype=bool)
    expired = np.asarray(expired, dtype=bool)
    if np.any(completed & ~holding) or np.any(expired & ~holding) or np.any(completed & expired):
        raise ValueError("only a UE holding a packet can complete or expire, not both")
    reward = np.zeros(holding.shape)
    reward[completed] = settings.success
    reward[expired] = -settings.failure
    if settings.kind == "v1":
        middle = holding & ~completed & ~expired
        shaped = intermediate(remaining, to_deadline, service, settings.progress, settings.lag)
        reward[middle] = shaped[middle]
    return reward


class PacketTracker:
    """Per-packet v1 shaping bookkeeping over one episode, keyed by (ue, arrival slot).

    Call ``observe`` once per slot with the per-UE state at the start of the slot
    and the slot's outcome. A packet's arrival slot is t - (period - to_deadline).
    """

    def __init__(self, period: NDArray, progress: float = 5.0, lag: float = 10.0) -> None:
        self.period = np.asarray(period, dtype=np.int64)
        self.progress, self.lag = progress, lag
        self.shaping: dict[tuple[int, int], float] = {}
        self.ratios: dict[tuple[int, int], list[float]] = {}

    def observe(self, t, remaining, to_deadline, service, completed, expired) -> None:
        remaining = np.asarray(remaining, dtype=np.float64)
        to_deadline = np.asarray(to_deadline)
        shaped = intermediate(remaining, to_deadline, service, self.progress, self.lag)
        target = remaining / np.maximum(to_deadline, 1)
        for ue in np.flatnonzero(remaining > 0):
            key = (int(ue), int(t - (self.period[ue] - to_deadline[ue])))
            self.shaping.setdefault(key, 0.0)
            self.ratios.setdefault(key, [])
            if completed[ue] or expired[ue]:
                continue
            self.shaping[key] += float(shaped[ue])
            if service[ue] > 0:
                self.ratios[key].append(float(service[ue] / target[ue]))

    def collect(self, jobs: list[dict]) -> dict:
        """Per-packet lists over the given settled job records (e.g. those in the window)."""
        out = {"tx_slots": [], "completion_fraction": [], "expired": 0, "shaping": [], "ratios": []}
        for job in jobs:
            key = (int(job["ue"]), int(job["arrival_slot"]))
            out["tx_slots"].append(int(job["tx_slots"]))
            out["shaping"].append(self.shaping.get(key, 0.0))
            out["ratios"] += self.ratios.get(key, [])
            if job["success"]:
                period = job["deadline_slot"] - job["arrival_slot"]
                out["completion_fraction"].append(
                    (job["completion_slot"] - job["arrival_slot"]) / period
                )
            else:
                out["expired"] += 1
        return out


def packet_summary(collected: list[dict]) -> dict:
    """Pool ``PacketTracker.collect`` outputs of several episodes into packet statistics."""
    tx = np.concatenate([np.asarray(c["tx_slots"], float) for c in collected] or [np.zeros(0)])
    frac = np.concatenate(
        [np.asarray(c["completion_fraction"], float) for c in collected] or [np.zeros(0)]
    )
    shaping = np.concatenate([np.asarray(c["shaping"], float) for c in collected] or [np.zeros(0)])
    ratios = np.concatenate([np.asarray(c["ratios"], float) for c in collected] or [np.zeros(0)])
    packets = tx.size

    def stat(values, fn):
        return float(fn(values)) if values.size else None

    return {
        "packets": int(packets),
        "expired_share": sum(c["expired"] for c in collected) / packets if packets else None,
        "tx_slots_per_packet": stat(tx, np.mean),
        "completion_fraction_mean": stat(frac, np.mean),
        "completion_fraction_p90": stat(frac, lambda v: np.percentile(v, 90)),
        "shaping_per_packet_mean": stat(shaping, np.mean),
        "shaping_per_packet_p90": stat(shaping, lambda v: np.percentile(v, 90)),
        "noncompleting_tx_slots": int(ratios.size),
        "noncompleting_ratio_p50": stat(ratios, np.median),
        "noncompleting_ratio_p90": stat(ratios, lambda v: np.percentile(v, 90)),
        "noncompleting_ratio_ge_1": stat(ratios, lambda v: np.mean(v >= 1.0)),
    }
