"""Survival time (EXP-DSA05-012, design §2, §3): consecutive deadline misses per UE.

Times are slot boundaries. A packet with deadline d is either decoded at a
delivery time c <= d (the boundary after its decoding slot) or settled as a miss
at d. With a survival time of ``tolerance`` transfer intervals (the UE's
period), a UE tolerates that many consecutive misses:

* survival-time state: entered when a miss is settled, left when a later packet
  of the UE is decoded; k is the number of consecutive misses (0 after a decode);
* outage: starts at the deadline of the (tolerance + 1)-th consecutive miss and
  ends at the delivery time of the next decoded packet (not at its deadline);
  further misses of the run extend it. Unavailability is the share of the
  evaluation window spent in outages, availability its complement.

``SurvivalTracker`` follows k and the outage flag per UE from local information
(the UE's own deadline settlements and decode feedback). ``SurvivalSettings``
configures the method components C1 (state input) and C3 (cost per outage slot).
``BoostPolicy`` is the R17-like rule (design §3.4). ``service_record`` measures
outages and unavailable slots of an episode from the environment's job records.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dsa_period.env import Outcome
from dsa_period.marl.features import DualSettings, LocalMemory, ObsArrays, Settlement, stack
from dsa_period.physics import PhysicalConfig, PowerMenu

STATE_DIM = 4  # one-hot min(k, 2) and the misses left before an outage, / tolerance
TOLERANCES = (1, 2)  # survival times measured in evaluation records (design §2)


@dataclass(frozen=True)
class SurvivalSettings:
    """Survival time in transfer intervals and the method components that use it."""

    tolerance: int = 1  # s: consecutive misses tolerated; s + 1 starts an outage
    features: bool = True  # C1: the actor sees its own survival state
    outage_slot_weight: float = 1.0  # C3: lambda_d, cost of each UE-slot in an outage

    def __post_init__(self) -> None:
        if self.tolerance < 1 or self.outage_slot_weight < 0:
            raise ValueError("need tolerance >= 1 and outage_slot_weight >= 0")


class SurvivalTracker:
    """Consecutive misses k_i and the outage flag of each UE of one environment."""

    def __init__(self, num_ues: int, tolerance: int) -> None:
        self.num_ues, self.tolerance = num_ues, tolerance
        self.reset()

    def reset(self) -> None:
        self.consecutive = np.zeros(self.num_ues, dtype=np.int64)
        self.outage = np.zeros(self.num_ues, dtype=np.bool_)

    def update(self, settlement: Settlement, obs: ObsArrays) -> None:
        """Apply the boundary just observed: decodes of the last slot, then settled misses.

        A decode and a settled miss of the same UE cannot meet at one boundary: a
        packet decoded in the last slot before its deadline settles as on time.
        """
        decoded = obs.prev_outcome == Outcome.DECODED
        self.consecutive[decoded] = 0
        self.outage[decoded] = False
        missed = np.asarray(settlement.violated, dtype=np.bool_)
        self.consecutive[missed] += 1
        self.outage |= missed & (self.consecutive > self.tolerance)

    @property
    def in_state(self) -> NDArray[np.bool_]:
        """In the survival-time state (outages included): k >= 1."""
        return self.consecutive >= 1

    def features(self) -> NDArray[np.float32]:
        """(N, STATE_DIM): one-hot of min(k, 2) and max(s - k, 0) / s."""
        out = np.zeros((self.num_ues, STATE_DIM), dtype=np.float32)
        out[np.arange(self.num_ues), np.minimum(self.consecutive, 2)] = 1.0
        out[:, 3] = np.maximum(self.tolerance - self.consecutive, 0) / self.tolerance
        return out


def settlement_memory(config: PhysicalConfig, menu: PowerMenu) -> LocalMemory:
    """A ``LocalMemory`` used only to detect deadline settlements (rules, wrappers)."""
    return LocalMemory(config, menu, config.num_ues, DualSettings("fixed", 0.01, weight=1.0))


class BoostPolicy:
    """Any policy plus the R17-like survival-state rule (EXP-DSA05-012 design §3.4).

    A UE with a packet in the survival-time state sends on the subchannel with its
    best current own gain at maximum power, whatever the inner policy chose; other
    UEs act as the inner policy says. It approximates the Rel-17 reaction (a more
    robust transmission after a failure) within the action space; PDCP duplication
    itself would need two subchannels in one slot. No random numbers are drawn.
    """

    def __init__(self, inner, config: PhysicalConfig, menu: PowerMenu) -> None:
        self.inner, self.config, self.menu = inner, config, menu
        self.name = f"boost[]@{getattr(inner, 'name', 'policy')}"
        self._memory = settlement_memory(config, menu)
        self.tracker = SurvivalTracker(config.num_ues, 1)

    def reset(self, rngs) -> None:
        self.inner.reset(rngs)
        self._memory.reset(np.zeros(self.config.num_ues))
        self.tracker.reset()
        self.boosted = np.zeros(self.config.num_ues, dtype=np.int64)

    def act(self, observations) -> NDArray[np.int64]:
        obs = stack(observations)
        self.tracker.update(self._memory.observe(obs), obs)
        actions = np.array(self.inner.act(observations), dtype=np.int64, copy=True)
        if obs.own_gain is None:
            raise ValueError("the boost rule needs scenario v2 observations (own gains)")
        boost = (obs.remaining > 0) & self.tracker.in_state
        for ue in np.flatnonzero(boost):
            actions[ue] = (int(np.argmax(obs.own_gain[ue])), self.menu.num_levels - 1)
            self.boosted[ue] += 1
        return actions


def service_record(
    jobs: list[dict], num_ues: int, warmup: int, horizon: int, tolerance: int
) -> dict[str, list[int]]:
    """Per UE: outages starting in (warmup, horizon] and unavailable slots within it.

    ``jobs`` are the environment's job records. Runs of consecutive misses use every
    released packet (warm-up included); a packet still pending at ``horizon`` ends
    the scan, and an outage still open then lasts until ``horizon``.
    """
    by_ue: list[list[dict]] = [[] for _ in range(num_ues)]
    for job in jobs:
        by_ue[job["ue"]].append(job)
    outages, down = [0] * num_ues, [0] * num_ues

    def add(ue: int, start: int, end: int) -> None:
        down[ue] += max(0, min(end, horizon) - max(start, warmup))

    for ue, rows in enumerate(by_ue):
        run, start = 0, None
        for job in sorted(rows, key=lambda j: j["deadline_slot"]):
            if job["completion_slot"] is not None:
                if start is not None:
                    add(ue, start, job["completion_slot"])
                run, start = 0, None
            elif job["success"] is False:
                run += 1
                if run == tolerance + 1:
                    start = job["deadline_slot"]
                    outages[ue] += int(warmup < start <= horizon)
            else:
                break  # pending at the horizon
        if start is not None:
            add(ue, start, horizon)
    return {"outages": outages, "unavailable_slots": down}
