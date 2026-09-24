"""Fixed-window evaluation of on-time completion and transmit energy.

The evaluated packet set is fixed by the arrival schedule, not by the policy:
every packet with warmup < deadline <= horizon, i.e. the packets whose whole
lifetime-ending boundary lies in the measured slots [warmup, horizon). Packets
still open at the horizon are right-censored and excluded, never counted late.
"""

from collections.abc import Sequence

import numpy as np


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def window_summary(
    jobs: Sequence[dict],
    energy_by_slot: np.ndarray,
    *,
    warmup_slots: int,
    horizon_slots: int,
    slot_duration_s: float,
    num_ues: int,
) -> dict:
    if not 0 <= warmup_slots < horizon_slots:
        raise ValueError("require 0 <= warmup_slots < horizon_slots")
    energy = np.asarray(energy_by_slot, dtype=np.float64)
    if energy.shape[0] < horizon_slots or energy.shape[1:] != (num_ues,):
        raise ValueError("energy_by_slot must cover every slot up to the horizon")
    window = [j for j in jobs if warmup_slots < j["deadline_slot"] <= horizon_slots]
    if any(j["success"] is None for j in window):
        raise ValueError("every packet in the window must be settled")
    slot_energy = energy[warmup_slots:horizon_slots]
    duration_s = (horizon_slots - warmup_slots) * slot_duration_s
    on_time = sum(j["success"] for j in window)
    latencies = np.array(
        [j["completion_slot"] - j["arrival_slot"] for j in window if j["success"]], dtype=float
    )

    per_ue = []
    for ue in range(num_ues):
        own = [j for j in window if j["ue"] == ue]
        success = sum(j["success"] for j in own)
        per_ue.append(
            {
                "ue": ue,
                "packets": len(own),
                "on_time": success,
                "ocr": _ratio(success, len(own)),
                "energy_j": float(slot_energy[:, ue].sum()),
                "average_power_w": float(slot_energy[:, ue].sum() / duration_s),
            }
        )
    per_class: dict[str, dict] = {}
    for job in window:
        entry = per_class.setdefault(job["traffic_class"], {"packets": 0, "on_time": 0})
        entry["packets"] += 1
        entry["on_time"] += int(job["success"])
    for entry in per_class.values():
        entry["ocr"] = _ratio(entry["on_time"], entry["packets"])

    ue_ocr = [row["ocr"] for row in per_ue if row["ocr"] is not None]
    window_energy = float(slot_energy.sum())
    return {
        "warmup_slots": warmup_slots,
        "horizon_slots": horizon_slots,
        "packets": len(window),
        "on_time": on_time,
        "violations": len(window) - on_time,
        "ocr": _ratio(on_time, len(window)),
        "worst_ue_ocr": min(ue_ocr) if ue_ocr else None,
        "window_energy_j": window_energy,
        "average_power_per_ue_w": window_energy / duration_s / num_ues,
        "packet_energy_j": float(sum(j["energy_j"] for j in window)),
        "energy_per_on_time_packet_j": _ratio(sum(j["energy_j"] for j in window), on_time),
        "dropped_bits": float(sum(j["dropped_bits"] for j in window)),
        "latency_slots_mean": float(latencies.mean()) if latencies.size else None,
        "latency_slots_p95": float(np.percentile(latencies, 95)) if latencies.size else None,
        "per_ue": per_ue,
        "per_class": per_class,
    }
