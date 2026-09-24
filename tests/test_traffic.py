import numpy as np
import pytest

from dsa_period.traffic import TrafficClass, TrafficPattern, build_pattern, pattern_from_record

CLASSES = (
    TrafficClass("fast", 10, 1000.0, 3),
    TrafficClass("slow", 40, 4000.0, 2),
)


def test_classes_are_assigned_in_order_with_phases_inside_each_period():
    pattern = build_pattern(CLASSES, 7)
    assert pattern.class_index.tolist() == [0, 0, 0, 1, 1]
    assert pattern.period_slots.tolist() == [10, 10, 10, 40, 40]
    assert pattern.packet_bits.tolist() == [1000.0] * 3 + [4000.0] * 2
    assert np.all((0 <= pattern.phase_slots) & (pattern.phase_slots < pattern.period_slots))
    assert pattern.offered_bits_per_slot() == pytest.approx(3 * 100 + 2 * 100)


def test_phases_are_seeded_and_replay_from_the_record():
    first, again, other = (
        build_pattern(CLASSES, 7),
        build_pattern(CLASSES, 7),
        build_pattern(CLASSES, 8),
    )
    assert first.phase_slots.tolist() == again.phase_slots.tolist()
    assert first.phase_slots.tolist() != other.phase_slots.tolist()
    replay = pattern_from_record(first.to_record())
    assert replay.to_record() == first.to_record()


@pytest.mark.parametrize(
    "phases, periods",
    [([0, 5], [5, 5]), ([-1, 0], [5, 5]), ([0, 0], [0, 5])],
)
def test_pattern_rejects_phases_outside_the_period(phases, periods):
    with pytest.raises(ValueError):
        TrafficPattern(("c",), np.zeros(2, int), np.array(periods), np.ones(2), np.array(phases))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "period_slots": 1, "packet_bits": 1.0, "num_ues": 1},
        {"name": "c", "period_slots": 0, "packet_bits": 1.0, "num_ues": 1},
        {"name": "c", "period_slots": 1, "packet_bits": 0.0, "num_ues": 1},
        {"name": "c", "period_slots": 1, "packet_bits": 1.0, "num_ues": -1},
    ],
)
def test_traffic_class_validation(kwargs):
    with pytest.raises(ValueError):
        TrafficClass(**kwargs)
