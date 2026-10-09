"""Information-boundary probe: AP broadcast flags and the yield rule (scripts/probe_broadcast.py)."""

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dsa_period import splits_v2
from dsa_period.capacity import run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.rules_v2 import make_policy_v2

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "scenario-v2.toml"
spec = importlib.util.spec_from_file_location("probe", ROOT / "scripts" / "probe_broadcast.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def short(n: int):
    scenario = load_scenario(CONFIG, n)
    return dataclasses.replace(scenario, horizon_slots=300, warmup_slots=40)


class Fixed:
    """Inner policy stub: always requests the same joint action."""

    name = "fixed"

    def __init__(self, actions):
        self.actions = np.array(actions, dtype=np.int64)

    def reset(self, rngs) -> None:
        pass

    def act(self, observations):
        return self.actions.copy()


def test_none_reproduces_the_protocol_episode():
    scenario = short(6)
    reference = run_capacity_episode(
        scenario, make_policy_v2("best_local", scenario.physical, scenario.menu), "dev", 1000
    )
    inner = make_policy_v2("best_local", scenario.physical, scenario.menu)
    out = probe.run(scenario, probe.Broadcast(inner, scenario, "none"), 1000)
    assert out["violations"] == reference["violations"]
    assert out["feasible"] == reference["feasible"]
    assert out["moved"] == out["silenced"] == 0


def _wrapper(mode, prev_urgent):
    scenario = short(4)
    wrapper = probe.Broadcast(Fixed(np.zeros((4, 2))), scenario, mode)
    wrapper.reset(splits_v2.execution_generators("dev", 4, 1000, 0))
    wrapper.prev_urgent = np.array(prev_urgent)
    return wrapper


def test_b2_blocks_subchannels_used_by_others():
    wrapper = _wrapper("B2", [False] * 4)
    prev = np.array([[0, 3], [0, 2], [2, 1], [3, 0]])  # UE 3 was OFF
    assert wrapper.blocked(prev, 0).tolist() == [True, False, True, False]  # UE 1 shares 0
    assert wrapper.blocked(prev, 2).tolist() == [True, False, False, False]  # alone on 2
    assert wrapper.blocked(prev, 3).tolist() == [True, False, True, False]


def test_b3_blocks_only_urgent_occupants_and_not_the_own_flag():
    wrapper = _wrapper("B3", [True, False, False, True])
    prev = np.array([[1, 3], [2, 2], [2, 1], [3, 0]])  # UE 3 urgent but was OFF
    assert wrapper.blocked(prev, 1).tolist() == [False, True, False, False]
    assert wrapper.blocked(prev, 0).tolist() == [False, False, False, False]  # its own flag


@pytest.mark.parametrize("mode", ["B2", "B2r"])
def test_yield_defers_or_moves_calm_ues_and_leaves_urgent_ones(mode):
    scenario = short(4)
    env = scenario.make_env(*scenario.instance(1000), splits_v2.fading_seeds("dev", 4, 1000, 0))
    observations = env.reset()
    for _ in range(100):  # until some UE among 0-2 waits with slack to spare
        observations = env.step(np.zeros((4, 2), dtype=np.int64)).observations
        calm = probe.slack(scenario, observations) >= probe.URGENT_SLACK
        if any(observations[ue].remaining_bits > 0 and calm[ue] for ue in range(3)):
            break
    waiting = np.array([o.remaining_bits > 0 for o in observations])
    request = np.array([[1, 5]] * 4)
    wrapper = probe.Broadcast(Fixed(request), scenario, mode)
    wrapper.reset(splits_v2.execution_generators("dev", 4, 1000, 0))
    # Nobody transmitted in the previous slot: nothing is blocked.
    assert np.array_equal(wrapper.act(observations), request)
    # Pretend another UE held subchannel 1 in the previous slot.
    held = [dataclasses.replace(o, previous_action=(1, 5) if u == 3 else (0, 0))
            for u, o in enumerate(observations)]  # fmt: skip
    actions = wrapper.act(held)
    urgent = wrapper.urgent
    assert any(waiting[ue] and not urgent[ue] for ue in range(3))  # the rule is exercised
    for ue in range(3):
        if not waiting[ue] or urgent[ue]:
            assert actions[ue].tolist() == [1, 5]
        elif mode == "B2":
            assert actions[ue].tolist() == [0, 0]
        else:
            gains = np.array(held[ue].own_gain_w)
            gains[1] = -np.inf
            assert actions[ue].tolist() == [int(np.argmax(gains)), 5]


def test_flag_loss_variants_hear_only_lone_or_strong_transmitters():
    scenario = short(4)
    env = scenario.make_env(*scenario.instance(1000), splits_v2.fading_seeds("dev", 4, 1000, 0))
    observations = env.reset()
    prev = np.array([[1, 3], [1, 2], [2, 1], [3, 0]])  # UEs 0 and 1 collide, UE 3 OFF
    bits = [5000.0, 200.0, 800.0, 0.0]
    held = [dataclasses.replace(o, previous_action=tuple(prev[u]), previous_service_bits=bits[u])
            for u, o in enumerate(observations)]  # fmt: skip
    expected = {
        "B3": [True, True, True, False],
        "B3c": [False, False, True, False],
        "B3s": [True, False, False, False],  # 0 dB = channel uses per slot bits
    }
    assert scenario.physical.channel_uses_per_slot == 1000
    for mode, heard in expected.items():
        wrapper = probe.Broadcast(Fixed(np.zeros((4, 2))), scenario, mode)
        wrapper.reset(splits_v2.execution_generators("dev", 4, 1000, 0))
        assert wrapper.received(prev, held).tolist() == heard
    # A lost flag does not block: UE 2 urgent and alone on 2 is heard by B3c,
    # urgent UE 1 collided on 1 is not.
    wrapper = _wrapper("B3c", [False, True, True, False])
    wrapper.heard = wrapper.received(prev, held)
    assert wrapper.blocked(prev, 3).tolist() == [False, False, True, False]
