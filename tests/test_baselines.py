import json
from pathlib import Path

import numpy as np
import pytest
from conftest import MENU, TOP, build_env, solo_bits

from dsa_period.baselines import act, estimate_link
from dsa_period.config import load_scenario
from dsa_period.experiments import scenario_smoke
from dsa_period.physics import shannon_bits

CONFIG = str(Path(__file__).parents[1] / "configs" / "scenario-default.toml")


def rngs(n, seed=0):
    return [np.random.Generator(np.random.PCG64(s)) for s in np.random.SeedSequence(seed).spawn(n)]


def test_link_estimate_recovers_own_gain_and_others_interference():
    env = build_env([[30.0, 0.0], [0.0, 60.0]], [5, 5], [1e9, 1e9], [0, 0], num_subchannels=2)
    obs = env.step([[0, TOP], [0, 2]]).observations
    gain, interference = estimate_link(obs[0], env.config, MENU)
    true_gain = env.channel.gains(0)
    noise = env.config.noise_power_w
    assert gain == pytest.approx(true_gain[0, 0], rel=1e-9)
    assert interference[0] == pytest.approx(noise + MENU.power_w[2] * true_gain[1, 0], rel=1e-9)
    assert interference[1] == pytest.approx(noise)


def test_lazy_pacing_keeps_the_channel_at_the_lowest_sufficient_power():
    s_top = solo_bits([20.0, 0.0])
    env = build_env([[20.0, 0.0]], [10], [5 * s_top], [0], num_subchannels=3)
    obs = env.step([[2, TOP]]).observations
    required = obs[0].remaining_bits / obs[0].slots_to_deadline
    gain = env.channel.gains(0)[0, 0]
    noise = env.config.noise_power_w
    expected = next(
        p
        for p in range(1, MENU.num_levels)
        if shannon_bits(env.config, MENU.power_w[p] * gain / noise) >= required
    )
    assert act("lazy_pacing", obs, env.config, MENU, rngs(1)).tolist() == [[2, expected]]
    assert expected < TOP


def test_idle_ues_choose_off_without_consuming_randomness():
    env = build_env([[20.0, 0.0]], [5], [1.0], [3])
    generators = rngs(1)
    before = generators[0].bit_generator.state
    for policy in ("random_max", "sticky_max", "lazy_pacing"):
        assert act(policy, env.observe(), env.config, MENU, generators).tolist() == [[0, 0]]
    assert generators[0].bit_generator.state == before


def test_policies_reject_unknown_names_and_shared_generators():
    env = build_env([[20.0, 0.0], [0.0, 20.0]], [5, 5], [1.0, 1.0], [0, 0])
    shared = np.random.default_rng(0)
    with pytest.raises(ValueError):
        act("oracle", env.observe(), env.config, MENU, rngs(2))
    with pytest.raises(ValueError):
        act("random_max", env.observe(), env.config, MENU, [shared, shared])


def test_default_config_builds_reproducible_instances():
    scenario = load_scenario(CONFIG)
    (top_a, traffic_a), (top_b, traffic_b) = scenario.instance(3), scenario.instance(3)
    assert top_a.to_record() == top_b.to_record()
    assert traffic_a.to_record() == traffic_b.to_record()
    assert scenario.instance(4)[0].to_record() != top_a.to_record()


def test_smoke_runs_end_to_end_and_refuses_to_overwrite(tmp_path, capsys):
    output = tmp_path / "smoke"
    scenario_smoke.main(["--config", CONFIG, "--instances", "1", "--output", str(output)])
    record = json.loads((output / "summary.json").read_text())
    assert record["formal"] is False and len(record["runs"]) == 3
    assert all(run["summary"]["packets"] > 0 for run in record["runs"])
    with pytest.raises(SystemExit):
        scenario_smoke.main(["--config", CONFIG, "--instances", "1", "--output", str(output)])
