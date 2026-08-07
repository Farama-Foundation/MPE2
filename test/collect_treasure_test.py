from __future__ import annotations

import numpy as np
import pytest

from mpe2 import collect_treasure_v1
from mpe2.collect_treasure.collect_treasure import Scenario


def _set_position(entity, x: float, y: float) -> None:
    entity.state.p_pos = np.array([x, y], dtype=float)
    entity.state.p_vel = np.zeros(2, dtype=float)


@pytest.fixture
def scenario_world():
    scenario = Scenario()
    world = scenario.make_world(num_collectors=1, num_deposits=1, num_treasures=1)
    scenario.reset_world(world, np.random.default_rng(123))
    return scenario, world


@pytest.fixture
def collect_treasure_env():
    env = collect_treasure_v1.parallel_env(
        num_collectors=1, num_deposits=1, num_treasures=1, max_cycles=5
    )
    env.reset(seed=123)
    yield env
    env.close()


def test_pickup_reward_is_preserved_after_post_step(scenario_world) -> None:
    scenario, world = scenario_world
    collector, deposit = world.agents
    treasure = world.landmarks[0]

    _set_position(collector, 0.0, 0.0)
    _set_position(treasure, 0.0, 0.0)
    _set_position(deposit, 0.9, 0.9)
    collector.holding = None
    treasure.alive = True
    treasure.type = 0

    scenario.post_step(world)

    assert collector.holding == 0
    assert treasure.alive is False
    assert scenario._global_collecting_reward(world) == 5.0
    assert scenario._global_deposit_reward(world) == 0.0
    assert scenario._global_reward(world) == 5.0


def test_delivery_reward_is_preserved_after_post_step(scenario_world) -> None:
    scenario, world = scenario_world
    collector, deposit = world.agents
    treasure = world.landmarks[0]

    _set_position(collector, 0.0, 0.0)
    _set_position(deposit, 0.0, 0.0)
    _set_position(treasure, 0.9, 0.9)
    collector.holding = 0
    deposit.d_i = 0

    scenario.post_step(world)

    assert collector.holding is None
    assert scenario._global_collecting_reward(world) == 0.0
    assert scenario._global_deposit_reward(world) == 5.0
    assert scenario._global_reward(world) == 5.0


def test_event_rewards_are_returned_by_parallel_step(collect_treasure_env) -> None:
    env = collect_treasure_env
    raw_env = env.unwrapped
    collector, deposit = raw_env.world.agents
    treasure = raw_env.world.landmarks[0]

    _set_position(collector, 0.0, 0.0)
    _set_position(treasure, 0.0, 0.0)
    _set_position(deposit, 1.0, 0.0)
    treasure.type = 0
    treasure.respawn_prob = 0.0

    _, pickup_rewards, _, _, _ = env.step({agent: 0 for agent in env.agents})

    assert pickup_rewards == pytest.approx({"collector_0": 4.9, "deposit_0": 4.9})

    _set_position(collector, 1.0, 0.0)
    _, delivery_rewards, _, _, _ = env.step({agent: 0 for agent in env.agents})

    assert delivery_rewards == pytest.approx({"collector_0": 5.0, "deposit_0": 5.0})

    _, next_rewards, _, _, _ = env.step({agent: 0 for agent in env.agents})

    assert next_rewards == pytest.approx({"collector_0": 0.0, "deposit_0": 0.0})
