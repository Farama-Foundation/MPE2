from __future__ import annotations

import numpy as np
import pytest

from mpe2 import (
    collect_treasure_v1,
    simple_adversary_v3,
    simple_crypto_v3,
    simple_formation_v1,
    simple_line_v1,
    simple_push_v3,
    simple_reference_v3,
    simple_speaker_listener_v4,
    simple_spread_v3,
    simple_tag_v3,
    simple_v3,
    simple_world_comm_v3,
)
from mpe2._mpe_utils.core import Agent, Landmark
from mpe2._mpe_utils.partial_observability import padded_relative_positions


def _agent_at(x: float, y: float) -> Agent:
    agent = Agent()
    agent.state.p_pos = np.array([x, y], dtype=np.float32)
    agent.state.p_vel = np.zeros(2, dtype=np.float32)
    agent.state.c = np.zeros(2, dtype=np.float32)
    return agent


def _landmark_at(x: float, y: float) -> Landmark:
    landmark = Landmark()
    landmark.state.p_pos = np.array([x, y], dtype=np.float32)
    landmark.state.p_vel = np.zeros(2, dtype=np.float32)
    return landmark


def test_compact_knn_keeps_nearest_first() -> None:
    observer = _agent_at(0.0, 0.0)
    entities = [
        _landmark_at(3.0, 0.0),
        _landmark_at(0.5, 0.0),
        _landmark_at(1.0, 0.0),
    ]

    positions = padded_relative_positions(observer, entities, 2)

    np.testing.assert_array_equal(positions, [[0.5, 0.0], [1.0, 0.0]])


def test_masked_knn_preserves_full_entity_slots() -> None:
    observer = _agent_at(0.0, 0.0)
    entities = [
        _landmark_at(3.0, 0.0),
        _landmark_at(0.5, 0.0),
        _landmark_at(1.0, 0.0),
    ]

    positions = padded_relative_positions(observer, entities, 2, knn_mode="masked")

    np.testing.assert_array_equal(positions, [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])


def test_radius_filters_before_optional_knn_cap() -> None:
    observer = _agent_at(0.0, 0.0)
    entities = [
        _landmark_at(0.75, 0.0),
        _landmark_at(0.25, 0.0),
        _landmark_at(2.0, 0.0),
    ]

    uncapped = padded_relative_positions(observer, entities, None, radius=1.0)
    capped = padded_relative_positions(observer, entities, 1, radius=1.0)
    masked = padded_relative_positions(
        observer, entities, None, radius=1.0, knn_mode="masked"
    )

    np.testing.assert_array_equal(uncapped, [[0.25, 0.0], [0.75, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(capped, [[0.25, 0.0]])
    np.testing.assert_array_equal(masked, [[0.75, 0.0], [0.25, 0.0], [0.0, 0.0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_agent_neighbors": 1},
        {"radius": 0.5},
        {"num_agent_neighbors": 3, "radius": 0.5},
    ],
)
def test_world_comm_compact_distinguishes_stationary_agent_roles(kwargs) -> None:
    env = simple_world_comm_v3.raw_env(
        num_good=2,
        num_adversaries=2,
        num_obstacles=0,
        num_food=1,
        num_forests=0,
        num_landmark_neighbors=1,
        **kwargs,
    )
    try:
        env.reset(seed=0)
        observer = env.world.agents[2]
        for agent in env.world.agents:
            agent.state.p_pos = np.array([2.0, 2.0])
            agent.state.p_vel = np.zeros(2)
        observer.state.p_pos = np.zeros(2)

        observations = []
        for index in (0, 1, 3):  # Leader, normal adversary, fellow good agent.
            neighbor = env.world.agents[index]
            neighbor.state.p_pos = np.array([0.05, 0.0])
            observations.append(env.observe(observer.name))
            neighbor.state.p_pos = np.array([2.0, 2.0])

        assert not np.array_equal(observations[0], observations[1])
        assert not np.array_equal(observations[1], observations[2])
        assert not np.array_equal(observations[0], observations[2])
        for observation in observations:
            assert env.observation_space(observer.name).contains(observation)
    finally:
        env.close()


def test_world_comm_compact_roles_follow_visibility_and_nearest_order() -> None:
    env = simple_world_comm_v3.raw_env(
        num_good=2,
        num_adversaries=3,
        num_obstacles=0,
        num_food=1,
        num_forests=1,
        num_agent_neighbors=5,
        num_landmark_neighbors=1,
        radius=0.5,
    )
    try:
        env.reset(seed=0)
        observer, fellow_good = env.world.agents[3:]
        for agent, position in zip(
            env.world.agents,
            [(0.2, 0.0), (0.0, 0.4), (0.8, 0.0), (0.0, 0.0), (0.1, 0.0)],
        ):
            agent.state.p_pos = np.array(position)
            agent.state.p_vel = np.zeros(2)
        env.world.forests[0].state.p_pos = env.world.agents[1].state.p_pos.copy()
        fellow_good.state.p_vel = np.array([1.0, 2.0])

        observation = env.observe(observer.name)
        # Self (4), one landmark position (2) and color (3), five agent positions (10).
        positions = observation[9:19].reshape(5, 2)
        roles = observation[19:34].reshape(5, 3)
        velocities = observation[35:].reshape(5, 2)  # Skip self_in_forest.
        np.testing.assert_array_equal(
            positions,
            np.array(
                [[0.1, 0.0], [0.2, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            roles, [[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        )
        np.testing.assert_array_equal(
            velocities, [[1, 2], [0, 0], [0, 0], [0, 0], [0, 0]]
        )
        assert env.observation_space(observer.name).contains(observation)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("env_module", "kwargs"),
    [
        (
            simple_spread_v3,
            {
                "N": 4,
                "num_agent_neighbors": 2,
                "num_landmark_neighbors": 2,
            },
        ),
        (
            simple_tag_v3,
            {
                "num_good": 1,
                "num_adversaries": 3,
                "num_obstacles": 2,
                "num_agent_neighbors": 2,
                "num_landmark_neighbors": 1,
            },
        ),
        (
            simple_adversary_v3,
            {
                "N": 3,
                "num_agent_neighbors": 2,
                "num_landmark_neighbors": 1,
            },
        ),
        (
            collect_treasure_v1,
            {
                "num_collectors": 4,
                "num_deposits": 2,
                "num_treasures": 4,
                "num_agent_neighbors": 2,
                "num_landmark_neighbors": 2,
            },
        ),
        (
            simple_push_v3,
            {"num_agent_neighbors": 1, "num_landmark_neighbors": 1},
        ),
        (
            simple_world_comm_v3,
            {"num_agent_neighbors": 3, "num_landmark_neighbors": 3},
        ),
    ],
)
def test_masked_mode_keeps_full_observation_shape(env_module, kwargs) -> None:
    full_kwargs = dict(kwargs)
    for key in ("num_agent_neighbors", "num_landmark_neighbors"):
        if key in full_kwargs:
            full_kwargs[key] = None
    full_env = env_module.env(**full_kwargs)
    masked_env = env_module.env(**kwargs, knn_mode="masked")
    full_env.reset(seed=0)
    masked_env.reset(seed=0)

    for agent in full_env.agents:
        assert (
            masked_env.observation_space(agent).shape
            == full_env.observation_space(agent).shape
        )

    full_env.close()
    masked_env.close()


@pytest.mark.parametrize(
    "env_module",
    [
        simple_spread_v3,
        simple_tag_v3,
        simple_adversary_v3,
        collect_treasure_v1,
        simple_push_v3,
        simple_world_comm_v3,
    ],
)
def test_radius_mode_has_fixed_declared_observation_shape(env_module) -> None:
    env = env_module.env(radius=0.5)
    env.reset(seed=0)

    for agent in env.agents:
        assert env.observe(agent).shape == env.observation_space(agent).shape

    env.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": 0.0},
        {"radius": float("inf")},
        {"knn_mode": "unknown"},
    ],
)
def test_partial_observability_arguments_are_validated(kwargs) -> None:
    with pytest.raises(AssertionError):
        simple_spread_v3.env(**kwargs)


@pytest.mark.parametrize(
    "env_module",
    [
        simple_v3,
        simple_crypto_v3,
        simple_formation_v1,
        simple_line_v1,
        simple_reference_v3,
        simple_speaker_listener_v4,
    ],
)
def test_landmark_only_or_nonspatial_environments_do_not_expose_po(env_module) -> None:
    with pytest.raises(TypeError):
        env_module.env(radius=0.5)
