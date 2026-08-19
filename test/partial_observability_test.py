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
