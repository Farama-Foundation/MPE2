from __future__ import annotations

import pytest
from pettingzoo.test import max_cycles_test, parallel_api_test
from pettingzoo.test.api_test import api_test
from pettingzoo.test.render_test import render_test
from pettingzoo.test.seed_test import parallel_seed_test, seed_test
from pettingzoo.test.state_test import state_test

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

partial_obs_envs = [
    # simple_tag: N nearest agents + landmarks (fewer than actual → zero-padding)
    [
        simple_tag_v3,
        dict(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            num_agent_neighbors=2,
            num_landmark_neighbors=1,
            max_cycles=50,
        ),
    ],
    # simple_tag: more neighbors than agents/landmarks → no padding needed
    [
        simple_tag_v3,
        dict(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            num_agent_neighbors=100,
            num_landmark_neighbors=100,
            max_cycles=50,
        ),
    ],
    # simple_tag: only agent neighbors limited, landmarks fully visible
    [
        simple_tag_v3,
        dict(
            num_good=2,
            num_adversaries=4,
            num_obstacles=3,
            num_agent_neighbors=2,
            max_cycles=50,
        ),
    ],
    # simple_tag: continuous actions with PO
    [
        simple_tag_v3,
        dict(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            num_agent_neighbors=2,
            num_landmark_neighbors=1,
            continuous_actions=True,
            max_cycles=50,
        ),
    ],
    # simple_spread: N nearest agents + landmarks with communication filtering
    [
        simple_spread_v3,
        dict(N=5, num_agent_neighbors=2, num_landmark_neighbors=2, max_cycles=50),
    ],
    # simple_spread: only landmark neighbors limited
    [
        simple_spread_v3,
        dict(N=4, num_landmark_neighbors=2, max_cycles=50),
    ],
    # simple_spread: continuous actions with PO
    [
        simple_spread_v3,
        dict(
            N=5,
            num_agent_neighbors=2,
            num_landmark_neighbors=2,
            continuous_actions=True,
            max_cycles=50,
        ),
    ],
    # simple_spread: more neighbors than exist → no padding needed (graceful)
    [
        simple_spread_v3,
        dict(N=3, num_agent_neighbors=100, num_landmark_neighbors=100, max_cycles=50),
    ],
    # simple_adversary: PO with disclaimer (solvability not guaranteed)
    [
        simple_adversary_v3,
        dict(N=2, num_agent_neighbors=2, num_landmark_neighbors=1, max_cycles=50),
    ],
    # simple_adversary: only agent neighbors limited
    [
        simple_adversary_v3,
        dict(N=3, num_agent_neighbors=2, max_cycles=50),
    ],
]

parameterized_envs = [
    [collect_treasure_v1, dict(max_cycles=50)],
    [
        collect_treasure_v1,
        dict(num_collectors=4, num_deposits=2, num_treasures=4, max_cycles=50),
    ],
    [
        collect_treasure_v1,
        dict(continuous_actions=True, max_cycles=50),
    ],
    [
        collect_treasure_v1,
        dict(
            num_collectors=3,
            num_deposits=1,
            num_treasures=3,
            continuous_actions=True,
            max_cycles=50,
        ),
    ],
    [simple_v3, dict(max_cycles=50)],
    [simple_v3, dict(continuous_actions=True, max_cycles=50)],
    [simple_push_v3, dict(max_cycles=50)],
    [
        simple_push_v3,
        dict(continuous_actions=True, max_cycles=50),
    ],
    [simple_crypto_v3, dict(max_cycles=50)],
    [
        simple_crypto_v3,
        dict(continuous_actions=True, max_cycles=50),
    ],
    [simple_speaker_listener_v4, dict(max_cycles=50)],
    [
        simple_speaker_listener_v4,
        dict(continuous_actions=True, max_cycles=50),
    ],
    [simple_adversary_v3, dict(N=4, max_cycles=50)],
    [
        simple_reference_v3,
        dict(local_ratio=0.2, max_cycles=50),
    ],
    [simple_formation_v1, dict(max_cycles=50)],
    [simple_formation_v1, dict(N=3, max_cycles=50)],
    [simple_formation_v1, dict(N=6, max_cycles=50)],
    [simple_formation_v1, dict(N=4, continuous_actions=True, max_cycles=50)],
    [simple_formation_v1, dict(N=4, terminate_on_success=True, max_cycles=50)],
    [simple_line_v1, dict(max_cycles=50)],
    [simple_line_v1, dict(N=3, max_cycles=50)],
    [simple_line_v1, dict(N=6, max_cycles=50)],
    [simple_line_v1, dict(N=4, continuous_actions=True, max_cycles=50)],
    [simple_line_v1, dict(N=4, terminate_on_success=True, max_cycles=50)],
    [simple_spread_v3, dict(N=5, max_cycles=50)],
    [
        simple_tag_v3,
        dict(num_good=5, num_adversaries=10, num_obstacles=4, max_cycles=50),
    ],
    [
        simple_tag_v3,
        dict(num_good=1, num_adversaries=1, num_obstacles=1, max_cycles=50),
    ],
    [
        simple_tag_v3,
        dict(
            num_good=5,
            num_adversaries=10,
            num_obstacles=4,
            continuous_actions=True,
            max_cycles=50,
        ),
    ],
    [
        simple_tag_v3,
        dict(
            num_good=1,
            num_adversaries=1,
            num_obstacles=1,
            continuous_actions=True,
            max_cycles=50,
        ),
    ],
    [
        simple_world_comm_v3,
        dict(
            num_good=5, num_adversaries=10, num_obstacles=4, num_food=3, max_cycles=50
        ),
    ],
    [
        simple_world_comm_v3,
        dict(num_good=1, num_adversaries=1, num_obstacles=1, num_food=1, max_cycles=50),
    ],
    [
        simple_world_comm_v3,
        dict(
            num_good=5,
            num_adversaries=10,
            num_obstacles=4,
            num_food=3,
            continuous_actions=True,
            max_cycles=50,
        ),
    ],
    [
        simple_world_comm_v3,
        dict(
            num_good=1,
            num_adversaries=1,
            num_obstacles=1,
            num_food=1,
            continuous_actions=True,
            max_cycles=50,
        ),
    ],
    [
        simple_adversary_v3,
        dict(N=4, continuous_actions=True, max_cycles=50),
    ],
    [
        simple_reference_v3,
        dict(local_ratio=0.2, continuous_actions=True, max_cycles=50),
    ],
    [
        simple_spread_v3,
        dict(N=5, continuous_actions=True, max_cycles=50),
    ],
]


@pytest.mark.parametrize(["env_module", "kwargs"], parameterized_envs)
def test_module(env_module, kwargs):
    _env = env_module.env(**kwargs)
    api_test(_env)
    parallel_api_test(env_module.parallel_env())
    max_cycles_test(env_module)
    parallel_seed_test(lambda: env_module.parallel_env(**kwargs), 500)
    seed_test(lambda: env_module.env(**kwargs), 500)

    render_test(lambda render_mode: env_module.env(render_mode=render_mode, **kwargs))
    state_test(env_module.env(), env_module.parallel_env())


@pytest.mark.parametrize(["env_module", "kwargs"], partial_obs_envs)
def test_partial_observability(env_module, kwargs):
    """Verify that partial-observability configurations satisfy PettingZoo API.
    """
    _env = env_module.env(**kwargs)
    api_test(_env)
    parallel_api_test(env_module.parallel_env(**kwargs))
    seed_test(lambda: env_module.env(**kwargs), 500)
    parallel_seed_test(lambda: env_module.parallel_env(**kwargs), 500)
    state_test(env_module.env(**kwargs), env_module.parallel_env(**kwargs))
    
    _env = env_module.env(**kwargs)
    _env.reset(seed=0)
    for agent in _env.agents:
        declared_shape = _env.observation_space(agent).shape
        obs = _env.observe(agent)
        assert obs.shape == declared_shape, (
            f"Agent {agent}: declared obs shape {declared_shape} != "
            f"actual shape {obs.shape}"
        )
