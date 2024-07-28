from __future__ import annotations

import pytest

from mpe2 import (
    simple_adversary_v3,
    simple_crypto_v3,
    simple_push_v3,
    simple_reference_v3,
    simple_speaker_listener_v4,
    simple_spread_v3,
    simple_tag_v3,
    simple_v3,
    simple_world_comm_v3,
)
from pettingzoo.test import max_cycles_test, parallel_api_test
from pettingzoo.test.api_test import api_test
from pettingzoo.test.render_test import render_test
from pettingzoo.test.seed_test import parallel_seed_test, seed_test
from pettingzoo.test.state_test import state_test

parameterized_envs = [
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
