from __future__ import annotations

import numpy as np

from mpe2 import simple_world_comm_v3


def test_world_comm_steps_without_food() -> None:
    env = simple_world_comm_v3.parallel_env(num_food=0)
    try:
        env.reset(seed=123)
        actions = {agent: 0 for agent in env.agents}

        _, rewards, _, _, _ = env.step(actions)

        assert rewards.keys() == actions.keys()
        assert all(np.isfinite(reward) for reward in rewards.values())
    finally:
        env.close()
