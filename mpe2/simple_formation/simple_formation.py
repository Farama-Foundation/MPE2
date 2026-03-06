# noqa: D212, D415
"""
# Simple Formation

This environment is part of the <a href='http://mpe2.farama.org/mpe2/'>MPE environments</a>. Please read that page first for general information.

| Import               | `from mpe2 import simple_formation_v1`       |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, ..., agent_N-1]`           |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (6)                                           |
| Observation Values   | (-inf, inf)                                   |

N agents must arrange themselves in a circle of radius 0.5 around a central landmark.
At each step the ideal circular positions are anchored to the agent with the smallest
angle from horizontal, and agents are assigned to positions via bipartite matching
(Hungarian algorithm). The shared reward is the negative mean distance from assigned
target positions, clipped to [0, 2].

Agent observations: `[self_vel, self_pos, landmark_rel_pos]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_formation_v1.env(N=4, max_cycles=25, continuous_actions=False, terminate_on_success=False)
```

`N`: number of agents

`max_cycles`: number of frames until the episode terminates

`continuous_actions`: whether action spaces are discrete (default) or continuous

`terminate_on_success`: when True, the episode ends as soon as every agent is within
  0.05 units of its assigned target position.

"""

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn
from scipy.optimize import linear_sum_assignment

from mpe2._mpe_utils.core import Agent, Landmark, World
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env


def _find_angle(pose):
    """Return angle of a 2D vector in [0, 2π)."""
    angle = np.arctan2(pose[1], pose[0])
    if angle < 0:
        angle += 2 * np.pi
    return angle


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=4,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        benchmark_data=False,
        terminate_on_success=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=dynamic_rescaling,
            benchmark_data=benchmark_data,
            terminate_on_success=terminate_on_success,
        )
        scenario = Scenario(terminate_on_success=terminate_on_success)
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=0,
            dynamic_rescaling=dynamic_rescaling,
            benchmark_data=benchmark_data,
        )
        self.metadata["name"] = "simple_formation_v1"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    TARGET_RADIUS = 0.5
    DIST_THRESHOLD = 0.05

    def __init__(self, terminate_on_success=False):
        self.terminate_on_success = terminate_on_success
        self._delta_dists = None
        self._joint_reward = 0.0

    def make_world(self, N=4):
        world = World()
        world.dim_c = 0
        world.agents = [Agent() for _ in range(N)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
        world.landmarks = [Landmark()]
        world.landmarks[0].name = "landmark_0"
        world.landmarks[0].collide = False
        world.landmarks[0].movable = False
        world.landmarks[0].size = 0.03
        return world

    def reset_world(self, world, np_random):
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
        world.landmarks[0].color = np.array([0.25, 0.25, 0.25])
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, 1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        world.landmarks[0].state.p_pos = np_random.uniform(-0.5, 0.5, world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)
        self._delta_dists = None
        self._joint_reward = 0.0

    def _compute_formation(self, world):
        """Compute target circle positions and bipartite matching distances."""
        N = len(world.agents)
        ideal_sep = (2 * np.pi) / N
        landmark_pos = world.landmarks[0].state.p_pos
        relative_poses = [a.state.p_pos - landmark_pos for a in world.agents]
        theta_min = min(_find_angle(p) for p in relative_poses)
        expected = [
            landmark_pos
            + self.TARGET_RADIUS
            * np.array(
                [
                    np.cos(theta_min + i * ideal_sep),
                    np.sin(theta_min + i * ideal_sep),
                ]
            )
            for i in range(N)
        ]
        dists = np.array(
            [
                [np.linalg.norm(a.state.p_pos - pos) for pos in expected]
                for a in world.agents
            ]
        )
        ri, ci = linear_sum_assignment(dists)
        self._delta_dists = dists[ri, ci]
        self._joint_reward = -float(np.mean(np.clip(self._delta_dists, 0, 2)))

    def reward(self, agent, world):
        return 0.0

    def global_reward(self, world):
        self._compute_formation(world)
        return self._joint_reward

    def is_terminal(self, world):
        if not self.terminate_on_success:
            return False
        if self._delta_dists is None:
            return False
        return bool(np.all(self._delta_dists < self.DIST_THRESHOLD))

    def observation(self, agent, world):
        entity_pos = [e.state.p_pos - agent.state.p_pos for e in world.landmarks]
        return np.concatenate([agent.state.p_vel, agent.state.p_pos] + entity_pos)
