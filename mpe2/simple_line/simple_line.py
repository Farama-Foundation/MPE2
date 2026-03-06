# noqa: D212, D415
"""
# Simple Line

This environment is part of the <a href='https://github.com/sumitsk/marl_transfer/tree/master/mape/multiagent/scenarios'>"Learning Transferable Cooperative Behavior in Multi-Agent Teams" Paper</a>.

| Import               | `from mpe2 import simple_line_v1`             |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, ..., agent_N-1]`           |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (8)                                           |
| Observation Values   | (-inf, inf)                                   |

N agents must arrange themselves in a line between two landmarks. At reset, the
two landmarks are placed at a fixed separation in a random direction; the ideal
agent positions are evenly spaced along that line. Agents are assigned to target
positions via bipartite matching (Hungarian algorithm). The shared reward is the
negative mean distance from assigned positions, clipped to [0, 2].

Agent observations: `[self_vel, self_pos, landmark_0_rel_pos, landmark_1_rel_pos]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_line_v1.env(N=4, max_cycles=25, continuous_actions=False, terminate_on_success=False)
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
        self.metadata["name"] = "simple_line_v1"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    DIST_THRESHOLD = 0.05
    # Line endpoints are separated by total_sep; agents fill the interior evenly.
    TOTAL_SEP = 1.25

    def __init__(self, terminate_on_success=False):
        self.terminate_on_success = terminate_on_success
        self._expected_positions = None
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
            agent.size = 0.03
        world.landmarks = [Landmark(), Landmark()]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark_{i}"
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.02
        return world

    def reset_world(self, world, np_random):
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, 1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # Place landmark 0 near the center
        world.landmarks[0].state.p_pos = np_random.uniform(-0.25, 0.25, world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

        # Place landmark 1 at TOTAL_SEP from landmark 0 in a random direction,
        # rotating until the position is within [-1, 1]^2.
        theta = np_random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)])
        loc = world.landmarks[0].state.p_pos + self.TOTAL_SEP * direction
        while not (abs(loc[0]) < 1.0 and abs(loc[1]) < 1.0):
            theta += np.radians(5)
            direction = np.array([np.cos(theta), np.sin(theta)])
            loc = world.landmarks[0].state.p_pos + self.TOTAL_SEP * direction

        world.landmarks[1].state.p_pos = loc
        world.landmarks[1].state.p_vel = np.zeros(world.dim_p)

        # Compute evenly-spaced ideal agent positions along the line
        N = len(world.agents)
        ideal_sep = self.TOTAL_SEP / (N - 1) if N > 1 else 0.0
        self._expected_positions = [
            world.landmarks[0].state.p_pos + i * ideal_sep * direction
            for i in range(N)
        ]

        self._delta_dists = None
        self._joint_reward = 0.0

    def _compute_line(self, world):
        """Compute bipartite matching distances against ideal line positions."""
        dists = np.array(
            [
                [
                    np.linalg.norm(a.state.p_pos - pos)
                    for pos in self._expected_positions
                ]
                for a in world.agents
            ]
        )
        ri, ci = linear_sum_assignment(dists)
        self._delta_dists = dists[ri, ci]
        self._joint_reward = -float(np.mean(np.clip(self._delta_dists, 0, 2)))

    def reward(self, agent, world):
        return 0.0

    def global_reward(self, world):
        self._compute_line(world)
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
