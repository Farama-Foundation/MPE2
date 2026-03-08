# noqa: D212, D415
"""
# Simple Spread

This environment is part of the <a href='http://mpe2.farama.org/mpe2/'>MPE environments</a>. Please read that page first for general information.

| Import               |      `from mpe2 import simple_spread_v3`      |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, agent_1, agent_2]`         |
| Agents               | 3                                             |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (18)                                          |
| Observation Values   | (-inf,inf)                                    |
| State Shape          | (54,)                                         |
| State Values         | (-inf,inf)                                    |


This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the
`local_ratio` parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False, dynamic_rescaling=False, curriculum=False, num_agent_neighbors=None, num_landmark_neighbors=None)
```



`N`:  number of agents and landmarks

`local_ratio`:  Weight applied to local reward and global reward. Global reward weight will always be 1 - local reward weight.

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen size

`curriculum`: Whether to enable curriculum learning mode. When enabled, training proceeds through
stages that gradually increase task difficulty. Use `env.unwrapped.advance_curriculum()` to move
to the next stage, or `env.unwrapped.set_curriculum_stage(n)` to jump to a specific stage.

Curriculum stages:
  - Stage 0: Agents receive no collision penalty — focus purely on covering landmarks.
  - Stage 1: Collision penalty is restored — agents must cover landmarks while avoiding each other.

To scale the number of agents/landmarks across stages, recreate the environment with a larger `N`
and reset the curriculum stage accordingly.

`terminate_on_success`: When `True`, the episode terminates as soon as every landmark is covered
by at least one agent (an agent is within distance 0.1 of the landmark). This gives a stronger
training signal than always running to `max_cycles`, and pairs naturally with curriculum learning.

`num_agent_neighbors`: **Partial observability.** Maximum number of *other agents* each agent
observes, selected by Euclidean distance (nearest first).  Observation slots beyond the
available count are zero-padded so the observation shape remains fixed.  Communication signals
are also filtered to the same N nearest agents.  ``None`` (default) = full observability.
simple_spread is generally solvable under PO – agents can learn locally-greedy covering
policies without needing global information.

`num_landmark_neighbors`: **Partial observability.** Maximum number of *landmarks* each agent
observes, selected by Euclidean distance (nearest first).  Zero-padded to a fixed size.
``None`` (default) = full observability.

"""

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from mpe2._mpe_utils.core import Agent, Landmark, World
from mpe2._mpe_utils.partial_observability import (
    padded_comms,
    padded_relative_positions,
)
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        benchmark_data=False,
        curriculum=False,
        terminate_on_success=False,
        num_agent_neighbors=None,
        num_landmark_neighbors=None,
    ):
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        assert num_agent_neighbors is None or (
            isinstance(num_agent_neighbors, int) and num_agent_neighbors > 0
        ), "num_agent_neighbors must be a positive integer or None."
        assert num_landmark_neighbors is None or (
            isinstance(num_landmark_neighbors, int) and num_landmark_neighbors > 0
        ), "num_landmark_neighbors must be a positive integer or None."
        EzPickle.__init__(
            self,
            N=N,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            benchmark_data=benchmark_data,
            curriculum=curriculum,
            terminate_on_success=terminate_on_success,
            num_agent_neighbors=num_agent_neighbors,
            num_landmark_neighbors=num_landmark_neighbors,
        )
        scenario = Scenario(
            curriculum=curriculum,
            terminate_on_success=terminate_on_success,
            num_agent_neighbors=num_agent_neighbors,
            num_landmark_neighbors=num_landmark_neighbors,
        )
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
            dynamic_rescaling=dynamic_rescaling,
            benchmark_data=benchmark_data,
        )
        self.metadata["name"] = "simple_spread_v3"

    @property
    def curriculum_stage(self):
        """Current curriculum stage (0-indexed). Only meaningful when curriculum=True."""
        return self.scenario.curriculum_stage

    def advance_curriculum(self):
        """Advance to the next curriculum stage. No-op if already at the final stage."""
        self.scenario.advance_curriculum()

    def set_curriculum_stage(self, stage):
        """Jump to a specific curriculum stage (0-indexed, clamped to valid range)."""
        self.scenario.set_curriculum_stage(stage)


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    # Curriculum stages for simple_spread.
    # Stage 0: no collision penalty — agents learn to cover landmarks first.
    # Stage 1: collision penalty enabled — agents learn to spread without colliding.
    CURRICULUM_STAGES = [
        {"collision_penalty": False},
        {"collision_penalty": True},
    ]

    # Distance threshold within which an agent is considered to "occupy" a landmark.
    CAPTURE_RADIUS = 0.1

    def __init__(self, curriculum=False, terminate_on_success=False, num_agent_neighbors=None, num_landmark_neighbors=None,):
        self.curriculum = curriculum
        self.curriculum_stage = 0
        self.terminate_on_success = terminate_on_success
        self.num_agent_neighbors = num_agent_neighbors
        self.num_landmark_neighbors = num_landmark_neighbors

    def advance_curriculum(self):
        """Move to the next curriculum stage. No-op at the final stage."""
        max_stage = len(self.CURRICULUM_STAGES) - 1
        if self.curriculum_stage < max_stage:
            self.curriculum_stage += 1

    def set_curriculum_stage(self, stage):
        """Set curriculum stage directly (clamped to valid range)."""
        max_stage = len(self.CURRICULUM_STAGES) - 1
        self.curriculum_stage = max(0, min(stage, max_stage))

    def make_world(self, N=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and a != agent:
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_terminal(self, world):
        """Return True when every landmark is covered by at least one agent.

        Only active when terminate_on_success=True. The capture radius matches
        the threshold used in benchmark_data so success is measured consistently.
        """
        if not self.terminate_on_success:
            return False
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            if min(dists) >= self.CAPTURE_RADIUS:
                return False
        return True

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions.
        # In curriculum stage 0, the collision penalty is suppressed so agents first learn to reach landmarks.
        rew = 0
        stage_config = self.CURRICULUM_STAGES[self.curriculum_stage]
        collision_penalty_active = (not self.curriculum) or stage_config["collision_penalty"]
        if agent.collide and collision_penalty_active:
            for a in world.agents:
                rew -= 1.0 * (self.is_collision(a, agent) and a != agent)
        return rew

    def global_reward(self, world):
        rew = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        return rew

    def observation(self, agent, world):
        """Return the observation vector for *agent*.

        Full observability (``num_*_neighbors=None``, default):

        Partial observability:
            Only the N nearest landmarks / other agents are included.
            Slots are zero-padded to maintain a *fixed* observation shape.
            Communication slots are aligned with agent position slots (same
            N nearest agents).
        """
        others = [other for other in world.agents if other is not agent]

        # lsndmarks
        if self.num_landmark_neighbors is None:
            entity_pos = [e.state.p_pos - agent.state.p_pos for e in world.landmarks]
        else:
            entity_pos = padded_relative_positions(
                agent, world.landmarks, self.num_landmark_neighbors
            )

        # Other agents + comm
        if self.num_agent_neighbors is None:
            other_pos = [o.state.p_pos - agent.state.p_pos for o in others]
            comm = [o.state.c for o in others]
        else:
            other_pos = padded_relative_positions(
                agent, others, self.num_agent_neighbors
            )
            comm = padded_comms(agent, others, self.num_agent_neighbors, world.dim_c)

        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        )
