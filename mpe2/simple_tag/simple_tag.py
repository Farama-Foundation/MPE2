# noqa: D212, D415
"""
# Simple Tag

This environment is part of the <a href='https://mpe2.farama.org/mpe2/'>MPE environments</a>. Please read that page first for general information.

| Import             |              `from mpe2 import simple_tag_v3`              |
|--------------------|------------------------------------------------------------|
| Actions            | Discrete/Continuous                                        |
| Parallel API       | Yes                                                        |
| Manual Control     | No                                                         |
| Agents             | `agents= [adversary_0, adversary_1, adversary_2, agent_0]` |
| Agents             | 4                                                          |
| Action Shape       | (5)                                                        |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (50))                            |
| Observation Shape  | (14),(16)                                                  |
| Observation Values | (-inf,inf)                                                 |
| State Shape        | (62,)                                                      |
| State Values       | (-inf,inf)                                                 |


This is a predator-prey environment. Good agents (green) are faster and receive a negative reward for being hit by adversaries (red) (-10 for each collision). Adversaries are slower and are rewarded for hitting good agents (+10 for each collision). Obstacles (large black circles) block the way. By
default, there is 1 good agent, 3 adversaries and 2 obstacles.

So that good agents don't run to infinity, they are also penalized for exiting the area by the following function:

``` python
def bound(x):
      if x < 0.9:
          return 0
      if x < 1.0:
          return (x - 0.9) * 10
      return min(np.exp(2 * x - 2), 10)
```

Agent and adversary observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]`

Agent and adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False, dynamic_rescaling=False, curriculum=False, num_agent_neighbors=None, num_landmark_neighbors=None)
```



`num_good`:  number of good agents

`num_adversaries`:  number of adversaries

`num_obstacles`:  number of obstacles

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen size

`curriculum`: Whether to enable curriculum learning mode. When enabled, prey (good agents) start
slow and become faster as stages advance, making them progressively harder to catch. Use
`env.unwrapped.advance_curriculum()` to move to the next stage, or
`env.unwrapped.set_curriculum_stage(n)` to jump to a specific stage. Stage changes take effect
on the next `env.reset()`.

`num_agent_neighbors`: **Partial observability.** Maximum number of *other agents* each agent
observes, selected by Euclidean distance (nearest first).  Observation slots beyond the
available count are zero-padded so the observation shape remains fixed.  ``None`` (default)
restores full observability (all agents observed) and preserves backwards-compatibility.
Under PO, velocity information is restricted to good agents visible within the neighbour
window; velocity slots for adversaries or padded slots are zero.

`num_landmark_neighbors`: **Partial observability.** Maximum number of *landmarks* (obstacles)
each agent observes, selected by Euclidean distance (nearest first).  Zero-padded to a fixed
size when fewer landmarks are available.  ``None`` (default) = full observability.

Curriculum stages (prey max_speed / accel as fraction of full speed 1.3 / 4.0):
  - Stage 0: 50% speed — prey is slow and easy to catch.
  - Stage 1: 75% speed — prey moves at moderate pace.
  - Stage 2: 100% speed — prey moves at full speed (normal game difficulty).

To scale the number of agents across stages, recreate the environment with updated `num_good` /
`num_adversaries` values and reset the curriculum stage accordingly.

`terminate_on_success`: When `True`, the episode terminates as soon as every good agent is
simultaneously caught (colliding with at least one adversary).

"""

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from mpe2._mpe_utils.core import Agent, Landmark, World
from mpe2._mpe_utils.partial_observability import (
    padded_relative_positions,
    padded_velocities,
)
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
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
        assert num_agent_neighbors is None or (
            isinstance(num_agent_neighbors, int) and num_agent_neighbors > 0
        ), "num_agent_neighbors must be a positive integer or None."
        assert num_landmark_neighbors is None or (
            isinstance(num_landmark_neighbors, int) and num_landmark_neighbors > 0
        ), "num_landmark_neighbors must be a positive integer or None."
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
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
        world = scenario.make_world(num_good, num_adversaries, num_obstacles)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
            benchmark_data=benchmark_data,
        )
        self.metadata["name"] = "simple_tag_v3"

    @property
    def curriculum_stage(self):
        """Current curriculum stage (0-indexed). Only meaningful when curriculum=True."""
        return self.scenario.curriculum_stage

    def advance_curriculum(self):
        """Advance to the next curriculum stage. Takes effect on the next env.reset()."""
        self.scenario.advance_curriculum()

    def set_curriculum_stage(self, stage):
        """Jump to a specific curriculum stage (0-indexed, clamped to valid range). Takes effect on the next env.reset()."""
        self.scenario.set_curriculum_stage(stage)


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    # Curriculum stages for simple_tag.
    # Prey (good agents) start slow, making them easier to catch.
    # Each stage increases prey speed toward full difficulty.
    CURRICULUM_STAGES = [
        {"prey_speed_factor": 0.5},   # Stage 0: prey at 50% speed — easy
        {"prey_speed_factor": 0.75},  # Stage 1: prey at 75% speed — moderate
        {"prey_speed_factor": 1.0},   # Stage 2: prey at 100% speed — full difficulty
    ]

    # Base kinematic values for prey (good agents)
    _PREY_BASE_MAX_SPEED = 1.3
    _PREY_BASE_ACCEL = 4.0

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

    def is_terminal(self, world):
        """Return True when every good agent is simultaneously caught by an adversary.

        Only active when terminate_on_success=True. A good agent is considered caught
        when it is currently colliding with at least one adversary. This gives a crisp
        success signal to the adversaries and works naturally with curriculum learning —
        early stages (slow prey) produce frequent, short episodes; later stages produce
        longer, harder-to-terminate episodes.
        """
        if not self.terminate_on_success:
            return False
        adversaries = self.adversaries(world)
        for prey in self.good_agents(world):
            if not any(self.is_collision(prey, adv) for adv in adversaries):
                return False
        return True

    def make_world(self, num_good=1, num_adversaries=3, num_obstacles=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
        # Apply curriculum speed scaling to prey (good agents).
        if self.curriculum:
            speed_factor = self.CURRICULUM_STAGES[self.curriculum_stage]["prey_speed_factor"]
            for agent in world.agents:
                if not agent.adversary:
                    agent.max_speed = self._PREY_BASE_MAX_SPEED * speed_factor
                    agent.accel = self._PREY_BASE_ACCEL * speed_factor

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents
                )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        """Return the observation vector for *agent*.

        * ``self_vel``       – agent's own velocity (dim_p,)
        * ``self_pos``       – agent's own position (dim_p,)
        * ``landmark_pos``   – relative positions of observed landmarks
        * ``other_agent_pos``– relative positions of observed other agents
        * ``good_agent_vel`` – velocities of visible good agents (zeros for
                               adversary slots or padded slots)

        Full observability (``num_*_neighbors=None``, default):
            All non-boundary landmarks are included; all other agents' positions are included;
            only good agents contribute velocities (matching legacy size).

        Partial observability:
            Only the N nearest landmarks / agents are included.  Slots are
            zero-padded to maintain a *fixed* observation shape.  Velocity
            slots for adversary agents within the neighbour window are zeros.
        """
        # Landmarks ---
        non_boundary_landmarks = [e for e in world.landmarks if not e.boundary]
        entity_pos = padded_relative_positions(
            agent, non_boundary_landmarks, self.num_landmark_neighbors
        )

        # Other agents ---
        others = [other for other in world.agents if other is not agent]

        if self.num_agent_neighbors is None:
            # Full observability
            other_pos = [o.state.p_pos - agent.state.p_pos for o in others]
            other_vel = [o.state.p_vel for o in others if not o.adversary]
        else:
            # Partial observability
            other_pos = padded_relative_positions(
                agent, others, self.num_agent_neighbors
            )
            other_vel = padded_velocities(
                agent,
                others,
                self.num_agent_neighbors,
                predicate=lambda e: not e.adversary,
            )

        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
        )
