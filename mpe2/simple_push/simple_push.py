# noqa: D212, D415
"""
# Simple Push

This environment is part of the <a href='https://mpe2.farama.org/mpe2/'>MPE2 environments</a>. Please read that page first for general information.

| Import             |      `from mpe2 import simple_push_v3`      |
|--------------------|---------------------------------------------|
| Actions            | Discrete/Continuous                         |
| Parallel API       | Yes                                         |
| Manual Control     | No                                          |
| Agents             | `agents= [adversary_0, agent_0]`            |
| Agents             | 2                                           |
| Action Shape       | (5)                                         |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5,))             |
| Observation Shape  | (8),(19)                                    |
| Observation Values | (-inf,inf)                                  |
| State Shape        | (27,)                                       |
| State Values       | (-inf,inf)                                  |


This environment has 1 good agent, 1 adversary, and 1 landmark. The good agent is rewarded based on the distance to the landmark. The adversary is rewarded if it is close to the landmark, and if the agent is far from the landmark (the difference of the distances). Thus the adversary must learn to
push the good agent away from the landmark.

Agent observation space: `[self_vel, goal_rel_position, goal_landmark_id, all_landmark_rel_positions, landmark_ids, other_agent_rel_positions]`

Adversary observation space: `[self_vel, all_landmark_rel_positions, other_agent_rel_positions]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

Adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_push_v3.env(max_cycles=25, continuous_actions=False, dynamic_rescaling=False, num_agent_neighbors=None, num_landmark_neighbors=None, radius=None, knn_mode="compact")
```



`max_cycles`:  number of frames (a step for each agent) until game terminates

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen size

`num_agent_neighbors`: Optional nearest-agent cap. There is only one other agent, so radius is
the more meaningful agent-visibility control.

`num_landmark_neighbors`: Optional nearest-landmark cap.

`radius`: Optional shared sensing radius for agents and landmarks, applied before the caps.

`knn_mode`: ``"compact"`` (default) stores visible entities nearest-first; ``"masked"`` keeps
the full stable entity slots and zeros hidden entities. Landmark colors remain aligned with
their position slots. The good agent's private goal-relative position is always retained.


"""

from __future__ import annotations

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from mpe2._mpe_utils.core import Agent, Landmark, World, _require_initialized
from mpe2._mpe_utils.partial_observability import (
    KNNMode,
    observed_entity_slots,
    validate_partial_observability,
)
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        max_cycles: int = 25,
        continuous_actions: bool = False,
        render_mode: str | None = None,
        dynamic_rescaling: bool = False,
        benchmark_data: bool = False,
        num_agent_neighbors: int | None = None,
        num_landmark_neighbors: int | None = None,
        radius: float | None = None,
        knn_mode: KNNMode = "compact",
    ) -> None:
        validate_partial_observability(
            num_agent_neighbors,
            num_landmark_neighbors,
            radius,
            knn_mode,
        )
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            benchmark_data=benchmark_data,
            num_agent_neighbors=num_agent_neighbors,
            num_landmark_neighbors=num_landmark_neighbors,
            radius=radius,
            knn_mode=knn_mode,
        )
        scenario = Scenario(
            num_agent_neighbors=num_agent_neighbors,
            num_landmark_neighbors=num_landmark_neighbors,
            radius=radius,
            knn_mode=knn_mode,
        )
        world = scenario.make_world()
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
        self.metadata["name"] = "simple_push_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class ExtendedLandmark(Landmark):
    def __init__(self) -> None:
        super().__init__()
        self.index: int = 0


class ExtendedAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.adversary: bool = False
        self._goal_a: ExtendedLandmark | None = None

    @property
    def goal_a(self) -> ExtendedLandmark:
        return _require_initialized(self._goal_a, "ExtendedAgent.goal_a")

    @goal_a.setter
    def goal_a(self, value: ExtendedLandmark | None) -> None:
        self._goal_a = value


class ExtendedWorld(World):
    def __init__(self) -> None:
        super().__init__()
        self.agents: list[ExtendedAgent] = []
        self.landmarks: list[ExtendedLandmark] = []


class Scenario(BaseScenario):
    def __init__(
        self,
        num_agent_neighbors: int | None = None,
        num_landmark_neighbors: int | None = None,
        radius: float | None = None,
        knn_mode: KNNMode = "compact",
    ) -> None:
        self.num_agent_neighbors = num_agent_neighbors
        self.num_landmark_neighbors = num_landmark_neighbors
        self.radius = radius
        self.knn_mode: KNNMode = knn_mode

    def make_world(self) -> ExtendedWorld:
        world = ExtendedWorld()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 2
        # add agents
        world.agents = [ExtendedAgent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
        # add landmarks
        world.landmarks = [ExtendedLandmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world: ExtendedWorld, np_random: np.random.Generator) -> None:
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.color[i + 1] += 0.8
            landmark.index = i
        # set goal landmark
        goal = world.landmarks[int(np_random.integers(len(world.landmarks)))]
        for i, agent in enumerate(world.agents):
            agent.goal_a = goal
            agent.color = np.array([0.25, 0.25, 0.25])
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                j = goal.index
                agent.color[j + 1] += 0.5
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent: ExtendedAgent, world: ExtendedWorld) -> float:
        # Agents are rewarded based on minimum agent distance to each landmark
        return (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )

    def agent_reward(self, agent: ExtendedAgent, world: ExtendedWorld) -> float:
        # the distance to the goal
        return -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))

    def adversary_reward(self, agent: ExtendedAgent, world: ExtendedWorld) -> float:
        # keep the nearest good agents away from the goal
        agent_dist = [
            np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
            for a in world.agents
            if not a.adversary
        ]
        pos_rew = min(agent_dist)
        # nearest_agent = world.good_agents[np.argmin(agent_dist)]
        # neg_rew = np.sqrt(np.sum(np.square(nearest_agent.state.p_pos - agent.state.p_pos)))
        neg_rew = np.sqrt(
            np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos))
        )
        # neg_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.good_agents])
        return pos_rew - neg_rew

    def observation(self, agent: ExtendedAgent, world: ExtendedWorld) -> np.ndarray:
        landmark_slots = observed_entity_slots(
            agent,
            world.landmarks,
            self.num_landmark_neighbors,
            self.radius,
            self.knn_mode,
        )
        entity_pos = [
            (
                np.zeros(world.dim_p)
                if entity is None
                else entity.state.p_pos - agent.state.p_pos
            )
            for entity in landmark_slots
        ]
        entity_color = [
            np.zeros(world.dim_color) if entity is None else entity.color
            for entity in landmark_slots
        ]

        others = [other for other in world.agents if other is not agent]
        agent_slots = observed_entity_slots(
            agent,
            others,
            self.num_agent_neighbors,
            self.radius,
            self.knn_mode,
        )
        other_pos = [
            (
                np.zeros(world.dim_p)
                if other is None
                else other.state.p_pos - agent.state.p_pos
            )
            for other in agent_slots
        ]
        if not agent.adversary:
            return np.concatenate(
                [agent.state.p_vel]
                + [agent.goal_a.state.p_pos - agent.state.p_pos]
                + [agent.color]
                + entity_pos
                + entity_color
                + other_pos
            )
        else:
            return np.concatenate([agent.state.p_vel] + entity_pos + other_pos)
