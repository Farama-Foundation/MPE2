# noqa: D212, D415
"""
# Simple World Comm

This environment is part of the <a href='https://mpe2.farama.org/mpe2/'>MPE2 environments</a>. Please read that page first for general information.

| Import             |                       `from mpe2 import simple_world_comm_v3`                       |
|--------------------|-------------------------------------------------------------------------------------|
| Actions            | Discrete/Continuous                                                                 |
| Parallel API       | Yes                                                                                 |
| Manual Control     | No                                                                                  |
| Agents             | `agents=[leadadversary_0, adversary_0, adversary_1, adversary_3, agent_0, agent_1]` |
| Agent Count        | 6                                                                                   |
| Action Shape       | (5),(20)                                                                            |
| Action Values      | Discrete(5),(20)/Box(0.0, 1.0, (5)), Box(0.0, 1.0, (9))                             |
| Observation Shape  | (28),(34)                                                                           |
| Observation Values | (-inf,inf)                                                                          |
| State Shape        | (192,)                                                                              |
| State Values       | (-inf,inf)                                                                          |


This environment is similar to simple_tag, except there is food (small blue balls) that the good agents are rewarded for being near, there are 'forests' that hide agents inside from being seen, and there is a 'leader adversary' that can see the agents at all times and can communicate with the
other adversaries to help coordinate the chase. By default, there are 2 good agents, 3 adversaries, 1 obstacles, 2 foods, and 2 forests.

In particular, the good agents reward, is -5 for every collision with an adversary, -2 x bound by the `bound` function described in simple_tag, +2 for every collision with a food, and -0.05 x minimum distance to any food. The adversarial agents are rewarded +5 for collisions and -0.1 x minimum
distance to a good agent. s

Good agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, self_in_forest, other_agent_velocities]`

Normal adversary observations:`[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities, self_in_forest, leader_comm]`

Adversary leader observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities, self_in_forest, leader_comm]`

*Note that when the forests prevent an agent from being seen, the observation of that agents relative position is set to (0,0).*

Good agent action space: `[no_action, move_left, move_right, move_down, move_up]`

Normal adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

Adversary leader discrete action space: `[say_0, say_1, say_2, say_3] X [no_action, move_left, move_right, move_down, move_up]`

Where X is the Cartesian product (giving a total action space of 50).

Adversary leader continuous action space: `[no_action, move_left, move_right, move_down, move_up, say_0, say_1, say_2, say_3]`

### Arguments

``` python
simple_world_comm_v3.env(num_good=2, num_adversaries=4, num_obstacles=1,
                num_food=2, max_cycles=25, num_forests=2, continuous_actions=False,
                dynamic_rescaling=False, num_agent_neighbors=None,
                num_landmark_neighbors=None, radius=None, knn_mode="compact")
```



`num_good`:  number of good agents

`num_adversaries`:  number of adversaries

`num_obstacles`:  number of obstacles

`num_food`:  number of food locations that good agents are rewarded at

`max_cycles`:  number of frames (a step for each agent) until game terminates

`num_forests`: number of forests that can hide agents inside from being seen

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen size

`num_agent_neighbors`: Optional nearest-agent cap, composed with the environment's existing
forest occlusion. Forest-hidden agents do not consume the cap.

`num_landmark_neighbors`: Optional nearest-landmark cap across obstacles, food, and forests.

`radius`: Optional shared sensing radius for agents and landmarks, applied before the caps.

`knn_mode`: ``"compact"`` (default) stores visible entities nearest-first and adds landmark
color/type features; ``"masked"`` preserves stable full slots and zeros entities hidden by
range, k, or forest occlusion. The leader's communication remains globally available.

"""

from __future__ import annotations

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from mpe2._mpe_utils.core import Agent, Entity, Landmark, World
from mpe2._mpe_utils.partial_observability import (
    KNNMode,
    nearest_entities,
    observed_entity_slots,
    validate_partial_observability,
)
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good: int = 2,
        num_adversaries: int = 4,
        num_obstacles: int = 1,
        num_food: int = 2,
        max_cycles: int = 25,
        num_forests: int = 2,
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
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            num_food=num_food,
            max_cycles=max_cycles,
            num_forests=num_forests,
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
        world = scenario.make_world(
            num_good, num_adversaries, num_obstacles, num_food, num_forests
        )
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
        self.metadata["name"] = "simple_world_comm_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class ExtendedAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.adversary: bool = False
        self.leader: bool = False


class ExtendedLandmark(Landmark):
    def __init__(self) -> None:
        super().__init__()
        self.boundary: bool = False


class ExtendedWorld(World):
    def __init__(self) -> None:
        super().__init__()
        self.agents: list[ExtendedAgent] = []
        self.landmarks: list[ExtendedLandmark] = []
        self.food: list[ExtendedLandmark] = []
        self.forests: list[ExtendedLandmark] = []


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

    def make_world(
        self,
        num_good_agents: int = 2,
        num_adversaries: int = 4,
        num_landmarks: int = 1,
        num_food: int = 2,
        num_forests: int = 2,
    ) -> ExtendedWorld:
        world = ExtendedWorld()
        # set any world properties first
        world.dim_c = 4
        # world.damping = 1
        num_good_agents = num_good_agents
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_landmarks
        num_food = num_food
        num_forests = num_forests
        # add agents
        world.agents = [ExtendedAgent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_index = i - 1 if i < num_adversaries else i - num_adversaries
            base_index = 0 if base_index < 0 else base_index
            base_name = "adversary" if agent.adversary else "agent"
            base_name = "leadadversary" if i == 0 else base_name
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.leader = True if i == 0 else False
            agent.silent = True if i > 0 else False
            agent.size = 0.075 if agent.adversary else 0.045
            agent.accel = 3.0 if agent.adversary else 4.0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [ExtendedLandmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        world.food = [ExtendedLandmark() for i in range(num_food)]
        for i, lm in enumerate(world.food):
            lm.name = "food %d" % i
            lm.collide = False
            lm.movable = False
            lm.size = 0.03
            lm.boundary = False
        world.forests = [ExtendedLandmark() for i in range(num_forests)]
        for i, lm in enumerate(world.forests):
            lm.name = "forest %d" % i
            lm.collide = False
            lm.movable = False
            lm.size = 0.3
            lm.boundary = False
        world.landmarks += world.food
        world.landmarks += world.forests
        # world.landmarks += self.set_boundaries(world)
        # world boundaries now penalized with negative reward
        return world

    def set_boundaries(self, world: ExtendedWorld) -> list[ExtendedLandmark]:
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                landmark = ExtendedLandmark()
                landmark.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(landmark)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                landmark = ExtendedLandmark()
                landmark.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(landmark)

        for i, l in enumerate(boundary_list):
            l.name = "boundary %d" % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list

    def reset_world(self, world: ExtendedWorld, np_random: np.random.Generator) -> None:
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.45, 0.95, 0.45])
                if not agent.adversary
                else np.array([0.95, 0.45, 0.45])
            )
            agent.color -= (
                np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.15, 0.65])
        for i, landmark in enumerate(world.forests):
            landmark.color = np.array([0.6, 0.9, 0.6])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.forests):
            landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent: ExtendedAgent, world: ExtendedWorld) -> int:
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1: Entity, agent2: Entity) -> bool:
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world: ExtendedWorld) -> list[ExtendedAgent]:
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world: ExtendedWorld) -> list[ExtendedAgent]:
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent: ExtendedAgent, world: ExtendedWorld) -> float:
        # Agents are rewarded based on minimum agent distance to each landmark
        # boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def outside_boundary(self, agent: ExtendedAgent) -> bool:
        if (
            agent.state.p_pos[0] > 1
            or agent.state.p_pos[0] < -1
            or agent.state.p_pos[1] > 1
            or agent.state.p_pos[1] < -1
        ):
            return True
        else:
            return False

    def agent_reward(self, agent: ExtendedAgent, world: ExtendedWorld) -> float:
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:
            for adv in adversaries:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 5

        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)  # 1 + (x - 1) * (x - 1)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= 2 * bound(x)

        for food in world.food:
            if self.is_collision(agent, food):
                rew += 2
        rew -= 0.05 * min(
            np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos)))
            for food in world.food
        )

        return rew

    def adversary_reward(self, agent: ExtendedAgent, world: ExtendedWorld) -> float:
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            rew -= 0.1 * min(
                np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                for a in agents
            )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 5
        return rew

    def observation2(self, agent: ExtendedAgent, world: ExtendedWorld) -> np.ndarray:
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
        )

    def observation(self, agent: ExtendedAgent, world: ExtendedWorld) -> np.ndarray:
        landmarks = [entity for entity in world.landmarks if not entity.boundary]
        landmark_slots = observed_entity_slots(
            agent,
            landmarks,
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
        entity_color = (
            [
                np.zeros(world.dim_color) if entity is None else entity.color
                for entity in landmark_slots
            ]
            if self.knn_mode == "compact"
            and (self.num_landmark_neighbors is not None or self.radius is not None)
            else []
        )

        in_forest = [np.array([-1]) for _ in range(len(world.forests))]
        agent_forests = [self.is_collision(agent, forest) for forest in world.forests]
        for i, is_inside in enumerate(agent_forests):
            if is_inside:
                in_forest[i] = np.array([1])

        others = [other for other in world.agents if other is not agent]

        def visible_through_forest(other: ExtendedAgent) -> bool:
            if agent.leader:
                return True
            other_forests = [
                self.is_collision(other, forest) for forest in world.forests
            ]
            return any(
                observer_inside and other_inside
                for observer_inside, other_inside in zip(agent_forests, other_forests)
            ) or (not any(agent_forests) and not any(other_forests))

        forest_visible = [other for other in others if visible_through_forest(other)]
        compact_agent_po = self.knn_mode == "compact" and (
            self.num_agent_neighbors is not None or self.radius is not None
        )
        if compact_agent_po:
            agent_cap = (
                self.num_agent_neighbors
                if self.num_agent_neighbors is not None
                else len(others)
            )
            agent_slots = observed_entity_slots(
                agent,
                forest_visible,
                agent_cap,
                self.radius,
                self.knn_mode,
            )
        else:
            selected = nearest_entities(
                agent,
                forest_visible,
                self.num_agent_neighbors,
                self.radius,
            )
            selected_ids = {id(other) for other in selected}
            agent_slots = [
                other if id(other) in selected_ids else None for other in others
            ]

        other_pos = [
            (
                np.zeros(world.dim_p)
                if other is None
                else other.state.p_pos - agent.state.p_pos
            )
            for other in agent_slots
        ]
        if compact_agent_po:
            other_vel = [
                (
                    other.state.p_vel
                    if other is not None and not other.adversary
                    else np.zeros(world.dim_p)
                )
                for other in agent_slots
            ]
        else:
            other_vel = [
                slot.state.p_vel if slot is not None else np.zeros(world.dim_p)
                for other, slot in zip(others, agent_slots)
                if not other.adversary
            ]

        comm = [world.agents[0].state.c]

        if agent.adversary and not agent.leader:
            return np.concatenate(
                [agent.state.p_vel]
                + [agent.state.p_pos]
                + entity_pos
                + entity_color
                + other_pos
                + other_vel
                + in_forest
                + comm
            )
        if agent.leader:
            return np.concatenate(
                [agent.state.p_vel]
                + [agent.state.p_pos]
                + entity_pos
                + entity_color
                + other_pos
                + other_vel
                + in_forest
                + comm
            )
        else:
            return np.concatenate(
                [agent.state.p_vel]
                + [agent.state.p_pos]
                + entity_pos
                + entity_color
                + other_pos
                + in_forest
                + other_vel
            )
