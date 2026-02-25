# noqa: D212, D415
"""
# Collect Treasure

This environment is part of the <a href='http://mpe2.farama.org/mpe2/'>MPE environments</a>. Please read that page first for general information.

| Import               | `from mpe2 import collect_treasure_v1`                                    |
|----------------------|---------------------------------------------------------------------------|
| Actions              | Discrete/Continuous                                                       |
| Parallel API         | Yes                                                                       |
| Manual Control       | No                                                                        |
| Agents               | `agents= [collector_0, ..., collector_5, deposit_0, deposit_1]`           |
| Agents               | 8 (default: 6 collectors + 2 deposits)                                    |
| Action Shape         | (5,)                                                                      |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5,))                                           |
| Observation Shape    | (86,) for collectors, (84,) for deposits (default config)                 |
| Observation Values   | (-inf, inf)                                                               |
| State Shape          | (684,) (default config)                                                   |
| State Values         | (-inf, inf)                                                               |

A cooperative multi-agent task in which collector agents must pick up treasures and deliver
them to the matching deposit agent.

There are two types of agents:
- **Collectors** (grey by default): roam the environment to pick up treasure landmarks and
  carry them to the appropriate deposit agent. A collector can hold at most one treasure at a
  time. When holding a treasure it changes color to match the treasure's type.
- **Deposits** (dark-tinted, color-coded by type): goal zones. Each deposit accepts only one
  treasure type. Deposits try to position themselves near collectors that are carrying matching
  treasure.

Treasures are landmarks that appear at random positions. When a collector touches a treasure
(is within collision distance), the collector picks it up and the treasure disappears. The
treasure then respawns at a new random location on the next step. When a collector carrying a
treasure touches the matching deposit agent, the treasure is delivered, the collector's
inventory is cleared, and it turns grey again.

**Reward structure (shared globally + shaped locally):**
- +5 global reward each time a collector is touching a live treasure while not holding anything
- +5 global reward each time a collector is touching its matching deposit while carrying the
  right treasure type
- -0.1 * distance shaping for collectors (toward nearest treasure or matching deposit)
- -0.1 * distance shaping for deposits (toward collectors carrying matching treasure, or
  toward the centroid of all collectors if none are carrying their type)
- -5 per pair of collectors that are overlapping (collision penalty)

**Observations:**
Each agent observes its own position and velocity. Collectors additionally observe a one-hot
encoding of what they are holding. Every agent then sees all other agents sorted by distance,
each described by relative position, velocity, and a 4-element encoding
[deposit_type_0, deposit_type_1, holding_type_0, holding_type_1]. Finally, every agent
observes all treasures sorted by distance, each described by relative position and a one-hot
type encoding. Dead (just-picked-up) treasures are shown with zero relative position and
zero type encoding.

**Agent action space:** `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
collect_treasure_v1.env(
    num_collectors=6,
    num_deposits=2,
    num_treasures=6,
    max_cycles=25,
    continuous_actions=False,
    dynamic_rescaling=False,
)
```

`num_collectors`: number of collector agents

`num_deposits`: number of deposit agents (also sets the number of treasure types)

`num_treasures`: number of treasure landmarks in the world

`max_cycles`: number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete (default) or continuous

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen
size

"""

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from mpe2._mpe_utils.core import Agent, Landmark, World
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env

# Visually distinct colors, one per deposit/treasure type (up to 6 types supported).
# These approximate the first entries of matplotlib's "tab10" / seaborn default palette.
_TYPE_COLORS = [
    np.array([0.212, 0.408, 0.776]),  # blue
    np.array([0.945, 0.553, 0.000]),  # orange
    np.array([0.169, 0.627, 0.173]),  # green
    np.array([0.839, 0.149, 0.157]),  # red
    np.array([0.580, 0.404, 0.741]),  # purple
    np.array([0.549, 0.337, 0.294]),  # brown
]


def _color_palette(n):
    if n > len(_TYPE_COLORS):
        raise ValueError(
            f"At most {len(_TYPE_COLORS)} deposit types are supported; got {n}."
        )
    return [c.copy() for c in _TYPE_COLORS[:n]]


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_collectors=6,
        num_deposits=2,
        num_treasures=6,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        benchmark_data=False,
    ):
        EzPickle.__init__(
            self,
            num_collectors=num_collectors,
            num_deposits=num_deposits,
            num_treasures=num_treasures,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=dynamic_rescaling,
            benchmark_data=benchmark_data,
        )
        scenario = Scenario()
        world = scenario.make_world(num_collectors, num_deposits, num_treasures)
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
        self.metadata["name"] = "collect_treasure_v1"

    # ------------------------------------------------------------------
    # Override _execute_world_step to inject post_step between physics
    # and reward computation.  The original SimpleEnv has no hook for
    # this, so we replicate the action-setup and reward logic here.
    # ------------------------------------------------------------------
    def _execute_world_step(self):
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        # Run pickup / deposit / respawn logic before computing rewards.
        self.scenario.post_step(self.world)

        for agent in self.world.agents:
            self.rewards[agent.name] = float(self.scenario.reward(agent, self.world))

        self._populate_benchmark_data()

    # ------------------------------------------------------------------
    # Override draw to skip non-alive landmarks (picked-up treasures are
    # teleported to [-999, -999] so we must not try to render them).
    # ------------------------------------------------------------------
    def draw(self):
        import pygame

        self.screen.fill((255, 255, 255))

        # Build the list of entities that are currently visible in the world.
        visible = [
            e
            for e in self.world.entities
            if not (hasattr(e, "alive") and not e.alive)
        ]
        if not visible:
            return

        all_poses = np.array([e.state.p_pos for e in visible])
        cam_range = np.max(np.abs(all_poses))
        if cam_range == 0:
            cam_range = 1.0

        scaling_factor = 0.9 * self.original_cam_range / cam_range

        for entity in visible:
            x, y = entity.state.p_pos
            y *= -1  # flip y to match the original pyglet convention
            x = (x / cam_range) * self.width // 2 * 0.9 + self.width // 2
            y = (y / cam_range) * self.height // 2 * 0.9 + self.height // 2

            radius = (
                entity.size * 350 * scaling_factor
                if self.dynamic_rescaling
                else entity.size * 350
            )

            pygame.draw.circle(self.screen, entity.color * 200, (x, y), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 1)
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # All agents in this environment are silent; no text to render.


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


class Scenario(BaseScenario):
    def make_world(self, num_collectors=6, num_deposits=2, num_treasures=6):
        if num_deposits < 1:
            raise ValueError("num_deposits must be >= 1")
        if num_collectors < 1:
            raise ValueError("num_collectors must be >= 1")
        if num_treasures < 1:
            raise ValueError("num_treasures must be >= 1")

        world = World()
        world.dim_c = 2  # communication channels (unused; kept for compatibility)

        world.num_collectors = num_collectors
        world.num_deposits = num_deposits
        world.treasure_types = list(range(num_deposits))
        world.treasure_colors = _color_palette(num_deposits)

        num_agents = num_collectors + num_deposits

        # ---- Agents: collectors first, then deposits ----
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            is_collector = i < num_collectors
            agent.collector = is_collector
            agent.name = (
                f"collector_{i}" if is_collector else f"deposit_{i - num_collectors}"
            )
            # All agents use "ghost" semantics: no physics push forces between them.
            # Inter-collector collisions are penalised via distance in the reward
            # function, matching the original MAAC environment behaviour.
            agent.collide = False
            agent.silent = True
            agent.size = 0.05 if is_collector else 0.075
            agent.accel = 1.5
            agent.max_speed = 1.0
            agent.initial_mass = 1.0 if is_collector else 2.25
            agent.holding = None  # None or an int treasure type (collectors only)

            if is_collector:
                agent.color = np.array([0.85, 0.85, 0.85])
            else:
                agent.d_i = i - num_collectors  # treasure type this deposit accepts
                agent.color = world.treasure_colors[agent.d_i] * 0.35

        # ---- Landmarks: treasures ----
        world.landmarks = [Landmark() for _ in range(num_treasures)]
        for i, lm in enumerate(world.landmarks):
            lm.name = f"treasure_{i}"
            lm.respawn_prob = 1.0
            lm.type = 0  # randomised in reset_world
            lm.color = world.treasure_colors[0].copy()
            lm.alive = True
            lm.collide = False
            lm.movable = False
            lm.size = 0.025

        return world

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def collectors(self, world):
        return [a for a in world.agents if a.collector]

    def deposits(self, world):
        return [a for a in world.agents if not a.collector]

    def treasures(self, world):
        return world.landmarks

    def _is_collision(self, entity1, entity2):
        delta = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta)))
        return dist < (entity1.size + entity2.size)

    def _get_agent_encoding(self, agent, world):
        """Return a 2*n_types vector: [deposit_type_enc | holding_type_enc]."""
        n = len(world.treasure_types)
        if agent.collector:
            deposit_enc = np.zeros(n)
            holding_enc = (np.arange(n) == agent.holding).astype(float)
        else:
            deposit_enc = (np.arange(n) == agent.d_i).astype(float)
            holding_enc = np.zeros(n)
        return np.concatenate([deposit_enc, holding_enc])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_world(self, world, np_random):
        # Store the seeded RNG so post_step can use it for reproducible respawns.
        world.np_random = np_random

        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1.0, 1.0, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.holding = None
            if agent.collector:
                agent.color = np.array([0.85, 0.85, 0.85])

        bound = 0.95
        for lm in world.landmarks:
            lm.type = int(np_random.integers(0, len(world.treasure_types)))
            lm.color = world.treasure_colors[lm.type].copy()
            lm.state.p_pos = np_random.uniform(-bound, bound, world.dim_p)
            lm.state.p_vel = np.zeros(world.dim_p)
            lm.alive = True

    # ------------------------------------------------------------------
    # Post-step: pickup, deposit, respawn
    # Called after physics (world.step) but before reward computation.
    # ------------------------------------------------------------------

    def post_step(self, world):
        self._reset_cached_rewards()
        np_random = world.np_random

        # --- Treasure pickup ---
        for lm in world.landmarks:
            if lm.alive:
                for collector in self.collectors(world):
                    if collector.holding is None and self._is_collision(lm, collector):
                        lm.alive = False
                        collector.holding = lm.type
                        collector.color = 0.85 * lm.color
                        # Move off-screen; draw() skips non-alive landmarks.
                        lm.state.p_pos = np.array([-999.0, -999.0])
                        break
            else:
                # Respawn each dead treasure every step (respawn_prob == 1.0 default).
                if np_random.uniform() <= lm.respawn_prob:
                    bound = 0.95
                    lm.state.p_pos = np_random.uniform(-bound, bound, world.dim_p)
                    lm.type = int(np_random.integers(0, len(world.treasure_types)))
                    lm.color = world.treasure_colors[lm.type].copy()
                    lm.alive = True

        # --- Deposit delivery ---
        for collector in self.collectors(world):
            if collector.holding is not None:
                for deposit in self.deposits(world):
                    if deposit.d_i == collector.holding and self._is_collision(
                        collector, deposit
                    ):
                        collector.holding = None
                        collector.color = np.array([0.85, 0.85, 0.85])
                        break

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _reset_cached_rewards(self):
        self._cached_global_collecting = None
        self._cached_global_deposit = None

    def _global_collecting_reward(self, world):
        """+5 for each collector currently touching a live treasure (not holding)."""
        if self._cached_global_collecting is None:
            rew = 0.0
            for lm in self.treasures(world):
                if lm.alive:
                    for c in self.collectors(world):
                        if c.holding is None and self._is_collision(c, lm):
                            rew += 5.0
            self._cached_global_collecting = rew
        return self._cached_global_collecting

    def _global_deposit_reward(self, world):
        """+5 for each collector currently touching its matching deposit."""
        if self._cached_global_deposit is None:
            rew = 0.0
            for d in self.deposits(world):
                for c in self.collectors(world):
                    if c.holding == d.d_i and self._is_collision(c, d):
                        rew += 5.0
            self._cached_global_deposit = rew
        return self._cached_global_deposit

    def _global_reward(self, world):
        return self._global_collecting_reward(world) + self._global_deposit_reward(world)

    def reward(self, agent, world):
        if agent.collector:
            return self._collector_reward(agent, world)
        return self._deposit_reward(agent, world)

    def _collector_reward(self, agent, world):
        rew = 0.0

        # Penalise overlaps between collectors (ghost physics, handled here).
        for other in self.collectors(world):
            if other is not agent and self._is_collision(agent, other):
                rew -= 5.0

        # Distance shaping: move toward nearest treasure or matching deposit.
        if agent.holding is None:
            alive = [lm for lm in self.treasures(world) if lm.alive]
            if alive:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(lm.state.p_pos - agent.state.p_pos)))
                    for lm in alive
                )
        else:
            matching = [d for d in self.deposits(world) if d.d_i == agent.holding]
            if matching:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(d.state.p_pos - agent.state.p_pos)))
                    for d in matching
                )

        rew += self._global_reward(world)
        return rew

    def _deposit_reward(self, agent, world):
        rew = 0.0

        # Collectors carrying this deposit's treasure type.
        relevant = [c for c in self.collectors(world) if c.holding == agent.d_i]

        if relevant:
            # Move toward the nearest collector carrying the right type.
            rew -= 0.1 * min(
                np.sqrt(np.sum(np.square(c.state.p_pos - agent.state.p_pos)))
                for c in relevant
            )
        else:
            # No matching carrier: move toward centroid of all collectors
            # (matches the original MAAC average-distance heuristic).
            all_c = self.collectors(world)
            if all_c:
                centroid = np.mean([c.state.p_pos for c in all_c], axis=0)
                rew -= 0.1 * np.linalg.norm(centroid - agent.state.p_pos)

        rew += self._global_reward(world)
        return rew

    def benchmark_data(self, agent, world):
        num_alive = sum(1 for lm in world.landmarks if lm.alive)
        if agent.collector:
            return (agent.holding, num_alive)
        else:
            carrying_matching = sum(
                1 for c in self.collectors(world) if c.holding == agent.d_i
            )
            return (carrying_matching, num_alive)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observation(self, agent, world):
        n_types = len(world.treasure_types)

        obs = [agent.state.p_pos.copy(), agent.state.p_vel.copy()]

        # Collectors encode what they are currently holding.
        if agent.collector:
            obs.append((np.arange(n_types) == agent.holding).astype(float))

        # All other agents, sorted by distance (closest first).
        others = sorted(
            (a for a in world.agents if a is not agent),
            key=lambda a: np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))),
        )
        for other in others:
            obs.append(other.state.p_pos - agent.state.p_pos)
            obs.append(other.state.p_vel.copy())
            obs.append(self._get_agent_encoding(other, world))

        # All treasures, alive ones first (sorted by distance), then dead ones.
        def _treasure_sort_key(lm):
            if lm.alive:
                return np.sqrt(np.sum(np.square(lm.state.p_pos - agent.state.p_pos)))
            return 1e6  # dead treasures sorted to the end

        for lm in sorted(self.treasures(world), key=_treasure_sort_key):
            if lm.alive:
                obs.append(lm.state.p_pos - agent.state.p_pos)
                obs.append((np.arange(n_types) == lm.type).astype(float))
            else:
                # Zero-pad dead treasures so the observation size stays fixed.
                obs.append(np.zeros(world.dim_p))
                obs.append(np.zeros(n_types))

        return np.concatenate(obs)
