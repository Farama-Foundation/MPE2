from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mpe2._mpe_utils.core import Agent, World


class BaseScenario(ABC):  # defines scenario upon which the world is built
    @abstractmethod
    def make_world(self, *args: Any, **kwargs: Any) -> World:
        """Create elements of the world."""
        raise NotImplementedError()

    @abstractmethod
    def reset_world(self, world: World, np_random: Any) -> None:
        """Create initial conditions of the world."""
        raise NotImplementedError()

    @abstractmethod
    def reward(self, agent: Agent, world: World) -> float:
        """Return reward for an agent."""
        raise NotImplementedError()

    @abstractmethod
    def observation(self, agent: Agent, world: World) -> np.ndarray:
        """Return observation for an agent."""
        raise NotImplementedError()

    def benchmark_data(self, agent: Agent, world: World) -> Any:
        """Return benchmark metrics for evaluation."""
        return {}

    def global_reward(self, world: World) -> float:
        """Return a shared reward when an environment blends local/global rewards."""
        return 0.0

    def post_step(self, world: World) -> None:
        """Run optional scenario logic after physics and before reward computation."""
        return

    @property
    def curriculum_stage(self) -> int:
        """Current curriculum stage for scenarios that support curriculum learning."""
        return 0

    @curriculum_stage.setter
    def curriculum_stage(self, stage: int) -> None:
        return

    def advance_curriculum(self) -> None:
        """Advance curriculum for scenarios that support it."""
        return

    def set_curriculum_stage(self, stage: int) -> None:
        """Set curriculum stage for scenarios that support it."""
        return

    def is_terminal(self, world: World) -> bool:
        """Return True to end the episode early on success."""
        return False
