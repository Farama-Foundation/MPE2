from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mpe2._mpe_utils.core import Agent, World


class BaseScenario:  # defines scenario upon which the world is built
    def make_world(self) -> World:  # create elements of the world
        raise NotImplementedError()

    def reset_world(
        self, world: World, np_random: Any
    ) -> None:  # create initial conditions of the world
        raise NotImplementedError()

    def benchmark_data(
        self, agent: Agent, world: World
    ) -> Any:  # return benchmark metrics for evaluation
        return {}

    def is_terminal(
        self, world: World
    ) -> bool:  # return True to end the episode early on success
        return False
