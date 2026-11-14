from __future__ import annotations

import numpy as np
import pytest

from mpe2 import simple_spread_v4


def _run_covering_landmarks_on_cycle(max_cycles: int, success_cycle: int):
    """Run simple_spread with no-op actions, covering every landmark before `success_cycle`."""
    env = simple_spread_v4.parallel_env(
        N=3, max_cycles=max_cycles, terminate_on_success=True
    )
    env.reset(seed=0)
    world = env.unwrapped.world
    cycles = 0
    while env.agents:
        if cycles == success_cycle - 1:
            for agent, landmark in zip(world.agents, world.landmarks):
                agent.state.p_pos = landmark.state.p_pos.copy()
                agent.state.p_vel = np.zeros(world.dim_p)
        _, _, terminations, truncations, _ = env.step({a: 0 for a in env.agents})
        cycles += 1
    env.close()
    return cycles, terminations, truncations


@pytest.mark.parametrize("max_cycles", [1, 2, 25])
def test_success_on_last_cycle_terminates(max_cycles: int) -> None:
    cycles, terminations, truncations = _run_covering_landmarks_on_cycle(
        max_cycles, success_cycle=max_cycles
    )

    assert cycles == max_cycles
    assert all(terminations.values())
    assert not any(truncations.values())


def test_success_before_last_cycle_terminates_early() -> None:
    cycles, terminations, truncations = _run_covering_landmarks_on_cycle(
        max_cycles=25, success_cycle=10
    )

    assert cycles == 10
    assert all(terminations.values())
    assert not any(truncations.values())


def test_no_success_is_truncated_at_max_cycles() -> None:
    cycles, terminations, truncations = _run_covering_landmarks_on_cycle(
        max_cycles=5, success_cycle=-1
    )

    assert cycles == 5
    assert not any(terminations.values())
    assert all(truncations.values())


@pytest.mark.parametrize(
    "env_name",
    ["simple_spread_v4", "simple_tag_v4", "simple_line_v2", "simple_formation_v2"],
)
@pytest.mark.parametrize("api", ["env", "parallel_env"])
@pytest.mark.parametrize("success_cycle", [1, 2, None])
def test_versioned_success_boundary(env_name, api, success_cycle, monkeypatch):
    """Exercise shared cycle handling in every affected environment and API."""
    from mpe2.all_modules import mpe_environments

    module = mpe_environments[f"mpe/{env_name}"]
    env = getattr(module, api)(max_cycles=2, terminate_on_success=True)
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        assert raw.metadata["name"] == env_name
        monkeypatch.setattr(
            raw.scenario, "is_terminal", lambda world: raw.steps == success_cycle
        )
        if api == "parallel_env":
            while env.agents:
                _, _, terminations, truncations, _ = env.step(
                    {a: 0 for a in env.agents}
                )
        else:
            for agent in env.agent_iter():
                _, _, terminated, truncated, _ = env.last()
                if terminated or truncated:
                    break
                env.step(0)
            terminations, truncations = raw.terminations, raw.truncations
        assert raw.steps == (success_cycle or 2)
        assert all(
            value == (success_cycle is not None) for value in terminations.values()
        )
        assert all(value == (success_cycle is None) for value in truncations.values())
    finally:
        env.close()
