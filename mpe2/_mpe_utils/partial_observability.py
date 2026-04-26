"""Partial observability (PO) utilities for MPE environments.

These helpers implement *N-nearest-neighbour* filtering: instead of an agent
observing *all* other agents and landmarks, it observes only the **N closest**
ones.  When fewer entities exist than requested, the remaining observation
slots are filled with **zeros** so that the observation shape stays fixed.

Currently used by simple_spread, simple_tag, and simple_adversary. Can be
extended to other environments as needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence, TypeVar

import numpy as np

if TYPE_CHECKING:
    from mpe2._mpe_utils.core import Agent, Entity

_EntityT = TypeVar("_EntityT", bound="Entity")


def nearest_entities(
    agent: Agent, entities: Sequence[_EntityT], n: int | None
) -> list[_EntityT]:
    """Return up to *n* entities nearest to *agent*, sorted closest-first."""
    if n is None:
        return list(entities)
    if not entities:
        return []
    dists = np.array(
        [np.linalg.norm(e.state.p_pos - agent.state.p_pos) for e in entities]
    )
    order = np.argsort(dists)[:n]
    return [entities[i] for i in order]


def padded_relative_positions(
    agent: Agent, entities: Sequence[Entity], n: int | None, dim_p: int = 2
) -> list[np.ndarray]:
    """Relative positions of the *n* nearest entities, zero-padded to *n* slots."""
    if n is None:
        return [e.state.p_pos - agent.state.p_pos for e in entities]
    selected = nearest_entities(agent, entities, n)
    positions = [e.state.p_pos - agent.state.p_pos for e in selected]
    while len(positions) < n:
        positions.append(np.zeros(dim_p))
    return positions


def padded_velocities(
    agent: Agent,
    entities: Sequence[_EntityT],
    n: int | None,
    predicate: Callable[[_EntityT], bool] | None = None,
    dim_p: int = 2,
) -> list[np.ndarray]:
    """Velocities of the *n* nearest entities, zero-padded to *n* slots."""
    if n is None:
        # Full-observability path: respect predicate-based filtering
        if predicate is None:
            return [e.state.p_vel.copy() for e in entities]
        # Original behaviour: only include matching entities (shorter list)
        return [e.state.p_vel.copy() for e in entities if predicate(e)]
    # PO path: fixed n slots aligned with position output
    selected = nearest_entities(agent, entities, n)
    velocities = []
    for e in selected:
        if predicate is None or predicate(e):
            velocities.append(e.state.p_vel.copy())
        else:
            velocities.append(np.zeros(dim_p))
    while len(velocities) < n:
        velocities.append(np.zeros(dim_p))
    return velocities


def padded_comms(
    agent: Agent, entities: Sequence[Agent], n: int | None, dim_c: int
) -> list[np.ndarray]:
    """Communication signals of the *n* nearest agents, zero-padded to *n* slots."""
    if n is None:
        return [e.state.c.copy() for e in entities]
    selected = nearest_entities(agent, entities, n)
    comms = [e.state.c.copy() for e in selected]
    while len(comms) < n:
        comms.append(np.zeros(dim_c))
    return comms
