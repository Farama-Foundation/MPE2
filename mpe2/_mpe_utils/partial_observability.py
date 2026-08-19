"""Partial observability (PO) utilities for MPE environments.

These helpers implement nearest-neighbour and radius filtering. Compact
observations contain nearest-first entity slots, while masked observations
retain the full-observation slot layout and zero unobserved entities.

Currently used by simple_spread, simple_tag, and simple_adversary. Can be
extended to other environments as needed.
"""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Callable, Literal, Sequence, TypeVar

import numpy as np

if TYPE_CHECKING:
    from mpe2._mpe_utils.core import Agent, Entity

_EntityT = TypeVar("_EntityT", bound="Entity")
KNNMode = Literal["compact", "masked"]


def validate_partial_observability(
    num_agent_neighbors: int | None = None,
    num_landmark_neighbors: int | None = None,
    radius: float | None = None,
    knn_mode: KNNMode = "compact",
) -> None:
    """Validate the common partial-observability constructor arguments."""
    assert num_agent_neighbors is None or (
        isinstance(num_agent_neighbors, int)
        and not isinstance(num_agent_neighbors, bool)
        and num_agent_neighbors > 0
    ), "num_agent_neighbors must be a positive integer or None."
    assert num_landmark_neighbors is None or (
        isinstance(num_landmark_neighbors, int)
        and not isinstance(num_landmark_neighbors, bool)
        and num_landmark_neighbors > 0
    ), "num_landmark_neighbors must be a positive integer or None."
    assert radius is None or (
        isinstance(radius, Real)
        and not isinstance(radius, bool)
        and np.isfinite(radius)
        and radius > 0
    ), "radius must be a positive finite number or None."
    assert knn_mode in (
        "compact",
        "masked",
    ), "knn_mode must be 'compact' or 'masked'."


def nearest_entities(
    agent: Agent,
    entities: Sequence[_EntityT],
    n: int | None,
    radius: float | None = None,
) -> list[_EntityT]:
    """Return entities visible by radius and optional nearest-neighbour cap."""
    if n is None and radius is None:
        return list(entities)
    if not entities:
        return []
    dists = np.array(
        [np.linalg.norm(e.state.p_pos - agent.state.p_pos) for e in entities]
    )
    order = np.argsort(dists)
    if radius is not None:
        order = order[dists[order] <= radius]
    if n is not None:
        order = order[:n]
    return [entities[i] for i in order]


def observed_entity_slots(
    agent: Agent,
    entities: Sequence[_EntityT],
    n: int | None,
    radius: float | None = None,
    knn_mode: KNNMode = "compact",
) -> list[_EntityT | None]:
    """Build fixed-size compact or identity-preserving masked entity slots."""
    selected = nearest_entities(agent, entities, n, radius)
    if knn_mode == "masked":
        selected_ids = {id(entity) for entity in selected}
        return [entity if id(entity) in selected_ids else None for entity in entities]
    if knn_mode != "compact":
        raise ValueError("knn_mode must be 'compact' or 'masked'.")

    slot_count = n if n is not None else len(entities)
    return selected + [None] * (slot_count - len(selected))


def padded_relative_positions(
    agent: Agent,
    entities: Sequence[Entity],
    n: int | None,
    dim_p: int = 2,
    *,
    radius: float | None = None,
    knn_mode: KNNMode = "compact",
) -> list[np.ndarray]:
    """Relative positions in compact or identity-preserving masked slots."""
    slots = observed_entity_slots(agent, entities, n, radius, knn_mode)
    return [
        np.zeros(dim_p) if entity is None else entity.state.p_pos - agent.state.p_pos
        for entity in slots
    ]


def padded_velocities(
    agent: Agent,
    entities: Sequence[_EntityT],
    n: int | None,
    predicate: Callable[[_EntityT], bool] | None = None,
    dim_p: int = 2,
    *,
    radius: float | None = None,
    knn_mode: KNNMode = "compact",
) -> list[np.ndarray]:
    """Velocities in compact or identity-preserving masked slots."""
    if n is None and radius is None:
        # Full-observability path: respect predicate-based filtering
        if predicate is None:
            return [e.state.p_vel.copy() for e in entities]
        # Original behaviour: only include matching entities (shorter list)
        return [e.state.p_vel.copy() for e in entities if predicate(e)]
    slots = observed_entity_slots(agent, entities, n, radius, knn_mode)
    if knn_mode == "masked":
        return [
            entity.state.p_vel.copy() if slot is not None else np.zeros(dim_p)
            for entity, slot in zip(entities, slots)
            if predicate is None or predicate(entity)
        ]

    velocities = []
    for entity in slots:
        if entity is not None and (predicate is None or predicate(entity)):
            velocities.append(entity.state.p_vel.copy())
        else:
            velocities.append(np.zeros(dim_p))
    return velocities


def padded_comms(
    agent: Agent,
    entities: Sequence[Agent],
    n: int | None,
    dim_c: int,
    *,
    radius: float | None = None,
    knn_mode: KNNMode = "compact",
) -> list[np.ndarray]:
    """Communication signals in compact or identity-preserving masked slots."""
    slots = observed_entity_slots(agent, entities, n, radius, knn_mode)
    return [
        np.zeros(dim_c) if entity is None else entity.state.c.copy() for entity in slots
    ]
