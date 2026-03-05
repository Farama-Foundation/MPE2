"""Partial observability (PO) utilities for MPE environments.

These helpers implement *N-nearest-neighbour* filtering: instead of an agent
observing *all* other agents and landmarks, it observes only the **N closest**
ones.  When fewer entities exist than requested, the remaining observation
slots are filled with **zeros** so that the observation shape stays fixed.

Currently used by simple_spread, simple_tag, and simple_adversary. Can be
extended to other environments as needed.
"""

import numpy as np


def nearest_entities(agent, entities, n):
    """Return up to *n* entities nearest to *agent*, sorted closest-first.
    """
    if n is None:
        return list(entities)
    if not entities:
        return []
    dists = np.array(
        [np.linalg.norm(e.state.p_pos - agent.state.p_pos) for e in entities]
    )
    order = np.argsort(dists)[:n]
    return [entities[i] for i in order]


def padded_relative_positions(agent, entities, n, dim_p=2):
    """Relative positions of the *n* nearest entities, zero-padded to *n* slots.
    """
    if n is None:
        return [e.state.p_pos - agent.state.p_pos for e in entities]
    selected = nearest_entities(agent, entities, n)
    positions = [e.state.p_pos - agent.state.p_pos for e in selected]
    while len(positions) < n:
        positions.append(np.zeros(dim_p))
    return positions


def padded_velocities(agent, entities, n, predicate=None, dim_p=2):
    """Velocities of the *n* nearest entities, zero-padded to *n* slots.
    """
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


def padded_comms(agent, entities, n, dim_c):
    """Communication signals of the *n* nearest agents, zero-padded to *n* slots.
    """
    if n is None:
        return [e.state.c.copy() for e in entities]
    selected = nearest_entities(agent, entities, n)
    comms = [e.state.c.copy() for e in selected]
    while len(comms) < n:
        comms.append(np.zeros(dim_c))
    return comms
