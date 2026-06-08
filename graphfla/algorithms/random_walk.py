import random
import numpy as np
from typing import Any, Dict, List, Optional

from ._search_cache import SearchCache


def random_walk(
    cache: SearchCache,
    start_node: Any,
    attribute: Optional[str] = None,
    walk_length: int = 100,
    neutral_neighbors: Optional[Dict[int, List[int]]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Performs a random walk on the landscape graph held by ``cache`` starting from
    a specified node, optionally logging a specified vertex attribute at each step.

    Parameters
    ----------
    cache : SearchCache
        Precomputed search data (graph + hoisted fitness vector). Build once
        with ``SearchCache(graph)`` and reuse across a batch of walks.

    start_node : int
        The index of the starting node for the random walk.

    attribute : str, optional
        The vertex attribute to log at each step of the walk. If None,
        only step and node are logged. ``"fitness"`` is served from the cache's
        hoisted vector; any other attribute is read from the graph.

    walk_length : int, default=100
        The length of the random walk.

    neutral_neighbors : dict, optional
        Mapping from node index to a list of neutral neighbor indices. When
        provided, the walker can also traverse neutral edges (equal-fitness
        neighbors that have no directed edge in the graph).

    seed : int, optional
        Seed for a local random number generator, making the walk reproducible.
        If None (default), the global ``random`` state is used.

    Returns
    -------
    np.ndarray
        An array of shape ``(steps, 2)`` or ``(steps, 3)`` containing the step
        number, node id, and optionally the logged attribute at each step.
    """
    g = cache.graph
    if start_node < 0 or start_node >= cache.n:
        raise ValueError(f"Node {start_node} not in graph")

    rand = random.Random(seed) if seed is not None else random

    has_attribute = attribute is not None and attribute in g.vs.attributes()
    if has_attribute:
        attr_vec = (
            cache.fitness if attribute == "fitness"
            else np.asarray(g.vs[attribute])
        )

    # Collect visited node ids in a plain int array (no per-step object writes /
    # attribute-proxy lookups); the attribute column is mapped in one vectorised
    # gather at the end.
    nodes = np.empty(walk_length, dtype=np.int64)
    node = start_node
    cnt = 0

    while cnt < walk_length:
        nodes[cnt] = node

        neighbors = g.neighbors(node, mode="all")
        if neutral_neighbors and node in neutral_neighbors:
            neighbors = list(set(neighbors) | set(neutral_neighbors[node]))

        if not neighbors:
            break

        node = rand.choice(neighbors)
        cnt += 1

    nodes = nodes[:cnt]
    steps = np.arange(cnt)
    if has_attribute:
        return np.column_stack((steps, nodes, attr_vec[nodes]))
    return np.column_stack((steps, nodes))
