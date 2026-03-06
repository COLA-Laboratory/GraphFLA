import pandas as pd
import igraph as ig
import random
import numpy as np
from typing import Any, Dict, List, Optional


def random_walk(
    graph: ig.Graph,
    start_node: Any,
    attribute: Optional[str] = None,
    walk_length: int = 100,
    neutral_neighbors: Optional[Dict[int, List[int]]] = None,
) -> np.ndarray:
    """
    Performs a random walk on a directed graph starting from a specified node,
    optionally logging a specified attribute at each step.

    Parameters
    ----------
    graph : ig.Graph
        The igraph Graph on which the random walk is performed.

    start_node : int
        The index of the starting node for the random walk.

    attribute : str, optional
        The vertex attribute to log at each step of the walk. If None,
        only nodes are logged.

    walk_length : int, default=100
        The length of the random walk.

    neutral_neighbors : dict, optional
        Mapping from node index to a list of neutral neighbor indices. When
        provided, the walker can also traverse neutral edges (equal-fitness
        neighbors that have no directed edge in the graph). Typically obtained
        from ``landscape._neutral_neighbors``.

    Returns
    -------
    np.ndarray
        An array of shape ``(steps, 2)`` or ``(steps, 3)`` containing the
        step number, node id, and optionally the logged attribute at each step.
    """
    if start_node < 0 or start_node >= graph.vcount():
        raise ValueError(f"Node {start_node} not in graph")

    has_attribute = attribute is not None and attribute in graph.vs.attributes()

    if has_attribute:
        logger = np.empty((walk_length, 3), dtype=object)
    else:
        logger = np.empty((walk_length, 2), dtype=object)

    node = start_node
    cnt = 0

    while cnt < walk_length:
        if has_attribute:
            logger[cnt] = [cnt, node, graph.vs[node][attribute]]
        else:
            logger[cnt] = [cnt, node]

        neighbors = graph.neighbors(node, mode="all")
        if neutral_neighbors and node in neutral_neighbors:
            neighbors = list(set(neighbors) | set(neutral_neighbors[node]))

        if not neighbors:
            break

        node = random.choice(neighbors)
        cnt += 1

    return logger[:cnt]
