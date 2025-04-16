import pandas as pd
import igraph as ig
import random
import numpy as np
from typing import Any, Optional


def random_walk(
    graph: ig.Graph,
    start_node: Any,
    attribute: Optional[str] = None,
    walk_length: int = 100,
) -> pd.DataFrame:
    """
    Performs an optimized random walk on a directed graph starting from a specified node,
    optionally logging a specified attribute at each step.

    Parameters:
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

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the step number, node id, and optionally the
        logged attribute at each step.
    """
    # Validate start_node exists in the graph
    if start_node < 0 or start_node >= graph.vcount():
        raise ValueError(f"Node {start_node} not in graph")

    # Check if attribute exists upfront
    has_attribute = attribute is not None and attribute in graph.vs.attributes()

    # Pre-allocate numpy array with appropriate dimensions
    if has_attribute:
        logger = np.empty((walk_length, 3), dtype=object)
    else:
        logger = np.empty((walk_length, 2), dtype=object)

    node = start_node
    cnt = 0

    while cnt < walk_length:
        # Log current node
        if has_attribute:
            logger[cnt] = [cnt, node, graph.vs[node][attribute]]
        else:
            logger[cnt] = [cnt, node]

        # Get neighbors and select next node
        neighbors = graph.neighbors(node, mode="all")
        if not neighbors:
            # No neighbors to move to, end the walk
            break

        # Choose next node randomly and update
        node = random.choice(neighbors)
        cnt += 1

    # Create and return appropriate DataFrame
    if has_attribute:
        return logger[:cnt]
    else:
        return logger[:cnt]
