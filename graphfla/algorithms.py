import pandas as pd
import networkx as nx
import random 
import numpy as np 
from typing import List, Any, Dict, Tuple, Set, Union, Optional

def local_search(
        graph: nx.DiGraph, 
        node: Any, 
        weight: str, 
        search_method: str = "best-improvement"
    ) -> Any:
    """
    Conducts a local search on a directed graph from a specified node, using a specified edge attribute 
    for decision-making regarding the next node.

    Parameters
    ----------
    graph : nx.DiGraph
        The directed graph where the search is carried out.

    node : Any
        The index of the starting node for the local search.

    weight : str
        The edge attribute key that helps determine the best move during the search.

    search_method : str
        Specifies the local search method. Available options:
        - 'best-improvement': Analyzes all adjacent nodes and chooses the one with the optimal 
          improvement in the weight attribute.
        - 'first-improvement': Chooses the first adjacent node that shows any improvement in the weight attribute.

    Returns
    -------
    Any: The index of the next node to move to, determining the search direction.
    """

    if search_method not in ["best-improvement", "first-improvement"]:
        raise ValueError(f"Unsupported search method: {search_method}")

    out_edges = graph.out_edges(node, data=True)
    if not out_edges:
        return None
    
    best_node = None
    best_value = float('-inf')

    if search_method == "best-improvement":
        for _, next_node, data in out_edges:
            value = data.get(weight, float('-inf'))
            if value > best_value:
                best_value = value
                best_node = next_node
    elif search_method == "first-improvement":
        current_value = graph.nodes[node].get(weight, float('-inf'))
        for _, next_node, data in out_edges:
            value = data.get(weight, float('-inf'))
            if value > current_value:
                return next_node

    return best_node

def hill_climb(
        graph: nx.DiGraph, 
        node: int, 
        weight: str, 
        verbose: int = 0,
        return_trace: bool = False,
        search_method: str = "best-improvement"
    ) -> Tuple[Any, int, List[int]]:
    """
    Performs hill-climbing local search on a directed graph starting from a specified node, using a particular
    edge attribute as a guide for climbing.

    Parameters
    ----------
    graph : nx.DiGraph
        The directed graph on which the hill climbing is performed.

    node : int
        The indice of the starting node for the hill climbing search.

    weight : str
        The edge attribute key used to determine the "weight" during climbing, which guides the search.

    verbose : int, default=0
        The verbosity level for logging progress, where 0 is silent and higher values increase the verbosity.
    
    return_trace: bool, default=False
        Whether to return the trace of the search as a list of node indices. 

    search_method : str
        Specifies the method of local search to use. Options include:
        - 'best-improvement': Evaluates all neighbors and selects the one with the most significant
          improvement in the weight attribute.
        - 'first-improvement': Selects the first neighbor that shows any improvement in the weight attribute.

    Returns
    -------
    Tuple[Any, int]
        A tuple containing:
        - The final local optimum node reached.
        - The total number of steps taken in the search process.

    Example
    -------
    ```python
    >>> lo, steps, trace = hill_climb(graph=landscape.graph, node=0, weight="delta_fit")
    >>> print(f"configuration visited: {trace}")
    >>> print(f"local optimum id: {lo}")
    configuration visited: [0, 1, 341, 681, 2041, 1701, 1706, 1705, 1365, 1370, 1390, 1730, 1750]
    local optimum id: 1750
    ```
    """

    step = 0
    visited = {node} 
    trace = [node]   

    if verbose:
        print(f"Hill climbing begins from {node}...")

    current_node = node
    next_node = local_search(graph, current_node, weight, search_method)

    while next_node is not None and next_node not in visited:
        visited.add(next_node)
        trace.append(next_node)
        step += 1

        if verbose:
            print(f"# step: {step}, move from {current_node} to {next_node}")

        current_node = next_node
        next_node = local_search(graph, current_node, weight, search_method)

    if verbose:
        print(f"Finished at node {current_node} with {step} step(s).")

    if return_trace:
        return current_node, step, trace
    else:
        return current_node, step

def random_walk(
        graph: nx.DiGraph, 
        start_node: Any, 
        attribute: Optional[str] = None, 
        walk_length: int = 100
    ) -> pd.DataFrame:
    """
    Performs an optimized random walk on a directed graph starting from a specified node, 
    optionally logging a specified attribute at each step.

    Parameters:
    - graph (nx.DiGraph): The directed graph on which the random walk is performed.
    - start_node: The starting node for the random walk.
    - attribute (str, optional): The node attribute to log at each step of the walk. If None, 
        only nodes are logged.
    - walk_length (int): The length of the random walk. Default is 100.

    Returns:
    - pd.DataFrame: A DataFrame containing the step number, node id, and optionally the 
        logged attribute at each step.
    """
    
    node = start_node
    logger = np.empty((walk_length, 3), dtype=object)
    cnt = 0

    while cnt < walk_length:
        if not graph.has_node(node):
            raise ValueError("Node not in graph")

        neighbors = list(graph.successors(node)) + list(graph.predecessors(node))
        if not neighbors:
            break

        node = random.choice(neighbors)
        if attribute:
            attr_value = graph.nodes[node].get(attribute, None)
            logger[cnt] = [cnt, node, attr_value]
        else:
            logger[cnt] = [cnt, node, None]
        cnt += 1

    if attribute:
        return pd.DataFrame(logger[:cnt], columns=["step", "node_id", attribute])
    else:
        return pd.DataFrame(logger[:cnt], columns=["step", "node_id"])