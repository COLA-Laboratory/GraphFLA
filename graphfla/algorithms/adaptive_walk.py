import random
from typing import List, Any, Tuple


def local_search(
    graph, node: Any, weight: str, search_method: str = "best-improvement"
) -> Any:
    """
    Conducts a local search on a directed graph from a specified node, using a specified edge attribute
    for decision-making regarding the next node.

    Parameters
    ----------
    graph : ig.Graph
        The directed graph where the search is carried out.

    node : Any
        The index of the starting node for the local search.

    weight : str
        The edge attribute key that helps determine the best move during the search.
        Note: For 'best-improvement' using node fitness directly, this is less relevant.

    search_method : str
        Specifies the local search method. Available options:
        - 'best-improvement': Analyzes all adjacent nodes and chooses the one with the highest
          fitness value. This essentially implements the greedy adaptive walks.
        - 'first-improvement': Randomly selects an adjacent node.
          This essentially implements adaptive walks with uniform fixation probability for fitness-increasing mutations.

    Returns
    -------
    Any: The index of the next node to move to, determining the search direction.
    """

    # Get all successor nodes in one call
    successors = graph.neighbors(node, mode="out")
    if not successors:
        return None

    if search_method == "best-improvement":
        # Get fitness of all successors directly
        return max(successors, key=lambda s: graph.vs[s]["fitness"])

    elif search_method == "first-improvement":
        # Randomly select a successor
        return random.choice(successors)

    else:
        raise ValueError(f"Unsupported search method: {search_method}")


def hill_climb(
    graph,
    node: int,
    weight: str,
    verbose: int = 0,
    return_trace: bool = False,
    search_method: str = "best-improvement",
) -> Tuple[Any, int, List[int]]:
    """
    Performs hill-climbing local search on a directed graph starting from a specified node, using a particular
    edge attribute as a guide for climbing.

    Parameters
    ----------
    graph : ig.Graph
        The directed graph on which the hill climbing is performed.

    node : int
        The index of the starting node for the hill climbing search.

    weight : str
        The edge attribute key used to determine the "weight" during climbing, which guides the search.

    verbose : int, default=0
        The verbosity level for logging progress, where 0 is silent and higher values increase the verbosity.

    return_trace: bool, default=False
        Whether to return the trace of the search as a list of node indices.

    search_method : str
        Specifies the method of local search to use. Options include:
        - 'best-improvement': Also known as greedy. Evaluates all neighbors and selects the one with the most significant
          improvement in the weight attribute.
        - 'first-improvement': Selects the first neighbor that shows any improvement in the weight attribute.

    Returns
    -------
    Tuple[Any, int]
        A tuple containing:
        - The final local optimum node reached.
        - The total number of steps taken in the search process.
    """
    # Check if node is already a local optimum (has no outgoing edges)
    if graph.degree(node, mode="out") == 0:
        if return_trace:
            return node, 0, [node]
        return node, 0

    # Initialize tracking
    step = 0
    visited = {node}
    trace = [node] if return_trace else None
    current_node = node

    # Determine if we need verbose output
    verbose_output = verbose > 0

    if verbose_output:
        print(f"Hill climbing begins from {node}...")

    while True:
        # Get next node efficiently
        next_node = local_search(graph, current_node, weight, search_method)

        # No better node or we've seen this node before - we've found an optimum or cycle
        if next_node is None or next_node in visited:
            break

        # Update state
        visited.add(next_node)
        if return_trace:
            trace.append(next_node)
        step += 1

        if verbose_output:
            print(f"# step: {step}, move from {current_node} to {next_node}")

        current_node = next_node

    if verbose_output:
        print(f"Finished at node {current_node} with {step} step(s).")

    if return_trace:
        return current_node, step, trace
    return current_node, step
