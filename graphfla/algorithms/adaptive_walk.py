import random
from typing import Any, List, Tuple

from ._search_cache import SearchCache


def local_search(
    cache: SearchCache, node: Any, search_method: str = "best-improvement"
) -> Any:
    """
    Take a single local-search step from ``node`` on the landscape graph held by
    ``cache``, using the precomputed fitness vector for decision-making.

    Parameters
    ----------
    cache : SearchCache
        Precomputed search data (graph + hoisted fitness vector). Build once
        with ``SearchCache(graph)`` and reuse across a batch of calls.

    node : Any
        The index of the starting node for the local search.

    search_method : str
        Specifies the local search method. Available options:
        - 'best-improvement': Analyzes all adjacent nodes and chooses the one with the highest
          fitness value. This essentially implements the greedy adaptive walks.
        - 'first-improvement': Randomly selects an adjacent node.
          This essentially implements adaptive walks with uniform fixation probability for fitness-increasing mutations.

    Returns
    -------
    Any: The index of the next node to move to, or ``None`` if ``node`` has no
    improving (out-edge) neighbours.
    """
    successors = cache.graph.neighbors(node, mode="out")
    if not successors:
        return None

    if search_method == "best-improvement":
        # fitness.__getitem__ replaces the per-successor graph.vs[s]["fitness"]
        # proxy lookup; max() keeps the first-maximum tie-break (successor order
        # matches the original graph.neighbors order).
        return max(successors, key=cache.fitness.__getitem__)

    if search_method == "first-improvement":
        return random.choice(successors)

    raise ValueError(f"Unsupported search method: {search_method}")


def hill_climb(
    cache: SearchCache,
    node: int,
    verbose: int = 0,
    return_trace: bool = False,
    search_method: str = "best-improvement",
) -> Tuple[Any, int, List[int]]:
    """
    Performs hill-climbing local search on the landscape graph held by ``cache``,
    starting from a specified node.

    Parameters
    ----------
    cache : SearchCache
        Precomputed search data (graph + hoisted fitness vector). Build once
        with ``SearchCache(graph)`` and reuse across a batch of climbs.

    node : int
        The index of the starting node for the hill climbing search.

    verbose : int, default=0
        The verbosity level for logging progress, where 0 is silent.

    return_trace: bool, default=False
        Whether to return the trace of the search as a list of node indices.

    search_method : str
        Specifies the method of local search to use ('best-improvement' or
        'first-improvement').

    Returns
    -------
    Tuple[Any, int] or Tuple[Any, int, List[int]]
        ``(final_local_optimum, n_steps)`` or, when ``return_trace`` is set,
        ``(final_local_optimum, n_steps, trace)``.
    """
    best = search_method == "best-improvement"
    if not best and search_method != "first-improvement":
        raise ValueError(f"Unsupported search method: {search_method}")

    g = cache.graph
    fit_get = cache.fitness.__getitem__

    if verbose > 0:
        print(f"Hill climbing begins from {node}...")

    step = 0
    trace = [node] if return_trace else None
    current = node

    # Every out-edge strictly increases fitness, so a climb is monotone and can
    # never revisit a node -- no cycle guard / visited set is needed. The walk
    # stops at the first node with no improving neighbour (out-degree 0).
    while True:
        successors = g.neighbors(current, mode="out")
        if not successors:
            break
        current = max(successors, key=fit_get) if best else random.choice(successors)
        step += 1
        if return_trace:
            trace.append(current)

    if verbose > 0:
        print(f"Finished at node {current} with {step} step(s).")

    if return_trace:
        return current, step, trace
    return current, step
