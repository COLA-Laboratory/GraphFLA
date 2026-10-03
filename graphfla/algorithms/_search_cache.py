import numpy as np


class SearchCache:
    """Precomputed read-only data for fast repeated traversal of a static graph.

    Store the graph and its fitness values once, then reuse the cache across
    HillClimb and RandomWalk instances. Neighbors are queried from the graph;
    fitness lookup uses cached Python and NumPy arrays. Rebuild the cache if
    the graph or its fitness values change.

    Parameters
    ----------
    graph : ig.Graph
        A built landscape graph carrying a per-vertex ``"fitness"`` attribute.
    """

    __slots__ = ("graph", "n", "fitness", "fitness_list")

    def __init__(self, graph):
        self.graph = graph
        self.n = graph.vcount()
        # `fitness_list` (plain Python floats) is the key for best-improvement
        # max() -- list indexing returns the existing float object, avoiding the
        # np.float64 boxing that ndarray.__getitem__ does on every access.
        # `fitness` (ndarray) is kept for RandomWalk's vectorised attr gather.
        self.fitness_list = list(graph.vs["fitness"])
        self.fitness = np.asarray(self.fitness_list, dtype=np.float64)
