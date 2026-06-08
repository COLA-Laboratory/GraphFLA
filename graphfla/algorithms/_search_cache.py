import numpy as np


class SearchCache:
    """Precomputed read-only data for fast repeated traversal of a static graph.

    The trajectory algorithms (``local_search``, ``hill_climb``, ``random_walk``)
    are called many times over the same landscape graph -- once per starting
    node, often for tens or hundreds of thousands of nodes. The original
    implementations read fitness through igraph's per-vertex Python attribute
    API (``graph.vs[i]["fitness"]``), which builds a ``Vertex`` proxy object and
    does a dict lookup *for every neighbour, every step, every call*. That
    per-successor proxy access dominates best-improvement cost.

    ``SearchCache`` hoists the per-vertex fitness vector into a single
    contiguous ``float64`` array once, so a best-improvement move becomes an
    O(degree) array gather (``max(succ, key=fitness.__getitem__)``) instead of
    one proxy lookup per neighbour. Construction is O(V) and costs only a few
    milliseconds even on the largest landscapes, so it does NOT introduce a
    crossover batch size below which it loses -- build it once and reuse it
    across the whole batch. The graph is retained for neighbour queries
    (``graph.neighbors``), whose order matches ``get_adjlist`` exactly, so
    best-improvement tie-breaks and stochastic RNG draws stay byte-identical.

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
        # `fitness` (ndarray) is kept for random_walk's vectorised attr gather.
        self.fitness_list = list(graph.vs["fitness"])
        self.fitness = np.asarray(self.fitness_list, dtype=np.float64)
