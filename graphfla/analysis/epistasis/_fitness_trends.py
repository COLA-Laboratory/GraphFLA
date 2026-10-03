"""Stable, edge-weighted fitness trends on the retained landscape graph."""

import warnings

import numpy as np
from scipy.stats import rankdata


_EDGE_BLOCK = 32768


class _BivariateMoments:
    """Merge centered block moments without subtracting large raw sums."""

    def __init__(self):
        self.n = 0
        self.mx = self.my = self.xx = self.yy = self.xy = 0.0

    def update(self, x, y):
        n = len(x)
        if not n:
            return
        mx = float(x[0] + np.mean(x - x[0]))
        my = float(y[0] + np.mean(y - y[0]))
        dx, dy = x - mx, y - my
        total = self.n + n
        weight = self.n * (n / total)
        shift_x, shift_y = mx - self.mx, my - self.my
        self.xx += float(dx @ dx) + shift_x * shift_x * weight
        self.yy += float(dy @ dy) + shift_y * shift_y * weight
        self.xy += float(dx @ dy) + shift_x * shift_y * weight
        self.mx += shift_x * (n / total)
        self.my += shift_y * (n / total)
        self.n = total

    def statistic(self, method):
        if self.n < 2 or self.xx == 0 or (method != "regression" and self.yy == 0):
            warnings.warn(
                "Fitness trend is undefined: fewer than two edges or constant "
                "background fitness (or constant effects for correlation).",
                UserWarning,
                stacklevel=3,
            )
            return float("nan")
        if method == "regression":
            return float(self.xy / self.xx)
        return float(np.clip((self.xy / np.sqrt(self.xx)) / np.sqrt(self.yy), -1, 1))


def _oriented_fitness(fitness, maximize):
    """Use one affine scale for background and effect, preserving the slope."""
    f = np.asarray(fitness, dtype=float)
    if not len(f):
        return f
    # Subtract before scaling to preserve small, representable differences at
    # a large offset. A power of two avoids rounding equal edge differences
    # unequally. Scale first only if the initial subtraction would overflow.
    with np.errstate(over="ignore"):
        anchor = f[np.argmin(np.abs(f))]
        shifted = f - anchor
    if not np.isfinite(shifted).all():
        exponent = int(np.frexp(np.max(np.abs(f)))[1])
        f = np.ldexp(f, -exponent)
        shifted = f - f[0]
    magnitude = np.max(np.abs(shifted))
    if magnitude:
        shifted = np.ldexp(shifted, -int(np.frexp(magnitude)[1]))
    return shifted if maximize else -shifted


def _edge_observations(endpoints, fitness, costs):
    source, target = endpoints.T
    effect = fitness[target] - fitness[source]
    if np.any(effect < 0):
        raise ValueError("Graph edges must point toward improving fitness.")
    selected = effect > 0
    # A deleterious move reverses the stored improving edge, so its
    # background is the target and its positive cost is the same gap.
    return fitness[target if costs else source][selected], effect[selected]


def _edge_blocks(graph, fitness, costs):
    for start in range(0, graph.ecount(), _EDGE_BLOCK):
        edges = graph.es[start : start + _EDGE_BLOCK]
        endpoints = np.asarray([edge.tuple for edge in edges], dtype=np.intp)
        yield _edge_observations(endpoints, fitness, costs)


def _edge_fitness_trend(landscape, method, *, costs):
    if method not in {"pearson", "spearman", "regression"}:
        raise ValueError("Method must be 'pearson', 'spearman', or 'regression'.")
    landscape._check_built()
    graph = landscape.graph
    if graph is None or "fitness" not in graph.vs.attributes():
        raise ValueError("Landscape graph or node 'fitness' attribute not found.")
    if not graph.is_directed():
        raise ValueError("A directed improving-edge graph is required.")
    raw = np.asarray(graph.vs["fitness"], dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Node fitness values must be finite.")
    active = np.asarray(graph.degree()) > 0
    fitness = np.zeros(len(raw))
    # Isolated observations have no statistical weight. Do not let their
    # potentially extreme fitness determine the numeric scale of edge data.
    fitness[active] = _oriented_fitness(raw[active], landscape.maximize)
    moments = _BivariateMoments()
    if method == "spearman":
        # Exact ranks require the entire edge population. The other methods
        # retain only vertex fitness and a fixed-size block of edges.
        if graph.ecount():
            endpoints = np.asarray(graph.get_edgelist(), dtype=np.intp)
            x, y = _edge_observations(endpoints, fitness, costs)
            moments.update(rankdata(x), rankdata(y))
    else:
        for x, y in _edge_blocks(graph, fitness, costs):
            moments.update(x, y)
    return moments.statistic(method)
