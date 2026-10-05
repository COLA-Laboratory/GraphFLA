"""Neighbor graphs of small search spaces and their stress-majorized layouts."""
import itertools
import math
import random


def product(*factors):
    """Return nodes and edges of a Cartesian product of variables.

    Parameters
    ----------
    *factors : tuple of (int, str)
        Number of values and kind of each variable. ``"categorical"`` links every
        pair of values; ``"ordinal"`` links adjacent values only.

    Returns
    -------
    nodes : list of tuple
        One value index per variable.
    edges : list of (int, int)
        Pairs of nodes that differ in one variable by a single allowed change.
    """
    nodes = list(itertools.product(*(range(size) for size, _ in factors)))
    edges = []
    for a, b in itertools.combinations(range(len(nodes)), 2):
        diff = [k for k in range(len(factors)) if nodes[a][k] != nodes[b][k]]
        if len(diff) == 1:
            k = diff[0]
            if factors[k][1] == "categorical" or abs(nodes[a][k] - nodes[b][k]) == 1:
                edges.append((a, b))
    return nodes, edges


def simplex(steps):
    """Return the compositions of three components in ``steps`` increments, linked by one transfer."""
    nodes = [(i, j, steps - i - j) for i in range(steps + 1) for j in range(steps + 1 - i)]
    edges = [(a, b) for a, b in itertools.combinations(range(len(nodes)), 2)
             if sorted(x - y for x, y in zip(nodes[a], nodes[b])) == [-1, 0, 1]]
    return nodes, edges


def distances(count, edges):
    """Return all-pairs shortest path lengths by breadth-first search."""
    adjacent = [[] for _ in range(count)]
    for a, b in edges:
        adjacent[a].append(b)
        adjacent[b].append(a)
    table = []
    for source in range(count):
        row = [math.inf] * count
        row[source], frontier = 0, [source]
        while frontier:
            following = []
            for node in frontier:
                for other in adjacent[node]:
                    if row[other] == math.inf:
                        row[other] = row[node] + 1
                        following.append(other)
            frontier = following
        table.append(row)
    return table


def stress_layout(count, edges, seed=0, iterations=400):
    """Return 2-D positions minimizing the stress of graph distances (SMACOF).

    Each pair is weighted by the inverse squared distance, which keeps
    neighborhoods tight while letting the whole graph fold into a round shape.
    """
    d = distances(count, edges)
    rng = random.Random(seed)
    x = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(count)]
    w = [[0.0 if i == j else d[i][j] ** -2 for j in range(count)] for i in range(count)]
    total = [sum(row) for row in w]
    for _ in range(iterations):
        updated = []
        for i in range(count):
            sx = sy = 0.0
            for j in range(count):
                if i == j:
                    continue
                dx, dy = x[i][0] - x[j][0], x[i][1] - x[j][1]
                norm = math.hypot(dx, dy) or 1e-9
                pull = w[i][j] * d[i][j] / norm
                sx += w[i][j] * x[j][0] + pull * dx
                sy += w[i][j] * x[j][1] + pull * dy
            updated.append([sx / total[i], sy / total[i]])
        x = updated
    return [tuple(p) for p in x]


def fit_disc(points, center=(0.5, 0.5), radius=0.4):
    """Center the points and scale them so the farthest lies on the given circle."""
    mx = sum(p[0] for p in points) / len(points)
    my = sum(p[1] for p in points) / len(points)
    far = max(math.hypot(p[0] - mx, p[1] - my) for p in points) or 1.0
    return [(center[0] + (p[0] - mx) / far * radius, center[1] + (p[1] - my) / far * radius) for p in points]
