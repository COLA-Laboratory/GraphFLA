"""Reusable graph walkers returning path and endpoint data as dictionaries."""

from __future__ import annotations

import random
from typing import Dict, Optional, Union
from numbers import Integral

import numpy as np

from ._search_cache import SearchCache
from ..exceptions import InvalidParameterError

_STRATEGIES = ("best-improvement", "first-improvement")


class Walk:
    """Base class for walks bound to a :class:`SearchCache`.

    Parameters
    ----------
    cache : SearchCache
        Precomputed graph + hoisted fitness vector. Build once and reuse.
    seed : int, optional
        Seed for this walk's random choices. ``None`` (default) uses the global
        ``random`` state.
    """

    def __init__(self, cache: SearchCache, *, seed: Optional[int] = None):
        self.cache = cache
        # An explicit seed isolates the walk from process-global random state.
        self._rng = random.Random(seed) if seed is not None else random

    def run(
        self, start: int
    ) -> Dict[str, Union[np.ndarray, int]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _check_start(self, start: int) -> None:
        if start < 0 or start >= self.cache.n:
            raise InvalidParameterError(
                f"start node {start} is out of range [0, {self.cache.n})."
            )


class HillClimb(Walk):
    """Greedy adaptive walk that follows improving (out-) edges to a local optimum.

    Parameters
    ----------
    cache : SearchCache
    strategy : {"best-improvement", "first-improvement"}, default="best-improvement"
        ``best-improvement`` always moves to the highest-fitness improving
        neighbour; ``first-improvement`` picks a uniformly random improving
        neighbour. Every out-edge strictly increases fitness, so the climb is
        monotone and never revisits a node.
    seed : int, optional
        Reproducibility for ``first-improvement`` (ignored by best-improvement).
    """

    def __init__(
        self,
        cache: SearchCache,
        *,
        strategy: str = "best-improvement",
        seed: Optional[int] = None,
    ):
        super().__init__(cache, seed=seed)
        if strategy not in _STRATEGIES:
            raise InvalidParameterError(
                f"strategy must be one of {_STRATEGIES}, got {strategy!r}."
            )
        self.strategy = strategy

    def run(self, start: int) -> Dict[str, Union[np.ndarray, int]]:
        """Return the path to a local optimum and its endpoint.

        Parameters
        ----------
        start : int
            Zero-based index of the starting node.

        Returns
        -------
        result : dict
            ``path`` is an integer ndarray of visited node indices, including
            start. ``final`` is the last node (int), and ``n_steps`` is the
            number of edges traversed (int).
        """
        self._check_start(start)
        g = self.cache.graph
        fit_get = self.cache.fitness_list.__getitem__
        best = self.strategy == "best-improvement"
        current = start
        path = [start]
        while True:
            successors = g.neighbors(current, mode="out")
            if not successors:
                break
            # best: first-maximum tie-break (successor order == graph.neighbors
            # order); first: uniform random improving neighbour.
            current = (
                max(successors, key=fit_get) if best else self._rng.choice(successors)
            )
            path.append(current)
        return {
            "path": np.asarray(path, dtype=np.int64),
            "final": int(current),
            "n_steps": len(path) - 1,
        }

    def descend(self, start: int) -> tuple:
        """Endpoint-only climb returning ``(final_node, n_steps)``.

        The fast path for batch basin computation: identical traversal to
        :meth:`run` but without materialising the visited path, so it carries no
        per-node list/array allocation. (The loop is duplicated rather than
        shared to keep this hot path allocation-free.)
        """
        g = self.cache.graph
        fit_get = self.cache.fitness_list.__getitem__
        best = self.strategy == "best-improvement"
        current = start
        steps = 0
        while True:
            successors = g.neighbors(current, mode="out")
            if not successors:
                break
            current = (
                max(successors, key=fit_get) if best else self._rng.choice(successors)
            )
            steps += 1
        return current, steps


class RandomWalk(Walk):
    """Unbiased random walk over the undirected neighbourhood for a fixed length.

    Parameters
    ----------
    cache : SearchCache
    length : int, default=100
        Number of nodes to visit (the walk stops early at a node with no
        neighbours).
    neutral_neighbors : dict, optional
        Mapping ``node -> list[int]`` of equal-fitness neighbours that have no
        directed edge; when given, the walker may also traverse them.
    seed : int, optional
        Reproducibility for the walk.
    """

    def __init__(
        self,
        cache: SearchCache,
        *,
        length: int = 100,
        neutral_neighbors: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(cache, seed=seed)
        if (
            isinstance(length, (bool, np.bool_))
            or not isinstance(length, Integral)
            or length < 1
        ):
            raise InvalidParameterError("length must be a positive integer.")
        self.length = int(length)
        self.neutral_neighbors = neutral_neighbors

    def run(self, start: int) -> Dict[str, Union[np.ndarray, int]]:
        """Return a random path and its endpoint.

        Parameters
        ----------
        start : int
            Zero-based index of the starting node.

        Returns
        -------
        result : dict
            ``path`` is an integer ndarray containing at most length visited
            nodes, including start. ``final`` is the last node (int), and
            ``n_steps`` is the number of edges traversed (int). An isolated
            start returns a one-node path with zero steps.
        """
        self._check_start(start)
        g = self.cache.graph
        nodes = np.empty(self.length, dtype=np.int64)
        node = start
        cnt = 0
        while cnt < self.length:
            nodes[cnt] = node
            neighbors = g.neighbors(node, mode="all")
            if self.neutral_neighbors and node in self.neutral_neighbors:
                neighbors = list(set(neighbors) | set(self.neutral_neighbors[node]))
            cnt += 1
            if not neighbors:
                break
            node = self._rng.choice(neighbors)
        return {"path": nodes[:cnt], "final": int(nodes[cnt - 1]), "n_steps": cnt - 1}
