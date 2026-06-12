"""Unified trajectory walks over a built landscape graph.

All walks share one contract: construct a walker bound to a :class:`SearchCache`
(plus walk-specific options), then call ``walker.run(start)`` to get a
:class:`WalkResult`. Building the walker once and reusing it across a batch of
start nodes keeps the per-node cost identical to the old free functions (the
object is constructed per *walk batch*, never per *step*).
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from ._search_cache import SearchCache
from ..exceptions import InvalidParameterError

_STRATEGIES = ("best-improvement", "first-improvement")


class WalkResult:
    """The outcome of a single walk.

    Attributes
    ----------
    path : np.ndarray
        Visited node ids, including the start node, in visitation order.

    Notes
    -----
    A plain ``__slots__`` class (not a dataclass) because walks are constructed
    in tight per-node batches and ``frozen`` dataclass ``__init__`` (which routes
    every field through ``object.__setattr__``) shows up in those hot loops.
    """

    __slots__ = ("path",)

    def __init__(self, path: np.ndarray):
        self.path = path

    @property
    def final(self) -> int:
        """The last node visited."""
        return int(self.path[-1])

    @property
    def n_steps(self) -> int:
        """Number of moves taken (``len(path) - 1``)."""
        return int(len(self.path) - 1)

    def __repr__(self) -> str:
        return f"WalkResult(n_steps={self.n_steps}, final={self.final})"


class Walk:
    """Base class for walks bound to a :class:`SearchCache`.

    Parameters
    ----------
    cache : SearchCache
        Precomputed graph + hoisted fitness vector. Build once and reuse.
    seed : int, optional
        Seed for this walk's random choices. ``None`` (default) uses the global
        ``random`` state, preserving the historical, unseeded behaviour.
    """

    def __init__(self, cache: SearchCache, *, seed: Optional[int] = None):
        self.cache = cache
        # seed=None -> the process-global `random` module (historical behaviour);
        # an explicit seed -> a private, reproducible generator.
        self._rng = random.Random(seed) if seed is not None else random

    def run(self, start: int) -> WalkResult:  # pragma: no cover - abstract
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

    def run(self, start: int) -> WalkResult:
        """Climb to a local optimum, recording the full visited path."""
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
        return WalkResult(np.asarray(path, dtype=np.int64))

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
        self.length = length
        self.neutral_neighbors = neutral_neighbors

    def run(self, start: int) -> WalkResult:
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
            if not neighbors:
                break
            node = self._rng.choice(neighbors)
            cnt += 1
        return WalkResult(nodes[:cnt])
