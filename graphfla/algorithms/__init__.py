# graphfla/algorithms/__init__.py
"""Trajectory simulation on fitness landscapes: the pure Walk classes.

Basin / optima / plateau construction was moved into ``graphfla.landscape`` as
build-internal steps; this package now holds only the search-cache and the
seedable Walk hierarchy (``HillClimb`` / ``RandomWalk``).
"""

from ._search_cache import SearchCache
from .walk import Walk, WalkResult, HillClimb, RandomWalk

__all__ = [
    "SearchCache",
    "Walk",
    "WalkResult",
    "HillClimb",
    "RandomWalk",
]
