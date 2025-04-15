# graphfla/algorithms/__init__.py
"""Methods for simulating evolution on fitness landscapes."""

from .adaptive_walk import local_search_igraph, hill_climb_igraph
from .random_walk import random_walk_igraph

__all__ = [
    "local_search_igraph",
    "hill_climb_igraph",
    "random_walk_igraph",
]
