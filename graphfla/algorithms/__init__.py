# graphfla/algorithms/__init__.py
"""Methods for simulating evolution on fitness landscapes."""

from .adaptive_walk import local_search, hill_climb
from .random_walk import random_walk

__all__ = [
    "local_search",
    "hill_climb",
    "random_walk",
]
