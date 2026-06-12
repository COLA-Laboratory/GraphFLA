"""Neighbor generation and edge construction for fitness landscapes.

Public entry point: :func:`build_edges`. Internals are split across the
_arrays / generators / _classify / _kernels / edges submodules.
"""

from .edges import build_edges, EdgeResult
from .generators import (
    NeighborGenerator,
    BooleanNeighborGenerator,
    SequenceNeighborGenerator,
    OrdinalNeighborGenerator,
    DefaultNeighborGenerator,
)

__all__ = [
    "build_edges",
    "EdgeResult",
    "NeighborGenerator",
    "BooleanNeighborGenerator",
    "SequenceNeighborGenerator",
    "OrdinalNeighborGenerator",
    "DefaultNeighborGenerator",
]
