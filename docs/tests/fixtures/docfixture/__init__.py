"""Fixture only; building documentation must never import this module."""

raise RuntimeError("A static documentation build imported the documented package!")

from .metric import metric, stream
from .models import Box

__all__ = ["metric", "stream", "Box"]
