"""GraphFLA errors, also catchable through their corresponding built-in types."""

from __future__ import annotations


class GraphFLAError(Exception):
    """Base class for every GraphFLA-specific error."""


class NotBuiltError(GraphFLAError, RuntimeError):
    """An operation needs a built landscape, but it has not been built yet.

    """


class InvalidParameterError(GraphFLAError, ValueError):
    """A parameter value is outside its allowed domain.

    Also a :class:`ValueError`.
    """


class DataValidationError(GraphFLAError, ValueError):
    """The input data ``(X, fitness)`` is malformed, inconsistent, or unusable."""
