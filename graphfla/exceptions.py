"""Custom exception types for GraphFLA.

A small taxonomy so callers can catch GraphFLA-specific failures precisely.
Each type also subclasses the built-in it semantically replaces, so existing
``except RuntimeError`` / ``except ValueError`` handlers (and the test suite)
keep working after the migration.
"""

from __future__ import annotations


class GraphFLAError(Exception):
    """Base class for every GraphFLA-specific error."""


class NotBuiltError(GraphFLAError, RuntimeError):
    """An operation needs a built landscape, but it has not been built yet.

    Subclasses :class:`RuntimeError` (the type historically raised by the
    ``_check_built`` guard) so existing handlers continue to catch it. This is
    GraphFLA's analogue of scikit-learn's ``NotFittedError``.
    """


class InvalidParameterError(GraphFLAError, ValueError):
    """A parameter value is outside its allowed domain.

    Subclasses :class:`ValueError` so callers catching ``ValueError`` still work.
    """


class DataValidationError(GraphFLAError, ValueError):
    """The input data ``(X, fitness)`` is malformed, inconsistent, or unusable."""
