"""Shared internal helpers for the analysis subpackage."""

from __future__ import annotations

import numpy as np


def _pythonize(value):
    """Recursively convert numpy scalar types to native Python types.

    Walks dicts/lists/tuples so a returned structure carries plain Python
    floats/ints (JSON-friendly, clean reprs) rather than numpy scalar objects.
    """
    if isinstance(value, dict):
        return {key: _pythonize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_pythonize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pythonize(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value
