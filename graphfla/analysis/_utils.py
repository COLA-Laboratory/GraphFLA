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


def _pack_rows(M):
    """Dense 0-based group id per distinct row of a small-int matrix, via
    mixed-radix packing into a single int64 key (one fast 1D ``np.unique``).

    Returns ``(ids, n_groups)``; ``(None, 0)`` when the radix product would
    overflow int64 (high-cardinality / many-column inputs) so callers can fall
    back to a dict/byte grouping.
    """
    nrows = M.shape[0]
    if M.shape[1] == 0:
        return np.zeros(nrows, dtype=np.intp), 1
    radices = M.max(axis=0).astype(np.int64) + 1
    prod = 1
    for r in radices:
        prod *= int(r)
        if prod > (1 << 62):
            return None, 0
    mult = np.ones(M.shape[1], dtype=np.int64)
    for i in range(M.shape[1] - 2, -1, -1):
        mult[i] = mult[i + 1] * int(radices[i + 1])
    key = M.astype(np.int64) @ mult
    uniq, inv = np.unique(key, return_inverse=True)
    return np.asarray(inv).reshape(-1).astype(np.intp), int(uniq.size)
