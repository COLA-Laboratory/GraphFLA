"""Low-level array / threshold helpers for edge construction (leaf module)."""

import numpy as np


def _neutral_abs_threshold(epsilon: float) -> float:
    """Inclusive bound on |Δf| for classifying a neighbor pair as neutral.

    For ``epsilon > 0``, returns the next float above *epsilon* so values that
    differ from *epsilon* only at the last ULP (common after arithmetic on
    fitness arrays) still count as neutral. For ``epsilon == 0``, returns
    ``0.0`` so only exactly-zero |Δf| is neutral.
    """
    eps = float(epsilon)
    if eps <= 0.0:
        return 0.0
    return float(np.nextafter(eps, np.inf))


# ===================================================================
# Neighbor generators
# ===================================================================


def _empty_edges() -> np.ndarray:
    """Return a canonical empty ``(0, 2)`` int64 edge array."""
    return np.empty((0, 2), dtype=np.int64)


def _empty_deltas() -> np.ndarray:
    """Return a canonical empty 1-D float64 delta-fit array."""
    return np.empty(0, dtype=np.float64)


def _stack_edges(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Stack aligned source/target index arrays into an ``(E, 2)`` int64 array."""
    if src.size == 0:
        return _empty_edges()
    out = np.empty((src.shape[0], 2), dtype=np.int64)
    out[:, 0] = src
    out[:, 1] = tgt
    return out


def _edge_arrays_from_lists(edge_list, delta_list):
    """Convert Python edge/delta lists (slow fallback paths) to ndarrays."""
    edges = (
        np.asarray(edge_list, dtype=np.int64) if edge_list else _empty_edges()
    )
    delta_fits = (
        np.asarray(delta_list, dtype=np.float64)
        if delta_list
        else _empty_deltas()
    )
    return edges, delta_fits


def _as_config_matrix(configs, configs_array=None):
    """Return a contiguous numeric configuration matrix."""
    if configs_array is not None:
        return np.ascontiguousarray(configs_array)
    config_list = configs.tolist() if hasattr(configs, "tolist") else list(configs)
    return np.ascontiguousarray(np.array(config_list))
