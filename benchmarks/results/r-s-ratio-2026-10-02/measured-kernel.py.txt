"""Numerically scaled additive fits with bounded design-matrix storage."""

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import lstsq, qr, LinAlgError


_DESIGN_BYTES = 16 * 1024**2
_QR_BYTES = 64 * 1024**2
_SLOPE_REL_TOL = 1e-12


def _undefined(message):
    warnings.warn(message + " Returning nan.", UserWarning, stacklevel=3)
    return float("nan")


def _predictors(landscape, data):
    """Store one vector per variable, not a full one-hot matrix."""
    n = len(data)
    columns = list(landscape.data_types)
    configs = getattr(landscape, "_configs_array", None)
    if configs is not None and configs.shape != (n, len(columns)):
        configs = None
    specs = []
    for j, name in enumerate(columns):
        kind = landscape.data_types[name]
        if kind == "categorical":
            cat = pd.Categorical(data[name]).remove_unused_categories()
            if np.any(cat.codes < 0):
                raise ValueError("Configuration states must not be missing.")
            if len(cat.categories) > 1:
                specs.append((cat.codes, len(cat.categories) - 1))
        elif kind in {"boolean", "ordinal"}:
            if kind == "boolean":
                values = np.asarray(data[name], dtype=bool).astype(float)
            elif configs is not None:
                values = np.asarray(configs[:, j], dtype=float)
            else:
                values = pd.Categorical(data[name], ordered=True).codes.astype(float)
            if not np.isfinite(values).all():
                raise ValueError("Configuration codes must be finite.")
            if n and np.any(values != values[0]):
                specs.append((values, 0))
        else:
            raise ValueError(f"Unsupported data type {kind!r} in column {name!r}.")
    return specs


def _design(specs, start, stop):
    width = 1 + sum(k or 1 for _, k in specs)
    matrix = np.ones((stop - start, width), dtype=float, order="F")
    j = 1
    for values, k in specs:
        if k:
            for state in range(1, k + 1):
                matrix[:, j] = values[start:stop] == state
                j += 1
        else:
            matrix[:, j] = values[start:stop]
            j += 1
    return matrix


def _additive_fit(specs, y):
    """Use SVD, compressing tall inputs by QR without normal equations."""
    n = len(y)
    width = 1 + sum(k or 1 for _, k in specs)
    rows = max(width + 1, _DESIGN_BYTES // (8 * (width + 1)))
    if n <= rows:
        design = _design(specs, 0, n)
        coefficients, _, rank, _ = lstsq(
            design,
            y,
            cond=max(n, width) * np.finfo(float).eps,
            check_finite=False,
        )
    else:
        # QR preserves least-squares geometry. Forming X.T @ X would square
        # the condition number and make rank decisions unreliable.
        reduced = np.empty((0, width + 1))
        for start in range(0, n, rows):
            stop = min(n, start + rows)
            block = np.column_stack((_design(specs, start, stop), y[start:stop]))
            augmented = np.vstack((reduced, block))
            reduced = qr(augmented, mode="r", check_finite=False, overwrite_a=True)[0][
                : width + 1
            ].copy()
        coefficients, _, rank, _ = lstsq(
            reduced[:, :width],
            reduced[:, width],
            cond=max(n, width) * np.finfo(float).eps,
            check_finite=False,
        )
    squared_error = 0.0
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        # einsum avoids a second N x p array and platform-specific matmul
        # overflow warnings on otherwise finite Accelerate inputs.
        residual = y[start:stop] - np.einsum(
            "ij,j->i", _design(specs, start, stop), coefficients
        )
        squared_error += float(np.dot(residual, residual))
    return coefficients, rank, np.sqrt(squared_error / n)


def roughness_slope_ratio(landscape):
    graph = getattr(landscape, "graph", None)
    columns = list(landscape.data_types or {})
    if graph is not None and all(name in graph.vs.attributes() for name in columns):
        # Degree, basin and path metadata are irrelevant to this regression.
        # Read only vertex-aligned inputs rather than copying every attribute.
        data = pd.DataFrame({name: graph.vs[name] for name in [*columns, "fitness"]})
    else:
        # Retain get_data's feature-label recovery for imported graphs.
        data = landscape.get_data()
    fitness = np.asarray(data["fitness"], dtype=float)
    if not np.isfinite(fitness).all():
        raise ValueError("Objective values must be finite.")
    if not len(fitness):
        return _undefined("The landscape contains no configurations.")
    if np.all(fitness == fitness[0]):
        return _undefined("Fitness is constant, so both roughness and slope are zero.")

    specs = _predictors(landscape, data)
    width = 1 + sum(k or 1 for _, k in specs)
    if len(fitness) < width or width == 1:
        return _undefined("The additive coefficients are not identifiable.")
    # Bound the quadratic workspace as well as the tall design blocks before
    # allocating either. This protects high-cardinality optimization inputs.
    if 8 * (width + 1) ** 2 > _QR_BYTES:
        return _undefined("The encoded model exceeds the 64 MiB QR workspace limit.")

    # Power-of-two scaling cannot overflow the subtraction, and preserves
    # represented small differences around a large offset. Normalize again
    # after subtracting the offset, before any squared quantities are formed.
    exponent = int(np.frexp(np.max(np.abs(fitness)))[1])
    y = np.ldexp(fitness, -exponent)
    y -= y[0]
    spread = np.ptp(y)
    y /= spread
    y -= np.mean(y)
    try:
        coefficients, rank, roughness = _additive_fit(specs, y)
    except LinAlgError:
        return _undefined("The additive least-squares solve did not converge.")
    if rank < width:
        return _undefined(
            "The additive design is rank deficient; slope is not identifiable."
        )
    if len(y) == width:
        warnings.warn(
            "The additive fit is saturated and has no residual degrees of freedom.",
            UserWarning,
            stacklevel=3,
        )
    slope = float(np.mean(np.abs(coefficients[1:])))
    if slope <= _SLOPE_REL_TOL:
        warnings.warn(
            "Slope 's' is zero or near zero. Returning inf.", UserWarning, stacklevel=3
        )
        return float("inf")
    return float(roughness / slope)
