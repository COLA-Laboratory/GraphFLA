"""One-call landscape metric profile -- the computed-metric analog of
:meth:`graphfla.landscape.Landscape.describe`.

``describe()`` reports cheap structural facts (sizes, optima counts); ``profile()``
runs the whole-landscape analysis metrics and returns a tidy ``pandas`` object:
a ``Series`` for one landscape, a ``DataFrame`` (one row each) for several.

Only metrics that need *just the landscape* and return a scalar or a fixed-field
struct are included; drill-down metrics (those needing a ``mutation`` / ``position``
/ ``lo``, or returning a variable-length table) are intentionally left out -- call
those functions directly. Selection is via ``groups`` or ``include`` (pick the base
set) and ``exclude`` (a filter that composes with either); ``params`` overrides
per-metric kwargs. Stochastic/parallel metrics receive shared ``seed`` / ``n_jobs``
/ ``time_budget`` automatically, but only if their signature accepts them.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .correlation import (
    basin_fitness_correlation,
    fdc,
    fitness_flattening_index,
    neighbor_fitness_correlation,
)
from .fitness import fitness_distribution
from .ruggedness import (
    autocorrelation,
    gradient_intensity,
    local_optima_ratio,
    r_s_ratio,
)
from .navigability import (
    global_optima_accessibility,
    mean_distance_to_global_optimum,
    mean_path_length_to_global_optimum,
)
from .robustness import evolvability_enhancing_mutations, neutrality
from .epistasis import (
    classify_epistasis,
    diminishing_returns_index,
    extradimensional_bypass,
    gamma,
    gamma_star,
    global_idiosyncratic_index,
    higher_order_epistasis,
    increasing_costs_index,
)


@dataclass(frozen=True)
class _Metric:
    """Registry entry. ``prefix`` set => structured (dataclass/dict) return."""

    name: str
    fn: object
    group: str
    prefix: Optional[str] = None          # column namespace for structured returns
    fields: Optional[tuple] = None        # which subkeys to keep (structured only)
    rename: Optional[dict] = None         # optional {subkey: short column name}


# The default portfolio: every whole-landscape, scalar-or-fixed-struct metric.
# Grouped by source module; order here is the order columns appear in the output.
_REGISTRY = (
    _Metric("fitness_distribution", fitness_distribution, "fitness", prefix="fitness",
            fields=("skewness", "kurtosis", "cv", "quartile_coefficient",
                    "median_mean_ratio", "relative_range", "cauchy_loc")),

    _Metric("local_optima_ratio", local_optima_ratio, "ruggedness"),
    _Metric("gradient_intensity", gradient_intensity, "ruggedness"),
    _Metric("autocorrelation", autocorrelation, "ruggedness"),
    _Metric("r_s_ratio", r_s_ratio, "ruggedness"),

    _Metric("neutrality", neutrality, "robustness"),
    _Metric("evolvability_enhancing_mutations", evolvability_enhancing_mutations, "robustness"),

    _Metric("fdc", fdc, "correlation"),
    _Metric("basin_fitness_correlation", basin_fitness_correlation, "correlation"),
    _Metric("neighbor_fitness_correlation", neighbor_fitness_correlation, "correlation"),
    _Metric("fitness_flattening_index", fitness_flattening_index, "correlation"),

    _Metric("global_optima_accessibility", global_optima_accessibility, "navigability"),
    _Metric("mean_path_length_to_global_optimum", mean_path_length_to_global_optimum,
            "navigability"),
    _Metric("mean_distance_to_global_optimum", mean_distance_to_global_optimum, "navigability"),

    _Metric("gamma", gamma, "epistasis"),
    _Metric("gamma_star", gamma_star, "epistasis"),
    _Metric("higher_order_epistasis", higher_order_epistasis, "epistasis"),
    _Metric("global_idiosyncratic_index", global_idiosyncratic_index, "epistasis"),
    _Metric("diminishing_returns_index", diminishing_returns_index, "epistasis"),
    _Metric("increasing_costs_index", increasing_costs_index, "epistasis"),
    _Metric("classify_epistasis", classify_epistasis, "epistasis", prefix="epistasis",
            fields=("magnitude", "sign", "reciprocal_sign", "positive", "negative")),
    _Metric("extradimensional_bypass", extradimensional_bypass, "epistasis", prefix="bypass",
            fields=("bypass_proportion", "average_bypass_length"),
            rename={"bypass_proportion": "proportion", "average_bypass_length": "avg_length"}),
)

_BY_NAME = {m.name: m for m in _REGISTRY}
_GROUPS = tuple(dict.fromkeys(m.group for m in _REGISTRY))


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _columns_for(m):
    """Output column names for a metric -- known up front, even when it fails."""
    if m.prefix is None:
        return (m.name,)
    rn = m.rename or {}
    return tuple(f"{m.prefix}.{rn.get(k, k)}" for k in m.fields)


def _flatten(m, value):
    cols = _columns_for(m)
    if value is None:                              # failed or skipped
        return {c: np.nan for c in cols}
    if m.prefix is None:                           # scalar
        return {m.name: _as_float(value)}
    data = asdict(value) if is_dataclass(value) else dict(value)
    rn = m.rename or {}
    return {f"{m.prefix}.{rn.get(k, k)}": _as_float(data.get(k)) for k in m.fields}


def _shared_kwargs(fn, **shared):
    params = inspect.signature(fn).parameters
    return {k: v for k, v in shared.items() if k in params}


def _select(groups, include, exclude):
    if include is not None and groups is not None:
        raise ValueError("pass at most one of `groups` or `include`, not both")
    if include is not None:
        include = [include] if isinstance(include, str) else list(include)
        unknown = [n for n in include if n not in _BY_NAME]
        if unknown:
            raise ValueError(
                f"unknown metric name(s) in include: {unknown}; see analysis.list_metrics()"
            )
        base = [_BY_NAME[n] for n in include]
    elif groups is not None:
        groups = [groups] if isinstance(groups, str) else list(groups)
        unknown = [g for g in groups if g not in _GROUPS]
        if unknown:
            raise ValueError(f"unknown group(s): {unknown}; valid groups are {list(_GROUPS)}")
        gset = set(groups)
        base = [m for m in _REGISTRY if m.group in gset]
    else:
        base = list(_REGISTRY)
    if exclude:
        exclude = [exclude] if isinstance(exclude, str) else list(exclude)
        unknown = [n for n in exclude if n not in _BY_NAME]
        if unknown:
            raise ValueError(f"unknown metric name(s) in exclude: {unknown}")
        eset = set(exclude)
        base = [m for m in base if m.name not in eset]
    return base


def profile(
    landscape,
    *,
    groups=None,
    include=None,
    exclude=None,
    params=None,
    n_jobs=-1,
    seed=None,
    time_budget=15.0,
    on_error="warn",
    include_structure=False,
    index=None,
):
    """Compute the whole-landscape metric profile.

    Parameters
    ----------
    landscape : Landscape or sequence of Landscape
        One built landscape -> a ``Series``; a list/tuple -> a ``DataFrame``,
        one row per landscape.
    groups : str or sequence of str, optional
        Restrict to these metric groups (one of ``analysis.list_metrics()``'s
        ``group`` values). Mutually exclusive with ``include``.
    include : str or sequence of str, optional
        Use exactly these metrics (by registry name). Mutually exclusive with
        ``groups``.
    exclude : str or sequence of str, optional
        Drop these metrics from whatever ``groups`` / ``include`` selected
        (or from the full default). Composes with either.
    params : dict, optional
        Per-metric keyword overrides, ``{metric_name: {kwarg: value}}``.
    n_jobs, seed, time_budget : optional
        Shared keywords forwarded to each metric *that accepts them*.
    on_error : {"warn", "raise", "ignore"}, default "warn"
        How to handle a metric that raises -- record NaN (warn/ignore) or
        propagate.
    include_structure : bool, default False
        Prepend the numeric fields of :meth:`Landscape.describe` as
        ``structure.*`` columns.
    index : optional
        Index for the returned ``DataFrame`` when ``landscape`` is a sequence.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Metric values (a flat, all-float index of metric names; structured
        metrics flatten to dotted columns such as ``epistasis.magnitude``).
    """
    if on_error not in ("warn", "raise", "ignore"):
        raise ValueError("on_error must be 'warn', 'raise', or 'ignore'")

    if isinstance(landscape, (list, tuple)):
        rows = [
            profile(
                ls, groups=groups, include=include, exclude=exclude, params=params,
                n_jobs=n_jobs, seed=seed, time_budget=time_budget, on_error=on_error,
                include_structure=include_structure,
            )
            for ls in landscape
        ]
        df = pd.DataFrame(rows)
        if index is not None:
            df.index = index
        return df

    metrics = _select(groups, include, exclude)
    params = params or {}
    out = {}
    if include_structure:
        for k, v in landscape.describe().items():
            if not isinstance(v, bool) and isinstance(v, (int, float)):
                out[f"structure.{k}"] = float(v)
    for m in metrics:
        kwargs = _shared_kwargs(m.fn, n_jobs=n_jobs, seed=seed, time_budget=time_budget)
        kwargs.update(params.get(m.name, {}))
        try:
            value = m.fn(landscape, **kwargs)
        except Exception as e:  # noqa: BLE001 -- one bad metric must not sink the profile
            if on_error == "raise":
                raise
            if on_error == "warn":
                warnings.warn(
                    f"profile: metric {m.name!r} failed ({type(e).__name__}: {e}); "
                    f"recording NaN.",
                    stacklevel=2,
                )
            value = None
        out.update(_flatten(m, value))
    return pd.Series(out, dtype=float)


def list_metrics():
    """Return the default metric portfolio as a DataFrame (for discoverability).

    Columns: ``group``, ``kind`` (scalar / struct), the output ``columns`` each
    metric contributes, and which shared kwargs (``n_jobs`` / ``seed`` /
    ``time_budget``) it accepts.
    """
    rows = []
    for m in _REGISTRY:
        sig = inspect.signature(m.fn).parameters
        rows.append({
            "metric": m.name,
            "group": m.group,
            "kind": "scalar" if m.prefix is None else "struct",
            "columns": ", ".join(_columns_for(m)),
            "n_jobs": "n_jobs" in sig,
            "seed": "seed" in sig,
            "time_budget": "time_budget" in sig,
        })
    return pd.DataFrame(rows).set_index("metric")
