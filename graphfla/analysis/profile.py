"""Compute selected landscape metrics as a Series or a table."""

from __future__ import annotations

import inspect
import logging
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from .._progress import track

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
from .robustness import evolvability_enhancing_fraction, neutrality
from .epistasis import (
    classify_epistasis,
    diminishing_returns_index,
    extradimensional_bypass,
    gamma,
    gamma_star,
    global_idiosyncratic_index,
    increasing_costs_index,
)


@dataclass(frozen=True)
class _Metric:
    """Registry entry. ``prefix`` set => dictionary return."""

    name: str
    fn: object
    group: str
    prefix: Optional[str] = None  # column namespace for structured returns
    fields: Optional[tuple] = None  # which subkeys to keep (structured only)
    rename: Optional[dict] = None  # optional {subkey: short column name}


# The default portfolio: every whole-landscape, scalar-or-fixed-struct metric.
# Grouped by source module; order here is the order columns appear in the output.
_REGISTRY = (
    _Metric(
        "fitness_distribution",
        fitness_distribution,
        "fitness",
        prefix="fitness",
        fields=(
            "skewness",
            "kurtosis",
            "cv",
            "quartile_coefficient",
            "median_mean_ratio",
            "relative_range",
            "cauchy_loc",
        ),
    ),
    _Metric("local_optima_ratio", local_optima_ratio, "ruggedness"),
    _Metric("gradient_intensity", gradient_intensity, "ruggedness"),
    _Metric("autocorrelation", autocorrelation, "ruggedness"),
    _Metric("r_s_ratio", r_s_ratio, "ruggedness"),
    _Metric("neutrality", neutrality, "robustness"),
    _Metric(
        "evolvability_enhancing_fraction", evolvability_enhancing_fraction, "robustness"
    ),
    _Metric("fdc", fdc, "correlation"),
    _Metric("basin_fitness_correlation", basin_fitness_correlation, "correlation"),
    _Metric(
        "neighbor_fitness_correlation", neighbor_fitness_correlation, "correlation"
    ),
    _Metric("fitness_flattening_index", fitness_flattening_index, "correlation"),
    _Metric("global_optima_accessibility", global_optima_accessibility, "navigability"),
    _Metric(
        "mean_path_length_to_global_optimum",
        mean_path_length_to_global_optimum,
        "navigability",
    ),
    _Metric(
        "mean_distance_to_global_optimum",
        mean_distance_to_global_optimum,
        "navigability",
    ),
    _Metric("gamma", gamma, "epistasis"),
    _Metric("gamma_star", gamma_star, "epistasis"),
    _Metric("global_idiosyncratic_index", global_idiosyncratic_index, "epistasis"),
    _Metric("diminishing_returns_index", diminishing_returns_index, "epistasis"),
    _Metric("increasing_costs_index", increasing_costs_index, "epistasis"),
    _Metric(
        "classify_epistasis",
        classify_epistasis,
        "epistasis",
        prefix="epistasis",
        fields=("magnitude", "sign", "reciprocal_sign", "positive", "negative"),
    ),
    _Metric(
        "extradimensional_bypass",
        extradimensional_bypass,
        "epistasis",
        prefix="bypass",
        fields=("bypass_proportion", "average_bypass_length"),
        rename={
            "bypass_proportion": "proportion",
            "average_bypass_length": "avg_length",
        },
    ),
)

_BY_NAME = {m.name: m for m in _REGISTRY}
_GROUPS = tuple(dict.fromkeys(m.group for m in _REGISTRY))


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _columns_for(metric):
    if metric.prefix is None:
        return (metric.name,)
    rename = metric.rename or {}
    return tuple(f"{metric.prefix}.{rename.get(k, k)}" for k in metric.fields)


def _flatten(metric, value):
    if value is None:
        return {column: np.nan for column in _columns_for(metric)}
    if metric.prefix is None:
        return {metric.name: _as_float(value)}
    return dict(
        zip(_columns_for(metric), (_as_float(value.get(k)) for k in metric.fields))
    )


def _select(metrics):
    if metrics is None:
        return list(_REGISTRY)
    names = [metrics] if isinstance(metrics, str) else list(metrics)
    selected = {}
    for name in names:
        if name in _BY_NAME:
            selected[name] = _BY_NAME[name]
        elif name in _GROUPS:
            selected.update((m.name, m) for m in _REGISTRY if m.group == name)
        else:
            raise ValueError(
                f"Unknown metric or group {name!r}. Groups: {list(_GROUPS)}. "
                f"Metrics: {list(_BY_NAME)}."
            )
    return list(selected.values())


def _validate_params(params):
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary of metric parameter dictionaries.")
    for name, options in params.items():
        if name not in _BY_NAME:
            raise ValueError(f"Unknown metric in params: {name!r}.")
        if not isinstance(options, dict):
            raise TypeError(f"params[{name!r}] must be a dictionary.")
        allowed = set(inspect.signature(_BY_NAME[name].fn).parameters) - {"landscape"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unknown parameter(s) for {name!r}: {sorted(unknown)}.")
    return params


def _interactive_default():
    """Auto-enable the display only in a REPL/notebook, never in plain scripts."""
    if hasattr(sys, "ps1"):  # standard interactive interpreter
        return True
    ipy = sys.modules.get("IPython")  # an already-running notebook/kernel
    if ipy is None:
        return False
    try:
        return ipy.get_ipython() is not None
    except Exception:
        return False


def _resolve_show(progress):
    return _interactive_default() if progress is None else bool(progress)


def _replay_warnings(console, caught):
    """Re-emit warnings captured during profiling, de-duplicated, as a tidy
    footnote -- genuine warnings, but printed after the bar instead of through it."""
    seen, uniq = set(), []
    for w in caught:
        key = (w.category, str(w.message))
        if key not in seen:
            seen.add(key)
            uniq.append(w)
    if not uniq:
        return
    if console is not None:
        console.print(f"[yellow]![/] [dim]{len(uniq)} warning(s) during profiling:[/]")
    for w in uniq:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


@contextmanager
def _muted_landscapes(landscapes, enabled):
    """Mute several landscapes' own verbose logging/bars while an outer display
    runs (multi-landscape branch); a no-op when *enabled* is false."""
    if not enabled:
        yield
        return
    saved = [(ls, getattr(ls, "verbose", None)) for ls in landscapes]
    for ls, v in saved:
        if v:
            try:
                ls.verbose = False
            except Exception:
                pass
    glog = logging.getLogger("graphfla")
    level = glog.level
    if glog.level < logging.WARNING:
        glog.setLevel(logging.WARNING)
    wctx = warnings.catch_warnings(record=True)
    caught = wctx.__enter__()
    warnings.simplefilter("always")
    warnings.filterwarnings("ignore", message=r".*ipywidgets.*")
    try:
        yield
    finally:
        for ls, v in saved:
            if v:
                try:
                    ls.verbose = v
                except Exception:
                    pass
        glog.setLevel(level)
        wctx.__exit__(None, None, None)
        _replay_warnings(None, list(caught))


def profile(
    landscape,
    *,
    metrics=None,
    params=None,
    seed=None,
    n_jobs=-1,
    progress=None,
) -> Union[pd.Series, pd.DataFrame]:
    r"""Return selected analysis metrics for one or more landscapes.

    Parameters
    ----------
    landscape : Landscape, list of Landscape or tuple of Landscape
        One built landscape returns a Series. A list or tuple returns a
        DataFrame with one row per landscape, in input order.
    metrics : str, sequence of str or None, default=None
        Metrics to compute. Pass a group name, a function name, or a list
        mixing both; for example, ``"ruggedness"``, ``["fdc", "gamma"]``,
        or ``["ruggedness", "gamma"]``. None computes all 21 metrics.
        Available groups and their function names are:

        - ``"fitness"``: ``fitness_distribution``.
        - ``"ruggedness"``: ``local_optima_ratio``, ``gradient_intensity``,
          ``autocorrelation``, ``r_s_ratio``.
        - ``"robustness"``: ``neutrality``, ``evolvability_enhancing_fraction``.
        - ``"correlation"``: ``fdc``, ``basin_fitness_correlation``,
          ``neighbor_fitness_correlation``, ``fitness_flattening_index``.
        - ``"navigability"``: ``global_optima_accessibility``,
          ``mean_path_length_to_global_optimum``,
          ``mean_distance_to_global_optimum``.
        - ``"epistasis"``: ``gamma``, ``gamma_star``,
          ``global_idiosyncratic_index``, ``diminishing_returns_index``,
          ``increasing_costs_index``, ``classify_epistasis``,
          ``extradimensional_bypass``.

        Groups expand in the order listed above; repeated metrics are computed
        once, at their first position. An empty list selects no metrics.
        Use function names here, not output fields such as
        ``"epistasis.magnitude"``. Functions returning variable-length tables
        or requiring a mutation, position or target must be called directly.
    params : dict or None, default=None
        Optional settings for individual metrics, keyed by function name.
        For example, ``{"autocorrelation": {"walk_length": 50},
        "neutrality": {"threshold": 0.05}}``. These settings override shared
        seed and n_jobs values. Use ``{"classify_epistasis":
        {"sample_cut_prob": 0}}`` for exact motif enumeration, or set that
        function's ``time_budget`` here to control automatic sampling.
        None uses each function's defaults. Settings for unselected metrics
        are not used; unknown function or parameter names raise ValueError.
    seed : int or None, default=None
        Shared random seed for metrics that sample. An integer makes sampling
        reproducible; None leaves each function's default randomness in place.
    n_jobs : int or None, default=-1
        Worker count for metrics that support parallel computation. Use -1
        for all available CPUs or 1 for serial execution. Landscapes themselves
        are processed sequentially.
    progress : bool or None, default=None
        Show a progress bar on stderr. None shows it in interactive sessions
        and notebooks, and hides it in scripts. True or False forces the choice.

    Returns
    -------
    values : pandas.Series or pandas.DataFrame
        Float results, with function names as labels for scalar metrics.
        Dictionary results expand into columns: ``fitness.<statistic>``,
        ``epistasis.<type>``, ``bypass.proportion`` and ``bypass.avg_length``.
        A DataFrame uses a default integer index; assign its index after the
        call if labels are needed. Undefined results are NaN. A metric that
        fails produces NaN and a warning, while the remaining metrics continue.

    Raises
    ------
    ValueError
        If a metric, group or parameter name is unknown.
    TypeError
        If params or one of its values is not a dictionary.

    See Also
    --------
    list_metrics : Inspect available metrics and their output column names.
    graphfla.landscape.Landscape.describe : Read structural counts and properties.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import profile
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> profile(landscape, metrics="local_optima_ratio", progress=False).to_dict()
    {'local_optima_ratio': 0.25}
    >>> selected = profile(landscape, metrics=["fdc", "gamma"],
    ...                    n_jobs=1, progress=False)
    >>> selected.index.tolist()
    ['fdc', 'gamma']
    >>> profile(landscape, metrics="neutrality",
    ...         params={"neutrality": {"threshold": 1.0}}, progress=False).to_dict()
    {'neutrality': 0.25}
    """
    selected = _select(metrics)
    params = _validate_params(params)
    show = _resolve_show(progress)
    if isinstance(landscape, (list, tuple)):
        columns = [column for metric in selected for column in _columns_for(metric)]
        with _muted_landscapes(landscape, show):
            rows = [
                profile(
                    item,
                    metrics=[m.name for m in selected],
                    params=params,
                    seed=seed,
                    n_jobs=n_jobs,
                    progress=False,
                )
                for item in track(
                    landscape, description="profile landscapes", verbose=show
                )
            ]
        return pd.DataFrame(rows, columns=columns, dtype=float)

    out = {}
    with _muted_landscapes([landscape], show):
        for metric in track(selected, description="profile metrics", verbose=show):
            accepted = inspect.signature(metric.fn).parameters
            kwargs = {
                k: v
                for k, v in {"seed": seed, "n_jobs": n_jobs}.items()
                if k in accepted
            }
            kwargs.update(params.get(metric.name, {}))
            try:
                value = metric.fn(landscape, **kwargs)
            except Exception as exc:
                warnings.warn(
                    f"profile: metric {metric.name!r} failed "
                    f"({type(exc).__name__}: {exc}); recording NaN.",
                    stacklevel=2,
                )
                value = None
            out.update(_flatten(metric, value))
    return pd.Series(out, dtype=float)


def list_metrics() -> pd.DataFrame:
    r"""Return metadata for the metrics available through profile.

    Returns
    -------
    metrics : pandas.DataFrame
        One row per registered metric, indexed by its public function name.
        Columns are ``group``, ``kind`` ("scalar" or "dict"), ``columns``
        (comma-separated profile output names), and the boolean flags ``n_jobs``,
        ``seed`` and ``time_budget`` indicating which parameters each function accepts.
        Functions requiring a mutation, position or target, and variable-length
        result tables, are not part of this registry.

    See Also
    --------
    profile : Compute the selected metrics for one or more landscapes.

    Examples
    --------
    >>> from graphfla.analysis import list_metrics
    >>> list_metrics().loc["neutrality", "group"]
    'robustness'
    """
    rows = []
    for m in _REGISTRY:
        sig = inspect.signature(m.fn).parameters
        rows.append(
            {
                "metric": m.name,
                "group": m.group,
                "kind": "scalar" if m.prefix is None else "dict",
                "columns": ", ".join(_columns_for(m)),
                "n_jobs": "n_jobs" in sig,
                "seed": "seed" in sig,
                "time_budget": "time_budget" in sig,
            }
        )
    return pd.DataFrame(rows).set_index("metric")
