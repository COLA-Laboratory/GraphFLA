import warnings
import random
import numpy as np
import pandas as pd

from typing import Union, List, Optional, Callable
from ..distances import mixed_distance


from ._utils import _pythonize


def _as_lo_list(lo: Union[int, List[int]]) -> List[int]:
    """Normalize the ``lo`` argument to a list of integer node indices.

    Accepts a single int or a list of ints (preserving the original
    ``isinstance(..., int)`` contract); raises ``TypeError`` otherwise.
    """
    if isinstance(lo, int):
        return [lo]
    if isinstance(lo, list) and all(isinstance(i, int) for i in lo):
        return list(lo)
    raise TypeError("Parameter 'lo' must be an integer or a list of integers.")


def _validate_local_optima(landscape, lo_indices: List[int]) -> None:
    """Raise if any index is out of range or is not a local optimum.

    Uses the ``is_lo`` vertex attribute when present, else falls back to the
    out-degree-0 definition (identical to the per-function checks it replaces).
    """
    vcount = landscape.graph.vcount()
    has_is_lo_attr = "is_lo" in landscape.graph.vs.attributes()
    for l_idx in lo_indices:
        if not 0 <= l_idx < vcount:
            raise ValueError(
                f"Invalid node index: {l_idx}. Must be between 0 and {vcount - 1}."
            )
        if has_is_lo_attr:
            if not landscape.graph.vs[l_idx]["is_lo"]:
                raise ValueError(f"Node {l_idx} is not a local optimum.")
        elif landscape.graph.outdegree(l_idx) != 0:
            raise ValueError(
                f"Node {l_idx} is not a local optimum (has outgoing edges)."
            )


def local_optima_accessibility(
    landscape, lo: Union[int, List[int]]
) -> pd.DataFrame:
    r"""Return accessibility fractions for the requested local optima.

    This metric represents the fraction of configurations in the landscape
    that can reach the specified local optimum (or optima) via any monotonic,
    fitness-improving path.

    The implementation uses graph traversal to find all nodes (configurations)
    that have a directed path to the local optimum in the landscape graph.
    These are the "ancestors" of the local optimum - configurations from which
    the LO can be reached by following fitness-improving moves.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.
    lo : int or list of int
        Index of the local optimum to analyze, or a list of indices when analyzing
        multiple local optima.

    Returns
    -------
    accessibility : pandas.DataFrame
        One row per requested local optimum, with columns:

        - ``local_optimum`` : the local-optimum node index.
        - ``accessibility`` : the fraction of configurations able to reach it
          monotonically (between 0.0 and 1.0), including the target itself.

        A single ``lo`` yields a one-row frame (no scalar-vs-list polymorphism).

    Raises
    ------
    RuntimeError
        If the graph is not initialized.
    ValueError
        If any provided index is not a local optimum.
    TypeError
        If lo is not an int or a list of ints.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import local_optima_accessibility
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> local_optima_accessibility(landscape, lo=3).accessibility.tolist()
    [1.0]
    """
    if landscape.graph is None:
        raise RuntimeError("Graph not initialized. Cannot calculate accessibility.")

    lo_indices = _as_lo_list(lo)

    if landscape.n_configs is None or landscape.n_configs == 0:
        warnings.warn(
            "Landscape has 0 configurations. Accessibility is 0.", RuntimeWarning
        )
        return pd.DataFrame(
            {"local_optimum": lo_indices, "accessibility": [0.0] * len(lo_indices)}
        )

    _validate_local_optima(landscape, lo_indices)

    try:
        # Ancestors = configs with a monotonic path to the LO.
        accessibilities = [
            len(landscape.graph.subcomponent(l_idx, mode="in")) / landscape.n_configs
            for l_idx in lo_indices
        ]
    except Exception as e:
        raise RuntimeError(f"An error occurred during accessibility calculation: {e}") from e

    return pd.DataFrame(
        {"local_optimum": lo_indices, "accessibility": accessibilities}
    )


def global_optima_accessibility(landscape) -> float:
    r"""Return the fraction of configurations that can reach the global optimum.

    This metric represents the fraction of configurations in the landscape
    that can reach the global optimum via any monotonic, fitness-improving path.

    Use :func:`local_optima_accessibility` with the selected global-optimum
    index; the target itself counts as reachable.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.

    Returns
    -------
    fraction : float
        The fraction of configurations able to reach the global optimum
        monotonically (value between 0.0 and 1.0).

    Raises
    ------
    RuntimeError
        If the global optimum has not been determined or the graph is not initialized.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import global_optima_accessibility
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> global_optima_accessibility(landscape)
    1.0
    """
    if landscape.graph is None:
        raise RuntimeError("Graph not initialized. Cannot calculate accessibility.")

    if landscape.go_index is None:
        try:
            landscape._compute_global_optimum()
        except Exception as e:
            raise RuntimeError(
                f"Failed to determine global optimum: {e}. Cannot calculate accessibility."
            )
        if landscape.go_index is None:
            raise RuntimeError(
                "Global optimum could not be determined. Cannot calculate accessibility."
            )

    df = local_optima_accessibility(landscape, lo=landscape.go_index)
    return float(df["accessibility"].iloc[0])


def mean_path_length_to_local_optima(
    landscape,
    lo: Optional[Union[int, List[int]]] = None,
    accessible: bool = True,
    n_samples: Optional[Union[int, float]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    r"""Return mean and variance of shortest path lengths to local optima.

    This function computes the shortest path length from each configuration to the specified local optima.
    If accessible=True, only monotonically fitness-improving paths are considered (using OUT mode in distances).
    Otherwise, any path regardless of fitness is considered (using ALL mode).

    For large landscapes, computing distances for all configurations can be computationally expensive.
    In such cases, a warning is raised, and the function can use sampling to approximate the results by setting n_samples.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.
    lo : int, list of int or None, default=None
        Index of the local optimum to analyze, or a list of indices when analyzing
        multiple local optima. If None, uses the global optimum.
    accessible : bool, default=True
        If True, only consider monotonically accessible (fitness-improving) paths.
        If False, ignore the direction of existing graph edges. Retained
        neutral pairs are not added as path edges.
    n_samples : int, float or None, default=None
        If provided, use sampling to approximate the results:
        - If float in (0, 1]: Sample this fraction of configurations.
        - If positive int: Sample at most this many configurations.
        - If None: Compute for all configurations (with warning for large landscapes).
    seed : int or None, default=None
        Seed for sampling configurations. An integer makes the sample
        reproducible; None uses the global Python ``random`` state.
        Ignored when ``n_samples=None``.

    Returns
    -------
    path_lengths : pandas.DataFrame
        One row per target local optimum, with columns ``local_optimum``,
        ``mean`` and ``variance`` of the shortest path lengths to it. When
        ``lo`` is None the single row is the global optimum. Infinite distances
        are excluded from the calculations (a row whose targets are all
        unreachable has ``mean``/``variance`` of NaN).

    Raises
    ------
    RuntimeError
        If the graph is not initialized or the target optima are not determined.
    ValueError
        If n_samples is invalid or any provided index is not a local optimum.
    TypeError
        If lo is not an int, a list of ints, or None.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import mean_path_length_to_local_optima
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> mean_path_length_to_local_optima(landscape, lo=3)["mean"].tolist()
    [1.0]
    """
    if landscape.graph is None:
        raise RuntimeError("Graph not initialized. Cannot calculate path lengths.")

    if lo is None:
        if landscape.go_index is None:
            try:
                landscape._compute_global_optimum()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to determine global optimum: {e}. Cannot calculate path lengths."
                )
            if landscape.go_index is None:
                raise RuntimeError(
                    "Global optimum could not be determined. Cannot calculate path lengths."
                )
        target_indices = [landscape.go_index]
    else:
        target_indices = _as_lo_list(lo)

    _validate_local_optima(landscape, target_indices)

    # OUT = monotonic fitness-improving paths only; ALL = any path.
    mode = "OUT" if accessible else "ALL"
    # d(source -> target) over `mode` edges equals d(target -> source) over the
    # REVERSED edges. So every source's distance to a target can be obtained from
    # ONE traversal outward from the target, instead of a separate BFS per source
    # (O(V+E) vs O(N*(V+E))) -- identical integer shortest-path lengths.
    reverse_mode = {"OUT": "IN", "ALL": "ALL"}[mode]

    n_configs = landscape.graph.vcount()

    if n_configs > 10000 and n_samples is None:
        warnings.warn(
            f"Computing path lengths for a large landscape ({n_configs} configurations) "
            "may be computationally expensive. Consider using sampling by setting n_samples.",
            RuntimeWarning,
        )

    if n_samples is not None:
        if isinstance(n_samples, float):  # fraction
            if not 0 < n_samples <= 1:
                raise ValueError(
                    "When n_samples is a float, it must be between 0 and 1."
                )
            sample_size = max(1, int(n_samples * n_configs))
        elif isinstance(n_samples, int):  # count
            if n_samples <= 0:
                raise ValueError("When n_samples is an integer, it must be positive.")
            sample_size = min(n_samples, n_configs)
        else:
            raise ValueError(
                "n_samples must be a float between 0 and 1 or a positive integer."
            )

        # Local RNG when a seed is given, else global state.
        rand = random.Random(seed) if seed is not None else random
        sampled_indices = rand.sample(range(n_configs), sample_size)
    else:
        sampled_indices = range(n_configs)

    means = []
    variances = []
    try:
        for target_idx in target_indices:
            # Single traversal outward from the target over reversed edges;
            # row[j] is the distance from sampled source j to target_idx along
            # `mode` edges (see reverse_mode above).
            flattened_distances = landscape.graph.distances(
                source=target_idx, target=sampled_indices, mode=reverse_mode
            )[0]

            # Exclude unreachable (infinite) distances.
            finite_distances = [d for d in flattened_distances if np.isfinite(d)]

            if len(finite_distances) == 0:
                means.append(np.nan)
                variances.append(np.nan)
            else:
                means.append(np.mean(finite_distances))
                variances.append(np.var(finite_distances))

        return pd.DataFrame(
            {"local_optimum": target_indices, "mean": means, "variance": variances}
        )

    except Exception as e:
        raise RuntimeError(f"An error occurred during path length calculation: {e}") from e


def mean_path_length_to_global_optimum(
    landscape,
    accessible: bool = True,
    n_samples: Optional[Union[int, float]] = None,
    seed: Optional[int] = None,
) -> float:
    r"""Return the mean shortest path length to the global optimum.

    This function computes the shortest path length from each configuration to the global optimum.
    It extracts the mean returned by :func:`mean_path_length_to_local_optima`.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.
    accessible : bool, default=True
        If True, only consider monotonically accessible (fitness-improving) paths.
        If False, ignore the direction of existing graph edges. Retained
        neutral pairs are not added as path edges.
    n_samples : int, float or None, default=None
        If provided, use sampling to approximate the results:
        - If float in (0, 1]: Sample this fraction of configurations.
        - If positive int: Sample at most this many configurations.
        - If None: Compute for all configurations (with warning for large landscapes).
    seed : int or None, default=None
        Seed for sampling configurations. An integer makes the sample
        reproducible; None uses the global Python ``random`` state.
        Ignored when ``n_samples=None``.

    Returns
    -------
    mean_length : float
        The mean shortest path length to the global optimum. Infinite distances are
        excluded from the calculation.

    Raises
    ------
    RuntimeError
        If the graph is not initialized or the global optimum is not determined.
    ValueError
        If n_samples is invalid.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import mean_path_length_to_global_optimum
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> mean_path_length_to_global_optimum(landscape)
    1.0
    """
    if landscape.graph is None:
        raise RuntimeError("Graph not initialized. Cannot calculate path lengths.")

    if landscape.go_index is None:
        try:
            landscape._compute_global_optimum()
        except Exception as e:
            raise RuntimeError(
                f"Failed to determine global optimum: {e}. Cannot calculate path lengths."
            )
        if landscape.go_index is None:
            raise RuntimeError(
                "Global optimum could not be determined. Cannot calculate path lengths."
            )

    df = mean_path_length_to_local_optima(
        landscape,
        lo=landscape.go_index,
        accessible=accessible,
        n_samples=n_samples,
        seed=seed,
    )
    return float(df["mean"].iloc[0])


def mean_distance_to_local_optima(
    landscape, lo: Union[int, List[int]], distance_func: Optional[Callable] = None
) -> pd.DataFrame:
    r"""Return mean configuration distances to the requested local optima.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.
    lo : int or list of int
        Index of the local optimum to analyze, or a list of indices when analyzing
        multiple local optima.
    distance_func : callable or None, default=None
        Callable ``distance_func(configs, target, data_types)`` returning one
        distance per row of the encoded configuration array. If None, use
        the landscape default distance metric. The target contributes zero
        for standard distance functions.

    Returns
    -------
    distances : pandas.DataFrame
        One row per requested local optimum, with columns ``local_optimum`` and
        ``mean_distance`` (the mean distance from all configurations to it). A
        single ``lo`` yields a one-row frame (no scalar-vs-list polymorphism).

    Raises
    ------
    RuntimeError
        If the graph is not initialized or required attributes are missing.
    ValueError
        If any provided index is not a local optimum.
    TypeError
        If lo is not an int or a list of ints.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import mean_distance_to_local_optima
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> mean_distance_to_local_optima(landscape, lo=3).mean_distance.tolist()
    [1.0]
    """
    if landscape.graph is None:
        raise RuntimeError("Graph not initialized. Cannot calculate distances.")

    if landscape.configs is None or landscape.data_types is None:
        raise RuntimeError("Required attributes (configs, data_types) are missing.")

    lo_indices = _as_lo_list(lo)
    _validate_local_optima(landscape, lo_indices)

    if distance_func is None:
        distance_func = getattr(
            landscape, "_get_default_distance_metric", lambda: mixed_distance
        )()

    configs = np.vstack(landscape.configs.values)

    mean_distances = [
        np.mean(distance_func(configs, configs[target_idx], landscape.data_types))
        for target_idx in lo_indices
    ]

    return pd.DataFrame(
        {"local_optimum": lo_indices, "mean_distance": mean_distances}
    )


def mean_distance_to_global_optimum(landscape, distance_func: Optional[Callable] = None) -> float:
    r"""Return the mean configuration distance to the global optimum.

    Reuse cached ``dist_go`` values only when ``distance_func=None``. An
    explicit callable is always evaluated and does not replace the cache.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.
    distance_func : callable or None, default=None
        Callable ``distance_func(configs, target, data_types)`` returning one
        distance per row of the encoded configuration array. If None, use
        the landscape default distance metric. The target contributes zero
        for standard distance functions.

    Returns
    -------
    mean_distance : float
        The mean distance from all configurations to the global optimum.

    Raises
    ------
    RuntimeError
        If the graph is not initialized, required attributes are missing, or the
        global optimum has not been determined.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import mean_distance_to_global_optimum
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> mean_distance_to_global_optimum(landscape)
    1.0
    """
    if landscape.graph is None:
        raise RuntimeError("Graph not initialized. Cannot calculate distances.")

    # A cached default distance must not override an explicit distance function.
    if distance_func is None and "dist_go" in landscape.graph.vs.attributes():
        distances = landscape.graph.vs["dist_go"]
        return _pythonize(np.mean(distances))

    if landscape.configs is None or landscape.data_types is None:
        raise RuntimeError("Required attributes (configs, data_types) are missing.")

    if distance_func is None:
        distance_func = getattr(
            landscape, "_get_default_distance_metric", lambda: mixed_distance
        )()

    configs = np.vstack(landscape.configs.values)
    go_config = configs[landscape.go_index]
    distances = distance_func(configs, go_config, landscape.data_types)
    return _pythonize(np.mean(distances))
