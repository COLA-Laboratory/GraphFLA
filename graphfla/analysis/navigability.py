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
    """
    Calculate the accessibility of one or more specified local optima (LOs).

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
        The fitness landscape object.
    lo : int or list[int]
        Index of the local optimum to analyze, or a list of indices when analyzing
        multiple local optima.

    Returns
    -------
    pandas.DataFrame
        One row per requested local optimum, with columns:

        - ``local_optimum`` : the local-optimum node index.
        - ``accessibility`` : the fraction of configurations able to reach it
          monotonically (between 0.0 and 1.0).

        A single ``lo`` yields a one-row frame (no scalar-vs-list polymorphism).

    Raises
    ------
    RuntimeError
        If the graph is not initialized.
    ValueError
        If any provided index is not a local optimum.
    TypeError
        If lo is not an int or a list of ints.
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
        raise RuntimeError(f"An error occurred during accessibility calculation: {e}")

    return pd.DataFrame(
        {"local_optimum": lo_indices, "accessibility": accessibilities}
    )


def global_optima_accessibility(landscape) -> float:
    """
    Calculate the accessibility of the global optimum (GO).

    This metric represents the fraction of configurations in the landscape
    that can reach the global optimum via any monotonic, fitness-improving path.

    This function relies on `local_optima_accessibility` by passing the
    global optimum index.

    Returns
    -------
    float
        The fraction of configurations able to reach the global optimum
        monotonically (value between 0.0 and 1.0).

    Raises
    ------
    RuntimeError
        If the global optimum has not been determined or the graph is not initialized.
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
    lo: Union[int, List[int]] = None,
    accessible: bool = True,
    n_samples: Optional[Union[int, float]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate the mean and variance of the shortest path lengths from configurations to local optima.

    This function computes the shortest path length from each configuration to the specified local optima.
    If accessible=True, only monotonically fitness-improving paths are considered (using OUT mode in distances).
    Otherwise, any path regardless of fitness is considered (using ALL mode).

    For large landscapes, computing distances for all configurations can be computationally expensive.
    In such cases, a warning is raised, and the function can use sampling to approximate the results by setting n_samples.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.
    lo : int or list[int], optional
        Index of the local optimum to analyze, or a list of indices when analyzing
        multiple local optima. If None, uses the global optimum.
    accessible : bool, default=True
        If True, only consider monotonically accessible (fitness-improving) paths.
        If False, consider any path regardless of fitness changes.
    n_samples : int or float, optional
        If provided, use sampling to approximate the results:
        - If float between 0 and 1: Sample this fraction of configurations.
        - If int > 1: Sample this specific number of configurations.
        - If None: Compute for all configurations (with warning for large landscapes).

    Returns
    -------
    pandas.DataFrame
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
        raise RuntimeError(f"An error occurred during path length calculation: {e}")


def mean_path_length_to_global_optimum(
    landscape,
    accessible: bool = True,
    n_samples: Optional[Union[int, float]] = None,
    seed: Optional[int] = None,
) -> float:
    """
    Calculate the mean and variance of the shortest path lengths from configurations to the global optimum.

    This function computes the shortest path length from each configuration to the global optimum.
    It is a convenience wrapper around the more general `path_lengths` function.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.
    accessible : bool, default=True
        If True, only consider monotonically accessible (fitness-improving) paths.
        If False, consider any path regardless of fitness changes.
    n_samples : int or float, optional
        If provided, use sampling to approximate the results:
        - If float between 0 and 1: Sample this fraction of configurations.
        - If int > 1: Sample this specific number of configurations.
        - If None: Compute for all configurations (with warning for large landscapes).

    Returns
    -------
    float
        The mean shortest path length to the global optimum. Infinite distances are
        excluded from the calculation.

    Raises
    ------
    RuntimeError
        If the graph is not initialized or the global optimum is not determined.
    ValueError
        If n_samples is invalid.
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
    """
    Calculate the mean distance from all configurations to one or more specified local optima.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.
    lo : int or list[int]
        Index of the local optimum to analyze, or a list of indices when analyzing
        multiple local optima.
    distance_func : callable, optional
        A function to calculate distances between configurations. If None, uses the
        default distance metric from the landscape based on its type.

    Returns
    -------
    pandas.DataFrame
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
    """
    Calculate the mean distance from all configurations to the global optimum.

    This function first checks if distances to the global optimum have already been
    calculated and stored as 'dist_go' in the graph's vertex attributes. If not, it
    calculates these distances using the provided or default distance function.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.
    distance_func : callable, optional
        A function to calculate distances between configurations. If None, uses the
        default distance metric from the landscape based on its type.

    Returns
    -------
    float
        The mean distance from all configurations to the global optimum.

    Raises
    ------
    RuntimeError
        If the graph is not initialized, required attributes are missing, or the
        global optimum has not been determined.
    """
    if landscape.graph is None:
        raise RuntimeError("Graph not initialized. Cannot calculate distances.")

    # Reuse cached dist_go if present.
    if "dist_go" in landscape.graph.vs.attributes():
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
