import warnings
import random
import numpy as np

from typing import Union, List, Optional, Callable
from ..distances import mixed_distance


def local_optima_accessibility(
    landscape, lo: Union[int, List[int]]
) -> Union[float, List[float]]:
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
    float or list[float]
        If lo is a single integer: The fraction of configurations able to reach the
        specified local optimum monotonically (value between 0.0 and 1.0).
        If lo is a list: A list of fractions, each representing the accessibility of
        the corresponding local optimum.

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

    if landscape.n_configs is None or landscape.n_configs == 0:
        warnings.warn(
            "Landscape has 0 configurations. Accessibility is 0.", RuntimeWarning
        )
        return (
            0.0
            if isinstance(lo, int)
            else [0.0] * (len(lo) if isinstance(lo, list) else 0)
        )

    # Process input and verify the local optima
    single_input = isinstance(lo, int)
    if single_input:
        lo_indices = [lo]  # Convert single integer to list for uniform processing
    elif isinstance(lo, list) and all(isinstance(i, int) for i in lo):
        lo_indices = lo
    else:
        raise TypeError("Parameter 'lo' must be an integer or a list of integers.")

    # Validate each index
    has_is_lo_attr = "is_lo" in landscape.graph.vs.attributes()
    for l_idx in lo_indices:
        # Verify that the index is valid
        if not 0 <= l_idx < landscape.graph.vcount():
            raise ValueError(
                f"Invalid node index: {l_idx}. Must be between 0 and {landscape.graph.vcount()-1}."
            )

        # Check if the specified node is actually a local optimum
        if has_is_lo_attr:
            if not landscape.graph.vs[l_idx]["is_lo"]:
                raise ValueError(f"Node {l_idx} is not a local optimum.")
        else:
            # If 'is_lo' attribute isn't available, check if node has out-degree 0
            if landscape.graph.outdegree(l_idx) != 0:
                raise ValueError(
                    f"Node {l_idx} is not a local optimum (has outgoing edges)."
                )

    # Calculate accessibility for each local optimum
    accessibilities = []
    try:
        for l_idx in lo_indices:
            # Find all ancestors of the specified local optimum
            ancestors_set = landscape.graph.subcomponent(l_idx, mode="in")
            # Calculate accessibility as the fraction of nodes that can reach the LO
            accessibility = len(ancestors_set) / landscape.n_configs
            accessibilities.append(accessibility)
    except Exception as e:
        raise RuntimeError(f"An error occurred during accessibility calculation: {e}")

    # Return either a single value or list based on input type
    return accessibilities[0] if single_input else accessibilities


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
        # Attempt to determine GO if not already done
        try:
            landscape._determine_global_optimum()
        except Exception as e:
            raise RuntimeError(
                f"Failed to determine global optimum: {e}. Cannot calculate accessibility."
            )
        if landscape.go_index is None:  # Check again after attempting
            raise RuntimeError(
                "Global optimum could not be determined. Cannot calculate accessibility."
            )

    # Delegate the calculation to local_optima_accessibility
    return local_optima_accessibility(landscape, lo=landscape.go_index)


def mean_path_lengths(
    landscape,
    lo: Union[int, List[int]] = None,
    accessible: bool = True,
    n_samples: Optional[Union[int, float]] = None,
) -> Union[dict, List[dict]]:
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
    dict or list[dict]
        If lo is a single integer or None: A dictionary containing the "mean" and "variance" of the shortest path lengths.
        If lo is a list: A list of dictionaries, each containing "mean" and "variance" for the corresponding local optimum.
        Infinite distances are excluded from the calculations.

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

    # Process input and handle defaults
    if lo is None:
        # Use global optimum by default
        if landscape.go_index is None:
            # Attempt to determine GO if not already done
            try:
                landscape._determine_global_optimum()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to determine global optimum: {e}. Cannot calculate path lengths."
                )
            if landscape.go_index is None:  # Check again after attempting
                raise RuntimeError(
                    "Global optimum could not be determined. Cannot calculate path lengths."
                )
        target_indices = [landscape.go_index]
        single_input = True
    elif isinstance(lo, int):
        target_indices = [lo]
        single_input = True
    elif isinstance(lo, list) and all(isinstance(i, int) for i in lo):
        target_indices = lo
        single_input = False
    else:
        raise TypeError(
            "Parameter 'lo' must be an integer, a list of integers, or None."
        )

    # Validate each index
    has_is_lo_attr = "is_lo" in landscape.graph.vs.attributes()
    for l_idx in target_indices:
        # Verify that the index is valid
        if not 0 <= l_idx < landscape.graph.vcount():
            raise ValueError(
                f"Invalid node index: {l_idx}. Must be between 0 and {landscape.graph.vcount()-1}."
            )

        # Check if the specified node is actually a local optimum
        if has_is_lo_attr:
            if not landscape.graph.vs[l_idx]["is_lo"]:
                raise ValueError(f"Node {l_idx} is not a local optimum.")
        else:
            # If 'is_lo' attribute isn't available, check if node has out-degree 0
            if landscape.graph.outdegree(l_idx) != 0:
                raise ValueError(
                    f"Node {l_idx} is not a local optimum (has outgoing edges)."
                )

    # Determine the mode for path calculation
    mode = "OUT" if accessible else "ALL"

    # Handle sampling for large landscapes
    n_configs = landscape.graph.vcount()

    # Issue warning for large landscapes without sampling
    if n_configs > 10000 and n_samples is None:
        warnings.warn(
            f"Computing path lengths for a large landscape ({n_configs} configurations) "
            "may be computationally expensive. Consider using sampling by setting n_samples.",
            RuntimeWarning,
        )

    # Determine which configurations to analyze
    if n_samples is not None:
        if isinstance(n_samples, float):
            # n_samples is a fraction
            if not 0 < n_samples <= 1:
                raise ValueError(
                    "When n_samples is a float, it must be between 0 and 1."
                )
            sample_size = max(1, int(n_samples * n_configs))
        elif isinstance(n_samples, int):
            # n_samples is a count
            if n_samples <= 0:
                raise ValueError("When n_samples is an integer, it must be positive.")
            sample_size = min(n_samples, n_configs)
        else:
            raise ValueError(
                "n_samples must be a float between 0 and 1 or a positive integer."
            )

        # Sample node indices
        sampled_indices = random.sample(range(n_configs), sample_size)
    else:
        # Use all configurations
        sampled_indices = range(n_configs)

    # Calculate path lengths for each target optimum
    results = []
    try:
        for target_idx in target_indices:
            # Get distances from each node to the target optimum
            path_lengths_results = landscape.graph.distances(
                source=sampled_indices, target=target_idx, mode=mode
            )

            # Flatten the result (distances returns a list of lists)
            flattened_distances = [lengths[0] for lengths in path_lengths_results]

            # Filter out infinite distances
            finite_distances = [d for d in flattened_distances if np.isfinite(d)]

            # Calculate mean and variance
            if len(finite_distances) == 0:
                results.append({"mean": np.nan, "variance": np.nan})
            else:
                mean_distance = np.mean(finite_distances)
                variance_distance = np.var(finite_distances)
                results.append({"mean": mean_distance, "variance": variance_distance})

        # Return either a single dict or list based on input type
        return results[0] if single_input else results

    except Exception as e:
        raise RuntimeError(f"An error occurred during path length calculation: {e}")


def mean_path_lengths_go(
    landscape, accessible: bool = True, n_samples: Optional[Union[int, float]] = None
) -> dict:
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
    dict
        A dictionary containing the "mean" and "variance" of the shortest path lengths
        to the global optimum. Infinite distances are excluded from the calculations.

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
        # Attempt to determine GO if not already done
        try:
            landscape._determine_global_optimum()
        except Exception as e:
            raise RuntimeError(
                f"Failed to determine global optimum: {e}. Cannot calculate path lengths."
            )
        if landscape.go_index is None:  # Check again after attempting
            raise RuntimeError(
                "Global optimum could not be determined. Cannot calculate path lengths."
            )

    # Delegate the calculation to path_lengths with go_index
    return mean_path_lengths(
        landscape, lo=landscape.go_index, accessible=accessible, n_samples=n_samples
    )


def accessible_fract(landscape):
    raise NotImplementedError(
        "The function 'accessible_fract' is not implemented yet. Please check back later."
    )


def mean_dist_lo(
    landscape, lo: Union[int, List[int]], distance_func: Optional[Callable] = None
) -> Union[float, List[float]]:
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
    float or list[float]
        If lo is a single integer: The mean distance from all configurations to the
        specified local optimum.
        If lo is a list: A list of mean distances, each representing the mean distance
        to the corresponding local optimum.

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

    # Process input and verify the local optima
    single_input = isinstance(lo, int)
    if single_input:
        lo_indices = [lo]  # Convert single integer to list for uniform processing
    elif isinstance(lo, list) and all(isinstance(i, int) for i in lo):
        lo_indices = lo
    else:
        raise TypeError("Parameter 'lo' must be an integer or a list of integers.")

    # Validate each index
    has_is_lo_attr = "is_lo" in landscape.graph.vs.attributes()
    for l_idx in lo_indices:
        # Verify that the index is valid
        if not 0 <= l_idx < landscape.graph.vcount():
            raise ValueError(
                f"Invalid node index: {l_idx}. Must be between 0 and {landscape.graph.vcount()-1}."
            )

        # Check if the specified node is actually a local optimum
        if has_is_lo_attr:
            if not landscape.graph.vs[l_idx]["is_lo"]:
                raise ValueError(f"Node {l_idx} is not a local optimum.")
        else:
            # If 'is_lo' attribute isn't available, check if node has out-degree 0
            if landscape.graph.outdegree(l_idx) != 0:
                raise ValueError(
                    f"Node {l_idx} is not a local optimum (has outgoing edges)."
                )

    # Use default distance function if none provided
    if distance_func is None:
        # Get the appropriate distance metric based on landscape type
        distance_func = getattr(
            landscape, "_get_default_distance_metric", lambda: mixed_distance
        )()

    # Get all configurations and convert to numpy array for efficient calculation
    configs = np.vstack(landscape.configs.values)

    # Calculate mean distances for each target optimum
    mean_distances = []
    for target_idx in lo_indices:
        # Get the configuration of the target optimum
        target_config = configs[target_idx]

        # Calculate distances from all configurations to the target
        distances = distance_func(configs, target_config, landscape.data_types)

        # Calculate mean distance
        mean_dist = np.mean(distances)
        mean_distances.append(mean_dist)

    # Return either a single value or list based on input type
    return mean_distances[0] if single_input else mean_distances


def mean_dist_go(landscape, distance_func: Optional[Callable] = None) -> float:
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

    # Check if distances to global optimum have already been calculated
    if "dist_go" in landscape.graph.vs.attributes():
        # Use pre-calculated distances
        distances = landscape.graph.vs["dist_go"]
        return np.mean(distances)

    # Otherwise, need to calculate distances
    if landscape.configs is None or landscape.data_types is None:
        raise RuntimeError("Required attributes (configs, data_types) are missing.")

    # Use default distance function if none provided
    if distance_func is None:
        # Get the appropriate distance metric based on landscape type
        distance_func = getattr(
            landscape, "_get_default_distance_metric", lambda: mixed_distance
        )()

    # Get all configurations and convert to numpy array for efficient calculation
    configs = np.vstack(landscape.configs.values)

    # Get the configuration of the global optimum
    go_config = configs[landscape.go_index]

    # Calculate distances from all configurations to the global optimum
    distances = distance_func(configs, go_config, landscape.data_types)

    # Calculate and return the mean distance
    return np.mean(distances)
