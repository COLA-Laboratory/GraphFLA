import random
import warnings


def _is_ancestor(G, start_node, target_node):
    """
    Checks if target_node is reachable from start_node by following directed
    edges in graph G using Depth-First Search.

    Parameters
    ----------
    G : igraph.Graph
        The directed graph.
    start_node : int
        The node ID to start the search from.
    target_node : int
        The node ID to check reachability for.

    Returns
    -------
    bool
        True if target_node is reachable from start_node, False otherwise.
    """
    if start_node == target_node:
        return True  # Node is reachable from itself

    stack = [start_node]
    visited = {start_node}  # Add start node to visited immediately

    while stack:
        node = stack.pop()
        # Get successors (outgoing neighbors) in igraph
        for successor in G.neighbors(node, mode="out"):
            if successor == target_node:
                return True
            if successor not in visited:
                visited.add(successor)
                stack.append(successor)
    return False


def global_optima_accessibility(landscape, approximate=False, n_samples=0.2) -> float:
    """
    Calculate or estimate the accessibility of the global optimum (GO).

    This metric represents the fraction of configurations in the landscape
    that can reach the global optimum via any monotonic, fitness-improving path.

    By default (`approximate=False`), it relies on the 'size_basin_accessible'
    attribute calculated by `determine_accessible_paths()`. If this
    attribute is not available (i.e., `determine_accessible_paths()` has
    not been run or `landscape._path_calculated` is False), this method will
    raise a RuntimeError unless `approximate` is set to True.

    If `approximate=True`, the accessibility is estimated by sampling
    a specified number or fraction (`n_samples`) of configurations and
    checking if the GO is reachable from each sample using graph traversal
    (specifically, checking if the GO is a successor via directed paths).

    Parameters
    ----------
    approximate : bool, default=False
        If True, estimate accessibility by sampling and graph traversal.
        If False, use the pre-calculated 'size_basin_accessible' attribute for the GO.
    n_samples : int or float, optional
        Specifies the number or fraction of configurations to sample for
        approximation. Required and must be set if `approximate=True`.
        - If int: The absolute number of configurations to sample (must be > 0).
        - If float: The fraction of total configurations to sample (must be in (0, 1]).
        Ignored if `approximate=False`.

    Returns
    -------
    float
        The fraction of configurations estimated or known to be able to
        reach the global optimum monotonically (value between 0.0 and 1.0).

    Raises
    ------
    RuntimeError
        If `approximate=False` and `determine_accessible_paths()` has not
        been successfully run (i.e., `landscape._path_calculated` is False).
        If the global optimum has not been determined.
        If the graph is not initialized.
    ValueError
        If `approximate=True` and `n_samples` is None or invalid (e.g., int <= 0,
        float not in (0, 1], or wrong type).
    """
    if landscape.graph is None:
        raise RuntimeError("Graph not initialized. Cannot calculate accessibility.")
    if landscape.go_index is None:
        # Attempt to determine GO if not already done
        landscape._determine_global_optimum()
        if landscape.go_index is None:  # Check again after attempting
            raise RuntimeError(
                "Global optimum could not be determined. Cannot calculate accessibility."
            )

    if landscape.n_configs is None or landscape.n_configs == 0:
        warnings.warn(
            "Landscape has 0 configurations. Accessibility is 0.", RuntimeWarning
        )
        return 0.0

    # --- Exact Calculation ---
    if not approximate:
        if not landscape._path_calculated:
            raise RuntimeError(
                "Exact global optima accessibility requires 'size_basin_accessible' data. "
                "Please run `landscape.determine_accessible_paths()` first, or set `approximate=True`."
            )

        try:
            # Get the pre-calculated size of the basin leading to the GO using igraph API
            vertex = landscape.graph.vs[landscape.go_index]
            size_basin_go = (
                vertex["size_basin_accessible"]
                if "size_basin_accessible" in vertex.attributes()
                else 0
            )

            if (
                "size_basin_accessible" not in landscape.graph.vs.attributes()
                and landscape.verbose
            ):
                # This case indicates an issue if _path_calculated is True
                warnings.warn(
                    "Global optimum node missing 'size_basin_accessible' attribute "
                    "despite _path_calculated=True. Calculation might be inaccurate. Returning 0.",
                    RuntimeWarning,
                )
                return 0.0
            elif size_basin_go == 0 and landscape.verbose:
                warnings.warn(
                    "Calculated 'size_basin_accessible' for GO is 0.", RuntimeWarning
                )

            accessibility = size_basin_go / landscape.n_configs
            return accessibility

        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred accessing exact accessibility: {e}"
            )

    # --- Approximate Calculation ---
    else:
        if n_samples is None:
            raise ValueError(
                "If approximate=True, 'n_samples' must be provided (int > 0 or float (0, 1])."
            )

        num_to_sample: int
        if isinstance(n_samples, int):
            if n_samples <= 0:
                raise ValueError("If 'n_samples' is int, it must be > 0.")
            if n_samples > landscape.n_configs:
                warnings.warn(
                    f"'n_samples' ({n_samples}) > total configurations ({landscape.n_configs}). Sampling all."
                )
                num_to_sample = landscape.n_configs
            else:
                num_to_sample = n_samples
        elif isinstance(n_samples, float):
            if not 0.0 < n_samples <= 1.0:
                raise ValueError("If 'n_samples' is float, it must be in (0, 1].")
            num_to_sample = max(
                1, int(n_samples * landscape.n_configs)
            )  # Ensure at least 1 sample
        else:
            raise ValueError("'n_samples' must be an integer (>0) or a float (0, 1].")

        # Instead of getting all nodes, we can directly sample from range(n_configs)
        # since igraph node indices are always 0...n-1
        if landscape.graph.vcount() == 0:
            warnings.warn("Graph has no nodes to sample from.", RuntimeWarning)
            return 0.0

        # Ensure we don't try to sample more nodes than exist
        actual_sample_size = min(num_to_sample, landscape.graph.vcount())
        if actual_sample_size < num_to_sample and landscape.verbose:
            print(
                f"  Reduced sample size to {actual_sample_size} due to available nodes."
            )

        if actual_sample_size == 0:
            warnings.warn(
                "Calculated sample size is 0. Cannot approximate.", RuntimeWarning
            )
            return 0.0

        # Sample directly from the possible node indices
        sample_nodes = random.sample(
            range(landscape.graph.vcount()), actual_sample_size
        )

        count_reaching_go = 0

        for start_node in sample_nodes:
            try:
                # Use the reachability check function
                if _is_ancestor(landscape.graph, start_node, landscape.go_index):
                    count_reaching_go += 1
            except Exception as e:
                # Handle potential errors during graph traversal for a specific node
                warnings.warn(
                    f"Reachability check failed for sample node {start_node}: {e}",
                    RuntimeWarning,
                )
                continue  # Skip this sample

        # Calculate approximation based on successful checks
        approx_accessibility = count_reaching_go / actual_sample_size

        return approx_accessibility
