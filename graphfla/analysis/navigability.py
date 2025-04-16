import warnings


def global_optima_accessibility(landscape) -> float:
    """
    Calculate the accessibility of the global optimum (GO).

    This metric represents the fraction of configurations in the landscape
    that can reach the global optimum via any monotonic, fitness-improving path.

    The implementation uses graph traversal to find all nodes (configurations)
    that have a directed path to the global optimum in the landscape graph.
    These are the "ancestors" of the global optimum - configurations from which
    the GO can be reached by following fitness-improving moves.

    Returns
    -------
    float
        The fraction of configurations able to reach the global optimum
        monotonically (value between 0.0 and 1.0).

    Raises
    ------
    RuntimeError
        If the global optimum has not been determined.
        If the graph is not initialized.
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

    # Find all ancestors of the global optimum (nodes that can reach the GO)
    # using igraph's subcomponent function with "in" mode
    try:
        ancestors_set = landscape.graph.subcomponent(landscape.go_index, mode="in")
        # Calculate accessibility as the fraction of nodes that can reach the GO
        accessibility = len(ancestors_set) / landscape.n_configs
        return accessibility
    except Exception as e:
        raise RuntimeError(f"An error occurred during accessibility calculation: {e}")
