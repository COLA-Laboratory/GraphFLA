"""Build-internal landscape computations.

Mutating helpers that populate derived node/edge attributes on a built
landscape: the global optimum, accessible-path basin sizes, distance to the
nearest global optimum, and neighbour fitness. They run during construction or
lazily on first property access. Keeping them here (rather than in
``graphfla.analysis``) makes the analysis package a set of pure readers that
never mutate the landscape, and removes the landscape -> analysis import.

Each takes the landscape instance as its first argument.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from .landscape import Landscape


def determine_global_optimum(landscape):
    """Identifies the global optimum node in the landscape graph using igraph."""
    if landscape.graph is None:
        raise RuntimeError("Graph is None.")  # Internal check
    if landscape.verbose:
        print(" - Determining global optimum...")

    if "fitness" not in landscape.graph.vs.attributes():
        warnings.warn(
            "Cannot determine global optimum: 'fitness' attribute missing from graph nodes.",
            RuntimeWarning,
        )
        landscape.go_index = None
        landscape.go = None
        return

    # asarray is cheaper than array here and gives identical arg-extrema.
    fitness_values = np.asarray(landscape.graph.vs["fitness"], dtype=np.float64)

    if landscape.maximize:
        landscape.go_index = int(np.argmax(fitness_values))
    else:
        landscape.go_index = int(np.argmin(fitness_values))

    try:
        landscape.go = landscape.graph.vs[landscape.go_index].attributes()
        if landscape.verbose:
            print(
                f"   - Global optimum found at index {landscape.go_index} with fitness {landscape.go['fitness']:.4f}."
            )
    except IndexError:
        warnings.warn(
            f"Global optimum index {landscape.go_index} not found in graph nodes. Resetting GO.",
            RuntimeWarning,
        )
        landscape.go_index = None
        landscape.go = None


def determine_accessible_paths(landscape):
    """Determines the size of basins based on accessible paths (ancestors).

    Identifies all nodes from which a local optimum can be reached via any
    fitness-increasing path. When plateaus are active, ancestors are computed
    for each plateau-level LO by taking the union of ancestors across all
    plateau members,         assigned to the canonical representative (min index).

    Notes
    -----
    This method is computationally intensive and might be slow for large
    landscapes.
    """
    if landscape.graph is None:
        raise RuntimeError("Graph is None.")

    if landscape.lo_index is None or not landscape.lo_index or landscape.n_lo is None:
        warnings.warn(
            "No local optima found or determined. Cannot calculate accessible paths.",
            RuntimeWarning,
        )
        landscape._path_calculated = False
        return

    if landscape.verbose:
        print(" - Determining accessible paths (ancestors)...")

    dict_size = defaultdict(int)

    try:
        if landscape._has_plateaus and landscape._peak_index:
            peak_iter = (
                tqdm(landscape._peak_index, desc="   - Finding ancestors")
                if landscape.verbose
                else landscape._peak_index
            )

            for rep in peak_iter:
                pid = int(landscape._node_to_plateau[rep])
                if pid >= 0 and pid in landscape.plateaus:
                    members = landscape.plateaus[pid]
                    ancestors = set()
                    for member in members:
                        ancestors.update(
                            landscape.graph.subcomponent(member, mode="in")
                        )
                    dict_size[rep] = len(ancestors)
                else:
                    ancestors_set = landscape.graph.subcomponent(rep, mode="in")
                    dict_size[rep] = len(ancestors_set)
        else:
            los_iter = (
                tqdm(landscape.lo_index, total=landscape.n_lo, desc="   - Finding ancestors")
                if landscape.verbose
                else landscape.lo_index
            )
            for lo in los_iter:
                ancestors_set = landscape.graph.subcomponent(lo, mode="in")
                dict_size[lo] = len(ancestors_set)

        size_basin_accessible_values = [0] * landscape.graph.vcount()
        for vertex_id, basin_size in dict_size.items():
            size_basin_accessible_values[vertex_id] = basin_size

        landscape.graph.vs["size_basin_accessible"] = size_basin_accessible_values

        landscape._path_calculated = True
        if landscape.verbose:
            print(
                f"   - Accessible paths calculated for {len(dict_size)} local optima."
            )
    except Exception as e:
        landscape._path_calculated = False
        warnings.warn(
            f"Error during accessible path calculation: {e}. "
            "'size_basin_accessible' attribute may be incomplete.",
            RuntimeWarning,
        )


def determine_dist_to_go(landscape, distance=None):
    """Calculates the distance from each node to the global optimum."""

    if (
        landscape.graph is None
        or landscape.configs is None
        or landscape.go_index is None
        or landscape.data_types is None
    ):
        if landscape.verbose:
            print("Skipping distance calculation - missing prerequisites")
        return

    if distance is None:
        distance = landscape._get_default_distance_metric()

    if landscape.verbose:
        print(
            f" - Calculating distances to global optimum using {distance.__name__}..."
        )

    try:
        n_vertices = landscape.graph.vcount()
        if len(landscape.configs) != n_vertices:
            warnings.warn(
                f"configs length ({len(landscape.configs)}) does not match graph "
                f"vertex count ({n_vertices}). Skipping distance calculation.",
                RuntimeWarning,
            )
            landscape._distance_calculated = False
            return

        if landscape._configs_array is not None and len(landscape._configs_array) == n_vertices:
            configs = landscape._configs_array
        else:
            configs = np.array(landscape.configs.tolist())

        # Distance to the NEAREST global optimum when several configs tie for
        # best fitness, per Jones & Forrest (1995); FDC uses the closest one.
        fitness = np.asarray(landscape.graph.vs["fitness"], dtype=float)
        best = float(fitness.max()) if landscape.maximize else float(fitness.min())
        go_indices = np.where(fitness == best)[0]
        if go_indices.size <= 1:
            distances = distance(configs, configs[landscape.go_index], landscape.data_types)
        else:
            distances = np.min(
                np.stack(
                    [distance(configs, configs[gi], landscape.data_types) for gi in go_indices],
                    axis=0,
                ),
                axis=0,
            )

        landscape.graph.vs["dist_go"] = distances.tolist()
        landscape._distance_calculated = True

        if landscape.verbose:
            print(
                "   - Distances to GO calculated and added as node attribute 'dist_go'."
            )

    except Exception as e:
        warnings.warn(f"Error calculating distances to GO: {e}", RuntimeWarning)
        landscape._distance_calculated = False


def determine_neighbor_fitness(landscape) -> "Landscape":
    """Calculates the mean fitness of neighbors for each node and the difference
    in mean neighbor fitness between connected nodes.

    This method adds two new attributes to the landscape graph:
    1. 'mean_neighbor_fit': A vertex attribute representing the mean fitness of all
    neighboring nodes.
    2. 'delta_mean_neighbor_fit': An edge attribute representing the difference in
    mean neighbor fitness along the improving (lower-fitness → higher-fitness)
    direction, i.e. mean_neighbor_fit(target) - mean_neighbor_fit(source).

    This can be useful for identifying evolvability-enhancing (EE) mutations as
    introduced in Wagner (2023).

    References
    ----------
    .. [Wagner 2023] Wagner, A. The role of evolvability in the evolution of
       complex traits. Nat Rev Genet 24, 1-16 (2023).
       https://doi.org/10.1038/s41576-023-00559-0

    Returns
    -------
    Landscape
        The landscape instance with the new attributes added, for
        method chaining.

    Raises
    ------
    RuntimeError
        If the landscape has not been built yet.
    """
    if landscape.graph is None:
        raise RuntimeError("Graph is None despite landscape being built.")

    if landscape.verbose:
        print("Calculating neighbor fitness metrics...")

    n_vertices = landscape.graph.vcount()

    # Pre-fetch fitness into a numpy array for fast neighbour lookups
    fitness_array = np.array(landscape.graph.vs["fitness"])

    # Step 1: mean neighbour fitness per node
    mean_neighbor_fit = np.full(n_vertices, np.nan)

    adj_list = landscape.graph.get_adjlist(mode="all")

    for vertex_idx in range(n_vertices):
        neighbors = adj_list[vertex_idx]
        if landscape._neutral_neighbors and vertex_idx in landscape._neutral_neighbors:
            neighbors = list(
                set(neighbors) | set(landscape._neutral_neighbors[vertex_idx])
            )

        if neighbors:
            mean_neighbor_fit[vertex_idx] = np.mean(fitness_array[neighbors])

    landscape.graph.vs["mean_neighbor_fit"] = mean_neighbor_fit.tolist()

    if landscape.verbose:
        print(f" - Added 'mean_neighbor_fit' attribute for {n_vertices} nodes")

    # Step 2: delta mean neighbour fitness along each edge
    n_edges = landscape.graph.ecount()

    edge_list = landscape.graph.get_edgelist()
    if edge_list:
        edge_array = np.array(edge_list)
        delta_mean_neighbor_fit = (
            mean_neighbor_fit[edge_array[:, 1]] - mean_neighbor_fit[edge_array[:, 0]]
        )
    else:
        delta_mean_neighbor_fit = np.zeros(n_edges)

    landscape.graph.es["delta_mean_neighbor_fit"] = delta_mean_neighbor_fit.tolist()

    if landscape.verbose:
        print(f" - Added 'delta_mean_neighbor_fit' attribute for {n_edges} edges")

    landscape._neighbor_fit_calculated = True
    return landscape
