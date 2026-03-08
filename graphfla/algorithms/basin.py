import warnings

import numpy as np
from tqdm import tqdm

from .adaptive_walk import hill_climb


def find_plateau_exit(landscape, plateau_id, visited_plateaus):
    """Find a node in a plateau that has an improving edge to an unvisited plateau.

    Returns
    -------
    tuple[int, int] or tuple[None, None]
        (exit_node_in_plateau, successor_outside_plateau), or (None, None)
        if the plateau has no external exit.
    """
    self = landscape

    for node in self._plateaus[plateau_id]:
        for successor in self.graph.successors(node):
            target_pid = int(self._node_to_plateau[successor])
            if target_pid != plateau_id and target_pid not in visited_plateaus:
                return node, successor
    return None, None


def plateau_aware_climb(landscape, start_node, initial_lo, initial_steps):
    """Extend a hill climb across neutral plateaus.

    After a standard hill_climb reaches a node with out-degree 0, this
    method checks whether that node's plateau has an exit to a different,
    higher-fitness region. If so, the climb continues from that exit.

    Parameters
    ----------
    start_node : int
        The original starting node (unused except conceptually).
    initial_lo : int
        The local optimum reached by the initial hill climb.
    initial_steps : int
        Steps taken in the initial hill climb.

    Returns
    -------
    tuple[int, int]
        (final_lo, total_steps)
    """
    self = landscape

    current_lo = initial_lo
    total_steps = initial_steps
    visited_plateaus = set()

    while True:
        pid = int(self._node_to_plateau[current_lo])
        if pid not in self._plateaus:
            break  # Singleton, not part of a multi-member plateau
        if pid in visited_plateaus:
            break
        visited_plateaus.add(pid)

        exit_node, exit_target = find_plateau_exit(landscape, pid, visited_plateaus)
        if exit_node is None:
            break  # Plateau is a true local optimum

        total_steps += 1
        next_lo, next_steps = hill_climb(self.graph, exit_target, "delta_fit")
        total_steps += next_steps
        current_lo = next_lo

    return current_lo, total_steps


def determine_basin_of_attraction(landscape):
    """Calculates the basin of attraction for each node via hill climbing.

    For each node, a greedy hill climb follows improving edges until a
    per-node local optimum is reached. When plateaus are present
    (``_has_plateaus``), the climb is extended across neutral plateaus
    using ``_plateau_aware_climb``, and basin assignments are canonicalized
    so that all members of a plateau-LO share the same representative
    (minimum-index member).
    """
    self = landscape

    if self.graph is None:
        raise RuntimeError("Graph is None.")
    if self.n_configs is None:
        raise RuntimeError("n_configs is None.")

    if self.verbose:
        print(" - Calculating basins of attraction via hill climbing...")

    n_vertices = self.graph.vcount()

    basin_indices = np.full(n_vertices, -1, dtype=np.int32)
    step_counts = np.zeros(n_vertices, dtype=np.int32)

    batch_size = 10000

    is_lo = (
        np.array(self.graph.vs["is_lo"])
        if "is_lo" in self.graph.vs.attributes()
        else None
    )

    batch_ranges = list(range(0, n_vertices, batch_size))
    nodes_iter = (
        tqdm(batch_ranges, desc="   - Hill climbing batches")
        if self.verbose
        else batch_ranges
    )

    for batch_start in nodes_iter:
        batch_end = min(batch_start + batch_size, n_vertices)

        for i in range(batch_start, batch_end):
            if is_lo is not None and is_lo[i]:
                basin_indices[i] = i
                step_counts[i] = 0
                continue

            try:
                lo, steps = hill_climb(self.graph, i, "delta_fit")
                if self._has_plateaus:
                    lo, steps = plateau_aware_climb(landscape, i, lo, steps)
                basin_indices[i] = lo
                step_counts[i] = steps
            except Exception as e:
                if self.verbose:
                    warnings.warn(
                        f"Hill climb failed for node {i}: {e}. "
                        "Assigning node to its own basin.",
                        RuntimeWarning,
                    )
                basin_indices[i] = i
                step_counts[i] = 0

    # Canonicalize: map plateau-LO members to a single representative
    if self._has_plateaus:
        plateau_representative = {}
        for pid, members in self._plateaus.items():
            plateau_representative[pid] = min(members)

        for i in range(n_vertices):
            lo_pid = int(self._node_to_plateau[basin_indices[i]])
            if lo_pid in plateau_representative:
                basin_indices[i] = plateau_representative[lo_pid]

    # Calculate basin sizes using vectorized numpy operations
    unique_basins, basin_counts = np.unique(basin_indices, return_counts=True)
    basin_size_lookup = np.zeros(n_vertices, dtype=np.int32)
    basin_size_lookup[unique_basins] = basin_counts
    size_basin_values = basin_size_lookup[basin_indices]

    # Calculate basin radii (max step count per basin) using numpy
    max_steps_lookup = np.zeros(n_vertices, dtype=np.int32)
    np.maximum.at(max_steps_lookup, basin_indices, step_counts)
    radius_basin_values = max_steps_lookup[basin_indices]

    self.graph.vs["basin_index"] = basin_indices.tolist()
    self.graph.vs["size_basin_greedy"] = size_basin_values.tolist()
    self.graph.vs["radius_basin_greedy"] = radius_basin_values.tolist()

    self._basin_calculated = True

    if self.verbose:
        print(f"   - Basins calculated for {len(unique_basins)} local optima.")
