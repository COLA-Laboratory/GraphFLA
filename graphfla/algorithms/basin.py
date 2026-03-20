import warnings

import numpy as np
from tqdm import tqdm

from .adaptive_walk import hill_climb


PLATEAU_EXIT_MODES = {"first-improvement", "best-improvement"}


def _validate_plateau_exit_mode(mode):
    if mode not in PLATEAU_EXIT_MODES:
        valid_modes = ", ".join(sorted(PLATEAU_EXIT_MODES))
        raise ValueError(
            f"Unsupported plateau exit mode: {mode!r}. "
            f"Expected one of: {valid_modes}."
        )


def find_plateau_exit(
    landscape, plateau_id, visited_plateaus, mode="first-improvement"
):
    """Find an exit that improves beyond the plateau best by more than epsilon.

    Uses ``landscape.epsilon`` (neutrality threshold): only edges to outside
    nodes whose fitness beats the plateau's best by strictly more than
    ``epsilon`` (maximize) or is lower by more than ``epsilon`` (minimize)
    count as a true escape.

    Parameters
    ----------
    mode : {"first-improvement", "best-improvement"}, default="first-improvement"
        Strategy used to select an exit among all qualifying candidates.
        ``first-improvement`` returns the first qualifying exit encountered
        in iteration order. ``best-improvement`` scans all candidates and
        returns the one with the largest fitness gain beyond the plateau best.

    Returns
    -------
    tuple[int, int] or tuple[None, None]
        (exit_node_in_plateau, successor_outside_plateau), or (None, None)
        if the plateau has no qualifying external exit.
    """
    self = landscape
    _validate_plateau_exit_mode(mode)
    eps = float(getattr(self, "epsilon", 0) or 0)
    fitness = np.asarray(self.graph.vs["fitness"])
    members = self.plateaus[plateau_id]
    plateau_best = (
        float(np.max(fitness[members]))
        if self.maximize
        else float(np.min(fitness[members]))
    )

    best_exit = None
    best_improvement = None
    best_tiebreak = None

    for node in members:
        for successor in self.graph.successors(node):
            target_pid = int(self._node_to_plateau[successor])
            if target_pid == plateau_id or target_pid in visited_plateaus:
                continue
            successor_fit = float(fitness[successor])
            if self.maximize:
                if successor_fit > plateau_best + eps:
                    if mode == "first-improvement":
                        return node, successor
                    improvement = successor_fit - plateau_best
                    tiebreak = (successor, node)
                    if (
                        best_improvement is None
                        or improvement > best_improvement
                        or (
                            improvement == best_improvement
                            and tiebreak < best_tiebreak
                        )
                    ):
                        best_improvement = improvement
                        best_tiebreak = tiebreak
                        best_exit = (node, successor)
            elif successor_fit < plateau_best - eps:
                if mode == "first-improvement":
                    return node, successor
                improvement = plateau_best - successor_fit
                tiebreak = (successor, node)
                if (
                    best_improvement is None
                    or improvement > best_improvement
                    or (improvement == best_improvement and tiebreak < best_tiebreak)
                ):
                    best_improvement = improvement
                    best_tiebreak = tiebreak
                    best_exit = (node, successor)

    return best_exit if best_exit is not None else (None, None)


def plateau_aware_climb(
    landscape, initial_lo, initial_steps, plateau_exit_mode="first-improvement"
):
    """Extend a hill climb across neutral plateaus.

    After a standard hill_climb reaches a node with out-degree 0, this
    method checks whether that node's plateau has an exit to a different,
    higher-fitness region. If so, the climb continues from that exit.

    Parameters
    ----------
    initial_lo : int
        The local optimum reached by the initial hill climb.
    initial_steps : int
        Steps taken in the initial hill climb.
    plateau_exit_mode : {"first-improvement", "best-improvement"}, default="first-improvement"
        Exit selection policy used when leaving a neutral plateau.

    Returns
    -------
    tuple[int, int]
        (final_lo, total_steps)
    """
    self = landscape
    _validate_plateau_exit_mode(plateau_exit_mode)

    current_lo = initial_lo
    total_steps = initial_steps
    visited_plateaus = set()

    while True:
        pid = int(self._node_to_plateau[current_lo])
        if pid < 0 or pid not in self.plateaus:
            break  # Singleton, not part of a multi-member plateau
        if pid in visited_plateaus:
            break
        visited_plateaus.add(pid)

        exit_node, exit_target = find_plateau_exit(
            landscape, pid, visited_plateaus, mode=plateau_exit_mode
        )
        if exit_node is None:
            break  # Plateau is a true local optimum

        total_steps += 1
        next_lo, next_steps = hill_climb(self.graph, exit_target, "delta_fit")
        total_steps += next_steps
        current_lo = next_lo

    return current_lo, total_steps


def determine_basin_of_attraction(landscape, plateau_exit_mode="first-improvement"):
    """Calculates the basin of attraction for each node via hill climbing.

    For each node, a greedy hill climb follows improving edges until a
    per-node local optimum is reached. When plateaus are present
    (``_has_plateaus``), the climb is extended across neutral plateaus
    using ``_plateau_aware_climb``. Plateau exits can be selected with
    ``plateau_exit_mode`` as either first- or best-improvement. Basin
    assignments are canonicalized
    so that all members of a plateau-LO share the same representative
    (minimum-index member).
    """
    self = landscape
    _validate_plateau_exit_mode(plateau_exit_mode)

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
                    lo, steps = plateau_aware_climb(
                        landscape,
                        lo,
                        steps,
                        plateau_exit_mode=plateau_exit_mode,
                    )
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
        for pid, members in self.plateaus.items():
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
