from math import comb
from typing import Protocol, Tuple, Dict, List, Any, Callable, runtime_checkable
import warnings

import numpy as np
from tqdm import tqdm


@runtime_checkable
class NeighborGenerator(Protocol):
    """Protocol defining the interface for neighbor generation."""

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """
        Generate neighbors for a given configuration.

        Parameters
        ----------
        config : tuple
            The configuration for which to find neighbors
        config_dict : dict
            Dictionary describing the encoding
        n_edit : int
            Edit distance for neighborhood definition

        Returns
        -------
        list[tuple]
            List of neighboring configurations
        """
        ...


class BooleanNeighborGenerator:
    """Generator for boolean neighbors (bit flips)."""

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate neighbors by flipping bits."""
        if n_edit != 1:
            warnings.warn(
                f"BooleanNeighborGenerator only supports n_edit=1 for single bit flips. "
                f"Received n_edit={n_edit}. Returning no neighbors.",
                UserWarning,
            )
            return []

        neighbors = []
        current_config_list = list(config)
        num_bits = len(current_config_list)

        for i in range(num_bits):
            neighbor_list = current_config_list.copy()
            neighbor_list[i] = 1 - neighbor_list[i]  # Flip bit
            neighbors.append(tuple(neighbor_list))

        return neighbors


class SequenceNeighborGenerator:
    """Generator for sequence neighbors (substitutions)."""

    def __init__(self, alphabet_size: int):
        """
        Initialize with the size of the alphabet.

        Parameters
        ----------
        alphabet_size : int
            Number of possible values at each position
        """
        self.alphabet_size = alphabet_size

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate neighbors by substituting at each position."""
        if n_edit != 1:
            warnings.warn(
                f"SequenceNeighborGenerator only supports n_edit=1 for single position substitutions. "
                f"Received n_edit={n_edit}. Returning no neighbors.",
                UserWarning,
            )
            return []

        neighbors = []
        current_config_list = list(config)
        num_positions = len(current_config_list)

        for i in range(num_positions):
            original_val = current_config_list[i]
            # Try each possible substitution at this position
            for new_val in range(self.alphabet_size):
                if new_val != original_val:
                    neighbor_list = current_config_list.copy()
                    neighbor_list[i] = new_val
                    neighbors.append(tuple(neighbor_list))

        return neighbors


class DefaultNeighborGenerator:
    """Default generator for mixed data types."""

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate neighbors based on data types in config_dict."""
        if n_edit != 1:
            warnings.warn(
                f"DefaultNeighborGenerator only fully supports n_edit=1. "
                f"Received n_edit={n_edit}.",
                UserWarning,
            )

        neighbors = []
        num_vars = len(config)

        for i in range(num_vars):
            info = config_dict[i]
            current_val = config[i]
            dtype = info["type"]

            if dtype == "boolean":
                # Flip the bit (0 to 1, 1 to 0)
                new_vals = [1 - current_val]
            elif dtype in ["categorical", "ordinal"]:
                # Iterate through all possible values
                max_val = info["max"]
                new_vals = [v for v in range(max_val + 1) if v != current_val]
            else:
                warnings.warn(
                    f"Unsupported dtype '{dtype}' in generate_neighbors, skipping var {i}",
                    RuntimeWarning,
                )
                continue

            # Create neighbor tuples
            for new_val in new_vals:
                neighbor_list = list(config)
                neighbor_list[i] = new_val
                neighbors.append(tuple(neighbor_list))

        return neighbors


# ---------------------------------------------------------------------------
# Neighborhood construction strategies (moved from landscape.Landscape).
# Logic is unchanged; parameters replace self.xxx for use from Landscape.
# ---------------------------------------------------------------------------


def select_neighborhood_strategy(
    n_configs: int, n_vars: int, config_dict: Dict, n_edit: int
) -> str:
    """Choose the fastest neighborhood strategy for the current dataset.

    Decision logic:

    1. If the condensed pairwise distance matrix (from ``pdist``) fits
       within ~4 GiB, use ``'pairwise'`` — it is the fastest option for
       small-to-moderate datasets thanks to scipy's optimized C code.
    2. Otherwise, estimate the per-config candidate count for the
       ``'active'`` strategy (``sum_{e=1}^{n_edit} C(n_vars, e) *
       (k-1)^e``, where *k* is the maximum alphabet size) and compare
       against the cost of a vectorised broadcast
       (``n_configs / vectorisation_factor``).  If broadcast is cheaper,
       use ``'broadcast'``; otherwise fall back to ``'active'``.

    Returns
    -------
    str
        ``'pairwise'``, ``'broadcast'``, or ``'active'``.
    """
    n = n_configs
    n_vars = n_vars or 1

    # --- Can we afford the full pairwise matrix? ---
    pairwise_bytes = n * (n - 1) // 2 * 8  # float64 condensed form
    max_pairwise_bytes = 4 * 1024 ** 3      # 4 GiB
    if pairwise_bytes <= max_pairwise_bytes:
        return "pairwise"

    # --- Active vs. broadcast heuristic ---
    if config_dict:
        k_max = max(cd["max"] + 1 for cd in config_dict.values())
    else:
        k_max = 2

    candidates_per_config = sum(
        comb(n_vars, e) * (k_max - 1) ** e
        for e in range(1, n_edit + 1)
    )
    active_cost = n * candidates_per_config

    # Empirical factor: NumPy vectorised element-wise ops are roughly
    # 30-50x faster than per-element Python/hash-table work.
    vectorisation_factor = 40
    broadcast_cost = n * n / vectorisation_factor

    if broadcast_cost < active_cost:
        return "broadcast"

    return "active"


def construct_neighborhoods_active(
    configs,
    config_dict,
    data,
    n_edit,
    epsilon,
    maximize,
    verbose,
    neighbor_generator: Callable,
):
    """Enumerate candidate mutant neighbors and check dataset membership.

    For each configuration, all possible single-edit (or ``n_edit``-edit)
    neighbors are generated and looked up in a hash set built from the
    dataset. Efficient when the dataset is dense relative to the
    mutational neighborhood (i.e. most proposed neighbors exist).
    """
    config_to_index: dict = dict(zip(configs, data.index))
    config_to_fitness: dict = dict(zip(configs, data["fitness"]))

    edges, delta_fits, neutral_pairs = [], [], []

    configs_iter = (
        tqdm(
            configs, total=len(configs),
            desc="# Constructing neighborhoods (active)",
        )
        if verbose
        else configs
    )

    for config_tuple in configs_iter:
        current_fit = config_to_fitness[config_tuple]
        current_id = config_to_index[config_tuple]

        neighbors = neighbor_generator(config_tuple, config_dict, n_edit)

        for neighbor_tuple in neighbors:
            neighbor_idx = config_to_index.get(neighbor_tuple)
            if neighbor_idx is not None:
                neighbor_fit = config_to_fitness[neighbor_tuple]
                delta_fit = current_fit - neighbor_fit
                abs_delta = abs(delta_fit)

                if abs_delta <= epsilon:
                    if current_id < neighbor_idx:
                        neutral_pairs.append((current_id, neighbor_idx))
                else:
                    is_improvement = (maximize and delta_fit < 0) or (
                        not maximize and delta_fit > 0
                    )
                    if is_improvement:
                        edges.append((current_id, neighbor_idx))
                        delta_fits.append(abs_delta)

    if verbose:
        print(f" - Identified {len(edges)} improving connections.")
        if neutral_pairs:
            print(f" - Identified {len(neutral_pairs)} neutral neighbor pairs.")
    return edges, delta_fits, neutral_pairs


def construct_neighborhoods_pairwise(
    data, n_edit, configs, epsilon, verbose, maximize
):
    """Compute the full pairwise Hamming distance matrix with ``pdist``.

    All configuration pairs whose distance is in ``(0, n_edit]`` are
    returned as edges or neutral pairs. Very fast for small-to-moderate
    datasets (~25 000 configurations) thanks to scipy's optimized C
    implementation, but memory scales as ``O(n_configs^2)``.
    """
    from scipy.spatial.distance import pdist

    n = len(data)
    fitness = data["fitness"].values
    configs_array = np.array(configs.tolist())
    n_vars = configs_array.shape[1]

    if verbose:
        print(f" - Computing pairwise distances for {n} configurations (pdist)...")

    hamming_fracs = pdist(configs_array, metric="hamming")

    # pdist('hamming') returns fraction of mismatches; compare directly
    # against the fractional threshold to avoid allocating a second array.
    frac_threshold = (n_edit + 0.5) / n_vars
    valid_mask = (hamming_fracs > 0) & (hamming_fracs <= frac_threshold)
    valid_indices = np.where(valid_mask)[0]
    del valid_mask  # free memory early

    if verbose:
        print(
            f" - Found {len(valid_indices)} neighbor pairs "
            f"within edit distance {n_edit}."
        )

    if len(valid_indices) == 0:
        return [], [], []

    # Convert condensed-matrix indices to (i, j) pairs — vectorised
    n_f = float(n)
    k = valid_indices.astype(np.float64)
    i_arr = (
        n_f - 2 - np.floor(
            np.sqrt(-8.0 * k + 4.0 * n_f * (n_f - 1) - 7.0) / 2.0 - 0.5
        )
    ).astype(np.intp)
    j_arr = (
        k + i_arr + 1
        - n_f * (n_f - 1) / 2
        + (n_f - i_arr) * ((n_f - i_arr) - 1) / 2
    ).astype(np.intp)

    return edges_from_pairs(i_arr, j_arr, fitness, epsilon, maximize, verbose)


def construct_neighborhoods_broadcast(
    data, n_edit, configs, epsilon, maximize, verbose
):
    """For each config, compute Hamming distance to all others via NumPy.

    Only the upper triangle (j > i) is evaluated, so each pair is
    processed exactly once. Suitable for large datasets with long
    sequences and sparse sampling where ``'active'`` would waste time
    generating candidates and ``'pairwise'`` would exceed memory.
    """
    n = len(data)
    fitness = data["fitness"].values
    configs_array = np.array(configs.tolist())

    all_edges, all_delta_fits, all_neutral = [], [], []

    outer_iter = (
        tqdm(range(n - 1), desc="# Constructing neighborhoods (broadcast)")
        if verbose
        else range(n - 1)
    )

    for i in outer_iter:
        remaining = configs_array[i + 1:]
        dists = np.count_nonzero(remaining != configs_array[i], axis=1)
        valid_local = np.where((dists > 0) & (dists <= n_edit))[0]

        if len(valid_local) == 0:
            continue

        j_indices = (i + 1 + valid_local).astype(np.intp)
        fit_diffs = fitness[i] - fitness[j_indices]
        abs_diffs = np.abs(fit_diffs)

        # --- neutral pairs ---
        neutral_mask = abs_diffs <= epsilon
        if np.any(neutral_mask):
            neutral_j = j_indices[neutral_mask]
            all_neutral.extend((i, int(j)) for j in neutral_j)

        # --- improving edges ---
        improving_mask = ~neutral_mask
        if np.any(improving_mask):
            imp_j = j_indices[improving_mask]
            imp_diffs = fit_diffs[improving_mask]
            imp_abs = abs_diffs[improving_mask]

            if maximize:
                src = np.where(imp_diffs < 0, i, imp_j)
                tgt = np.where(imp_diffs < 0, imp_j, i)
            else:
                src = np.where(imp_diffs > 0, i, imp_j)
                tgt = np.where(imp_diffs > 0, imp_j, i)

            all_edges.extend(zip(src.tolist(), tgt.tolist()))
            all_delta_fits.extend(imp_abs.tolist())

    if verbose:
        print(f" - Identified {len(all_edges)} improving connections.")
        if all_neutral:
            print(
                f" - Identified {len(all_neutral)} neutral neighbor pairs."
            )
    return all_edges, all_delta_fits, all_neutral


def edges_from_pairs(i_arr, j_arr, fitness, epsilon, maximize, verbose):
    """Classify an array of (i, j) neighbor pairs into edges and neutrals.

    Parameters
    ----------
    i_arr, j_arr : np.ndarray[int]
        Matched arrays of pair indices (i < j for each element).
    fitness : np.ndarray[float]
        Fitness values indexed by node id.
    epsilon : float
        Neutrality threshold.
    maximize : bool
        Optimization direction.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    tuple[list, list, list]
        ``(edges, delta_fits, neutral_pairs)`` with the same semantics as
        ``_construct_neighborhoods``.
    """
    fit_i = fitness[i_arr]
    fit_j = fitness[j_arr]
    deltas = fit_i - fit_j
    abs_deltas = np.abs(deltas)

    # --- neutral pairs ---
    neutral_mask = abs_deltas <= epsilon
    neutral_pairs = list(
        zip(i_arr[neutral_mask].tolist(), j_arr[neutral_mask].tolist())
    )

    # --- improving edges ---
    imp_mask = ~neutral_mask
    if not np.any(imp_mask):
        if verbose:
            print(f" - Identified 0 improving connections.")
            if neutral_pairs:
                print(
                    f" - Identified {len(neutral_pairs)} neutral neighbor pairs."
                )
        return [], [], neutral_pairs

    imp_i = i_arr[imp_mask]
    imp_j = j_arr[imp_mask]
    imp_deltas = deltas[imp_mask]
    imp_abs = abs_deltas[imp_mask]

    if maximize:
        src = np.where(imp_deltas < 0, imp_i, imp_j)
        tgt = np.where(imp_deltas < 0, imp_j, imp_i)
    else:
        src = np.where(imp_deltas > 0, imp_i, imp_j)
        tgt = np.where(imp_deltas > 0, imp_j, imp_i)

    edges = list(zip(src.tolist(), tgt.tolist()))
    delta_fits = imp_abs.tolist()

    if verbose:
        print(f" - Identified {len(edges)} improving connections.")
        if neutral_pairs:
            print(
                f" - Identified {len(neutral_pairs)} neutral neighbor pairs."
            )
    return edges, delta_fits, neutral_pairs
