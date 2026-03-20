"""Neighbor generation and edge construction for fitness landscapes.

This module provides two concerns:

1. **Neighbor generators** — strategy classes that enumerate adjacent
   configurations in a given encoding space (boolean, sequence, or
   mixed-type).

2. **Edge construction** — routines that build directed improving edges
   (and identify neutral pairs) from a dataset of configurations and
   fitness values, using one of several computational strategies.

The single public entry point for edge construction is :func:`build_edges`.
"""

from dataclasses import dataclass
from math import comb
from typing import Protocol, Tuple, Dict, List, Callable, runtime_checkable
import warnings

import numpy as np
from tqdm import tqdm


def _neutral_abs_threshold(epsilon: float) -> float:
    """Inclusive bound on |Δf| for classifying a neighbor pair as neutral.

    For ``epsilon > 0``, returns the next float above *epsilon* so values that
    differ from *epsilon* only at the last ULP (common after arithmetic on
    fitness arrays) still count as neutral. For ``epsilon == 0``, returns
    ``0.0`` so only exactly-zero |Δf| is neutral.
    """
    eps = float(epsilon)
    if eps <= 0.0:
        return 0.0
    return float(np.nextafter(eps, np.inf))


# ===================================================================
# Neighbor generators
# ===================================================================


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
            The configuration for which to find neighbors.
        config_dict : dict
            Dictionary describing the encoding.
        n_edit : int
            Edit distance for neighborhood definition.

        Returns
        -------
        list[tuple]
            List of neighboring configurations.
        """
        ...


class BooleanNeighborGenerator:
    """Neighbor generator for boolean spaces (single bit flips)."""

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate neighbors by flipping bits."""
        if n_edit != 1:
            raise ValueError(
                f"BooleanNeighborGenerator only supports n_edit=1 "
                f"(single-bit flips). Received n_edit={n_edit}. "
                f"Use neighborhood_strategy='pairwise' or 'broadcast' if you need "
                f"Hamming neighborhoods with n_edit>1."
            )

        return [
            config[:i] + (1 - config[i],) + config[i + 1 :]
            for i in range(len(config))
        ]


class SequenceNeighborGenerator:
    """Neighbor generator for discrete-alphabet sequences (substitutions)."""

    def __init__(self, alphabet_size: int):
        """
        Initialize with the size of the alphabet.

        Parameters
        ----------
        alphabet_size : int
            Number of possible values at each position.
        """
        self.alphabet_size = alphabet_size

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate neighbors by substituting at each position."""
        if n_edit != 1:
            raise ValueError(
                f"SequenceNeighborGenerator only supports n_edit=1 "
                f"(single-position substitutions). Received n_edit={n_edit}. "
                f"Use neighborhood_strategy='pairwise' or 'broadcast' for "
                f"Hamming neighborhoods with n_edit>1."
            )

        neighbors = []
        for i, original in enumerate(config):
            prefix, suffix = config[:i], config[i + 1 :]
            for val in range(self.alphabet_size):
                if val != original:
                    neighbors.append(prefix + (val,) + suffix)
        return neighbors


class DefaultNeighborGenerator:
    """Neighbor generator for mixed data types (boolean / categorical / ordinal)."""

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate neighbors based on data types in config_dict."""
        if n_edit != 1:
            raise ValueError(
                f"DefaultNeighborGenerator only supports n_edit=1. "
                f"Received n_edit={n_edit}. "
                f"Use neighborhood_strategy='pairwise' or 'broadcast' for "
                f"Hamming neighborhoods with n_edit>1."
            )

        neighbors = []
        for i in range(len(config)):
            info = config_dict[i]
            current = config[i]
            dtype = info["type"]

            if dtype == "boolean":
                new_vals = [1 - current]
            elif dtype in ("categorical", "ordinal"):
                new_vals = [v for v in range(info["max"] + 1) if v != current]
            else:
                warnings.warn(
                    f"Unsupported dtype '{dtype}', skipping variable {i}.",
                    RuntimeWarning,
                )
                continue

            for val in new_vals:
                neighbor = list(config)
                neighbor[i] = val
                neighbors.append(tuple(neighbor))
        return neighbors


# ===================================================================
# Edge construction — public API
# ===================================================================


@dataclass(frozen=True)
class EdgeResult:
    """Container for the output of :func:`build_edges`."""

    edges: List[Tuple[int, int]]
    delta_fits: List[float]
    neutral_pairs: List[Tuple[int, int]]
    strategy: str


def build_edges(
    *,
    configs,
    config_dict,
    data,
    n_configs: int,
    n_vars: int,
    n_edit: int,
    strategy: str,
    epsilon: float,
    maximize: bool,
    verbose: bool,
    neighbor_generator: Callable,
    configs_array=None,
) -> EdgeResult:
    """Build improving edges and neutral pairs from a configuration dataset.

    This is the single public entry point for neighborhood construction.
    It validates inputs, resolves the ``'auto'`` strategy, and dispatches
    to the appropriate implementation.

    Parameters
    ----------
    configs : pandas.Series
        Mapping from node index to configuration tuple.
    config_dict : dict
        Encoding metadata keyed by variable index.
    data : pandas.DataFrame
        Must contain a ``'fitness'`` column.
    n_configs, n_vars, n_edit : int
        Dataset dimensions and edit-distance threshold.
    strategy : str
        One of ``'auto'``, ``'active'``, ``'pairwise'``, ``'broadcast'``.
    epsilon : float
        Neutrality threshold.
    maximize : bool
        Optimization direction.
    verbose : bool
        Whether to print progress.
    neighbor_generator : callable
        Bound ``generate`` method of a :class:`NeighborGenerator`.
    configs_array : numpy.ndarray, optional
        Pre-computed numeric configuration matrix.

    Returns
    -------
    EdgeResult
    """
    if configs is None or config_dict is None:
        raise RuntimeError("Cannot build edges: configs/config_dict missing.")
    if n_configs is None:
        raise RuntimeError("n_configs not set before edge construction.")

    valid_strategies = {"auto", "active", "pairwise", "broadcast"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from {sorted(valid_strategies)}."
        )

    resolved = strategy
    if resolved == "auto":
        resolved = _select_strategy(n_configs, n_vars, config_dict, n_edit)
        if verbose:
            print(f" - Auto-selected '{resolved}' neighborhood strategy.")

    kwargs = dict(
        configs=configs,
        config_dict=config_dict,
        data=data,
        n_edit=n_edit,
        epsilon=epsilon,
        maximize=maximize,
        verbose=verbose,
        neighbor_generator=neighbor_generator,
        configs_array=configs_array,
    )

    if resolved == "pairwise":
        edges, delta_fits, neutral_pairs = _build_pairwise(**kwargs)
    elif resolved == "broadcast":
        edges, delta_fits, neutral_pairs = _build_broadcast(**kwargs)
    else:
        edges, delta_fits, neutral_pairs = _build_active(**kwargs)

    return EdgeResult(
        edges=edges,
        delta_fits=delta_fits,
        neutral_pairs=neutral_pairs,
        strategy=resolved,
    )


# ===================================================================
# Strategy selection (private)
# ===================================================================


def _select_strategy(
    n_configs: int, n_vars: int, config_dict: Dict, n_edit: int
) -> str:
    """Choose the fastest strategy for the dataset dimensions.

    1. ``'pairwise'`` if the condensed distance matrix fits in ~4 GiB.
    2. Otherwise compare estimated cost of ``'broadcast'`` (vectorised
       per-row Hamming) vs. ``'active'`` (candidate enumeration + hash
       lookup) and pick the cheaper one.
    """
    n = n_configs
    n_vars = n_vars or 1

    pairwise_bytes = n * (n - 1) // 2 * 8  # float64 condensed form
    if pairwise_bytes <= 4 * 1024 ** 3:
        return "pairwise"

    k_max = (
        max(cd["max"] + 1 for cd in config_dict.values()) if config_dict else 2
    )
    candidates_per_config = sum(
        comb(n_vars, e) * (k_max - 1) ** e for e in range(1, n_edit + 1)
    )
    active_cost = n * candidates_per_config

    vectorisation_factor = 40
    broadcast_cost = n * n / vectorisation_factor

    return "broadcast" if broadcast_cost < active_cost else "active"


# ===================================================================
# Strategy implementations (private)
# ===================================================================


def _build_active(
    *,
    configs,
    config_dict,
    data,
    n_edit,
    epsilon,
    maximize,
    verbose,
    neighbor_generator,
    configs_array,
):
    """Enumerate candidate neighbors and look them up via hash set.

    Three fast-paths are tried in order:

    1. Byte-map lookup for boolean generators (single bit flips).
    2. Byte-map lookup for sequence generators (single substitutions).
    3. Generic tuple-based lookup for arbitrary generators.
    """
    fitness = data["fitness"].to_numpy(copy=False)
    edges, delta_fits, neutral_pairs = [], [], []

    generator_obj = getattr(neighbor_generator, "__self__", None)
    can_use_bytemap = (
        n_edit == 1
        and configs_array is not None
        and configs_array.ndim == 2
        and configs_array.size > 0
        and np.issubdtype(configs_array.dtype, np.integer)
        and int(np.max(configs_array)) <= np.iinfo(np.uint8).max
    )

    if can_use_bytemap and isinstance(generator_obj, BooleanNeighborGenerator):
        _active_boolean_bytemap(
            configs_array, fitness, epsilon, maximize, verbose,
            edges, delta_fits, neutral_pairs,
        )
    elif can_use_bytemap and isinstance(generator_obj, SequenceNeighborGenerator):
        _active_sequence_bytemap(
            configs_array, fitness, epsilon, maximize, verbose,
            generator_obj.alphabet_size, edges, delta_fits, neutral_pairs,
        )
    else:
        _active_generic(
            configs, config_dict, fitness, n_edit, epsilon, maximize, verbose,
            neighbor_generator, edges, delta_fits, neutral_pairs,
        )

    if verbose:
        print(f" - Identified {len(edges)} improving connections.")
        if neutral_pairs:
            print(f" - Identified {len(neutral_pairs)} neutral neighbor pairs.")
    return edges, delta_fits, neutral_pairs


def _build_pairwise(
    *,
    data,
    n_edit,
    configs,
    epsilon,
    verbose,
    maximize,
    configs_array,
    config_dict=None,
    neighbor_generator=None,
):
    """Full pairwise Hamming via ``pdist``.  O(n^2) memory."""
    from scipy.spatial.distance import pdist

    n = len(data)
    fitness = data["fitness"].values
    configs_array = _as_config_matrix(configs, configs_array)
    n_vars = configs_array.shape[1]

    if verbose:
        print(f" - Computing pairwise distances for {n} configurations (pdist)...")

    hamming_fracs = pdist(configs_array, metric="hamming")

    frac_threshold = (n_edit + 0.5) / n_vars
    valid_mask = (hamming_fracs > 0) & (hamming_fracs <= frac_threshold)
    valid_indices = np.where(valid_mask)[0]
    del valid_mask

    if verbose:
        print(
            f" - Found {len(valid_indices)} neighbor pairs "
            f"within edit distance {n_edit}."
        )

    if len(valid_indices) == 0:
        return [], [], []

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

    return _classify_pairs(i_arr, j_arr, fitness, epsilon, maximize, verbose)


def _build_broadcast(
    *,
    data,
    n_edit,
    configs,
    epsilon,
    maximize,
    verbose,
    configs_array,
    config_dict=None,
    neighbor_generator=None,
):
    """Per-row vectorised Hamming, upper triangle only."""
    n = len(data)
    fitness = data["fitness"].values
    configs_array = _as_config_matrix(configs, configs_array)
    n_vars = configs_array.shape[1]
    # Avoid allocating (n-i) x n_vars boolean arrays per row; chunk along j.
    max_chunk_bytes = 128 * 1024**2
    chunk_rows = max(1, max_chunk_bytes // max(n_vars, 1))

    edges, delta_fits, neutral = [], [], []
    neutral_eps = _neutral_abs_threshold(epsilon)

    outer = (
        tqdm(range(n - 1), desc="# Constructing neighborhoods (broadcast)")
        if verbose
        else range(n - 1)
    )

    row_view = configs_array

    for i in outer:
        row = row_view[i]
        j_start = i + 1
        while j_start < n:
            j_end = min(j_start + chunk_rows, n)
            block = row_view[j_start:j_end]
            dists = np.count_nonzero(block != row, axis=1)
            valid_local = np.where((dists > 0) & (dists <= n_edit))[0]

            if len(valid_local) == 0:
                j_start = j_end
                continue

            j_indices = (j_start + valid_local).astype(np.intp)
            fit_diffs = fitness[i] - fitness[j_indices]
            abs_diffs = np.abs(fit_diffs)

            neutral_mask = abs_diffs <= neutral_eps
            if np.any(neutral_mask):
                neutral_j = j_indices[neutral_mask]
                neutral.extend((i, int(j)) for j in neutral_j)

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

                edges.extend(zip(src.tolist(), tgt.tolist()))
                delta_fits.extend(imp_abs.tolist())

            j_start = j_end

    if verbose:
        print(f" - Identified {len(edges)} improving connections.")
        if neutral:
            print(f" - Identified {len(neutral)} neutral neighbor pairs.")
    return edges, delta_fits, neutral


# ===================================================================
# Helpers (private)
# ===================================================================


def _as_config_matrix(configs, configs_array=None):
    """Return a contiguous numeric configuration matrix."""
    if configs_array is not None:
        return np.ascontiguousarray(configs_array)
    config_list = configs.tolist() if hasattr(configs, "tolist") else list(configs)
    return np.ascontiguousarray(np.array(config_list))


def _classify_pairs(i_arr, j_arr, fitness, epsilon, maximize, verbose):
    """Partition (i, j) neighbor pairs into improving edges and neutrals.

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
        ``(edges, delta_fits, neutral_pairs)``.
    """
    fit_i = fitness[i_arr]
    fit_j = fitness[j_arr]
    deltas = fit_i - fit_j
    abs_deltas = np.abs(deltas)

    neutral_eps = _neutral_abs_threshold(epsilon)
    neutral_mask = abs_deltas <= neutral_eps
    neutral_pairs = list(
        zip(i_arr[neutral_mask].tolist(), j_arr[neutral_mask].tolist())
    )

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


# --- Active strategy fast-path helpers --------------------------------


def _active_boolean_bytemap(
    configs_array, fitness, epsilon, maximize, verbose,
    edges, delta_fits, neutral_pairs,
):
    """Byte-map lookup specialised for single bit flips."""
    rows = np.ascontiguousarray(configs_array, dtype=np.uint8)
    index = {row.tobytes(): idx for idx, row in enumerate(rows)}
    get = index.get
    append_edge = edges.append
    append_delta = delta_fits.append
    append_neutral = neutral_pairs.append
    neutral_eps = _neutral_abs_threshold(epsilon)

    it = (
        tqdm(
            range(len(rows)), total=len(rows),
            desc="# Constructing neighborhoods (active)",
        )
        if verbose
        else range(len(rows))
    )

    for cid in it:
        current_fit = fitness[cid]
        config_row = rows[cid]
        key = bytearray(config_row)

        for i, val in enumerate(config_row):
            val = int(val)
            key[i] = 1 - val
            nid = get(bytes(key))
            key[i] = val
            if nid is None:
                continue

            delta = current_fit - fitness[nid]
            abs_delta = abs(delta)

            if abs_delta <= neutral_eps:
                if cid < nid:
                    append_neutral((cid, nid))
            else:
                if (maximize and delta < 0) or (not maximize and delta > 0):
                    append_edge((cid, nid))
                    append_delta(abs_delta)


def _active_sequence_bytemap(
    configs_array, fitness, epsilon, maximize, verbose, alphabet_size,
    edges, delta_fits, neutral_pairs,
):
    """Byte-map lookup specialised for single-position substitutions."""
    rows = np.ascontiguousarray(configs_array, dtype=np.uint8)
    index = {row.tobytes(): idx for idx, row in enumerate(rows)}
    get = index.get
    append_edge = edges.append
    append_delta = delta_fits.append
    append_neutral = neutral_pairs.append
    neutral_eps = _neutral_abs_threshold(epsilon)

    it = (
        tqdm(
            range(len(rows)), total=len(rows),
            desc="# Constructing neighborhoods (active)",
        )
        if verbose
        else range(len(rows))
    )

    for cid in it:
        current_fit = fitness[cid]
        config_row = rows[cid]
        key = bytearray(config_row)

        for i, orig in enumerate(config_row):
            orig = int(orig)
            for val in range(alphabet_size):
                if val == orig:
                    continue

                key[i] = val
                nid = get(bytes(key))
                if nid is None:
                    continue

                delta = current_fit - fitness[nid]
                abs_delta = abs(delta)

                if abs_delta <= neutral_eps:
                    if cid < nid:
                        append_neutral((cid, nid))
                else:
                    if (maximize and delta < 0) or (not maximize and delta > 0):
                        append_edge((cid, nid))
                        append_delta(abs_delta)
            key[i] = orig


def _active_generic(
    configs, config_dict, fitness, n_edit, epsilon, maximize, verbose,
    neighbor_generator, edges, delta_fits, neutral_pairs,
):
    """Generic tuple-based lookup for arbitrary neighbor generators."""
    if hasattr(configs, "to_numpy"):
        config_values = configs.to_numpy(copy=False)
    else:
        config_values = configs

    index = {cfg: idx for idx, cfg in enumerate(config_values)}
    get = index.get
    append_edge = edges.append
    append_delta = delta_fits.append
    append_neutral = neutral_pairs.append
    neutral_eps = _neutral_abs_threshold(epsilon)

    it = (
        tqdm(
            config_values, total=len(config_values),
            desc="# Constructing neighborhoods (active)",
        )
        if verbose
        else config_values
    )

    for cid, cfg in enumerate(it):
        current_fit = fitness[cid]

        for neighbor in neighbor_generator(cfg, config_dict, n_edit):
            nid = get(neighbor)
            if nid is None:
                continue

            delta = current_fit - fitness[nid]
            abs_delta = abs(delta)

            if abs_delta <= neutral_eps:
                if cid < nid:
                    append_neutral((cid, nid))
            else:
                if (maximize and delta < 0) or (not maximize and delta > 0):
                    append_edge((cid, nid))
                    append_delta(abs_delta)
