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


class OrdinalNeighborGenerator:
    """Neighbor generator for ordinal spaces (±1 step on the ordinal scale).

    For an ordinal variable with allowed encoded values ``0..max``, each
    configuration has at most two neighbors per position: one ``+1`` step
    (if ``current < max``) and one ``-1`` step (if ``current > 0``).
    This corresponds to a Manhattan-distance-1 neighborhood on each axis,
    which is the standard definition of an ordinal-landscape neighborhood
    in the literature.
    """

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate ±1-step neighbors at each ordinal position."""
        if n_edit != 1:
            raise ValueError(
                f"OrdinalNeighborGenerator only supports n_edit=1 "
                f"(single ±1 step on the ordinal scale). Received n_edit={n_edit}. "
                f"Use neighborhood_strategy='pairwise' or 'broadcast' for "
                f"larger Hamming-style neighborhoods (note: those use Hamming, "
                f"not Manhattan, distance)."
            )

        neighbors = []
        for i in range(len(config)):
            info = config_dict[i]
            current = int(config[i])
            max_val = int(info["max"])
            for delta in (-1, 1):
                new_val = current + delta
                if 0 <= new_val <= max_val:
                    neighbor = list(config)
                    neighbor[i] = new_val
                    neighbors.append(tuple(neighbor))
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
            elif dtype == "categorical":
                new_vals = [v for v in range(info["max"] + 1) if v != current]
            elif dtype == "ordinal":
                # ±1 step on the ordinal scale, as in OrdinalNeighborGenerator.
                cur = int(current)
                max_val = int(info["max"])
                new_vals = [v for v in (cur - 1, cur + 1) if 0 <= v <= max_val]
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

    Four fast-paths are tried in order:

    1. Byte-map lookup for boolean generators (single bit flips).
    2. Byte-map lookup for sequence generators (single substitutions).
    3. Mixed-radix vectorised lookup for ordinal generators (±1 steps).
    4. Generic tuple-based lookup for arbitrary generators.
    """
    fitness = data["fitness"].to_numpy(copy=False)
    edges, delta_fits, neutral_pairs = [], [], []

    generator_obj = getattr(neighbor_generator, "__self__", None)
    numeric_configs_array = (
        n_edit == 1
        and configs_array is not None
        and configs_array.ndim == 2
        and configs_array.size > 0
        and np.issubdtype(configs_array.dtype, np.integer)
    )
    can_use_bytemap = (
        numeric_configs_array
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
    elif (
        numeric_configs_array
        and isinstance(generator_obj, OrdinalNeighborGenerator)
        and _active_ordinal_vectorized(
            configs_array, config_dict, fitness, epsilon, maximize, verbose,
            edges, delta_fits, neutral_pairs,
        )
    ):
        pass  # handled by the vectorised ordinal fast-path
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


def _mixed_radix_keys(configs_array, base):
    """Encode each config row as a unique mixed-radix integer key.

    For a row ``c`` of per-position codes in ``0..base-1``, the key is
    ``sum_j c[j] * base**j`` (little-endian, position 0 is the least
    significant digit). Because every code is strictly below *base*, the
    mapping is a bijection over the represented configurations, so keys are
    unique whenever the rows are unique.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        ``(keys, place_values)`` as ``int64`` arrays, or ``(None, None)`` if
        the largest representable key would overflow signed 64-bit range
        (``base**n_vars > 2**63``), signalling the caller to fall back to the
        exact Python implementation.
    """
    n_vars = configs_array.shape[1]
    # Overflow guard: the maximum key is base**n_vars - 1. Refuse if even the
    # place value of the most significant digit would not fit in int64.
    if n_vars >= 64 or base ** n_vars > (1 << 63):
        return None, None

    place_values = np.array(
        [base ** j for j in range(n_vars)], dtype=np.int64
    )
    keys = configs_array.astype(np.int64) @ place_values
    return keys, place_values


def _active_bytemap_vectorized(
    configs_array, fitness, neutral_eps, maximize, base,
    edges, delta_fits, neutral_pairs,
):
    """Vectorised single-substitution neighbour finder via mixed-radix keys.

    Enumerates exactly the same ordered ``(source_row -> neighbour_row)`` visits
    that the per-cell Python loop would (every row, every position ``p``, every
    alternative value ``v != current_code[p]``) but resolves all rows for a
    given ``(p, v)`` in one ``searchsorted`` lookup. The classification of each
    visit is identical to the baseline:

    * ``delta = fitness[source] - fitness[neighbour]``;
    * ``abs(delta) <= neutral_eps`` -> neutral pair, recorded as ``(src, nbr)``
      only when ``src < nbr`` (so each undirected neutral pair is kept once);
    * otherwise, if the source is the worse endpoint
      (``(maximize and delta < 0) or (not maximize and delta > 0)``), a directed
      edge ``src -> nbr`` is emitted with ``delta_fit = abs(delta)``.

    Returns
    -------
    bool
        ``True`` if vectorisation ran (results appended to the output lists);
        ``False`` if the mixed-radix key would overflow and the caller must use
        the exact Python fallback instead.
    """
    keys, place_values = _mixed_radix_keys(configs_array, base)
    if keys is None:
        return False

    n, n_vars = configs_array.shape
    row_idx = np.arange(n)

    order = np.argsort(keys, kind="stable")
    skeys = keys[order]

    src_list = []        # source row indices, in visit order
    nbr_list = []        # matched neighbour row indices

    for p in range(n_vars):
        place = int(place_values[p])
        codes_p = configs_array[:, p].astype(np.int64)
        for v in range(base):
            # Source rows are exactly those whose code at position p differs
            # from v (the loop's "val != orig" / bit-flip condition).
            src_mask = codes_p != v
            if not src_mask.any():
                continue
            src_rows = row_idx[src_mask]
            nbr_keys = keys[src_rows] + (v - codes_p[src_mask]) * place

            pos = np.searchsorted(skeys, nbr_keys)
            # Verify exact key match (searchsorted gives an insertion point;
            # only positions in-range whose key equals the query are real hits).
            valid = pos < n
            if not valid.any():
                continue
            pos_valid = pos[valid]
            exact = skeys[pos_valid] == nbr_keys[valid]
            if not exact.any():
                continue

            hit_src = src_rows[valid][exact]
            hit_nbr = order[pos_valid[exact]]
            src_list.append(hit_src)
            nbr_list.append(hit_nbr)

    if not src_list:
        return True

    src = np.concatenate(src_list)
    nbr = np.concatenate(nbr_list)

    delta = fitness[src] - fitness[nbr]
    abs_delta = np.abs(delta)

    neutral_mask = abs_delta <= neutral_eps
    if neutral_mask.any():
        nsrc = src[neutral_mask]
        nnbr = nbr[neutral_mask]
        keep = nsrc < nnbr  # dedup: record each undirected neutral pair once
        if keep.any():
            neutral_pairs.extend(
                zip(nsrc[keep].tolist(), nnbr[keep].tolist())
            )

    imp_mask = ~neutral_mask
    if imp_mask.any():
        idelta = delta[imp_mask]
        # Worse-endpoint emission, matching the baseline direction guard.
        if maximize:
            edge_mask = idelta < 0
        else:
            edge_mask = idelta > 0
        if edge_mask.any():
            esrc = src[imp_mask][edge_mask]
            enbr = nbr[imp_mask][edge_mask]
            edge_abs = abs_delta[imp_mask][edge_mask]
            edges.extend(zip(esrc.tolist(), enbr.tolist()))
            delta_fits.extend(edge_abs.tolist())

    return True


def _active_boolean_bytemap(
    configs_array, fitness, epsilon, maximize, verbose,
    edges, delta_fits, neutral_pairs,
):
    """Byte-map lookup specialised for single bit flips."""
    neutral_eps = _neutral_abs_threshold(epsilon)
    rows = np.ascontiguousarray(configs_array, dtype=np.uint8)
    if _active_bytemap_vectorized(
        rows, fitness, neutral_eps, maximize, 2,
        edges, delta_fits, neutral_pairs,
    ):
        return

    _active_boolean_bytemap_loop(
        rows, fitness, neutral_eps, maximize, verbose,
        edges, delta_fits, neutral_pairs,
    )


def _active_boolean_bytemap_loop(
    rows, fitness, neutral_eps, maximize, verbose,
    edges, delta_fits, neutral_pairs,
):
    """Exact Python-loop fallback for single bit flips (overflow path)."""
    index = {row.tobytes(): idx for idx, row in enumerate(rows)}
    get = index.get
    append_edge = edges.append
    append_delta = delta_fits.append
    append_neutral = neutral_pairs.append

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
    neutral_eps = _neutral_abs_threshold(epsilon)
    rows = np.ascontiguousarray(configs_array, dtype=np.uint8)
    if _active_bytemap_vectorized(
        rows, fitness, neutral_eps, maximize, alphabet_size,
        edges, delta_fits, neutral_pairs,
    ):
        return

    _active_sequence_bytemap_loop(
        rows, fitness, neutral_eps, maximize, verbose, alphabet_size,
        edges, delta_fits, neutral_pairs,
    )


def _active_sequence_bytemap_loop(
    rows, fitness, neutral_eps, maximize, verbose, alphabet_size,
    edges, delta_fits, neutral_pairs,
):
    """Exact Python-loop fallback for substitutions (overflow path)."""
    index = {row.tobytes(): idx for idx, row in enumerate(rows)}
    get = index.get
    append_edge = edges.append
    append_delta = delta_fits.append
    append_neutral = neutral_pairs.append

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


def _active_ordinal_vectorized(
    configs_array, config_dict, fitness, epsilon, maximize, verbose,
    edges, delta_fits, neutral_pairs,
):
    """Vectorised ±1-step (Manhattan-1) lookup for ordinal generators.

    Equivalent to :func:`_active_generic` driven by
    :class:`OrdinalNeighborGenerator` with ``n_edit == 1``, but replaces the
    per-configuration Python loop with a mixed-radix integer encoding plus
    ``searchsorted`` to locate neighbours in bulk.

    Each configuration is mapped to a single integer key over the per-variable
    cardinalities ``card_j = config_dict[j]["max"] + 1``. For every variable
    *j* and direction *d* in ``(+1, -1)`` the neighbour key is ``key +
    d * radix_j`` (computed only for rows whose code stays in ``[0, card_j-1]``),
    and an exact ``searchsorted`` match recovers the neighbour row. This
    enumerates exactly the directed ``(source, neighbour)`` lookups that succeed
    in the generic path, so the classification rules below reproduce its edge
    set, neutral-pair set, and edge/Δf correspondence identically.

    Returns
    -------
    bool
        ``True`` if the fast-path ran (results appended in-place); ``False`` if
        it bailed out *before appending anything* (e.g. mixed-radix overflow),
        so the caller can fall back to :func:`_active_generic`.
    """
    n_vars = configs_array.shape[1]

    # Per-variable cardinalities from config_dict (the same source the generic
    # generator reads via OrdinalNeighborGenerator: config_dict[j]["max"]).
    cards = np.empty(n_vars, dtype=np.int64)
    try:
        for j in range(n_vars):
            cards[j] = int(config_dict[j]["max"]) + 1
    except (KeyError, TypeError, ValueError):
        return False  # malformed config_dict -> let the generic path handle it
    if np.any(cards <= 0):
        return False

    # Mixed-radix place values: radix_j = prod(card_0 .. card_{j-1}).
    # Guard against int64 overflow of the full key space prod(card) > 2**63.
    radices = np.empty(n_vars, dtype=np.int64)
    acc = 1  # Python int: exact, no overflow during the product itself
    for j in range(n_vars):
        radices[j] = acc
        acc *= int(cards[j])
        if acc > (1 << 63) - 1:
            return False  # key space exceeds int64 -> fall back to generic

    # Committed to the fast-path from here on (only appends follow).
    codes = configs_array.astype(np.int64, copy=False)
    keys = codes @ radices  # (n_rows,) int64 mixed-radix key per configuration

    order = np.argsort(keys, kind="stable")
    skeys = keys[order]

    neutral_eps = _neutral_abs_threshold(epsilon)

    # Collect every directed (source_row, neighbour_row) adjacency, mirroring
    # the generic loop's successful lookups, then classify all at once.
    src_parts, nbr_parts = [], []
    for j in range(n_vars):
        col = codes[:, j]
        radix_j = radices[j]
        card_j = cards[j]
        for d in (1, -1):
            new_code = col + d
            valid = (new_code >= 0) & (new_code <= card_j - 1)
            if not np.any(valid):
                continue
            src_rows = np.nonzero(valid)[0]
            nbr_keys = keys[src_rows] + d * radix_j
            pos = np.searchsorted(skeys, nbr_keys)
            in_range = pos < skeys.shape[0]
            if not np.any(in_range):
                continue
            pos = pos[in_range]
            src_rows = src_rows[in_range]
            match = skeys[pos] == nbr_keys[in_range]
            if not np.any(match):
                continue
            src_parts.append(src_rows[match])
            nbr_parts.append(order[pos[match]])

    if verbose:
        print(" - Constructing neighborhoods (ordinal vectorised)...")

    if not src_parts:
        if verbose:
            print(" - Identified 0 improving connections.")
        return True

    src = np.concatenate(src_parts)
    nbr = np.concatenate(nbr_parts)

    delta = fitness[src] - fitness[nbr]
    abs_delta = np.abs(delta)

    # Neutral: |Δf| <= eps, deduped to the canonical (cid < nid) endpoint,
    # exactly as the generic loop records each neutral pair once.
    neutral_mask = abs_delta <= neutral_eps
    if np.any(neutral_mask):
        n_src = src[neutral_mask]
        n_nbr = nbr[neutral_mask]
        keep = n_src < n_nbr
        if np.any(keep):
            neutral_pairs.extend(
                zip(n_src[keep].tolist(), n_nbr[keep].tolist())
            )

    # Improving: recorded once, from the worse endpoint (source), matching the
    # generic direction test (maximize: delta < 0; minimize: delta > 0).
    imp_mask = ~neutral_mask
    if maximize:
        imp_mask &= delta < 0
    else:
        imp_mask &= delta > 0
    if np.any(imp_mask):
        edges.extend(zip(src[imp_mask].tolist(), nbr[imp_mask].tolist()))
        delta_fits.extend(abs_delta[imp_mask].tolist())

    if verbose:
        print(f" - Identified {len(edges)} improving connections.")
        if neutral_pairs:
            print(f" - Identified {len(neutral_pairs)} neutral neighbor pairs.")

    return True
