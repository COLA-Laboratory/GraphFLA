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
from typing import Protocol, Tuple, Dict, List, Callable, Union, runtime_checkable
import warnings

import numpy as np
import pandas as pd
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


def _empty_edges() -> np.ndarray:
    """Return a canonical empty ``(0, 2)`` int64 edge array."""
    return np.empty((0, 2), dtype=np.int64)


def _empty_deltas() -> np.ndarray:
    """Return a canonical empty 1-D float64 delta-fit array."""
    return np.empty(0, dtype=np.float64)


def _stack_edges(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Stack aligned source/target index arrays into an ``(E, 2)`` int64 array."""
    if src.size == 0:
        return _empty_edges()
    out = np.empty((src.shape[0], 2), dtype=np.int64)
    out[:, 0] = src
    out[:, 1] = tgt
    return out


def _edge_arrays_from_lists(edge_list, delta_list):
    """Convert Python edge/delta lists (slow fallback paths) to ndarrays."""
    edges = (
        np.asarray(edge_list, dtype=np.int64) if edge_list else _empty_edges()
    )
    delta_fits = (
        np.asarray(delta_list, dtype=np.float64)
        if delta_list
        else _empty_deltas()
    )
    return edges, delta_fits


@dataclass(frozen=True)
class EdgeResult:
    """Container for the output of :func:`build_edges`.

    ``edges`` holds directed ``(source, target)`` node-index pairs and
    ``delta_fits`` the aligned ``|Δfitness|`` edge weights (``delta_fits[i]`` is
    the weight of edge ``i``). Both are kept in whichever container the chosen
    producer emits:

    * the ``active`` strategy returns an ``(E, 2)`` ``int64`` ndarray of edges
      and a 1-D ``float64`` ndarray of weights (no per-edge Python objects);
    * the ``pairwise`` / ``broadcast`` strategies return a Python list of
      ``(int, int)`` tuples and a list of floats.

    The empty case is always the canonical ``(0, 2)`` / ``(0,)`` ndarrays.
    igraph 0.11 ingests either container, so :meth:`Landscape._build_graph`
    consumes both without conversion. ``neutral_pairs`` is always a Python list
    of ``(int, int)`` tuples.
    """

    edges: Union[np.ndarray, List[Tuple[int, int]]]
    delta_fits: Union[np.ndarray, List[float]]
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
    # ``configs`` (the per-row tuple ``Series``) may be ``None`` when the caller
    # defers its construction; the numeric ``configs_array`` is then the source
    # of truth and the only path that still needs the tuples (``_active_generic``)
    # derives them from it on demand.
    if (configs is None and configs_array is None) or config_dict is None:
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

    edges, delta_fits = _normalize_edge_output(edges, delta_fits)

    return EdgeResult(
        edges=edges,
        delta_fits=delta_fits,
        neutral_pairs=neutral_pairs,
        strategy=resolved,
    )


def _normalize_edge_output(edges, delta_fits):
    """Canonicalise the empty case while preserving the producer's container.

    The ``active`` producer returns ``(E, 2)`` int64 / 1-D float64 ndarrays;
    the ``pairwise``/``broadcast`` producers return Python lists of
    ``(source, target)`` tuples and floats. Both forms are accepted directly by
    igraph 0.11's edge-list / edge-attr ingestion, so they are passed through
    unchanged here. Converting the list producers to arrays is deliberately
    avoided: ``np.asarray`` over a large list of tuples is costly and igraph
    ingests a Python tuple list faster than an ndarray, so forcing arrays would
    regress those (small-cardinality, dense-pairwise) datasets.

    Only the empty case is normalised, to the canonical ``(0, 2)`` / ``(0,)``
    ndarray shapes, so :meth:`Landscape._build_graph` has one empty sentinel.
    """
    edges_out = edges if len(edges) else _empty_edges()
    delta_out = delta_fits if len(delta_fits) else _empty_deltas()
    return edges_out, delta_out


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

    Returns ``(edges, delta_fits, neutral_pairs)`` where ``edges`` is an
    ``(E, 2)`` int64 ndarray and ``delta_fits`` the aligned 1-D float64 ndarray.
    ``neutral_pairs`` stays a Python list of ``(int, int)`` tuples.
    """
    fitness = data["fitness"].to_numpy(copy=False)
    neutral_pairs: List[Tuple[int, int]] = []

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

    edges = delta_fits = None  # set by whichever path runs

    if can_use_bytemap and isinstance(generator_obj, BooleanNeighborGenerator):
        edges, delta_fits = _active_boolean_bytemap(
            configs_array, fitness, epsilon, maximize, verbose, neutral_pairs,
        )
    elif can_use_bytemap and isinstance(generator_obj, SequenceNeighborGenerator):
        edges, delta_fits = _active_sequence_bytemap(
            configs_array, fitness, epsilon, maximize, verbose,
            generator_obj.alphabet_size, neutral_pairs,
        )
    elif numeric_configs_array and isinstance(
        generator_obj, OrdinalNeighborGenerator
    ):
        result = _active_ordinal_vectorized(
            configs_array, config_dict, fitness, epsilon, maximize, verbose,
            neutral_pairs,
        )
        if result is not None:  # None => mixed-radix overflow, fall back
            edges, delta_fits = result

    if edges is None:
        # Generic Python fallback builds lists; convert at the boundary. This is
        # the only ``active`` path that consumes the per-row configuration
        # *tuples*; if the caller deferred the tuple ``Series`` (passing only the
        # numeric ``configs_array``), derive the equivalent tuple iterable here so
        # the hash-lookup keys match the generator's tuple output exactly.
        generic_configs = configs
        if generic_configs is None:
            # Materialise a concrete list of tuples (``_active_generic`` iterates
            # the configurations twice and calls ``len`` on them).
            generic_configs = list(map(tuple, configs_array.tolist()))
        edge_list: List[Tuple[int, int]] = []
        delta_list: List[float] = []
        _active_generic(
            generic_configs, config_dict, fitness, n_edit, epsilon, maximize,
            verbose, neighbor_generator, edge_list, delta_list, neutral_pairs,
        )
        edges, delta_fits = _edge_arrays_from_lists(edge_list, delta_list)

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
    """Blocked upper-triangle Hamming, low peak memory.

    Computes exactly the same neighbor pairs as a full
    ``pdist(metric="hamming")`` scan (Hamming distance in ``[1, n_edit]``)
    without ever allocating the ``n*(n-1)/2`` float64 condensed array that
    drives ``pdist``'s peak memory (~1.3 GB for n~18k). The upper triangle is
    split into independent pieces, each computed and reduced to compact index
    arrays before the next is touched, so peak memory is bounded by a single
    block instead of the whole matrix:

    * **within-block diagonal** ``[i0, i1) x [i0, i1)`` via ``pdist`` (no
      wasted work, condensed form decoded to ``(i, j)``);
    * **below-block rectangle** ``[i0, i1) x [i1, n)`` via ``cdist`` — every
      pair here is needed, so there is no redundant computation.

    ``pdist`` and ``cdist`` share the same per-pair C cost, and reducing each
    block immediately with ``np.flatnonzero`` keeps both runtime and memory
    low. The matched ``(i, j)`` arrays are handed to ``_classify_pairs`` once,
    which pairs each edge with its ``delta_fit`` in lockstep, so the edge set,
    per-edge delta, and neutral set are identical to the ``pdist`` path.

    For the dominant ``n_edit == 1`` case the O(n^2) Hamming scan is replaced by
    masked-position grouping (:func:`_build_masked_grouping`), which finds the
    exact same Hamming-1 pairs in ``O(n * n_vars)`` and feeds the *same*
    ``_classify_pairs``, so the result is identical. The blocked pdist/cdist
    path below is retained for ``n_edit > 1`` (Hamming distance up to n_edit).
    """
    if n_edit == 1:
        return _build_masked_grouping(
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

    from scipy.spatial.distance import pdist, cdist

    n = len(data)
    fitness = data["fitness"].values
    configs_array = _as_config_matrix(configs, configs_array)
    n_vars = configs_array.shape[1]

    if verbose:
        print(
            f" - Computing pairwise distances for {n} configurations "
            f"(blocked Hamming)..."
        )

    if n < 2:
        return [], [], []

    frac_threshold = (n_edit + 0.5) / n_vars

    # Bound the largest float64 distance buffer to ``budget_bytes``. Block rows
    # are chosen so the diagonal condensed array (~b^2/2 * 8) fits the budget,
    # and the below-block rectangle is further split into tail-column chunks so
    # its (b * cols * 8) buffer also fits. This keeps peak memory in the tens of
    # MB regardless of n (vs pdist's full n^2/2 * 8 array) while keeping the
    # per-block work large enough to amortise Python-loop overhead. When the
    # whole condensed array already fits the budget (small n) there is a single
    # diagonal block and no chunking, so behaviour matches the plain pdist path.
    budget_bytes = 64 * 1024**2
    budget_cells = max(1, budget_bytes // 8)
    block_rows = max(2, int(budget_cells**0.5))

    i_parts: List[np.ndarray] = []
    j_parts: List[np.ndarray] = []

    for i0 in range(0, n, block_rows):
        i1 = min(i0 + block_rows, n)
        b = i1 - i0

        # Diagonal block: upper triangle within [i0, i1) via pdist (no waste).
        if b >= 2:
            hf = pdist(configs_array[i0:i1], metric="hamming")
            local = np.flatnonzero((hf > 0) & (hf <= frac_threshold))
            del hf
            if len(local):
                bf = float(b)
                k = local.astype(np.float64)
                ri = (
                    bf - 2 - np.floor(
                        np.sqrt(-8.0 * k + 4.0 * bf * (bf - 1) - 7.0) / 2.0 - 0.5
                    )
                ).astype(np.intp)
                rj = (
                    k + ri + 1
                    - bf * (bf - 1) / 2
                    + (bf - ri) * ((bf - ri) - 1) / 2
                ).astype(np.intp)
                i_parts.append(ri + i0)
                j_parts.append(rj + i0)

        # Below-block rectangle: [i0, i1) x [i1, n). Every pair is needed.
        # Chunk the tail columns so the cdist buffer stays within the budget.
        if i1 < n:
            col_chunk = max(1, budget_cells // b)
            j_start = i1
            while j_start < n:
                j_end = min(j_start + col_chunk, n)
                n_cols = j_end - j_start
                d = cdist(
                    configs_array[i0:i1], configs_array[j_start:j_end],
                    metric="hamming",
                )
                flat = np.flatnonzero(((d > 0) & (d <= frac_threshold)).ravel())
                del d
                if len(flat):
                    # Row-major flat index -> (row in block, col in tail chunk).
                    i_parts.append((i0 + flat // n_cols).astype(np.intp))
                    j_parts.append((j_start + flat % n_cols).astype(np.intp))
                j_start = j_end

    if not j_parts:
        if verbose:
            print(f" - Found 0 neighbor pairs within edit distance {n_edit}.")
        return [], [], []

    i_arr = np.concatenate(i_parts)
    j_arr = np.concatenate(j_parts)
    del i_parts, j_parts

    if verbose:
        print(
            f" - Found {len(j_arr)} neighbor pairs "
            f"within edit distance {n_edit}."
        )

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
    """Per-row vectorised Hamming, upper triangle only.

    For ``n_edit == 1`` the per-row O(n^2) scan is replaced by masked-position
    grouping (:func:`_build_masked_grouping`), which finds the exact same
    Hamming-1 pairs in ``O(n * n_vars)`` and reuses ``_classify_pairs``, so the
    result is identical. The per-row loop below is kept for ``n_edit > 1``.
    """
    if n_edit == 1:
        return _build_masked_grouping(
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


# --- Masked-position grouping (universal Hamming-1 neighbour finder) ----

# Deterministic seed for the random int64 fingerprint weights. Fixing it keeps
# the fingerprint (and therefore the candidate-grouping order, though never the
# final exact pair set) reproducible across runs and processes.
_MASKED_GROUPING_SEED = 0x9E3779B97F4A7C15

# Max candidate pairs whose two configuration rows are gathered at once during
# the Hamming-1 collision verification. Bounds the transient
# ``rows[cand_i]`` / ``rows[cand_j]`` buffers (each ``chunk * n_vars`` of the
# compact verify dtype) so peak memory stays flat regardless of how many
# candidates a position produces, while keeping each block large enough to
# amortise numpy dispatch.
_MASKED_VERIFY_CHUNK = 1_000_000


def _within_run_pairs(group_id, n):
    """All within-group index pairs (a < b) for elements sharing a group id.

    ``group_id`` is a non-decreasing integer label per element (i.e. elements
    are already arranged so equal labels are contiguous). For every group of
    size ``c`` this enumerates its ``c*(c-1)/2`` ordered index pairs without a
    Python loop, by decoding a per-group condensed (upper-triangular) index —
    the same closed-form inverse used by :func:`_build_pairwise` to turn a
    ``pdist`` position into ``(i, j)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(a_idx, b_idx)`` int64 arrays of positions (into the same ordering as
        ``group_id``) with ``a_idx < b_idx`` and ``group_id[a] == group_id[b]``.
    """
    empty = np.empty(0, dtype=np.intp)
    if n < 2:
        return empty, empty

    # Run boundaries: a new run starts where the label changes.
    change = np.empty(n, dtype=bool)
    change[0] = True
    np.not_equal(group_id[1:], group_id[:-1], out=change[1:])
    starts = np.flatnonzero(change)
    counts = np.diff(np.append(starts, n))

    big = counts >= 2
    starts = starts[big]
    counts = counts[big].astype(np.int64)
    if starts.size == 0:
        return empty, empty

    npairs = counts * (counts - 1) // 2
    total = int(npairs.sum())
    if total == 0:
        return empty, empty

    # Map each of the ``total`` pairs back to its group and its offset within
    # that group's pair block.
    pair_grp = np.repeat(np.arange(starts.size), npairs)
    block_start = np.zeros(starts.size, dtype=np.int64)
    np.cumsum(npairs[:-1], out=block_start[1:])
    pair_off = np.arange(total, dtype=np.int64) - block_start[pair_grp]

    # Inverse of the condensed (upper-triangular) index within a group of size
    # ``c``: recover the (a, b) with 0 <= a < b < c. Same formula family as the
    # pdist decode in _build_pairwise (here per-group, with local size ``c``).
    cf = counts[pair_grp].astype(np.float64)
    off = pair_off.astype(np.float64)
    a = (
        cf - 2
        - np.floor(np.sqrt(-8.0 * off + 4.0 * cf * (cf - 1) - 7.0) / 2.0 - 0.5)
    ).astype(np.int64)
    b = (
        off + a + 1
        - cf * (cf - 1) // 2
        + (cf - a) * ((cf - a) - 1) // 2
    ).astype(np.int64)

    base = starts[pair_grp].astype(np.int64)
    return (base + a).astype(np.intp), (base + b).astype(np.intp)


def _compact_verify_view(a):
    """Return a narrow-dtype copy of ``a`` for the Hamming collision check.

    The verification only tests element equality, so the column codes can be
    held in the smallest unsigned integer width that fits ``a``'s value range.
    Narrowing from int64 to uint8 (the usual case: protein/DNA/boolean codes are
    well under 256) shrinks both the gathered ``rows[cand_i]`` buffers and the
    per-element ``!=`` work ~8x, which dominates the verification cost. When the
    input is already ``uint8`` it is returned unchanged (no copy).
    """
    if a.dtype == np.uint8:
        return a
    vmin = int(a.min())
    vmax = int(a.max())
    if vmin >= 0:
        if vmax <= 0xFF:
            return a.astype(np.uint8)
        if vmax <= 0xFFFF:
            return a.astype(np.uint16)
        if vmax <= 0xFFFFFFFF:
            return a.astype(np.uint32)
    return a.astype(np.int64, copy=False)


def _verify_hamming1(rows, cand_i, cand_j):
    """Keep only candidate pairs that are *exactly* Hamming-1, chunked.

    ``cand_i`` / ``cand_j`` index matched candidate rows that share a masked
    fingerprint (necessary but not sufficient for Hamming distance 1, since the
    scalar fingerprint can collide). This confirms each is truly Hamming-1 by
    counting differing columns over ``rows`` (the compact verify view), in
    blocks of :data:`_MASKED_VERIFY_CHUNK` so the gathered
    ``rows[cand_i_block]`` / ``rows[cand_j_block]`` arrays — the verification's
    peak transient — never exceed one block, bounding peak memory independently
    of the candidate count.

    Returns the boolean keep mask aligned with ``cand_i`` / ``cand_j``.
    """
    m = cand_i.shape[0]
    keep = np.empty(m, dtype=bool)
    chunk = _MASKED_VERIFY_CHUNK
    for s in range(0, m, chunk):
        e = s + chunk if s + chunk < m else m
        hdist = np.count_nonzero(
            rows[cand_i[s:e]] != rows[cand_j[s:e]], axis=1
        )
        np.equal(hdist, 1, out=keep[s:e])
    return keep


def _masked_grouping_pairs(configs_array):
    """Find every unordered Hamming-1 pair via masked-position grouping.

    Two configurations are Hamming-1 neighbours iff they are identical at all
    positions except exactly one. For each position ``p`` this groups the rows
    by their *masked* row (the row with position ``p`` removed); within a group
    every pair differs only at ``p`` (a candidate neighbour), and — because the
    configurations are unique — each true Hamming-1 pair lives in exactly one
    such position-group, so it is found exactly once.

    Grouping never materialises the masked row. Each masked row is reduced to a
    scalar int64 fingerprint: with fixed random weights ``R`` (deterministically
    seeded, shape ``n_vars``), the full fingerprint is ``h = configs @ R`` and
    the position-``p`` masked fingerprint is ``h - configs[:, p] * R[p]`` (all
    int64, wraparound is intentional and self-consistent). Rows with equal
    masked fingerprint form a candidate group. The groups are found by
    :func:`pandas.factorize` — an O(n) hash that maps each distinct masked
    fingerprint to a small dense ``int32`` code — followed by a stable sort of
    those narrow codes (numpy's radix sort over 4 bytes, far cheaper than the
    8-byte radix sort an int64 ``argsort`` of the raw fingerprints would pay).
    Positions whose masked fingerprints are *all distinct* (no two rows share
    one) are skipped immediately, since they can contribute no pair. Total cost
    is ``O(n_vars * n)`` plus ``O(n_edges)``, independent of the alphabet size
    and of sparsity.

    Equal fingerprint is *necessary but not sufficient* (hashes can collide), so
    every emitted candidate pair is verified to have Hamming distance exactly 1
    (``count_nonzero(configs[i] != configs[j]) == 1``, via
    :func:`_verify_hamming1`); pairs failing the check are dropped. The result is
    therefore EXACT regardless of hash collisions. The verification compares a
    compact narrow-dtype view of the rows (:func:`_compact_verify_view`) in
    bounded chunks, so it is both fast and memory-flat.

    Parameters
    ----------
    configs_array : np.ndarray, shape ``(n, n_vars)``
        Integer configuration matrix of unique rows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(i_arr, j_arr)`` int64 index arrays of every verified Hamming-1 pair,
        with ``i_arr < j_arr`` elementwise. Each undirected pair appears once.
    """
    a = np.ascontiguousarray(configs_array)
    n = a.shape[0]
    n_vars = a.shape[1] if a.ndim == 2 else 0
    empty = np.empty(0, dtype=np.intp)
    if n < 2 or n_vars == 0:
        return empty, empty

    ai = a.astype(np.int64, copy=False)
    # Narrow view used only for the equality-based Hamming verification; keeping
    # it separate from the int64 ``ai`` (needed for the wraparound fingerprint
    # arithmetic) makes the verification ~8x cheaper in both time and memory.
    averify = _compact_verify_view(a)

    # Fixed random int64 weights; deterministic so the grouping is reproducible.
    rng = np.random.default_rng(_MASKED_GROUPING_SEED)
    info = np.iinfo(np.int64)
    weights = rng.integers(
        info.min, info.max, size=n_vars, endpoint=True, dtype=np.int64
    )

    # Full fingerprint h = ai @ weights, with int64 wraparound (intended).
    with np.errstate(over="ignore"):
        h = (ai * weights[np.newaxis, :]).sum(axis=1)

    i_parts: List[np.ndarray] = []
    j_parts: List[np.ndarray] = []

    for p in range(n_vars):
        # Masked fingerprint: remove position p's contribution. Same modular
        # int64 arithmetic as h, so two rows identical except at p share it.
        with np.errstate(over="ignore"):
            masked = h - ai[:, p] * weights[p]

        # Group rows by equal masked fingerprint. ``pd.factorize`` is an O(n)
        # hash returning a dense int32 code per row (one per distinct
        # fingerprint); sorting those narrow codes is a 4-byte radix sort, much
        # cheaper than an 8-byte argsort of the raw int64 fingerprints.
        codes, uniques = pd.factorize(masked, sort=False)
        if uniques.shape[0] == n:
            # Every masked fingerprint is distinct -> no two rows agree on all
            # but position p, so this position yields no candidate pair.
            continue
        codes = codes.astype(np.int32, copy=False)

        order = np.argsort(codes, kind="stable")
        # Group id in sorted order is just the (non-decreasing) sorted codes;
        # _within_run_pairs only needs equal labels to be contiguous.
        gid = codes[order]

        ca, cb = _within_run_pairs(gid, n)
        if ca.size == 0:
            continue

        cand_i = order[ca]
        cand_j = order[cb]

        # Collision safety: keep only candidates that are truly Hamming-1.
        # Verified in bounded chunks against the compact view, so the result is
        # exact regardless of fingerprint collisions and peak memory stays flat.
        keep = _verify_hamming1(averify, cand_i, cand_j)
        if not keep.any():
            continue

        ki = cand_i[keep]
        kj = cand_j[keep]
        lo = np.minimum(ki, kj)
        hi = np.maximum(ki, kj)
        i_parts.append(lo)
        j_parts.append(hi)

    if not i_parts:
        return empty, empty

    i_arr = np.concatenate(i_parts)
    j_arr = np.concatenate(j_parts)
    return i_arr, j_arr


def _build_masked_grouping(
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
    """Hamming-1 edge builder via masked-position grouping (``n_edit == 1``).

    A universal, alphabet- and sparsity-independent replacement for the
    O(n^2)/O(n*n_vars*alphabet) Hamming scans on high-dimensional sparse data.
    Finds every Hamming-1 pair with :func:`_masked_grouping_pairs` (exact, with
    collision verification), then hands the matched ``(i, j)`` arrays to the
    shared :func:`_classify_pairs`, so the edge directions, neutral pairs, and
    per-edge ``delta_fit`` are byte-for-byte identical to the ``pairwise`` and
    ``broadcast`` strategies.

    Returns ``(edges, delta_fits, neutral_pairs)`` in the same Python-list
    container the ``pairwise``/``broadcast`` producers use.
    """
    fitness = data["fitness"].to_numpy(copy=False)
    matrix = _as_config_matrix(configs, configs_array)
    n = matrix.shape[0]

    if verbose:
        print(
            f" - Finding Hamming-1 neighbors for {n} configurations "
            f"(masked-position grouping)..."
        )

    if n < 2:
        return [], [], []

    i_arr, j_arr = _masked_grouping_pairs(matrix)

    if i_arr.size == 0:
        if verbose:
            print(" - Found 0 neighbor pairs within edit distance 1.")
        return [], [], []

    if verbose:
        print(
            f" - Found {len(j_arr)} neighbor pairs within edit distance 1."
        )

    return _classify_pairs(i_arr, j_arr, fitness, epsilon, maximize, verbose)


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


def _classify_pairs_to_arrays(
    i_arr, j_arr, fitness, neutral_eps, maximize, neutral_pairs,
):
    """Array-returning twin of :func:`_classify_pairs` for the active path.

    Applies the *same* classification rules as :func:`_classify_pairs` to the
    undirected ``(i, j)`` pairs (``i < j``): a pair is neutral when
    ``|Δf| <= neutral_eps`` (recorded once, as ``(i, j)``), otherwise it yields
    one directed edge from the worse to the better endpoint with
    ``delta_fit = |Δf|``. The direction test
    (``maximize``: worse means ``f[i] < f[j]``) is identical, so the edge set,
    neutral set, and per-edge Δf match :func:`_classify_pairs` exactly.

    Differs only in *containers*: it returns the active path's ``(E, 2)`` int64
    edge ndarray and 1-D float64 ``delta_fits`` ndarray, and appends neutral
    pairs to ``neutral_pairs`` in place (matching the other active fast-paths),
    instead of building Python lists. ``neutral_eps`` is the already-resolved
    inclusive bound (:func:`_neutral_abs_threshold`), as used elsewhere on the
    active path.
    """
    deltas = fitness[i_arr] - fitness[j_arr]
    abs_deltas = np.abs(deltas)

    neutral_mask = abs_deltas <= neutral_eps
    if neutral_mask.any():
        neutral_pairs.extend(
            zip(i_arr[neutral_mask].tolist(), j_arr[neutral_mask].tolist())
        )

    imp_mask = ~neutral_mask
    if not imp_mask.any():
        return _empty_edges(), _empty_deltas()

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

    edges = _stack_edges(src, tgt)
    delta_fits = np.ascontiguousarray(imp_abs, dtype=np.float64)
    return edges, delta_fits


def _active_masked_grouping(
    rows, fitness, neutral_eps, maximize, verbose, neutral_pairs,
):
    """Active-path Hamming-1 finder via masked-position grouping (ndarrays).

    Drop-in replacement for the per-cell Python-loop fallback used when the
    mixed-radix int key overflows int64 (high-dimensional sparse boolean /
    sequence landscapes). Finds every Hamming-1 pair with
    :func:`_masked_grouping_pairs` and classifies them with
    :func:`_classify_pairs_to_arrays`, so the edge set, neutral pairs, and
    per-edge Δf are identical to the loop fallback it replaces (and to the
    ``pairwise``/``broadcast`` strategies), while running in ``O(n * n_vars)``
    instead of ``O(n * n_vars * alphabet)``.

    Returns ``(edges, delta_fits)`` ndarrays; ``neutral_pairs`` is mutated in
    place.
    """
    n = rows.shape[0]
    if verbose:
        print(
            f" - Finding Hamming-1 neighbors for {n} configurations "
            f"(masked-position grouping)..."
        )
    if n < 2:
        return _empty_edges(), _empty_deltas()

    i_arr, j_arr = _masked_grouping_pairs(rows)
    if i_arr.size == 0:
        return _empty_edges(), _empty_deltas()

    return _classify_pairs_to_arrays(
        i_arr, j_arr, fitness, neutral_eps, maximize, neutral_pairs,
    )


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


# Largest mixed-radix key space (``base ** n_vars`` distinct keys) for which a
# direct-index lookup table (key -> row, ``-1`` = absent) is built instead of
# binary search. Combinatorially (near-)complete landscapes have a key space of
# the same order as the row count, so the table costs ~8 bytes/cell here (16 M
# cells -> 128 MB worst case) yet turns every neighbour lookup into an O(1)
# gather, replacing the O(M log n) ``searchsorted`` (which profiling shows is
# ~95% of the edge-finding cost) and the argsort. Sparse high-dimensional spaces
# exceed this and fall back to the per-cell ``searchsorted`` loop, which has no
# table to allocate.
_LUT_MAX_CELLS = 16_000_000

# Max candidate neighbour keys materialised per generation block on the LUT
# path. Bounds the transient ``nbr_keys`` + gathered-row buffers so peak memory
# stays at/below the all-at-once key array while keeping blocks large enough to
# amortise numpy dispatch.
_BYTEMAP_CHUNK_CANDIDATES = 4_000_000


def _bytemap_lut_block(
    keys, lut, a_row, codes_block, place_block, row_offset,
    src_list, nbr_list,
):
    """Resolve one batched block of single-substitution candidates via a LUT.

    For the row range starting at ``row_offset`` and the positions in
    ``codes_block`` / ``place_block``, build every ``(row, position,
    alternative-value)`` neighbour key, look each up in the direct-index table
    ``lut`` (``lut[key]`` is the neighbour row, or ``-1`` when absent), and
    append the hit source/neighbour row indices to ``src_list`` / ``nbr_list``.

    Parameters
    ----------
    keys : np.ndarray
        Mixed-radix key per row (int64).
    lut : np.ndarray
        Direct-index table of length ``base ** n_vars`` with ``lut[keys[r]] = r``
        and ``-1`` elsewhere.
    a_row : np.ndarray
        ``arange(base - 1)`` reused across blocks (alternative-value index).
    codes_block : np.ndarray, shape ``(rb, pb)``
        int64 per-position codes for the ``rb`` rows and ``pb`` positions.
    place_block : np.ndarray, shape ``(pb,)``
        Mixed-radix place values for the block's positions.
    row_offset : int
        Index of the first row in the block (added back to recover global ids).
    """
    rb, pb = codes_block.shape
    n_alt = a_row.shape[0]

    # Neighbour key for (row r, local position pl, alt index a):
    #   key[r] + (v - code) * place,   v = a + (a >= code)
    # so the value delta (v - code) is  a - code + (a >= code). Every neighbour
    # key stays in [0, base**n_vars), so the LUT gather is always in range.
    cb = codes_block[:, :, None]                       # (rb, pb, 1)
    a3 = a_row[None, None, :]                           # (1, 1, n_alt)
    vdelta = (a3 - cb) + (a3 >= cb)                     # (rb, pb, n_alt)
    del cb
    nbr_keys = (
        keys[row_offset:row_offset + rb][:, None, None]
        + vdelta * place_block[None, :, None]
    ).reshape(-1)
    del vdelta

    nbr_rows = lut[nbr_keys]
    del nbr_keys  # not needed past the gather; free before collecting hits
    hit = np.flatnonzero(nbr_rows >= 0)
    if hit.size == 0:
        return
    # Source row recovered from the flat index: layout is row-major over
    # (rb, pb, n_alt), so the row stride is pb * n_alt.
    src_list.append(row_offset + hit // (pb * n_alt))
    nbr_list.append(nbr_rows[hit])


def _bytemap_lut_adjacency(keys, place_values, codes, n, n_vars, n_alt, base):
    """Enumerate all directed single-substitution adjacencies via a LUT.

    Builds the direct-index table once, then generates candidate neighbour keys
    in memory-bounded blocks (grouping whole positions up to the candidate
    budget, splitting a single over-budget position over row chunks) and gathers
    the matches. Returns ``(src, nbr)`` concatenated hit arrays, or ``None`` if
    the key space exceeds :data:`_LUT_MAX_CELLS` (caller uses ``searchsorted``).
    """
    key_space = base ** n_vars
    if key_space > _LUT_MAX_CELLS:
        return None

    # key -> row table; -1 marks an absent configuration. int64 keeps row ids
    # exact for any realistic n while the table itself is small (key_space is
    # bounded above).
    lut = np.full(key_space, -1, dtype=np.int64)
    lut[keys] = np.arange(n)

    a_row = np.arange(n_alt, dtype=np.int64)
    src_list: List[np.ndarray] = []
    nbr_list: List[np.ndarray] = []

    rows_per_pos = n * n_alt
    p = 0
    while p < n_vars:
        if rows_per_pos > _BYTEMAP_CHUNK_CANDIDATES:
            # Single position too wide: split its rows into chunks.
            row_block = max(1, _BYTEMAP_CHUNK_CANDIDATES // n_alt)
            r0 = 0
            while r0 < n:
                r1 = min(r0 + row_block, n)
                _bytemap_lut_block(
                    keys, lut, a_row,
                    codes[r0:r1, p:p + 1], place_values[p:p + 1],
                    r0, src_list, nbr_list,
                )
                r0 = r1
            p += 1
            continue

        # Pack as many whole positions as fit in the candidate budget.
        max_group = max(1, _BYTEMAP_CHUNK_CANDIDATES // rows_per_pos)
        p_end = min(p + max_group, n_vars)
        _bytemap_lut_block(
            keys, lut, a_row,
            codes[:, p:p_end], place_values[p:p_end],
            0, src_list, nbr_list,
        )
        p = p_end

    del lut
    if not src_list:
        return (
            np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp),
        )
    return np.concatenate(src_list), np.concatenate(nbr_list)


def _bytemap_searchsorted_adjacency(keys, place_values, configs_array, n, n_vars, base):
    """Enumerate all directed single-substitution adjacencies via ``searchsorted``.

    Sparse fallback used when the key space is too large for a direct-index
    table. For each ``(position, value)`` the source rows (whose code differs
    from ``value``) are resolved in one binary search over the sorted keys.
    Returns ``(src, nbr)`` concatenated hit arrays.
    """
    order = np.argsort(keys, kind="stable")
    skeys = keys[order]
    row_idx = np.arange(n)

    src_list: List[np.ndarray] = []
    nbr_list: List[np.ndarray] = []

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
            # searchsorted gives an insertion point; only in-range positions
            # whose key equals the query are real hits.
            valid = pos < n
            if not valid.any():
                continue
            pos_valid = pos[valid]
            exact = skeys[pos_valid] == nbr_keys[valid]
            if not exact.any():
                continue

            src_list.append(src_rows[valid][exact])
            nbr_list.append(order[pos_valid[exact]])

    if not src_list:
        return (
            np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp),
        )
    return np.concatenate(src_list), np.concatenate(nbr_list)


def _active_bytemap_vectorized(
    configs_array, fitness, neutral_eps, maximize, base, neutral_pairs,
):
    """Vectorised single-substitution neighbour finder via mixed-radix keys.

    Enumerates exactly the same ``(source_row -> neighbour_row)`` adjacencies
    that the per-cell Python loop would (every row, every position ``p``, every
    alternative value ``v != current_code[p]``) and classifies them identically,
    but resolves the lookups in bulk. Two interchangeable enumeration backends
    produce the *same* adjacency set:

    * **direct-index table** (default for combinatorially dense key spaces):
      a ``key -> row`` table turns every neighbour lookup into an O(1) gather,
      avoiding both the argsort and the ``searchsorted`` that dominate the cost.
      Candidates for each ``(row, p)`` are generated directly via the index remap
      ``v = a + (a >= code)`` (``a in 0..base-2``), so no self-pair is ever
      materialised, in memory-bounded blocks;
    * **binary search** (sparse fallback, when ``base ** n_vars`` would make the
      table too large): one ``searchsorted`` per ``(position, value)``.

    The classification of each adjacency is identical to the baseline:

    * ``delta = fitness[source] - fitness[neighbour]``;
    * ``abs(delta) <= neutral_eps`` -> neutral pair, recorded as ``(src, nbr)``
      only when ``src < nbr`` (so each undirected neutral pair is kept once);
    * otherwise, if the source is the worse endpoint
      (``(maximize and delta < 0) or (not maximize and delta > 0)``), a directed
      edge ``src -> nbr`` is emitted with ``delta_fit = abs(delta)``.

    Because every neighbour differs in exactly one position, each ordered
    adjacent ``(src, nbr)`` pair is produced exactly once regardless of backend
    or iteration order, so the resulting edge set, neutral set, and the edge/Δf
    correspondence match the per-cell loop exactly. Neutral pairs are appended to
    the ``neutral_pairs`` list in place.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        ``(edges, delta_fits)`` where ``edges`` is an ``(E, 2)`` int64 ndarray
        and ``delta_fits`` the aligned 1-D float64 ndarray, when vectorisation
        ran; ``None`` if the mixed-radix key would overflow and the caller must
        use the exact Python fallback instead.
    """
    keys, place_values = _mixed_radix_keys(configs_array, base)
    if keys is None:
        return None

    n, n_vars = configs_array.shape

    # A single-value alphabet has no alternative substitutions, so there are no
    # candidate neighbours at all (the per-cell loop's inner range is empty).
    n_alt = base - 1
    if n_alt <= 0:
        return _empty_edges(), _empty_deltas()

    codes = configs_array.astype(np.int64, copy=False)

    result = _bytemap_lut_adjacency(
        keys, place_values, codes, n, n_vars, n_alt, base,
    )
    if result is None:  # key space too large for a direct-index table
        result = _bytemap_searchsorted_adjacency(
            keys, place_values, configs_array, n, n_vars, base,
        )
    src, nbr = result

    if src.size == 0:
        return _empty_edges(), _empty_deltas()

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
    # Worse-endpoint emission, matching the baseline direction guard.
    if maximize:
        imp_mask &= delta < 0
    else:
        imp_mask &= delta > 0
    if imp_mask.any():
        edges = _stack_edges(src[imp_mask], nbr[imp_mask])
        delta_fits = np.ascontiguousarray(abs_delta[imp_mask])
        return edges, delta_fits

    return _empty_edges(), _empty_deltas()


def _active_boolean_bytemap(
    configs_array, fitness, epsilon, maximize, verbose, neutral_pairs,
):
    """Byte-map lookup specialised for single bit flips.

    Returns ``(edges, delta_fits)`` as ndarrays; ``neutral_pairs`` is mutated
    in place.
    """
    neutral_eps = _neutral_abs_threshold(epsilon)
    rows = np.ascontiguousarray(configs_array, dtype=np.uint8)
    result = _active_bytemap_vectorized(
        rows, fitness, neutral_eps, maximize, 2, neutral_pairs,
    )
    if result is not None:
        return result

    # Mixed-radix int key overflowed int64 (high-dim sparse). Use masked-
    # position grouping instead of the O(n * n_vars) per-cell Python loop; both
    # produce the identical edge/neutral/Δf result, but grouping is far faster.
    return _active_masked_grouping(
        rows, fitness, neutral_eps, maximize, verbose, neutral_pairs,
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
    neutral_pairs,
):
    """Byte-map lookup specialised for single-position substitutions.

    Returns ``(edges, delta_fits)`` as ndarrays; ``neutral_pairs`` is mutated
    in place.
    """
    neutral_eps = _neutral_abs_threshold(epsilon)
    rows = np.ascontiguousarray(configs_array, dtype=np.uint8)
    result = _active_bytemap_vectorized(
        rows, fitness, neutral_eps, maximize, alphabet_size, neutral_pairs,
    )
    if result is not None:
        return result

    # Mixed-radix int key overflowed int64 (high-dim sparse). Use masked-
    # position grouping instead of the O(n * n_vars * alphabet) per-cell Python
    # loop; both produce the identical edge/neutral/Δf result, far faster.
    return _active_masked_grouping(
        rows, fitness, neutral_eps, maximize, verbose, neutral_pairs,
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
    neutral_pairs,
):
    """Vectorised ±1-step (Manhattan-1) lookup for ordinal generators.

    Equivalent to :func:`_active_generic` driven by
    :class:`OrdinalNeighborGenerator` with ``n_edit == 1``, but replaces the
    per-configuration Python loop with a mixed-radix integer encoding and a bulk
    neighbour lookup.

    Each configuration is mapped to a single integer key over the per-variable
    cardinalities ``card_j = config_dict[j]["max"] + 1``. For every variable
    *j* and direction *d* in ``(+1, -1)`` the neighbour key is ``key +
    d * radix_j`` (computed only for rows whose code stays in ``[0, card_j-1]``),
    and the neighbour row is recovered either by a direct gather through a dense
    inverse-key index (when the key space is small enough to address) or, for
    large sparse key spaces, by an exact ``searchsorted`` match. Both branches
    enumerate exactly the directed ``(source, neighbour)`` lookups that succeed
    in the generic path, so the classification rules below reproduce its edge
    set, neutral-pair set, and edge/Δf correspondence identically.

    Neutral pairs are appended to the ``neutral_pairs`` list in place.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        ``(edges, delta_fits)`` where ``edges`` is an ``(E, 2)`` int64 ndarray
        and ``delta_fits`` the aligned 1-D float64 ndarray, when the fast-path
        ran; ``None`` if it bailed out *before appending anything* (e.g.
        mixed-radix overflow), so the caller can fall back to
        :func:`_active_generic`.
    """
    n_vars = configs_array.shape[1]

    # Per-variable cardinalities from config_dict (the same source the generic
    # generator reads via OrdinalNeighborGenerator: config_dict[j]["max"]).
    cards = np.empty(n_vars, dtype=np.int64)
    try:
        for j in range(n_vars):
            cards[j] = int(config_dict[j]["max"]) + 1
    except (KeyError, TypeError, ValueError):
        return None  # malformed config_dict -> let the generic path handle it
    if np.any(cards <= 0):
        return None

    # Mixed-radix place values: radix_j = prod(card_0 .. card_{j-1}).
    # Guard against int64 overflow of the full key space prod(card) > 2**63.
    radices = np.empty(n_vars, dtype=np.int64)
    acc = 1  # Python int: exact, no overflow during the product itself
    for j in range(n_vars):
        radices[j] = acc
        acc *= int(cards[j])
        if acc > (1 << 63) - 1:
            return None  # key space exceeds int64 -> fall back to generic

    # Committed to the fast-path from here on. ``acc`` now holds the exact key
    # space size prod(card) (== max key + 1); every key is in ``[0, acc)`` and,
    # because rows are unique, the keys are distinct.
    keyspace = acc
    codes = configs_array.astype(np.int64, copy=False)
    n = codes.shape[0]
    keys = codes @ radices  # (n_rows,) int64 mixed-radix key per configuration

    neutral_eps = _neutral_abs_threshold(epsilon)

    if verbose:
        print(" - Constructing neighborhoods (ordinal vectorised)...")

    # Collect every directed (source_row, neighbour_row) adjacency, mirroring
    # the generic loop's successful lookups, then classify all at once.
    #
    # Lookup of the ±1 neighbour key uses a dense inverse index when the key
    # space is small enough to address directly, otherwise a sorted-key
    # ``searchsorted``. Both branches enumerate exactly the same successful
    # ``(source, neighbour)`` adjacencies, so the classification tail below is
    # identical regardless of which ran.
    #
    # Direct addressing replaces an O(log n) binary search per candidate with an
    # O(1) gather. The inverse-index array costs ``keyspace * 8`` bytes, so it is
    # only built when bounded by both an absolute floor (64 MiB) and ``4 * n``
    # entries — the latter ties its footprint to the dataset size (and hence to
    # the configs/edge arrays already held), so peak memory cannot blow up on a
    # sparse, high-cardinality lattice; such cases fall back to ``searchsorted``.
    use_dense_inverse = keyspace <= max(1 << 23, 4 * n)

    src_parts, nbr_parts = [], []
    if use_dense_inverse:
        # inv[k] = row id whose key is k, or -1 if absent. Every key < keyspace.
        inv = np.full(keyspace, -1, dtype=np.int64)
        inv[keys] = np.arange(n)
        for j in range(n_vars):
            col = codes[:, j]
            radix_j = int(radices[j])
            card_j = int(cards[j])
            for d in (1, -1):
                # Valid sources: +1 needs code < card-1, -1 needs code > 0.
                # The resulting neighbour key then stays within [0, keyspace),
                # so inv[...] is always an in-bounds lookup.
                valid = col != (card_j - 1) if d == 1 else col != 0
                if not np.any(valid):
                    continue
                src_rows = np.nonzero(valid)[0]
                nbr_rows = inv[keys[src_rows] + d * radix_j]
                hit = nbr_rows >= 0
                if not np.any(hit):
                    continue
                src_parts.append(src_rows[hit])
                nbr_parts.append(nbr_rows[hit])
        del inv
    else:
        order = np.argsort(keys, kind="stable")
        skeys = keys[order]
        for j in range(n_vars):
            col = codes[:, j]
            radix_j = int(radices[j])
            card_j = int(cards[j])
            for d in (1, -1):
                valid = col != (card_j - 1) if d == 1 else col != 0
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

    if not src_parts:
        if verbose:
            print(" - Identified 0 improving connections.")
        return _empty_edges(), _empty_deltas()

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
        edges = _stack_edges(src[imp_mask], nbr[imp_mask])
        delta_fits = np.ascontiguousarray(abs_delta[imp_mask])
    else:
        edges, delta_fits = _empty_edges(), _empty_deltas()

    if verbose:
        print(f" - Identified {len(edges)} improving connections.")
        if neutral_pairs:
            print(f" - Identified {len(neutral_pairs)} neutral neighbor pairs.")

    return edges, delta_fits
