"""Edge-construction compute kernels.

The neighbourhood-strategy builders (active / pairwise / broadcast) and
their supporting grouping, Hamming-1 verification and bytemap-adjacency
kernels. The public entry point lives in ``edges.py``; these are the
routines it dispatches to. (``scipy`` is imported lazily inside
``_build_pairwise`` for the n_edit > 1 path only.)
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from .._progress import track

from ._arrays import (
    _neutral_abs_threshold,
    _empty_edges,
    _empty_deltas,
    _edge_arrays_from_lists,
    _as_config_matrix,
)
from ._classify import (
    _classify_pairs,
    _classify_pairs_to_arrays,
)
from .generators import (
    BooleanNeighborGenerator,
    SequenceNeighborGenerator,
    OrdinalNeighborGenerator,
    DefaultNeighborGenerator,
)
import logging

logger = logging.getLogger(__name__)
_PAIRWISE_BLOCK_BYTES = 64 * 1024**2
_BROADCAST_CHUNK_BYTES = 128 * 1024**2


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
    3. Mixed-radix lookup for ordinal and mixed-type generators.
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

    # Subclasses may override adjacency; fast paths implement built-ins only.
    if can_use_bytemap and type(generator_obj) is BooleanNeighborGenerator:
        edges, delta_fits = _active_boolean_bytemap(
            configs_array, fitness, epsilon, maximize, verbose, neutral_pairs,
        )
    elif can_use_bytemap and type(generator_obj) is SequenceNeighborGenerator:
        edges, delta_fits = _active_sequence_bytemap(
            configs_array, fitness, epsilon, maximize, verbose,
            generator_obj.alphabet_size, neutral_pairs,
        )
    elif numeric_configs_array and type(generator_obj) in (
        OrdinalNeighborGenerator, DefaultNeighborGenerator
    ):
        result = _active_discrete(
            configs_array, config_dict, fitness, epsilon, maximize, neutral_pairs,
            ordinal_only=type(generator_obj) is OrdinalNeighborGenerator,
        )
        if result is not None:  # None => mixed-radix overflow, fall back
            edges, delta_fits = result

    if edges is None:
        # Generic Python fallback: the only active path consuming config tuples.
        # If the tuple Series was deferred, derive tuples from configs_array so
        # hash-lookup keys match the generator's tuple output exactly.
        generic_configs = configs
        if generic_configs is None:
            # _active_generic iterates the configs twice and calls len on them.
            generic_configs = list(map(tuple, configs_array.tolist()))
        edge_list: List[Tuple[int, int]] = []
        delta_list: List[float] = []
        _active_generic(
            generic_configs, config_dict, fitness, n_edit, epsilon, maximize,
            verbose, neighbor_generator, edge_list, delta_list, neutral_pairs,
        )
        edges, delta_fits = _edge_arrays_from_lists(edge_list, delta_list)

    if verbose:
        logger.info(f" - Identified {len(edges)} improving connections.")
        if neutral_pairs:
            logger.info(f" - Identified {len(neutral_pairs)} neutral neighbor pairs.")
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
        logger.info(
            f" - Computing pairwise distances for {n} configurations "
            f"(blocked Hamming)..."
        )

    if n < 2:
        return [], [], []

    frac_threshold = (n_edit + 0.5) / n_vars

    # Bound each float64 distance buffer to ``budget_bytes`` (block rows sized so
    # the diagonal ~b^2/2*8 fits; tail columns chunked so b*cols*8 fits). Keeps
    # peak memory in the tens of MB vs pdist's full n^2/2*8 array; small n falls
    # to a single block with no chunking, matching the plain pdist path.
    budget_bytes = _PAIRWISE_BLOCK_BYTES
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
            logger.info(f" - Found 0 neighbor pairs within edit distance {n_edit}.")
        return [], [], []

    i_arr = np.concatenate(i_parts)
    j_arr = np.concatenate(j_parts)
    del i_parts, j_parts

    if verbose:
        logger.info(
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
    max_chunk_bytes = _BROADCAST_CHUNK_BYTES
    chunk_rows = max(1, max_chunk_bytes // max(n_vars, 1))

    edges, delta_fits, neutral = [], [], []
    neutral_eps = _neutral_abs_threshold(epsilon)

    outer = track(
        range(n - 1),
        description="Constructing neighborhoods (broadcast)",
        verbose=verbose,
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
        logger.info(f" - Identified {len(edges)} improving connections.")
        if neutral:
            logger.info(f" - Identified {len(neutral)} neutral neighbor pairs.")
    return edges, delta_fits, neutral


# ===================================================================
# Helpers (private)
# ===================================================================


# --- Masked-position grouping (universal Hamming-1 neighbour finder) ----

# Deterministic seed for the random int64 fingerprint weights; keeps grouping
# order reproducible across runs (never affects the final exact pair set).
_MASKED_GROUPING_SEED = 0x9E3779B97F4A7C15

# Max candidate pairs whose rows are gathered at once during Hamming-1
# verification. Bounds the transient row buffers so peak memory stays flat.
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

    # Map each pair back to its group and its offset within the group's block.
    pair_grp = np.repeat(np.arange(starts.size), npairs)
    block_start = np.zeros(starts.size, dtype=np.int64)
    np.cumsum(npairs[:-1], out=block_start[1:])
    pair_off = np.arange(total, dtype=np.int64) - block_start[pair_grp]

    # Inverse condensed (upper-triangular) index within a group of size c:
    # recover (a, b) with 0 <= a < b < c. Same decode as _build_pairwise.
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
    fingerprint to a dense code numbered ``0..K-1`` — followed by a stable
    ``argsort`` of those codes. Because the codes are densely numbered, numpy's
    radix sort adapts to their (small) value range and skips the all-zero high
    bytes, so the sort is as cheap as a narrow-int sort and far cheaper than an
    argsort of the raw high-entropy int64 fingerprints (no explicit down-cast is
    needed). Positions whose masked fingerprints are *all distinct* (no two rows
    share one) are skipped immediately, since they can contribute no pair. Total
    cost is ``O(n_vars * n)`` plus ``O(n_edges)``, independent of the alphabet
    size and of sparsity.

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

    # Compact narrow-dtype view, reused for both verification and the per-position
    # fingerprint columns (usually input ``a`` itself, uint8, no copy). Never
    # materialise a full int64 fingerprint matrix: it would be 8x ``a`` (~680 MB
    # for HIS7) and make every strided ``[:, p]`` gather cache-hostile; instead
    # each column is cast to int64 on the fly, ~8x cheaper.
    averify = _compact_verify_view(a)

    # Fixed random int64 weights; deterministic so the grouping is reproducible.
    rng = np.random.default_rng(_MASKED_GROUPING_SEED)
    info = np.iinfo(np.int64)
    weights = rng.integers(
        info.min, info.max, size=n_vars, endpoint=True, dtype=np.int64
    )

    # Contract directly into one vector, without an n-by-position product buffer.
    h = np.einsum("ij,j->i", averify, weights, dtype=np.int64)

    i_parts: List[np.ndarray] = []
    j_parts: List[np.ndarray] = []

    for p in range(n_vars):
        # Masked fingerprint: drop position p's contribution (same modular int64
        # arithmetic as h), so rows identical except at p share it.
        with np.errstate(over="ignore"):
            masked = h - averify[:, p].astype(np.int64) * weights[p]

        # Group by equal masked fingerprint. factorize is an O(n) hash to dense
        # codes 0..K-1; numpy's radix argsort adapts to that range (skips zero
        # high bytes), cheaper than argsorting the raw high-entropy int64s.
        codes, uniques = pd.factorize(masked, sort=False)
        if uniques.shape[0] == n:
            # All fingerprints distinct -> no candidate pair at this position.
            continue

        order = np.argsort(codes, kind="stable")
        # Sorted codes are the group ids; _within_run_pairs only needs equal
        # labels to be contiguous.
        gid = codes[order]

        ca, cb = _within_run_pairs(gid, n)
        if ca.size == 0:
            continue

        cand_i = order[ca]
        cand_j = order[cb]

        # Collision safety: keep only candidates that are truly Hamming-1, so the
        # result is exact regardless of fingerprint collisions.
        keep = _verify_hamming1(averify, cand_i, cand_j)
        keep &= averify[cand_i, p] != averify[cand_j, p]
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
        logger.info(
            f" - Finding Hamming-1 neighbors for {n} configurations "
            f"(masked-position grouping)..."
        )

    if n < 2:
        return [], [], []

    i_arr, j_arr = _masked_grouping_pairs(matrix)

    if i_arr.size == 0:
        if verbose:
            logger.info(" - Found 0 neighbor pairs within edit distance 1.")
        return [], [], []

    if verbose:
        logger.info(
            f" - Found {len(j_arr)} neighbor pairs within edit distance 1."
        )

    return _classify_pairs(i_arr, j_arr, fitness, epsilon, maximize, verbose)


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
        logger.info(
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
    # Overflow guard: max key is base**n_vars - 1; refuse if it exceeds int64.
    if n_vars >= 64 or base ** n_vars > (1 << 63):
        return None, None

    place_values = np.array(
        [base ** j for j in range(n_vars)], dtype=np.int64
    )
    keys = configs_array.astype(np.int64) @ place_values
    return keys, place_values


# Largest mixed-radix key space for which a direct-index LUT (key -> row, -1 =
# absent) is built instead of binary search. ~8 bytes/cell (16 M -> 128 MB worst
# case) buys an O(1) gather, replacing the O(M log n) searchsorted (~95% of
# edge-finding cost per profiling). Sparse spaces exceed this and use the loop.
_LUT_MAX_CELLS = 16_000_000

# Max candidate neighbour keys materialised per block on the LUT path. Bounds the
# transient key + gathered-row buffers to ~the all-at-once key array.
_BYTEMAP_CHUNK_CANDIDATES = 4_000_000


def _bytemap_lut_block(
    keys, lut, a_row, codes_block, place_block, row_offset,
    src_list, nbr_list,
):
    """Append unique unordered neighbor pairs from one bounded lookup block."""
    rb, pb = codes_block.shape
    n_alt = a_row.shape[0]

    # Neighbour key for (row r, position pl, alt a): key[r] + (v - code) * place
    # with v = a + (a >= code), so vdelta = a - code + (a >= code). Keys stay in
    # [0, base**n_vars), so the LUT gather is always in range.
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
    del nbr_keys  # free before collecting hits
    hit = np.flatnonzero(nbr_rows >= 0)
    if hit.size == 0:
        return
    # Source row from the row-major (rb, pb, n_alt) flat index: row stride pb*n_alt.
    source = row_offset + hit // (pb * n_alt)
    neighbor = nbr_rows[hit]
    keep = source < neighbor
    src_list.append(source[keep])
    nbr_list.append(neighbor[keep])


def _bytemap_lut_adjacency(keys, place_values, codes, n, n_vars, n_alt, base):
    """Find unordered neighbor pairs with a dense index, or return None if too large."""
    key_space = base ** n_vars
    if key_space > _LUT_MAX_CELLS:
        return None

    # key -> row table; -1 marks an absent configuration.
    index_dtype = np.int32 if n <= np.iinfo(np.int32).max else np.int64
    lut = np.full(key_space, -1, dtype=index_dtype)
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
    """Find unordered neighbor pairs by binary search in a sparse key space."""
    order = np.argsort(keys, kind="stable")
    skeys = keys[order]
    row_idx = np.arange(n)

    src_list: List[np.ndarray] = []
    nbr_list: List[np.ndarray] = []

    for p in range(n_vars):
        place = int(place_values[p])
        codes_p = configs_array[:, p].astype(np.int64)
        for v in range(base):
            # Source rows: those whose code at p differs from v ("val != orig").
            src_mask = codes_p != v
            if not src_mask.any():
                continue
            src_rows = row_idx[src_mask]
            nbr_keys = keys[src_rows] + (v - codes_p[src_mask]) * place

            pos = np.searchsorted(skeys, nbr_keys)
            # Insertion point; only in-range positions whose key matches are hits.
            valid = pos < n
            if not valid.any():
                continue
            pos_valid = pos[valid]
            exact = skeys[pos_valid] == nbr_keys[valid]
            if not exact.any():
                continue

            source = src_rows[valid][exact]
            neighbor = order[pos_valid[exact]]
            keep = source < neighbor
            src_list.append(source[keep])
            nbr_list.append(neighbor[keep])

    if not src_list:
        return (
            np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp),
        )
    return np.concatenate(src_list), np.concatenate(nbr_list)


def _active_bytemap_vectorized(
    configs_array, fitness, neutral_eps, maximize, base, neutral_pairs,
):
    """Find and classify each single-substitution neighbor pair once.

    Return None on key overflow so the caller can use exact masked grouping."""
    keys, place_values = _mixed_radix_keys(configs_array, base)
    if keys is None:
        return None

    n, n_vars = configs_array.shape

    # Single-value alphabet has no alternative substitutions -> no neighbours.
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

    return _classify_pairs_to_arrays(
        src, nbr, fitness, neutral_eps, maximize, neutral_pairs,
    )


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

    # Mixed-radix key overflowed int64 (high-dim sparse): masked-position
    # grouping gives the identical result far faster than the per-cell loop.
    return _active_masked_grouping(
        rows, fitness, neutral_eps, maximize, verbose, neutral_pairs,
    )


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

    # Mixed-radix key overflowed int64 (high-dim sparse): masked-position
    # grouping gives the identical result far faster than the per-cell loop.
    return _active_masked_grouping(
        rows, fitness, neutral_eps, maximize, verbose, neutral_pairs,
    )


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

    it = track(
        config_values,
        total=len(config_values),
        description="Constructing neighborhoods (active)",
        verbose=verbose,
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


def _active_discrete(
    configs_array,
    config_dict,
    fitness,
    epsilon,
    maximize,
    neutral_pairs,
    *,
    ordinal_only=False,
):
    """Find single-site neighbors in mixed categorical, Boolean and ordinal data."""
    n, n_vars = configs_array.shape
    try:
        cards = [int(config_dict[j]["max"]) + 1 for j in range(n_vars)]
        kinds = [
            "ordinal" if ordinal_only else config_dict[j]["type"] for j in range(n_vars)
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if any(card <= 0 for card in cards) or any(
        kind not in {"ordinal", "categorical", "boolean"} for kind in kinds
    ):
        return None

    radices, keyspace = [], 1
    for card in cards:
        radices.append(keyspace)
        keyspace *= card
        if keyspace > np.iinfo(np.int64).max:
            return None
    codes = configs_array.astype(np.int64, copy=False)
    keys = codes @ np.asarray(radices, dtype=np.int64)
    dense = keyspace <= min(_LUT_MAX_CELLS, max(65536, 8 * n))
    if dense:
        inverse = np.full(keyspace, -1, dtype=np.int64)
        inverse[keys] = np.arange(n)
    else:
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]

    sources, neighbors = [], []
    for j, (card, radix, kind) in enumerate(zip(cards, radices, kinds)):
        column = codes[:, j]
        ordinal = kind == "ordinal"
        # Increasing the allele code visits each undirected pair exactly once.
        for target in (None,) if ordinal else range(1, card):
            source = np.flatnonzero(column < (card - 1 if ordinal else target))
            if not source.size:
                continue
            delta = 1 if ordinal else target - column[source]
            query = keys[source] + delta * radix
            if dense:
                neighbor = inverse[query]
                hit = neighbor >= 0
            else:
                position = np.searchsorted(sorted_keys, query)
                bounded = np.minimum(position, n - 1)
                hit = (position < n) & (sorted_keys[bounded] == query)
                neighbor = order[bounded]
            sources.append(source[hit])
            neighbors.append(neighbor[hit])
    if not sources:
        return _empty_edges(), _empty_deltas()
    source, neighbor = np.concatenate(sources), np.concatenate(neighbors)
    return _classify_pairs_to_arrays(
        np.minimum(source, neighbor),
        np.maximum(source, neighbor),
        fitness,
        _neutral_abs_threshold(epsilon),
        maximize,
        neutral_pairs,
    )
