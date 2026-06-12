"""Pair classification: improving vs. neutral vs. worsening edges."""

import numpy as np

from ._arrays import (
    _neutral_abs_threshold,
    _empty_edges,
    _empty_deltas,
    _stack_edges,
)
import logging

logger = logging.getLogger(__name__)


def _classify_undirected_core(i_arr, j_arr, fitness, neutral_eps, maximize):
    """Split undirected ``(i, j)`` pairs (``i < j``) into improving endpoints
    and neutral pairs.

    Shared numeric core of :func:`_classify_pairs` (list output) and
    :func:`_classify_pairs_to_arrays` (ndarray output): a pair is neutral when
    ``|Δf| <= neutral_eps``; otherwise it yields one directed edge from the
    worse to the better endpoint (``maximize``: worse means ``f[i] < f[j]``)
    with ``delta_fit = |Δf|``. ``neutral_eps`` is the already-resolved inclusive
    bound (:func:`_neutral_abs_threshold`).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(src, tgt, imp_abs, neutral_i, neutral_j)`` — improving edge endpoints
        with their ``|Δf|`` weights, and the neutral pair endpoints (each kept
        once, with ``neutral_i < neutral_j``). Any array may be empty.
    """
    deltas = fitness[i_arr] - fitness[j_arr]
    abs_deltas = np.abs(deltas)

    neutral_mask = abs_deltas <= neutral_eps
    neutral_i = i_arr[neutral_mask]
    neutral_j = j_arr[neutral_mask]

    imp_mask = ~neutral_mask
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

    return src, tgt, imp_abs, neutral_i, neutral_j


def _classify_pairs(i_arr, j_arr, fitness, epsilon, maximize, verbose):
    """Partition (i, j) neighbor pairs into improving edges and neutrals.

    List-returning adapter over :func:`_classify_undirected_core` (resolves
    ``epsilon`` to the inclusive neutral bound, packages the result as Python
    lists, and prints progress). Used by the ``pairwise`` / ``broadcast`` /
    masked-grouping producers.

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
    neutral_eps = _neutral_abs_threshold(epsilon)
    src, tgt, imp_abs, neutral_i, neutral_j = _classify_undirected_core(
        i_arr, j_arr, fitness, neutral_eps, maximize
    )

    neutral_pairs = list(zip(neutral_i.tolist(), neutral_j.tolist()))
    edges = list(zip(src.tolist(), tgt.tolist()))
    delta_fits = imp_abs.tolist()

    if verbose:
        logger.info(f" - Identified {len(edges)} improving connections.")
        if neutral_pairs:
            logger.info(
                f" - Identified {len(neutral_pairs)} neutral neighbor pairs."
            )
    return edges, delta_fits, neutral_pairs


def _classify_pairs_to_arrays(
    i_arr, j_arr, fitness, neutral_eps, maximize, neutral_pairs,
):
    """Array-returning twin of :func:`_classify_pairs` for the active path.

    Calls the shared :func:`_classify_undirected_core` and packages the result
    in the active path's containers: an ``(E, 2)`` int64 edge ndarray and a 1-D
    float64 ``delta_fits`` ndarray, appending the neutral ``(i, j)`` pairs to
    ``neutral_pairs`` in place. ``neutral_eps`` is the already-resolved inclusive
    bound (:func:`_neutral_abs_threshold`), as used elsewhere on the active path.
    The edge set, neutral set, and per-edge Δf match :func:`_classify_pairs`
    exactly; only the containers differ.
    """
    src, tgt, imp_abs, neutral_i, neutral_j = _classify_undirected_core(
        i_arr, j_arr, fitness, neutral_eps, maximize
    )

    if neutral_i.size:
        neutral_pairs.extend(zip(neutral_i.tolist(), neutral_j.tolist()))

    if src.size == 0:
        return _empty_edges(), _empty_deltas()

    edges = _stack_edges(src, tgt)
    delta_fits = np.ascontiguousarray(imp_abs, dtype=np.float64)
    return edges, delta_fits


def _classify_directed_adjacency(
    src, nbr, fitness, neutral_eps, maximize, neutral_pairs,
):
    """Classify directed ``(src -> nbr)`` adjacencies into improving edges.

    Shared classification tail of the vectorised active fast-paths
    (:func:`_active_bytemap_vectorized`, :func:`_active_ordinal_vectorized`),
    which both enumerate every directed single-change adjacency (each undirected
    neighbour appears in *both* orientations). A pair is neutral when
    ``|Δf| <= neutral_eps`` and is recorded once, deduped to the ``src < nbr``
    endpoint; otherwise the worse endpoint emits one directed edge
    (``maximize``: worse means ``f[src] < f[nbr]``, i.e. ``Δ < 0``) with
    ``delta_fit = |Δf|`` — so each improving edge is produced exactly once.
    ``neutral_pairs`` is mutated in place.

    Returns ``(edges, delta_fits)`` as the active path's ``(E, 2)`` int64 /
    1-D float64 ndarrays.
    """
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
    # Worse endpoint emits the edge (matches the baseline direction guard).
    if maximize:
        imp_mask &= delta < 0
    else:
        imp_mask &= delta > 0
    if imp_mask.any():
        edges = _stack_edges(src[imp_mask], nbr[imp_mask])
        delta_fits = np.ascontiguousarray(abs_delta[imp_mask])
        return edges, delta_fits

    return _empty_edges(), _empty_deltas()
