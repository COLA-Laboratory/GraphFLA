"""Preparation-only EE fraction using GraphFLA's degree-bucketed moment kernel.

This avoids a dense ``n_nodes x n_sites x 3`` moment cube and the Python
node-by-site loop in the first bounded prototype. It leaves package APIs,
graphs, and genotype populations unchanged.
"""

from __future__ import annotations

import time


def bounded_ee_fraction_fast(
    landscape,
    fdr=0.01,
    *,
    progress_callback=None,
):
    """Compute the exact GraphFLA evolvability-enhancing fraction.

    The full graph and exact neutral-neighbor population are retained. Each
    unordered improving or neutral pair is tested once; the paired opposite
    orientation has identical two-sided p-values, and duplicating every
    hypothesis leaves its Benjamini-Hochberg adjusted value unchanged.

    Parameters
    ----------
    landscape : graphfla.landscape.Landscape
        Built landscape with retained encoded configurations.
    fdr : float, default=0.01
        Benjamini-Hochberg false-discovery-rate threshold.
    progress_callback : callable, optional
        Called with ``(stage_name, elapsed_seconds, details_dict)`` at stage
        boundaries. This is for preparation-run progress reporting only.

    Returns
    -------
    value : float
        EE fraction over the unchanged ordered-neighbor denominator.
    details : dict
        Pair counts, implementation marker, and stage timings.
    """
    import numpy as np
    from graphfla.analysis._evolvability import (
        _bh_adjusted_pvalues,
        _ee_pvalues,
        _nonfocal_moments,
        _validate_fdr,
    )

    started = time.perf_counter()
    timings = {}

    def report(stage, stage_start, **details):
        elapsed = time.perf_counter() - stage_start
        timings[stage] = elapsed
        if progress_callback is not None:
            progress_callback(stage, elapsed, details)

    _validate_fdr(fdr)
    graph = landscape.graph
    if graph is None or not graph.is_directed():
        raise ValueError("A directed improving graph is required.")
    if landscape.data_types is None or landscape._configs_array is None:
        raise ValueError("Encoded configuration columns are required.")
    configs = np.asarray(landscape._configs_array)
    if configs.ndim != 2 or configs.shape[0] != graph.vcount():
        raise ValueError("Encoded configurations do not align with the graph.")
    fitness = np.asarray(graph.vs["fitness"], dtype=np.float64)
    if not np.isfinite(fitness).all():
        raise ValueError("Fitness values must be finite.")
    if not landscape.maximize:
        fitness = -fitness
    if len(fitness):
        fitness = fitness - fitness[0]

    stage_start = time.perf_counter()
    improving_count = graph.ecount()
    pairs = np.empty((improving_count, 2), dtype=np.intp)
    for index, edge in enumerate(graph.es):
        pairs[index] = (edge.source, edge.target)
    if improving_count and index + 1 != improving_count:
        raise RuntimeError("Could not extract every improving edge.")
    neutral_neighbors = getattr(landscape, "_neutral_neighbors", None) or {}
    neutral_pairs = [
        (source, target)
        for source, neighbors in neutral_neighbors.items()
        for target in neighbors
        if source < target
    ]
    neutral_count = len(neutral_pairs)
    if neutral_count:
        pairs = np.vstack((pairs, np.asarray(neutral_pairs, dtype=np.intp)))
    pair_count = len(pairs)
    neutral_flags = np.zeros(pair_count, dtype=bool)
    neutral_flags[improving_count:] = True
    report(
        "pairs_ready", stage_start,
        improving_pairs=improving_count,
        neutral_pairs=neutral_count,
        variables=int(configs.shape[1]),
        nodes=int(configs.shape[0]),
    )
    if not pair_count:
        return float("nan"), {
            "improving_edge_pairs": 0,
            "neutral_neighbor_pairs": 0,
            "ordered_pair_denominator": 0,
            "testable_pairs": 0,
            "implementation": "degree-bucketed exact GraphFLA nonfocal-moment kernel",
            "timings_seconds": timings,
        }

    stage_start = time.perf_counter()
    # This is the exact optimized kernel used by GraphFLA's public EE
    # implementation. It returns both endpoint orientations but groups only
    # observed (node, focal-site) moments, rather than allocating N x P x 3.
    source, target, position, left, right = _nonfocal_moments(
        configs, fitness, pairs, fitness_variance=None
    )
    if len(source) != 2 * pair_count:
        raise RuntimeError("Moment kernel returned an unexpected orientation count.")
    # The pair-level decision below combines both directional predicates, so
    # one orientation per unordered pair is sufficient.
    source = source[:pair_count]
    target = target[:pair_count]
    position = position[:pair_count]
    left = left[:pair_count]
    right = right[:pair_count]
    report(
        "moments_ready", stage_start,
        pair_count=pair_count,
        oriented_pairs=int(2 * pair_count),
        moment_rows=int(len(left)),
    )

    stage_start = time.perf_counter()
    delta_fitness = fitness[target] - fitness[source]
    delta_neighbor = right[:, 0] - left[:, 0]
    variance = left[:, 1] + right[:, 1]
    n = np.minimum(left[:, 2], right[:, 2])
    p_effect = _ee_pvalues(delta_neighbor - delta_fitness, variance, n)
    p_zero = _ee_pvalues(delta_neighbor, variance, n)
    scale = np.maximum.reduce((
        np.abs(left[:, 0]), np.abs(right[:, 0]),
        np.abs(fitness[source]), np.abs(fitness[target]),
    ))
    roundoff = 8 * np.finfo(float).eps * scale
    ee_beneficial = (
        delta_neighbor - np.maximum(0.0, delta_fitness) > roundoff
    )
    ee_deleterious = (
        -delta_neighbor - np.maximum(0.0, -delta_fitness) > roundoff
    )
    ee_neutral = np.abs(delta_neighbor) > roundoff
    ee_beneficial &= ~neutral_flags
    ee_deleterious &= ~neutral_flags
    q_effect = _bh_adjusted_pvalues(p_effect)
    q_zero = _bh_adjusted_pvalues(p_zero)
    testable = np.isfinite(p_effect) | np.isfinite(p_zero)
    report("tests_adjusted", stage_start, testable_pairs=int(testable.sum()))
    if not testable.any():
        return float("nan"), {
            "improving_edge_pairs": improving_count,
            "neutral_neighbor_pairs": neutral_count,
            "ordered_pair_denominator": 2 * pair_count,
            "testable_pairs": 0,
            "implementation": "degree-bucketed exact GraphFLA nonfocal-moment kernel",
            "timings_seconds": timings,
        }
    count = np.count_nonzero((q_effect <= fdr) & ee_beneficial)
    count += np.count_nonzero((q_zero <= fdr) & ee_deleterious)
    count += np.count_nonzero((q_zero <= fdr) & ee_neutral & neutral_flags)
    value = float(count / (2 * pair_count))
    total_elapsed = time.perf_counter() - started
    if progress_callback is not None:
        progress_callback("complete", total_elapsed, {"ee_count": int(count)})
    return value, {
        "improving_edge_pairs": improving_count,
        "neutral_neighbor_pairs": neutral_count,
        "ordered_pair_denominator": 2 * pair_count,
        "testable_pairs": int(testable.sum()),
        "ee_ordered_pairs": int(count),
        "implementation": "degree-bucketed exact GraphFLA nonfocal-moment kernel",
        "timings_seconds": timings,
        "total_seconds": total_elapsed,
    }
