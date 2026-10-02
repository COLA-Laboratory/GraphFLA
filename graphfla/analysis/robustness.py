from scipy.stats import binomtest
from itertools import combinations
from numbers import Real
import warnings

import numpy as np
import pandas as pd

from ._utils import _pythonize, _pack_rows
from ._evolvability import _landscape_ee_statistics, _validate_fdr
import logging

logger = logging.getLogger(__name__)


def _row_group_ids(M):
    """Dense 0-based id per distinct row of a small-int matrix: fast int64 mixed-
    radix packing, with a byte-view ``np.unique`` fallback for wide / high-
    cardinality inputs (so grouping always succeeds)."""
    if M.shape[1] == 0:
        return np.zeros(M.shape[0], dtype=np.intp), 1
    ids, n = _pack_rows(M)
    if ids is not None:
        return ids, n
    Mc = np.ascontiguousarray(M)
    void = Mc.view([("", Mc.dtype)] * Mc.shape[1]).reshape(-1)
    uniq, inv = np.unique(void, return_inverse=True)
    return np.asarray(inv).reshape(-1).astype(np.intp), int(uniq.size)


def _mutation_effects_for_position(X, f_arr, f_std, position, test_type):
    """Fitness effects of every allele-pair mutation at one ``position``, sharing a
    single background grouping (genetic background = all other positions).

    Each (allele, background) is a unique genotype, so per-allele fitness is
    gathered into a background-indexed array and the effect of the mutation
    ``A -> B`` is ``f_B - f_A`` over their shared backgrounds. Returns a list of
    per-pair result dicts.
    """
    pos_vals = X[position].to_numpy()
    bg_cols = [c for c in X.columns if c != position]
    if bg_cols:
        codes = np.column_stack(
            [pd.factorize(X[c])[0] for c in bg_cols]
        ).astype(np.int64)
        bg_ids, n_bg = _row_group_ids(codes)
    else:
        bg_ids = np.zeros(len(X), dtype=np.intp)
        n_bg = 1

    unique_values = sorted(pd.Series(pos_vals).dropna().unique())
    fit_by_val = {}
    for v in unique_values:
        arr = np.full(n_bg, np.nan)
        rows = np.flatnonzero(pos_vals == v)
        arr[bg_ids[rows]] = f_arr[rows]
        fit_by_val[v] = arr

    results = []
    for A, B in combinations(unique_values, 2):
        fa = fit_by_val[A]
        fb = fit_by_val[B]
        mask = ~(np.isnan(fa) | np.isnan(fb))  # shared genetic backgrounds
        # Effect of the mutation as labelled (mutation_from=A -> mutation_to=B),
        # i.e. f_B - f_A, matching ``fitness_effect_distribution``.
        diff = fb[mask] - fa[mask]
        n_trials = int(diff.size)
        if n_trials == 0:
            median_effect = np.nan
            mean_effect = np.nan
            p_value, significant = np.nan, False
        else:
            median_effect = float(np.median(np.abs(diff))) / f_std
            mean_effect = float(diff.mean())
            if test_type == "positive":
                successes = int(np.count_nonzero(diff > 0))
            else:  # "negative"; validated by the caller
                successes = int(np.count_nonzero(diff < 0))
            test_result = binomtest(successes, n_trials, p=0.5, alternative="greater")
            p_value = test_result.pvalue
            significant = test_result.pvalue < 0.05
        results.append(_pythonize({
            "mutation_from": A,
            "mutation_to": B,
            "median_abs_effect": median_effect,
            "mean_effect": mean_effect,
            "p_value": p_value,
            "significant": significant,
        }))
    return results


def _ee_fraction(statistics, effect_type="all"):
    """Aggregate without changing the full testing family or denominator."""
    if statistics.empty or not statistics["testable"].any():
        warnings.warn(
            "No testable EE mutations: each endpoint needs at least two "
            "neighbors outside the mutated position.",
            RuntimeWarning,
            stacklevel=3,
        )
        return float("nan")
    selected = statistics["ee"].to_numpy().copy()
    effect = statistics["delta_fitness"].to_numpy()
    if effect_type == "beneficial":
        selected &= effect > 0
    elif effect_type == "deleterious":
        selected &= effect < 0
    elif effect_type == "neutral":
        selected &= effect == 0
    return float(np.count_nonzero(selected) / len(statistics))


def evolvability_enhancing_fraction(
    landscape, *, fdr=0.01, effect_type="all"
) -> float:
    """Return the fraction of evolvability-enhancing directed mutations.

    A mutation is evolvability-enhancing (EE) when the increase in mean
    non-focal neighbor fitness significantly exceeds the larger of zero and
    its own fitness effect [1]_. A mutation may be any one-variable change
    represented in the landscape, including changes in nonbiological data.

    Parameters
    ----------
    landscape : Landscape
        Built landscape with unique, nonmissing configurations, finite fitness
        values and one-site neighbor pairs. Both orientations of each graph
        pair and retained neutral pair are evaluated once. Discarded vertices
        and neighbor pairs are not reconstructed. Fitness is negated when
        ``landscape.maximize=False`` so positive effects indicate improvement.
    fdr : float, default=0.01
        Benjamini-Hochberg false discovery rate, strictly between 0 and 1.
        Corrections use all ordered pairs, before selecting an effect type.
    effect_type : {"all", "beneficial", "deleterious", "neutral"}, default="all"
        Effects to include in the numerator, classified by the sign of the
        unrounded fitness change. The denominator always includes every
        represented ordered neighbor pair. "all" combines all three classes.

    Returns
    -------
    fraction : float
        Significant EE mutations of the selected type divided by all ordered
        neighbor pairs. Untestable pairs remain in the denominator and are not
        counted as EE. The value lies in [0, 1] when defined; return NaN if no
        pair is testable. A type with no EE mutations returns zero if at least
        one pair in the full landscape is testable.

    Raises
    ------
    graphfla.exceptions.NotBuiltError
        If the landscape has not been built.
    ValueError
        If fdr or effect_type is invalid, configurations are missing or
        duplicated, fitness is nonfinite, or a pair does not differ at one site.

    Warns
    -----
    RuntimeWarning
        If no pair has at least two non-focal neighbors at each endpoint.

    See Also
    --------
    evolvability_effects : Per-mutation results, test definitions and
        neighborhood-variance conventions.

    References
    ----------
    .. [1] Wagner, A. Evolvability-enhancing mutations in the fitness landscapes
           of an RNA and a protein. Nat. Commun. 14, 3624 (2023).
           https://doi.org/10.1038/s41467-023-39321-8

    Examples
    --------
    An additive landscape has no EE mutations: the neighborhood increase
    equals the focal mutation's own fitness benefit.

    >>> from itertools import product
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import evolvability_enhancing_fraction
    >>> X = list(product([0, 1], repeat=3))
    >>> landscape = BooleanLandscape()
    >>> _ = landscape.build_from_data(X, [sum(x) for x in X], verbose=False)
    >>> evolvability_enhancing_fraction(landscape)
    0.0
    >>> evolvability_enhancing_fraction(landscape, effect_type="beneficial")
    0.0
    """
    landscape._check_built()
    _validate_fdr(fdr)
    if not isinstance(effect_type, str) or effect_type not in (
        "all", "beneficial", "deleterious", "neutral"
    ):
        raise ValueError(
            "effect_type must be 'all', 'beneficial', 'deleterious' or 'neutral'."
        )
    return _ee_fraction(_landscape_ee_statistics(landscape, fdr=fdr), effect_type)


def evolvability_effects(landscape, *, fdr=0.01) -> pd.DataFrame:
    """Return EE statistics for each directed one-site mutation.

    Each row represents a change between two configurations in a particular
    background. The reverse change occupies a separate row. Exclude all
    changes at the focal position from both endpoints' neighborhoods.

    Parameters
    ----------
    landscape : Landscape
        Built landscape satisfying the input requirements of
        :func:`evolvability_enhancing_fraction`. The existing graph and retained
        neutral adjacency define which changes are evaluated.
    fdr : float, default=0.01
        Benjamini-Hochberg false discovery rate, strictly between 0 and 1.
        This changes the EE decisions, not the raw or adjusted p-values.

    Returns
    -------
    effects : pandas.DataFrame
        One row per ordered neighbor pair, sorted by source_id and target_id,
        with a RangeIndex. An empty result retains the same column schema:

        - ``source_id``, ``target_id`` : int
            Vertex IDs matching ``landscape.get_data().index``.
        - ``position``, ``source_allele``, ``target_allele`` : object
            Configuration column name and original allele labels.
        - ``effect_type`` : str
            "beneficial", "deleterious" or "neutral", using the sign of the
            unrounded fitness effect in the landscape's optimization direction.
        - ``delta_fitness``, ``delta_neighbor_fitness``, ``excess`` : float
            Focal effect, non-focal neighborhood mean difference and
            ``delta_neighbor_fitness - max(0, delta_fitness)``, in fitness units.
        - ``n_source_neighbors``, ``n_target_neighbors`` : int
            Non-focal neighborhood sizes. Both must be at least two for a test.
        - ``p_effect``, ``p_zero`` : float
            Two-sided p-values for neighborhood difference equal to the focal
            effect or zero, respectively. Untestable pairs have NaN values.
        - ``q_effect``, ``q_zero`` : float
            BH adjusted p-values, corrected separately over all ordered pairs.
            Untestable pairs enter each family as p=1 but retain NaN in output.
        - ``is_ee`` : pandas nullable boolean
            EE classification; NA for an untestable pair. Positive focal effects
            use q_effect; other effects use q_zero. A rejection must also have
            positive excess, allowing for floating-point roundoff.
        - ``status`` : str
            "ok" or "insufficient_neighbors".

    Raises
    ------
    graphfla.exceptions.NotBuiltError
        If the landscape has not been built.
    ValueError
        If fdr or the landscape inputs are invalid.

    See Also
    --------
    evolvability_enhancing_fraction : Aggregate EE counts over all ordered pairs.

    Notes
    -----
    Two-sided one-sample t tests use the sum of both neighborhood population
    variances (ddof=0), n=min(k_source, k_target) and df=n-1. The strict EE
    criterion is delta_neighbor_fitness > max(0, delta_fitness).

    Neighborhood variability describes differences among configurations, not
    repeated-measurement uncertainty. This interface does not model experimental
    errors. Applying the same screening procedure to other domains does not
    establish statistical calibration for their sampling or dependence structure.

    Examples
    --------
    >>> from itertools import product
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import evolvability_effects
    >>> X = list(product([0, 1], repeat=3))
    >>> landscape = BooleanLandscape()
    >>> _ = landscape.build_from_data(X, [sum(x) for x in X], verbose=False)
    >>> effects = evolvability_effects(landscape)
    >>> len(effects)
    24
    >>> bool(effects["is_ee"].any())
    False
    """
    landscape._check_built()
    _validate_fdr(fdr)
    statistics = _landscape_ee_statistics(landscape, fdr=fdr)
    effect = statistics["delta_fitness"].to_numpy()
    statistics["effect_type"] = np.where(
        effect > 0, "beneficial", np.where(effect < 0, "deleterious", "neutral")
    )
    statistics["is_ee"] = statistics["ee"].astype("boolean").mask(
        ~statistics["testable"], pd.NA
    )
    statistics["status"] = np.where(
        statistics["testable"], "ok", "insufficient_neighbors"
    )
    effects = statistics.rename(columns={
        "source": "source_id", "target": "target_id",
        "n_source": "n_source_neighbors", "n_target": "n_target_neighbors",
    })
    columns = [
        "source_id", "target_id", "position", "source_allele", "target_allele",
        "effect_type", "delta_fitness", "delta_neighbor_fitness", "excess",
        "n_source_neighbors", "n_target_neighbors", "p_effect", "p_zero",
        "q_effect", "q_zero", "is_ee", "status",
    ]
    return effects[columns].sort_values(
        ["source_id", "target_id"], ignore_index=True
    )


def evolvability_enhancing_mutations(landscape, epsilon=0, auto_calculate=True):
    """Return the EE fraction through the deprecated compatibility interface.

    Use :func:`evolvability_enhancing_fraction` for new analyses. This entry
    retains its original parameters and neighbor-cache preparation behavior.

    Parameters
    ----------
    landscape : Landscape
        Built landscape satisfying the requirements of
        :func:`evolvability_enhancing_fraction`.
    epsilon : float, default=0
        Finite, nonnegative minimum excess above the EE criterion, in fitness
        units. Nonzero values are a legacy extension of the paper's definition.
    auto_calculate : bool, default=True
        Prepare missing ``landscape.neighbor_fitness`` attributes. If False,
        raise RuntimeError when those attributes are absent.

    Returns
    -------
    fraction : float
        Combined EE count over all ordered neighbor pairs at fixed FDR 0.01.
        Return NaN with a warning if no pair is testable.

    Raises
    ------
    graphfla.exceptions.NotBuiltError
        If the landscape has not been built.
    ValueError
        If epsilon or the landscape inputs are invalid.
    RuntimeError
        If ``auto_calculate=False`` and neighbor-fitness attributes are absent.

    Warns
    -----
    FutureWarning
        On each call, directing callers to the new scalar interface.
    RuntimeWarning
        If no pair is testable.

    See Also
    --------
    evolvability_enhancing_fraction : Scalar interface with fdr and effect_type.
    evolvability_effects : Per-mutation evidence and classification.
    """
    warnings.warn(
        "evolvability_enhancing_mutations is deprecated; use "
        "evolvability_enhancing_fraction instead.",
        FutureWarning,
        stacklevel=2,
    )
    landscape._check_built()
    if (not isinstance(epsilon, Real) or isinstance(epsilon, (bool, np.bool_))
            or not np.isfinite(epsilon) or epsilon < 0):
        raise ValueError("epsilon must be finite and nonnegative.")

    if "delta_mean_neighbor_fit" not in landscape.graph.es.attributes():
        if auto_calculate:
            if landscape.verbose:
                logger.info("Neighbor fitness metrics not found. Computing them...")
            landscape.neighbor_fitness  # lazily computes mean/delta neighbor fitness
        else:
            raise RuntimeError(
                "Neighbor fitness metrics haven't been calculated. "
                "Either access landscape.neighbor_fitness first "
                "or set auto_calculate=True."
            )

    return _ee_fraction(_landscape_ee_statistics(landscape, epsilon))


def neutrality(landscape, threshold: float = 0.01) -> float:
    """
    Calculate the neutrality index of the landscape using an igraph-based graph.
    It assesses the proportion of neighbors with fitness values within a given threshold,
    indicating the presence of neutral areas in the landscape.

    When the landscape has a plateau layer (``epsilon > 0``), neutral neighbors
    stored during construction are included alongside the graph-based neighbors.
    This ensures that equal-fitness pairs — which have no directed edge — are
    still counted toward the neutrality metric.

    Parameters
    ----------
    landscape : object
        An object which contains an igraph.Graph in its 'graph' attribute. It is assumed
        that each vertex of the graph has a 'fitness' attribute.
    threshold : float, default=0.01
        The fitness difference threshold for neighbors to be considered neutral.

    Returns
    -------
    neutrality : float
        The neutrality index, ranging from 0 to 1. A higher value indicates more neutrality.
    """
    g = landscape.graph
    neutral_nn = getattr(landscape, '_neutral_neighbors', None) or {}
    fitness_values = g.vs["fitness"]
    neutral_pairs = 0
    total_pairs = 0

    for v in range(g.vcount()):
        fitness = fitness_values[v]

        # Directed-graph neighbors (improving + worsening edges)
        graph_neighbors = set(g.neighbors(v))

        # Include neutral neighbors from plateau layer if available
        all_neighbors = graph_neighbors | set(neutral_nn.get(v, []))

        for neighbor in all_neighbors:
            neighbor_fitness = fitness_values[neighbor]
            if abs(fitness - neighbor_fitness) <= threshold:
                neutral_pairs += 1
            total_pairs += 1

    # Undefined when there are no neighbour pairs at all.
    neutrality_val = neutral_pairs / total_pairs if total_pairs > 0 else float("nan")

    return _pythonize(neutrality_val)


def single_mutation_effects(
    landscape, position: str, test_type: str = "positive"
) -> pd.DataFrame:
    """
    Assess the fitness effects of all possible mutations at a single position across all genetic backgrounds.

    Parameters
    ----------
    landscape : Landscape
        The Landscape object containing the data and graph.

    position : str
        The name of the position (variable) to assess mutations for.

    test_type : str, default='positive'
        The type of significance test to perform. Must be 'positive' or
        'negative', i.e. whether a majority of backgrounds show a fitness
        increase or a decrease under the mutation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing mutation pairs, median absolute fitness effect,
        mean fitness effect, p-values, and significance flags.

    Notes
    -----
    Effects are signed as ``mutation_to`` minus ``mutation_from`` in each shared
    genetic background, the same convention as
    :func:`~graphfla.analysis.fitness_effect_distribution`.
    """

    if test_type not in ("positive", "negative"):
        raise ValueError("test_type must be 'positive' or 'negative'")

    data = landscape.get_data()
    X = data[list(landscape.data_types.keys())]
    f = data["fitness"]
    # Vectorised over a shared background grouping; the per-pair work is now cheap,
    # so this runs serially (the previous per-pair joblib fan-out was net-negative).
    results = _mutation_effects_for_position(
        X, f.to_numpy(), f.std(), position, test_type
    )
    return pd.DataFrame(results)


def all_mutation_effects(
    landscape, test_type: str = "positive"
) -> pd.DataFrame:
    """
    Assess the fitness effects of all possible mutations across all positions in the landscape.

    Parameters
    ----------
    landscape : Landscape
        The Landscape object containing the data and graph.

    test_type : str, default='positive'
        The type of significance test to perform. Must be 'positive' or
        'negative', i.e. whether a majority of backgrounds show a fitness
        increase or a decrease under the mutation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing, for each position and mutation pair, the median
        absolute fitness effect, mean fitness effect, p-values, and significance
        flags.

    Notes
    -----
    Effects are signed as ``mutation_to`` minus ``mutation_from`` in each shared
    genetic background, the same convention as
    :func:`~graphfla.analysis.fitness_effect_distribution`.
    """

    if test_type not in ("positive", "negative"):
        raise ValueError("test_type must be 'positive' or 'negative'")

    data = landscape.get_data()
    X = data[list(landscape.data_types.keys())]
    f = data["fitness"]
    f_arr = f.to_numpy()
    f_std = f.std()
    # Compute the shared data once and run positions serially: each position's
    # vectorised computation is cheap, and the old per-position joblib fan-out
    # pickled the whole landscape per worker (net-negative; see ANALYSIS bench).
    frames = [
        pd.DataFrame(
            _mutation_effects_for_position(X, f_arr, f_std, position, test_type)
        )
        for position in X.columns
    ]
    return pd.concat(frames, ignore_index=True)
