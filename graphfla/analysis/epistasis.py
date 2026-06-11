from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr, pearsonr
from typing import Literal
from collections import defaultdict
from joblib import Parallel, delayed

import numpy as np
import igraph as ig
import pandas as pd
import warnings
import itertools
import math
import copy
import random
import contextlib


@contextlib.contextmanager
def _seeded_igraph(seed):
    """Temporarily route igraph's RNG through a locally-seeded generator.

    Makes the sampling in ``motifs_randesu(cut_prob=...)`` reproducible when a
    seed is given. Restores igraph's default generator afterwards so global
    state is left untouched. A no-op when ``seed is None``.
    """
    if seed is None:
        yield
        return
    ig.set_random_number_generator(random.Random(seed))
    try:
        yield
    finally:
        ig.set_random_number_generator(None)


def _pythonize(value):
    if isinstance(value, dict):
        return {key: _pythonize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_pythonize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pythonize(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


# Module-level workers for process-based parallelism (must be importable so
# joblib's loky backend can pickle them; large code arrays are auto-memmapped).
def _idiosyncratic_position_worker(Xcodes, f, std_baseline, j, min_pairs):
    """Idiosyncratic indices for ALL allele pairs at position ``j``, sharing ONE
    background grouping (computed once, not per pair).

    Gives values identical to :func:`idiosyncratic_index`'s core: keep-first dedup
    per background (a no-op since each (allele, background) is a unique genotype),
    analytic random-pair baseline, and NaN for too-few-background mutations so they
    are excluded from -- rather than dilute -- the landscape average. Returns the
    list of per-mutation indices for this position.
    """
    P = Xcodes.shape[1]
    other = np.delete(np.arange(P), j)
    col = Xcodes[:, j]
    alleles = np.unique(col)  # sorted codes -> same mutation set/order as before
    bg_ids, n_bg = _pack_rows(Xcodes[:, other])
    out = []
    if bg_ids is not None:
        # Per-allele fitness indexed by background id; one O(V) fill per allele.
        fit = []
        for a in alleles:
            arr = np.full(n_bg, np.nan)
            rows = np.flatnonzero(col == a)
            arr[bg_ids[rows]] = f[rows]
            fit.append(arr)
        for ai in range(len(alleles)):
            fa = fit[ai]
            for bi in range(ai + 1, len(alleles)):
                fb = fit[bi]
                mask = ~(np.isnan(fa) | np.isnan(fb))  # shared backgrounds
                if int(np.count_nonzero(mask)) < min_pairs:
                    out.append(np.nan)
                    continue
                eff = fb[mask] - fa[mask]
                out.append(float(np.std(eff) / std_baseline))
    else:
        # High-dim fallback: per-allele dict grouping, still shared across pairs.
        bgcols = Xcodes[:, other]
        dicts = []
        for a in alleles:
            d = {}
            for i in np.flatnonzero(col == a):
                k = bgcols[i].tobytes()
                if k not in d:
                    d[k] = f[i]
            dicts.append(d)
        for ai in range(len(alleles)):
            da = dicts[ai]
            for bi in range(ai + 1, len(alleles)):
                db = dicts[bi]
                common = da.keys() & db.keys()
                if len(common) < min_pairs:
                    out.append(np.nan)
                    continue
                eff = np.fromiter(
                    (db[k] - da[k] for k in common), dtype=float, count=len(common)
                )
                out.append(float(np.std(eff) / std_baseline))
    return out


def _pack_rows(M):
    """Dense 0-based group id per distinct row of a small-int matrix, via
    mixed-radix packing into a single int64 key (one fast 1D ``np.unique``).

    Returns ``(ids, n_groups)``; ``(None, 0)`` when the radix product would
    overflow int64 (high-cardinality / many-column inputs) so callers can fall
    back to a dict/byte grouping.
    """
    nrows = M.shape[0]
    if M.shape[1] == 0:
        return np.zeros(nrows, dtype=np.intp), 1
    radices = M.max(axis=0).astype(np.int64) + 1
    prod = 1
    for r in radices:
        prod *= int(r)
        if prod > (1 << 62):
            return None, 0
    mult = np.ones(M.shape[1], dtype=np.int64)
    for i in range(M.shape[1] - 2, -1, -1):
        mult[i] = mult[i + 1] * int(radices[i + 1])
    key = M.astype(np.int64) @ mult
    uniq, inv = np.unique(key, return_inverse=True)
    return np.asarray(inv).reshape(-1).astype(np.intp), int(uniq.size)


def _gamma_pair_via_dict(Xcodes, f, p1, p2, alleles1, alleles2, other):
    """Original dict-grouping gamma pair contribution; the high-dimensional
    fallback used when the background does not pack into an int64 key."""
    col1 = Xcodes[:, p1]
    col2 = Xcodes[:, p2]
    bg = Xcodes[:, other]
    groups = {}
    for i in range(Xcodes.shape[0]):
        key = (col1[i], col2[i])
        d = groups.get(key)
        if d is None:
            d = groups[key] = {}
        bk = bg[i].tobytes()
        if bk not in d:
            d[bk] = f[i]
    num = den = snum = sden = 0.0
    for ai in range(len(alleles1)):
        for aj in range(ai + 1, len(alleles1)):
            a, A_ = alleles1[ai], alleles1[aj]
            for bi in range(len(alleles2)):
                for bj in range(bi + 1, len(alleles2)):
                    b, B_ = alleles2[bi], alleles2[bj]
                    g_ab = groups.get((a, b))
                    g_Ab = groups.get((A_, b))
                    g_aB = groups.get((a, B_))
                    g_AB = groups.get((A_, B_))
                    if not (g_ab and g_Ab and g_aB and g_AB):
                        continue
                    common = g_ab.keys() & g_Ab.keys() & g_aB.keys() & g_AB.keys()
                    if not common:
                        continue
                    common = list(common)
                    n = len(common)
                    bvec = np.fromiter((g_ab[k] - g_Ab[k] for k in common), float, n)
                    Bvec = np.fromiter((g_aB[k] - g_AB[k] for k in common), float, n)
                    num += float(np.dot(bvec, Bvec))
                    den += 0.5 * float(np.dot(bvec, bvec) + np.dot(Bvec, Bvec))
                    sb = np.sign(bvec)
                    sB = np.sign(Bvec)
                    snum += float(np.dot(sb, sB))
                    sden += 0.5 * float(np.count_nonzero(sb) + np.count_nonzero(sB))
    return num, den, snum, sden


def _gamma_position_pair_worker(Xcodes, f, p1, p2, alleles1, alleles2, other):
    """Pooled gamma / gamma* contributions for one ordered position pair (p1, p2).

    Implements the Ferretti et al. (2016) correlation of fitness effects: for the
    p1-mutation, correlate its effect on backgrounds with allele ``b`` at p2 with
    its effect on backgrounds with allele ``B`` at p2, across all shared genetic
    backgrounds. The correlation is *non-centered* (a raw second-moment ratio, as
    in eq. 3 of the paper), so it equals +1 for additive landscapes rather than
    being undefined. Returns the partial sums ``(num, den, snum, sden)`` to be
    pooled across all ordered pairs by :func:`_gamma_statistics`, giving
    ``gamma = num / den`` and ``gamma_star = snum / sden``.
    """
    # Group nodes by background. When the background columns pack into an int64
    # key (boolean / DNA / ordinal / low-dimensional protein) this is a fast 1D
    # unique and the allele-quadruple correlation vectorises over a fitness grid.
    # For high-cardinality, many-column backgrounds (high-dim protein) packing
    # overflows int64 -- there the original dict grouping is faster, so fall back.
    bg_ids, n_bg = _pack_rows(Xcodes[:, other])
    if bg_ids is None:
        return _gamma_pair_via_dict(Xcodes, f, p1, p2, alleles1, alleles2, other)

    col1 = Xcodes[:, p1]
    col2 = Xcodes[:, p2]
    A1 = len(alleles1)
    A2 = len(alleles2)
    # alleles1/alleles2 are sorted-unique, so searchsorted gives the local index.
    a1_local = np.searchsorted(alleles1, col1)
    a2_local = np.searchsorted(alleles2, col2)
    # G[bg, i, j] = fitness of the genotype (alleles1[i] @ p1, alleles2[j] @ p2, bg);
    # NaN where that genotype is absent. Each genotype is unique -> no cell collides.
    G = np.full((n_bg, A1, A2), np.nan)
    G[bg_ids, a1_local, a2_local] = f

    num = den = snum = sden = 0.0
    # Same allele-quadruple loops as the dict path, but each (num,den,snum,sden)
    # update is vectorised over the background axis instead of set intersections.
    for ai in range(A1):
        for aj in range(ai + 1, A1):
            for bi in range(A2):
                g_ai_bi = G[:, ai, bi]
                g_aj_bi = G[:, aj, bi]
                for bj in range(bi + 1, A2):
                    bvec = g_ai_bi - g_aj_bi          # effect at p2=bi: f(a,b)-f(A,b)
                    Bvec = G[:, ai, bj] - G[:, aj, bj]  # effect at p2=bj: f(a,B)-f(A,B)
                    mask = ~(np.isnan(bvec) | np.isnan(Bvec))  # shared backgrounds
                    if not mask.any():
                        continue
                    bv = bvec[mask]
                    Bv = Bvec[mask]
                    num += float(np.dot(bv, Bv))
                    den += 0.5 * float(np.dot(bv, bv) + np.dot(Bv, Bv))
                    sb = np.sign(bv)
                    sB = np.sign(Bv)
                    snum += float(np.dot(sb, sB))
                    sden += 0.5 * float(np.count_nonzero(sb) + np.count_nonzero(sB))
    return num, den, snum, sden


def _assign_roles_for_epistasis_igraph(graph, squares):
    """Assigns roles within collected square motif instances."""
    squares_with_roles = []
    if "fitness" not in graph.vs.attributes():
        raise ValueError(
            "igraph.Graph must have a 'fitness' vertex attribute for role assignment."
        )

    for square_nodes in squares:
        if len(square_nodes) != 4:
            continue  # defensive: should already be filtered

        try:
            nodes_in_square = list(square_nodes)
            fitness_values_list = graph.vs[nodes_in_square]["fitness"]
            fitness_values = {
                node: fitness
                for node, fitness in zip(nodes_in_square, fitness_values_list)
            }

            double_mutant = max(fitness_values, key=fitness_values.get)
            all_predecessors = graph.predecessors(double_mutant)
            square_set = set(nodes_in_square)
            single_mutants = [p for p in all_predecessors if p in square_set]

            if len(single_mutants) != 2:
                continue  # not the expected square structure

            wild_type_set = square_set - set(single_mutants) - {double_mutant}
            if len(wild_type_set) != 1:
                continue  # WT not uniquely identifiable
            wild_type = list(wild_type_set)[0]

            single_mutants.sort()  # stable ordering

            squares_with_roles.append(
                {
                    "wild_type": wild_type,
                    "single_mutant_1": single_mutants[0],
                    "single_mutant_2": single_mutants[1],
                    "double_mutant": double_mutant,
                    "fitness_values": fitness_values,
                }
            )
        except Exception as e:
            print(
                f"WARN: Could not process square {square_nodes} for role assignment: {e}"
            )
            continue

    return squares_with_roles


def _calculate_pos_neg_epistasis_igraph(squares_with_roles):
    """Calculates positive/negative epistasis from squares with assigned roles."""
    if not squares_with_roles:
        return {"positive epistasis": 0.0, "negative epistasis": 0.0}

    data_for_df = []
    for square_role_info in squares_with_roles:
        fit_vals = square_role_info["fitness_values"]
        try:
            data_for_df.append(
                {
                    "ab": fit_vals[square_role_info["wild_type"]],
                    "aB": fit_vals[square_role_info["single_mutant_1"]],
                    "Ab": fit_vals[square_role_info["single_mutant_2"]],
                    "AB": fit_vals[square_role_info["double_mutant"]],
                }
            )
        except KeyError as e:
            continue  # skip squares with missing role data

    if not data_for_df:
        return {"positive epistasis": 0.0, "negative epistasis": 0.0}

    df_squares = pd.DataFrame(data_for_df)
    effect_mut1_b = df_squares["Ab"] - df_squares["ab"]
    effect_mut2_a = df_squares["aB"] - df_squares["ab"]
    effect_both = df_squares["AB"] - df_squares["ab"]

    positive_count = (effect_both > (effect_mut1_b + effect_mut2_a)).sum()
    total_squares = len(df_squares)

    perc_positive = positive_count / total_squares if total_squares > 0 else 0.0
    perc_negative = 1.0 - perc_positive

    return _pythonize({
        "positive epistasis": perc_positive,
        "negative epistasis": perc_negative,
    })


def classify_epistasis(landscape, approximate=False, sample_cut_prob=0.2, seed=None):
    """
    Calculates proportions of five epistasis types using 4-node motifs in an igraph graph.

    Determines magnitude, sign, and reciprocal sign epistasis based on counts/estimates
    of motifs 19, 52, 66. Determines positive and negative epistasis by analyzing
    the fitness relationships within instances of these motifs.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object, containing landscape.graph as an igraph.Graph
        with a "fitness" vertex attribute.
    approximate : bool, optional
        If True, estimates motif counts and uses a sample of motif instances
        for positive/negative epistasis calculation. Faster but less accurate.
        Defaults to False (exact counts and all relevant instances).
    sample_cut_prob : float, optional
        The probability used for pruning the search tree at each level during
        sampling when approximate=True. Higher values -> faster, less accurate.
        Defaults to 0.2.

    Returns
    -------
    dict
        A dictionary containing proportions for:
        - "magnitude epistasis": The magnitude of the combined fitness effect of mutations
        differs from the sum of their individual effects, but the direction relative to
        single mutants or wild-type may not change sign.
        - "sign epistasis": The sign of the fitness effect of at least one mutation changes depending
        on the presence of other mutations. For example, a mutation beneficialon its own becomes
        deleterious when combined with another specific mutation.
        - "reciprocal sign epistasis": A specific form of sign epistasis where the sign of the effect
        of *each* mutation depends on the allele state at the other locus.
        - "positive epistasis": The combined fitness effect of mutations is greater than the sum of
        their individual effects, often referred to as synergistic epistasis.
        - "negative epistasis": The combined fitness effect of mutations is less than the sum of their
        individual effects, often referred to as antagonistic epistasis.

        Returns zero proportions if relevant counts/instances are zero or cannot be processed.

    Raises
    ------
    AttributeError
        If landscape.graph is not an igraph.Graph object or does not exist.
    ValueError
        If sample_cut_prob is not between 0 and 1, or if fitness attribute missing.
    """
    motif_size = 4
    square_indices = {19, 52, 66}  # set for O(1) membership in callback

    if not hasattr(landscape, "graph") or not isinstance(landscape.graph, ig.Graph):
        raise AttributeError(
            "Input 'landscape' must have a 'graph' attribute that is an igraph.Graph object."
        )
    if "fitness" not in landscape.graph.vs.attributes():
        raise ValueError("igraph.Graph must have a 'fitness' vertex attribute.")
    if approximate and not 0.0 <= sample_cut_prob <= 1.0:
        raise ValueError("sample_cut_prob must be between 0.0 and 1.0")

    collected_square_instances = defaultdict(list)
    cut_prob_vector = [sample_cut_prob] * motif_size if approximate else None

    if approximate:
        def motif_collector_callback_approx(graph, vertices, isoclass):
            if isoclass in square_indices:
                collected_square_instances[isoclass].append(tuple(sorted(vertices)))
            return False  # continue search

        # Seed the sampling RNG so approximate results are reproducible.
        with _seeded_igraph(seed):
            # Run 1: estimated counts for mag/sign/recip proportions
            estimated_motif_counts = landscape.graph.motifs_randesu(
                size=motif_size, cut_prob=cut_prob_vector
            )
            # Run 2: collect a sample of square instances
            landscape.graph.motifs_randesu(
                size=motif_size,
                cut_prob=cut_prob_vector,
                callback=motif_collector_callback_approx,
            )

        reci_sign_count = (
            np.nan_to_num(estimated_motif_counts[19])
            if len(estimated_motif_counts) > 19
            else 0
        )
        sign_count = (
            np.nan_to_num(estimated_motif_counts[52])
            if len(estimated_motif_counts) > 52
            else 0
        )
        mag_count = (
            np.nan_to_num(estimated_motif_counts[66])
            if len(estimated_motif_counts) > 66
            else 0
        )

    else:  # exact calculation
        def motif_collector_callback_exact(graph, vertices, isoclass):
            if isoclass in square_indices:
                collected_square_instances[isoclass].append(tuple(sorted(vertices)))
            return False  # continue search

        landscape.graph.motifs_randesu(
            size=motif_size, callback=motif_collector_callback_exact
        )

        reci_sign_count = len(collected_square_instances.get(19, []))
        sign_count = len(collected_square_instances.get(52, []))
        mag_count = len(collected_square_instances.get(66, []))

    # --- Step 2: Calculate Mag/Sign/Recip Proportions ---
    total_mag_sign_recip = reci_sign_count + sign_count + mag_count
    if total_mag_sign_recip == 0:
        mag_sign_recip_props = {
            "magnitude epistasis": 0.0,
            "sign epistasis": 0.0,
            "reciprocal sign epistasis": 0.0,
        }
    else:
        mag_sign_recip_props = {
            "magnitude epistasis": mag_count / total_mag_sign_recip,
            "sign epistasis": sign_count / total_mag_sign_recip,
            "reciprocal sign epistasis": reci_sign_count / total_mag_sign_recip,
        }

    # --- Step 4: Assign Roles within Collected Squares ---
    all_collected_squares = []
    for idx in square_indices:
        all_collected_squares.extend(collected_square_instances.get(idx, []))

    if not all_collected_squares:
        pos_neg_props = {"positive epistasis": 0.0, "negative epistasis": 0.0}
    else:
        squares_with_roles = _assign_roles_for_epistasis_igraph(
            landscape.graph, all_collected_squares
        )

        # --- Step 5: Calculate Positive/Negative Epistasis Proportions ---
        if not squares_with_roles:
            pos_neg_props = {"positive epistasis": 0.0, "negative epistasis": 0.0}
        else:
            pos_neg_props = _calculate_pos_neg_epistasis_igraph(squares_with_roles)

    # --- Step 6: Combine Results ---
    final_results = {**mag_sign_recip_props, **pos_neg_props}
    return _pythonize(final_results)


def idiosyncratic_index(landscape, mutation, min_pairs: int = 3):
    """
    Calculates the idiosyncratic index for the fitness landscape proposed in [1].

    The idiosyncratic index of a specific genetic mutation quantifies the sensitivity
    of a specific mutation to idiosyncratic epistasis. It is defined as the
    variation in the fitness difference between genotypes that differ by the mutation,
    relative to the variation in the fitness difference between random genotype pairs.
    We compute this for the entire fitness landscape by averaging it across individual
    mutations.

    The index is typically in [0, 1] (0 = no idiosyncrasy); a mutation whose effect
    varies more across backgrounds than random genotype pairs can exceed 1.

    For more information, please refer to the original paper:

    [1] Daniel M. Lyons et al, "Idiosyncratic epistasis creates universals in mutational
    effects and evolutionary trajectories", Nat. Ecol. Evo., 2020.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    mutation : tuple(A, pos, B)
        A tuple containing:
        - A: The original variable value (allele) at the given position.
        - pos: The position in the configuration where the mutation occurs.
        - B: The new variable value (allele) after the mutation.

    min_pairs : int, default=3
        Minimum number of shared genetic backgrounds required to estimate the
        index. Mutations with fewer background-matched pairs yield an unstable
        effect-variance estimate and return NaN (so they are excluded from any
        landscape average rather than biasing it toward zero).

    Returns
    -------
    float
        The calculated idiosyncratic index, or NaN when it cannot be estimated
        (fewer than ``min_pairs`` shared backgrounds).
    """
    A, pos, B = mutation

    data = landscape.get_data()
    X = data[list(landscape.data_types.keys())]
    f = data["fitness"]

    unique_alleles = X[pos].unique()
    if A not in unique_alleles:
        raise ValueError(
            f"Original allele '{A}' not found at position '{pos}'. Available: {unique_alleles}"
        )
    if B not in unique_alleles:
        raise ValueError(
            f"New allele '{B}' not found at position '{pos}'. Available: {unique_alleles}"
        )

    X_A = X[X[pos] == A]
    X_B = X[X[pos] == B]

    if X_A.empty or X_B.empty:
        print(
            f"Warning: No genotypes found for allele '{A}' or '{B}' at position '{pos}'. Returning 0.0."
        )
        return 0.0

    background_cols = [col for col in X.columns if col != pos]

    if not background_cols:
        X_A_backgrounds = pd.Series([tuple()] * len(X_A), index=X_A.index)
        X_B_backgrounds = pd.Series([tuple()] * len(X_B), index=X_B.index)
    else:
        X_A_backgrounds = X_A[background_cols].apply(tuple, axis=1)
        X_B_backgrounds = X_B[background_cols].apply(tuple, axis=1)

    df_A = pd.DataFrame({"background": X_A_backgrounds, "fitness_A": f.loc[X_A.index]})
    df_B = pd.DataFrame({"background": X_B_backgrounds, "fitness_B": f.loc[X_B.index]})

    df_A = df_A.drop_duplicates(subset="background", keep="first").set_index(
        "background"
    )
    df_B = df_B.drop_duplicates(subset="background", keep="first").set_index(
        "background"
    )

    df_merged = pd.merge(df_A, df_B, left_index=True, right_index=True, how="inner")

    if df_merged.empty:
        return _pythonize(np.nan)

    mutation_effects = df_merged["fitness_B"] - df_merged["fitness_A"]
    n_pairs = len(mutation_effects)

    if n_pairs < min_pairs:
        return _pythonize(np.nan)

    all_fitness_values = f.values
    if len(all_fitness_values) <= 1 or np.all(
        all_fitness_values == all_fitness_values[0]
    ):
        return 0.0

    # Random-pair baseline uses the exact closed form Var(diff) = 2*Var(f)
    # (i.i.d. draws): deterministic and avoids a near-zero denominator vs sampling.
    std_mutation_effect = np.std(mutation_effects)
    std_random_diff = np.sqrt(2.0) * np.std(all_fitness_values)

    idiosyncratic_val = std_mutation_effect / std_random_diff

    return _pythonize(idiosyncratic_val)


def global_idiosyncratic_index(landscape, n_jobs=-1, seed=None, min_pairs: int = 3):
    """
    Calculates the global idiosyncratic index for the entire fitness landscape using parallel processing.

    This function extends the individual mutation idiosyncratic index from Lyons et al. (2020)
    to provide a global measure by averaging across all possible mutations in the landscape.
    The global index quantifies the overall sensitivity of the landscape to idiosyncratic
    epistasis.

    The index is typically in [0, 1], with higher values indicating stronger idiosyncratic
    effects; individual mutations whose effects vary more than random genotype pairs can push
    the average above 1.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.
    n_jobs : int, optional
        Number of parallel jobs to use. Default is -1 (all available cores).
    seed : int, optional
        Accepted for API consistency with other stochastic functions, but the
        index is computed deterministically (analytic random-pair baseline), so
        this has no effect on the result.
    min_pairs : int, default=3
        Minimum number of shared genetic backgrounds for a mutation to contribute
        (passed to :func:`idiosyncratic_index`).

    Returns
    -------
    float
        The overall idiosyncratic index (average across all mutations).

    References
    ----------
    .. [1] Daniel M. Lyons et al, "Idiosyncratic epistasis creates universals in mutational
       effects and evolutionary trajectories", Nat. Ecol. Evo., 2020.
    """
    data = landscape.get_data()
    X = data[list(landscape.data_types.keys())]
    f = data["fitness"].to_numpy(dtype=float)

    # Flat landscape: mirror idiosyncratic_index (every mutation 0.0) -> avg 0.0, not NaN.
    if len(f) <= 1 or np.all(f == f[0]):
        return _pythonize(0.0)
    std_baseline = float(np.sqrt(2.0) * np.std(f))

    # Sorted-allele codes (match original sorted iteration); memmapped to workers.
    Xcodes = np.column_stack(
        [pd.Categorical(X[c], categories=sorted(X[c].unique())).codes for c in X.columns]
    ).astype(np.int32)

    # Process-based (loky) parallelism over POSITIONS: each task computes all of a
    # position's allele-pair indices from a single shared background grouping
    # (vs. regrouping once per allele pair), then we flatten.
    per_pos = Parallel(n_jobs=n_jobs)(
        delayed(_idiosyncratic_position_worker)(Xcodes, f, std_baseline, j, min_pairs)
        for j in range(Xcodes.shape[1])
    )
    values = [v for sub in per_pos for v in sub]
    # Too-few-background mutations are NaN and excluded from the mean (padding
    # 0.0 would bias sparse landscapes toward zero idiosyncrasy).
    if not values or np.all(np.isnan(values)):
        return _pythonize(np.nan)
    return _pythonize(float(np.nanmean(values)))


def diminishing_returns_index(
    landscape,
    method: Literal["pearson", "spearman", "regression"] = "pearson",
) -> float:
    """Measures diminishing returns epistasis in a fitness landscape.

    Diminishing returns epistasis occurs when the fitness benefit of new
    beneficial mutations decreases as the background fitness increases. This
    function quantifies this trend by calculating the correlation between the
    fitness of each genotype (node) and the average fitness improvement
    provided by its direct successors (fitter one-mutant neighbors). A
    significant negative correlation indicates diminishing returns.

    Parameters
    ----------
    landscape : Landscape
        An initialized and built fitness landscape object. The landscape graph
        must have a 'fitness' attribute for each node.
    method : {'pearson', 'spearman', 'regression'}, default='pearson'
        The method used to calculate the diminishing returns index.
        'pearson' for Pearson correlation coefficient,
        'spearman' for Spearman rank correlation coefficient,
        'regression' for the slope of a linear regression.

    Returns
    -------
    correlation_or_slope : float
        For 'pearson' or 'spearman': The correlation coefficient between node fitness
        and average successor fitness improvement.
        For 'regression': The slope of the linear regression.
        Returns NaN if calculation is not possible.

    Raises
    ------
    RuntimeError
        If the landscape object has not been built.
    ValueError
        If the graph is missing or the 'fitness' attribute is not found.
        If the correlation method is invalid.
    """
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found."
            " Landscape must be built first."
        )

    # Mean improvement toward the optimum across each node's improving out-edges.
    # `delta_fit` is |Δfitness| -- the positive improvement magnitude on every
    # improving edge (both maximize and minimize) -- so the per-node mean is simply
    # the delta_fit-weighted out-strength / out-degree: one C-level pass, no
    # edge-list materialisation (fast and memory-light). Fallback recomputes from
    # fitness via the sparse adjacency when delta_fit is absent. NaN = local optima.
    fitness = np.asarray(landscape.graph.vs["fitness"], dtype=float)
    node_fitnesses = fitness
    outdeg = np.asarray(landscape.graph.outdegree(), dtype=float)
    if "delta_fit" in landscape.graph.es.attributes():
        per_node = np.asarray(
            landscape.graph.strength(mode="out", weights="delta_fit"), dtype=float
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_successor_improvement = np.where(outdeg > 0, per_node / outdeg, np.nan)
    else:
        mean_succ_fit = landscape.graph.get_adjacency_sparse().dot(fitness)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_succ_fit = np.where(outdeg > 0, mean_succ_fit / outdeg, np.nan)
        avg_successor_improvement = (
            mean_succ_fit - fitness if landscape.maximize
            else fitness - mean_succ_fit
        )
    nodes_with_successors = int(np.count_nonzero(outdeg > 0))

    if nodes_with_successors < 2:
        warnings.warn(
            "Not enough nodes with successors to calculate correlation for diminishing returns.",
            UserWarning,
        )
        return np.nan

    node_fitnesses_series = pd.Series(node_fitnesses)
    avg_improvement_series = pd.Series(avg_successor_improvement)

    mask = ~avg_improvement_series.isna()
    if mask.sum() < 2:
        warnings.warn(
            "Not enough valid data points after NaN omission to calculate correlation.",
            UserWarning,
        )
        return np.nan
    node_fitnesses = node_fitnesses_series[mask]
    avg_improvement = avg_improvement_series[mask]

    if method == "pearson":
        corr_func = pearsonr
    elif method == "spearman":
        corr_func = spearmanr
    elif method == "regression":
        try:
            X = np.array(node_fitnesses).reshape(-1, 1)
            y = np.array(avg_improvement)

            X_with_const = np.column_stack((np.ones(X.shape[0]), X))  # add intercept

            beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
            slope = beta[1]

            n = len(X)
            if n <= 2:
                return slope

            y_pred = X_with_const.dot(beta)
            residual_SS = np.sum((y - y_pred) ** 2)
            X_mean = np.mean(X)
            X_var = np.sum((X.reshape(-1) - X_mean) ** 2)

            if X_var == 0:
                return slope

            return slope
        except Exception as e:
            warnings.warn(f"Could not calculate regression: {e}", UserWarning)
            return np.nan
    else:
        raise ValueError("Method must be 'pearson', 'spearman', or 'regression'")

    try:
        correlation, _ = corr_func(node_fitnesses, avg_improvement)
        return _pythonize(correlation)
    except Exception as e:
        warnings.warn(f"Could not calculate correlation: {e}", UserWarning)
        return np.nan


def increasing_costs_index(
    landscape,
    method: Literal["pearson", "spearman", "regression"] = "pearson",
) -> float:
    """Measures increasing cost epistasis in a fitness landscape.

    Increasing cost epistasis occurs when the fitness cost (reduction) of
    deleterious mutations increases as the background fitness increases. This
    function quantifies this trend by calculating the correlation between the
    fitness of each genotype (node) and the average fitness cost incurred
    by mutations leading *to* that node from its direct predecessors (less fit
    one-mutant neighbors). A significant positive correlation indicates
    increasing cost.

    Parameters
    ----------
    landscape : Landscape
        An initialized and built fitness landscape object. The landscape graph
        must have a 'fitness' attribute for each node.
    method : {'pearson', 'spearman', 'regression'}, default='pearson'
        The method used to calculate the increasing costs index.
        'pearson' for Pearson correlation coefficient,
        'spearman' for Spearman rank correlation coefficient,
        'regression' for the slope of a linear regression.

    Returns
    -------
    correlation_or_slope : float
        For 'pearson' or 'spearman': The correlation coefficient between node fitness
        and average predecessor fitness cost.
        For 'regression': The slope of the linear regression.
        Returns NaN if calculation is not possible.

    Raises
    ------
    RuntimeError
        If the landscape object has not been built.
    ValueError
        If the graph is missing or the 'fitness' attribute is not found.
        If the correlation method is invalid.
    """
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found."
            " Landscape must be built first."
        )

    # Mirror of diminishing_returns_index over IN-edges: mean cost across each
    # node's improving predecessors. delta_fit is the positive cost magnitude on
    # every improving edge, so the per-node mean is the delta_fit-weighted
    # in-strength / in-degree (fast, memory-light). Fallback via the transposed
    # sparse adjacency when delta_fit is absent. NaN for source nodes.
    fitness = np.asarray(landscape.graph.vs["fitness"], dtype=float)
    node_fitnesses = fitness
    indeg = np.asarray(landscape.graph.indegree(), dtype=float)
    if "delta_fit" in landscape.graph.es.attributes():
        per_node = np.asarray(
            landscape.graph.strength(mode="in", weights="delta_fit"), dtype=float
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_predecessor_cost = np.where(indeg > 0, per_node / indeg, np.nan)
    else:
        mean_pred_fit = landscape.graph.get_adjacency_sparse().T.dot(fitness)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_pred_fit = np.where(indeg > 0, mean_pred_fit / indeg, np.nan)
        avg_predecessor_cost = (
            fitness - mean_pred_fit if landscape.maximize
            else mean_pred_fit - fitness
        )
    nodes_with_predecessors = int(np.count_nonzero(indeg > 0))

    if nodes_with_predecessors < 2:
        warnings.warn(
            "Not enough nodes with predecessors to calculate correlation for increasing cost.",
            UserWarning,
        )
        return np.nan

    node_fitnesses_series = pd.Series(node_fitnesses)
    avg_cost_series = pd.Series(avg_predecessor_cost)

    mask = ~avg_cost_series.isna()
    if mask.sum() < 2:
        warnings.warn(
            "Not enough valid data points after NaN omission to calculate correlation.",
            UserWarning,
        )
        return np.nan
    node_fitnesses = node_fitnesses_series[mask]
    avg_cost = avg_cost_series[mask]

    if method == "pearson":
        corr_func = pearsonr
    elif method == "spearman":
        corr_func = spearmanr
    elif method == "regression":
        try:
            X = np.array(node_fitnesses).reshape(-1, 1)
            y = np.array(avg_cost)

            X_with_const = np.column_stack((np.ones(X.shape[0]), X))  # add intercept

            beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
            slope = beta[1]

            return slope
        except Exception as e:
            warnings.warn(f"Could not calculate regression: {e}", UserWarning)
            return np.nan
    else:
        raise ValueError("Method must be 'pearson', 'spearman', or 'regression'")

    try:
        correlation, _ = corr_func(node_fitnesses, avg_cost)
        return _pythonize(correlation)
    except Exception as e:
        warnings.warn(f"Could not calculate correlation: {e}", UserWarning)
        return np.nan


def _gamma_statistics(landscape, n_jobs=-1):
    """Calculate both gamma statistics for internal reuse."""
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found."
            " Landscape must be built first."
        )

    df = landscape.get_data()
    X = df[list(landscape.data_types.keys())]

    if landscape.n_vars < 2:
        warnings.warn(
            "Gamma statistics require at least 2 variables so that fitness "
            f"effects of one mutation can be compared; this landscape has "
            f"{landscape.n_vars}. Returning NaN.",
            UserWarning,
        )
        return {"gamma": np.nan, "gamma_star": np.nan}

    # Appearance-order codes (match original df[pos].unique() iteration); memmapped to workers.
    f = df["fitness"].to_numpy(dtype=float)
    Xcodes = np.column_stack([pd.factorize(X[c])[0] for c in X.columns]).astype(np.int32)
    P = Xcodes.shape[1]
    alleles = [np.unique(Xcodes[:, j]) for j in range(P)]
    position_pairs = [(p1, p2) for p1 in range(P) for p2 in range(P) if p1 != p2]

    # Process-based (loky) parallelism over ordered position pairs; thread backend was GIL-bound.
    results = Parallel(n_jobs=n_jobs)(
        delayed(_gamma_position_pair_worker)(
            Xcodes, f, p1, p2, alleles[p1], alleles[p2], np.delete(np.arange(P), [p1, p2])
        )
        for p1, p2 in position_pairs
    )

    # Pool over all ordered pairs (both orderings cover both square sides) into
    # the single global non-centered correlation of Ferretti et al. (2016).
    num = sum(r[0] for r in results)
    den = sum(r[1] for r in results)
    snum = sum(r[2] for r in results)
    sden = sum(r[3] for r in results)

    return {
        "gamma": num / den if den else np.nan,
        "gamma_star": snum / sden if sden else np.nan,
    }


def gamma_statistic(landscape, n_jobs=-1):
    """
    Calculates the gamma and gamma_star statistics for a fitness landscape.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object containing fitness data.
    n_jobs : int, optional
        Number of parallel jobs to use. Default is -1 (all available cores).

    Returns
    -------
    float
        The traditional gamma statistic value. Values close to -1 or 1 indicate
        strong epistatic interactions in magnitude, while values close to 0 indicate
        weak or no epistasis.

    Notes
    -----
    - The gamma statistic measures the correlation between fitness effects of mutations
      across different genetic backgrounds, providing a measure of epistatic interactions
      in the landscape.
    - It is computed as the non-centered (raw second-moment) correlation of fitness
      effects pooled over all square motifs, following eq. (3) of Ferretti et al.
      (2016). Hence a purely additive landscape gives gamma = 1, a House-of-Cards
      landscape gives gamma ~ 0, and a reciprocal-sign-dominated landscape gives
      gamma < 0.
    - The gamma_star statistic focuses only on sign consistency, ignoring the magnitude
      of fitness effects. It indicates whether mutations tend to have consistent
      directional effects across different genetic backgrounds.

    References
    ----------
    .. [1] L. Ferretti et al., "Measuring epistasis in fitness landscapes: The
       correlation of fitness effects of mutations", J. Theor. Biol. 396, 132-143 (2016).
    """

    stats = _gamma_statistics(landscape, n_jobs=n_jobs)
    return _pythonize(stats["gamma"])


def gamma_star(landscape, n_jobs=-1):
    """
    Calculate the gamma-star statistic for a fitness landscape.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object containing fitness data.
    n_jobs : int, optional
        Number of parallel jobs to use. Default is -1 (all available cores).

    Returns
    -------
    float
        The gamma-star statistic that only considers sign consistency.
        Values close to 1 indicate consistent sign epistasis across
        backgrounds, values close to -1 indicate opposing sign patterns,
        and values close to 0 indicate random sign patterns.
    """
    stats = _gamma_statistics(landscape, n_jobs=n_jobs)
    return _pythonize(stats["gamma_star"])


def higher_order_epistasis(landscape, order=2, verbose=False, n_jobs=1):
    """
    Calculates the fraction of variance in fitness that can be explained
    by interactions between variables up to the specified order using polynomial regression.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object to analyze.
    order : int, optional
        The maximum order of polynomial features to consider. This controls the degree
        of the polynomial, where an order of k allows for modeling interactions between
        up to k variables. Must be between 1 and the total number of variables in the landscape.
        Default is 2 (quadratic terms and pairwise interactions).
    verbose : bool, optional
        Whether to print progress information. Default is False.
    n_jobs : int, optional
        Number of CPU cores used by the underlying linear regression. Default is 1.

    Returns
    -------
    float
        The R² score representing the fraction of variance explained by
        polynomial terms up to the specified order. Values closer to 1.0 indicate
        stronger epistasis of the given order.

    Notes
    -----
    This function uses polynomial regression with degree=order to model interactions
    up to the specified order. The resulting R² score indicates how well these
    interactions explain the observed fitness values.

    A high R² score suggests that most of the fitness variance can be
    explained by considering interactions up to the specified order,
    indicating strong epistatic effects of that order in the landscape.

    """
    try:
        from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
    except ImportError:
        raise ImportError(
            "This function requires scikit-learn. "
            "Please install it with 'pip install scikit-learn'."
        )

    landscape._check_built()

    if landscape.configs is None or len(landscape.configs) == 0:
        raise ValueError("Landscape has no configuration data.")

    if not isinstance(order, int):
        raise TypeError(f"Order must be an integer, got {type(order).__name__}")

    if order < 1:
        raise ValueError(f"Order must be at least 1, got {order}")

    if order > landscape.n_vars:
        raise ValueError(
            f"Order cannot exceed the number of variables in the landscape "
            f"({landscape.n_vars}), got {order}"
        )

    if verbose:
        print(f"Calculating order-{order} epistasis using polynomial regression...")

    X = np.vstack(landscape.configs.values)
    y = np.array(landscape.graph.vs["fitness"])

    # Boolean is already 0/1; other types need one-hot with a reference level
    # dropped for a numerically stable design matrix.
    if verbose:
        print(f"Encoding {X.shape[1]} variables...")

    if landscape.kind == "boolean":
        X_encoded = np.asarray(X, dtype=np.float64)
    else:
        encoder = OneHotEncoder(
            sparse_output=False,
            drop="first",
            dtype=np.float64,
        )
        try:
            X_encoded = encoder.fit_transform(X)
        except Exception as e:
            raise ValueError(f"Failed to one-hot encode configurations: {e}")

    if verbose:
        print(f"Encoded data shape: {X_encoded.shape}")
        print(f"Creating polynomial features of degree {order}...")

    # Use interaction-only features and let LinearRegression handle the intercept.
    poly = PolynomialFeatures(
        degree=order,
        include_bias=False,
        interaction_only=True,
    )
    model = LinearRegression(n_jobs=n_jobs)

    try:
        if verbose:
            print(f"Fitting polynomial regression model...")
        X_poly = poly.fit_transform(X_encoded)
        model.fit(X_poly, y)
        # Manual dot instead of np.matmul: Accelerate (macOS arm64) emits
        # spurious RuntimeWarnings on finite inputs.
        coefficients = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        y_pred = (
            np.sum(
                np.asarray(X_poly, dtype=np.float64) * coefficients,
                axis=1,
                dtype=np.float64,
            )
            + float(model.intercept_)
        )
        r2 = r2_score(y, y_pred)
    except Exception as e:
        raise RuntimeError(f"Error fitting polynomial regression model: {e}")

    if verbose:
        print(f"Order-{order} epistasis R² score: {r2:.4f}")

    return _pythonize(r2)


def walsh_hadamard_coefficient(landscape, max_order=2, max_cells=1e9, chunk_size=1000):
    """
    Compute Walsh-Hadamard coefficients for a fitness landscape.

    This function calculates Walsh-Hadamard coefficients for base and interaction terms
    up to a specified order using the ensemble encoding approach from the extended
    Walsh-Hadamard transform. The coefficients quantify the contribution of individual
    mutations and their interactions to the overall fitness.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object containing genotype-fitness data.
    max_order : int, default=2
        Maximum interaction order to consider. Higher orders capture more complex
        epistatic interactions but increase computational cost.
    max_cells : float, default=1e9
        Maximum matrix cells permitted to prevent excessive memory usage during
        interaction feature generation.
    chunk_size : int, default=1000
        Chunk size for H matrix construction to optimize memory usage for large datasets.

    Returns
    -------
    dict
        A dictionary with sorted coefficients organized by interaction order:
        - Keys are integers representing interaction orders (0 for wildtype,
          1 for single mutations, 2 for pairwise interactions, etc.)
        - Values are dictionaries mapping feature names to their coefficients

        Feature names use the format ``{original}_{position}_{mutant}`` for
        single mutations (e.g. ``0_12_1`` = position 12, mutation from 0 to 1).
        Pairwise and higher-order interactions join mutations with ``-``
        (e.g. ``0_10_1-0_11_1`` = interaction between positions 10 and 11).

    Raises
    ------
    RuntimeError
        If the landscape has not been built.
    ValueError
        If memory limit is exceeded during computation or if input data is invalid.

    Notes
    -----
    The Walsh-Hadamard transform provides a complete decomposition of the fitness
    function into additive and epistatic components. Higher-order coefficients
    represent increasingly complex epistatic interactions between mutations.

    Examples
    --------
    >>> # Assuming 'landscape' is a built Landscape object
    >>> coefficients = walsh_hadamard_coefficient(landscape, max_order=3)
    >>> print(f"Wildtype coefficient: {coefficients[0]['WT']}")
    >>> print(f"Single mutation effects: {list(coefficients[1].keys())}")
    >>> print(f"Pairwise interactions: {list(coefficients[2].keys())}")
    """

    landscape._check_built()

    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found. "
            "Landscape must be built first."
        )

    data = landscape.get_data()
    X = data[list(landscape.data_types.keys())]
    f = data["fitness"].values

    # Walsh-Hadamard transform operates on one symbol per position, so encode
    # each landscape type to a per-position string.
    if landscape.kind in ["boolean"]:
        X_strings = ["".join(map(str, row.astype(int))) for _, row in X.iterrows()]
    elif landscape.kind in ["dna", "rna", "protein"]:
        if hasattr(landscape, "configs") and landscape.configs is not None:
            # reconstruct original sequences from encoded configs
            X_strings = []
            for config_tuple in landscape.configs.values:
                if landscape.kind == "dna":
                    alphabet = ["A", "C", "G", "T"]
                elif landscape.kind == "rna":
                    alphabet = ["A", "C", "G", "U"]
                else:  # protein
                    alphabet = list("ACDEFGHIKLMNPQRSTVWY")

                sequence = "".join([alphabet[int(pos)] for pos in config_tuple])
                X_strings.append(sequence)
        else:
            # fallback: treat as categorical
            X_strings = ["".join(map(str, row)) for _, row in X.iterrows()]
    else:
        # One symbol per variable so multi-digit codes don't split across positions;
        # +1 so no code collides with "0", which marks WT-matching positions downstream.
        codes = X.apply(lambda col: pd.factorize(col)[0] + 1).to_numpy()
        X_strings = ["".join(chr(48 + c) for c in row) for row in codes]

    if landscape.kind == "boolean":
        wildtype = "0" * landscape.n_vars
    else:
        wildtype = X_strings[0]  # first sequence is the reference WT

    wildtype_split = [c for c in wildtype]

    X_df = pd.DataFrame([list(seq) for seq in X_strings])

    enc = OneHotEncoder(
        handle_unknown="ignore", drop=np.array(wildtype_split), dtype=int
    )
    enc.fit(X_df)

    # Feature names "{original}_{position}_{mutant}", e.g. "0_12_1" (pos 12, 0->1).
    one_hot_names = []
    for i, feature_name in enumerate(enc.get_feature_names_out()):
        pos = int(feature_name.split("_")[0][1:])
        state = feature_name.split("_")[1]
        one_hot_names.append(f"{wildtype_split[pos]}_{pos+1}_{state}")

    Xoh = pd.DataFrame(enc.transform(X_df).toarray(), columns=one_hot_names)

    Xoh = pd.concat([pd.DataFrame({"WT": [1] * len(Xoh)}), Xoh], axis=1)

    Xohi = _generate_interactions(Xoh, max_order, max_cells)

    Xensemble = _ensemble_encode_features(
        X_strings, Xohi.columns, wildtype, X_df, chunk_size
    )

    # Direct least-squares solve, not normal equations: more stable and avoids
    # spurious macOS arm64 Accelerate matmul warnings from pinv(X^T X).
    Xensemble_values = Xensemble.to_numpy(dtype=np.float64, copy=False)
    coefficients, *_ = np.linalg.lstsq(
        Xensemble_values,
        np.asarray(f, dtype=np.float64),
        rcond=None,
    )

    coef_dict = {}
    for i, feature_name in enumerate(Xohi.columns):
        if feature_name == "WT":
            order = 0
        elif "-" in feature_name:
            order = len(feature_name.split("-"))  # "-"-joined mutations
        else:
            order = 1

        if order not in coef_dict:
            coef_dict[order] = {}

        coef_dict[order][feature_name] = coefficients[i]

    for order in coef_dict:
        coef_dict[order] = dict(sorted(coef_dict[order].items()))

    return coef_dict


def _generate_interactions(Xoh, max_order, max_cells):
    """Generate interaction features up to max_order with memory optimization."""
    if max_order < 2:
        return copy.deepcopy(Xoh)

    # observed mutations (drop WT and unobserved columns)
    mut_count = list(Xoh.sum(axis=0))
    pheno_mut = [
        Xoh.columns[i]
        for i in range(len(Xoh.columns))
        if mut_count[i] != 0 and Xoh.columns[i] != "WT"
    ]

    # group mutations by position; name format original_position_mutant ("0_12_1")
    def _get_position(mut_name):
        return mut_name.split("_")[1]

    all_pos = list(set([_get_position(i) for i in pheno_mut]))
    all_pos_mut = {p: [j for j in pheno_mut if _get_position(j) == p] for p in all_pos}

    # enumerate all theoretical interaction features
    all_features = {}
    int_order_dict = {}

    for n in range(2, max_order + 1):
        all_features[n] = []
        pos_comb = list(itertools.combinations(sorted(all_pos_mut.keys(), key=int), n))
        for p in pos_comb:
            all_features[n] += [
                "-".join(c) for c in itertools.product(*[all_pos_mut[j] for j in p])
            ]
        int_order_dict[n] = len(all_features[n])

    print(
        "... Total theoretical features (order:count): "
        + ", ".join(
            [
                str(i) + ":" + str(int_order_dict[i])
                for i in sorted(int_order_dict.keys())
            ]
        )
    )

    all_features_flat = list(itertools.chain(*list(all_features.values())))

    int_list = []
    int_list_names = []
    int_order_dict_retained = {}

    for c in all_features_flat:
        c_split = c.split("-")  # "-"-joined mutations, e.g. "0_10_1-0_11_1"
        int_col = (Xoh.loc[:, c_split].sum(axis=1) == len(c_split)).astype(int)

        if sum(int_col) >= 0:  # min-observation threshold kept at 0 as in original
            int_list.append(int_col)
            int_list_names.append(c)

            order = len(c_split)
            if order not in int_order_dict_retained:
                int_order_dict_retained[order] = 1
            else:
                int_order_dict_retained[order] += 1

        if len(int_list) * len(Xoh) > max_cells:
            print(
                f"Error: Too many interaction terms: number of feature matrix cells >{max_cells:>.0e}"
            )
            raise ValueError("Memory limit exceeded")

    print(
        "... Total retained features (order:count): "
        + ", ".join(
            [
                str(i)
                + ":"
                + str(int_order_dict_retained[i])
                + " ("
                + str(round(int_order_dict_retained[i] / int_order_dict[i] * 100, 1))
                + "%)"
                for i in sorted(int_order_dict_retained.keys())
            ]
        )
    )

    if len(int_list) > 0:
        Xint = pd.concat(int_list, axis=1)
        Xint.columns = int_list_names
        # restore original feature ordering
        Xint = Xint.loc[:, [i for i in all_features_flat if i in Xint.columns]]
        Xohi = pd.concat([Xoh, Xint], axis=1)
    else:
        Xohi = copy.deepcopy(Xoh)

    return Xohi


def _ensemble_encode_features(X, feature_names, wildtype, X_df, chunk_size):
    """Ensemble encode features using Walsh-Hadamard transform with chunking optimization."""

    # mask each variant against WT: positions matching WT become "0"
    geno_list = []
    for seq in X:
        masked = "".join(x if x != y else "0" for x, y in zip(seq, wildtype))
        geno_list.append(masked)

    coef_list = [
        _coefficient_to_sequence(coef, len(wildtype)) for coef in feature_names
    ]

    # number of observed states per position
    state_counts = X_df.apply(lambda col: col.value_counts(), axis=0)
    state_list = [(state_counts[col] > 0).sum() for col in state_counts.columns]

    print("Construction time for H_matrix...")
    hmat_inv = _H_matrix_chunker(
        str_geno=geno_list,
        str_coef=coef_list,
        num_states=state_list,
        invert=True,
        chunk_size=chunk_size,
    )

    vmat_inv = _V_matrix(str_coef=coef_list, num_states=state_list, invert=True)

    # V is diagonal: H @ V == column-scaling H, exactly. Avoids warning-prone
    # np.matmul on macOS arm64 Accelerate.
    return pd.DataFrame(hmat_inv * np.diag(vmat_inv), columns=feature_names)


def _coefficient_to_sequence(coefficient, length):
    """Convert coefficient string to sequence representation.

    Expects format "original_position_mutant" (e.g. "0_12_1") or multiple joined
    with "-" (e.g. "0_10_1-0_11_1" for pairwise).
    """
    coefficient_seq = ["0"] * length

    if coefficient == "WT":
        return "".join(coefficient_seq)

    for mut in coefficient.split("-"):
        parts = mut.split("_")
        if len(parts) >= 3:
            _orig, pos_str, state = parts[0], parts[1], parts[2]
            pos = int(pos_str) - 1  # 1-indexed to 0-indexed
            if 0 <= pos < length:
                coefficient_seq[pos] = state

    return "".join(coefficient_seq)


def _H_matrix_chunker(str_geno, str_coef, num_states=2, invert=False, chunk_size=1000):
    """Construct Walsh-Hadamard matrix in chunks (memory optimization)."""
    if len(str_geno) < chunk_size:
        return _H_matrix(str_geno, str_coef, num_states, invert)

    hmat_list = []
    for i in range(math.ceil(len(str_geno) / chunk_size)):
        from_i = i * chunk_size
        to_i = min((i + 1) * chunk_size, len(str_geno))
        hmat_list.append(_H_matrix(str_geno[from_i:to_i], str_coef, num_states, invert))

    return np.concatenate(hmat_list, axis=0)


def _H_matrix(str_geno, str_coef, num_states=2, invert=False):
    """Construct Walsh-Hadamard matrix."""
    string_length = len(str_geno[0])

    if isinstance(num_states, int):
        num_states = [float(num_states)] * string_length
    else:
        num_states = [float(i) for i in num_states]

    # to numeric (ord codes); "0" -> "." in coefs to mark unconstrained positions
    str_coef_num = [[ord(j) for j in i.replace("0", ".")] for i in str_coef]
    str_geno_num = [[ord(j) for j in i] for i in str_geno]

    num_statesi = np.repeat([num_states], len(str_geno) * len(str_coef), axis=0)
    str_genobi = np.repeat(str_geno_num, len(str_coef), axis=0)
    str_coefbi = np.transpose(
        np.tile(np.transpose(np.asarray(str_coef_num)), len(str_geno))
    )

    str_genobi_eq_str_coefbi = str_genobi == str_coefbi
    row_factor2 = str_genobi_eq_str_coefbi.sum(axis=1)

    if invert:
        row_factor1 = np.prod(str_genobi_eq_str_coefbi * (num_statesi - 2) + 1, axis=1)
        return (row_factor1 * np.power(-1, row_factor2) / np.prod(num_states)).reshape(
            (len(str_geno), -1)
        )
    else:
        row_factor1 = (
            np.logical_or(
                np.logical_or(str_genobi_eq_str_coefbi, str_genobi == ord("0")),
                str_coefbi == ord("."),
            ).sum(axis=1)
            == string_length
        ).astype(float)
        return (row_factor1 * np.power(-1, row_factor2)).reshape((len(str_geno), -1))


def _V_matrix(str_coef, num_states=2, invert=False):
    """Construct diagonal weighting matrix."""
    string_length = len(str_coef[0])

    if isinstance(num_states, int):
        num_states = [float(num_states)] * string_length
    else:
        num_states = [float(i) for i in num_states]

    str_coef_dot = [i.replace("0", ".") for i in str_coef]
    V = np.zeros((len(str_coef), len(str_coef)))

    for i in range(len(str_coef)):
        factor1 = int(
            np.prod(
                [
                    c
                    for a, b, c in zip(str_coef_dot[i], str_coef[i], num_states)
                    if ord(a) != ord(b)
                ]
            )
        )
        factor2 = sum(
            [1 for a, b in zip(str_coef_dot[i], str_coef[i]) if ord(a) == ord(b)]
        )

        if invert:
            V[i, i] = factor1 * np.power(-1, factor2)
        else:
            V[i, i] = 1 / (factor1 * np.power(-1, factor2))

    return V


def extradimensional_bypass_analysis(landscape, approximate=False, sample_cut_prob=0.2, seed=None):
    """
    Analyzes extradimensional bypasses in reciprocal sign epistasis motifs.

    For each motif representing reciprocal sign epistasis (type 19), this function
    identifies whether accessible evolutionary paths exist that bypass the direct
    path between the double mutant nodes. Such indirect paths are called
    extradimensional bypasses and allow evolution to traverse fitness valleys
    that would otherwise be inaccessible under strong selection.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object, containing landscape.graph as an igraph.Graph
        with a "fitness" vertex attribute.
    approximate : bool, optional
        If True, uses sampling to find motif instances. Faster but less accurate.
        Defaults to False (exact enumeration of all instances).
    sample_cut_prob : float, optional
        The probability used for pruning the search tree at each level during
        sampling when approximate=True. Higher values -> faster, less accurate.
        Defaults to 0.2.

    Returns
    -------
    dict
        A dictionary containing:
        - "bypass_proportion": The proportion of reciprocal sign epistasis motifs
          for which an extradimensional bypass exists (float between 0 and 1).
        - "average_bypass_length": The average length of extradimensional bypasses
          for motifs where such bypasses exist. Returns NaN if no bypasses exist.
        - "total_motifs": Total number of type 19 motifs analyzed.
        - "motifs_with_bypass": Number of motifs that have extradimensional bypasses.

    Raises
    ------
    AttributeError
        If landscape.graph is not an igraph.Graph object or does not exist.
    ValueError
        If sample_cut_prob is not between 0 and 1, or if fitness attribute missing.

    Notes
    -----
    Reciprocal sign epistasis occurs when both the wildtype (ab) and double mutant (AB)
    have higher fitness than both single mutants (aB, Ab). This creates a fitness valley
    that prevents direct evolutionary access between ab and AB. Extradimensional bypasses
    are indirect paths through the broader fitness landscape that circumvent this valley.
    """

    # --- Validate Input ---
    if not hasattr(landscape, "graph") or not isinstance(landscape.graph, ig.Graph):
        raise AttributeError(
            "Input 'landscape' must have a 'graph' attribute that is an igraph.Graph object."
        )
    if "fitness" not in landscape.graph.vs.attributes():
        raise ValueError("igraph.Graph must have a 'fitness' vertex attribute.")
    if approximate and not 0.0 <= sample_cut_prob <= 1.0:
        raise ValueError("sample_cut_prob must be between 0.0 and 1.0")

    # --- Find Type 19 Motifs (Reciprocal Sign Epistasis) ---
    try:
        motif_19_instances = get_motif_node_indices(
            landscape.graph,
            motif_size=4,
            target_motif_type=19,
            approximate=approximate,
            sample_cut_prob=sample_cut_prob,
            seed=seed,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to find motif instances: {e}")

    if not motif_19_instances:
        return _pythonize({
            "bypass_proportion": 0.0,
            "average_bypass_length": np.nan,
            "total_motifs": 0,
            "motifs_with_bypass": 0,
        })

    # --- Analyze Each Motif for Extradimensional Bypasses ---
    total_motifs = len(motif_19_instances)

    # Identify (ab, AB) per motif from a precomputed fitness array, caching each
    # AB's predecessor set (one high-fitness node anchors many reciprocal-sign
    # squares). AB = fittest node (first on ties, matching the previous dict-max);
    # ab = first remaining node that is not a direct predecessor of AB.
    fit = np.asarray(landscape.graph.vs["fitness"], dtype=float)
    pred_cache = {}
    ABs_by_ab = defaultdict(list)
    for motif_nodes in motif_19_instances:
        nodes = list(motif_nodes)
        AB = nodes[int(np.argmax(fit[nodes]))]
        pred = pred_cache.get(AB)
        if pred is None:
            pred = pred_cache[AB] = set(landscape.graph.predecessors(AB))
        ab = next((n for n in nodes if n != AB and n not in pred), None)
        if ab is None:
            continue
        ABs_by_ab[ab].append(AB)

    # One forward traversal per UNIQUE ab -- the SAME improving (OUT) direction the
    # original used from the wildtype (whose descendant set is typically far smaller
    # than the double-mutant's ancestor set, so going from AB instead would be
    # slower), now covering all of that ab's motifs in a single BFS. igraph forbids
    # duplicate targets, so query distinct ABs once and count every motif occurrence
    # (approximate sampling can repeat a motif; the original processed the list
    # element-wise). A finite distance means an extradimensional bypass exists.
    bypass_lengths = []
    motifs_with_bypass = 0
    for ab, AB_list in ABs_by_ab.items():
        uniq_ABs = list(dict.fromkeys(AB_list))
        row = landscape.graph.distances(source=[ab], target=uniq_ABs, mode="out")[0]
        dist_map = dict(zip(uniq_ABs, row))
        for AB in AB_list:
            distance = dist_map[AB]
            if not np.isinf(distance):
                bypass_lengths.append(distance)
                motifs_with_bypass += 1

    # --- Calculate Results ---
    bypass_proportion = motifs_with_bypass / total_motifs if total_motifs > 0 else 0.0
    average_bypass_length = np.mean(bypass_lengths) if bypass_lengths else np.nan

    return _pythonize({
        "bypass_proportion": bypass_proportion,
        "average_bypass_length": average_bypass_length,
        "total_motifs": total_motifs,
        "motifs_with_bypass": motifs_with_bypass,
    })


def get_motif_node_indices(
    graph, motif_size=4, target_motif_type=19, approximate=False, sample_cut_prob=0.2,
    seed=None,
):
    """
    Find all instances of a specific motif type and return their node indices.

    Parameters
    ----------
    graph : igraph.Graph
        The igraph object to search for motifs
    motif_size : int
        Size of motifs to search for (default 4)
    target_motif_type : int
        The specific motif ID to collect (e.g., 19, 52, 66)
    approximate : bool, optional
        If True, uses sampling to find motif instances. Faster but less accurate.
        Defaults to False (exact enumeration of all instances).
    sample_cut_prob : float, optional
        The probability used for pruning the search tree at each level during
        sampling when approximate=True. Higher values -> faster, less accurate.
        Defaults to 0.2.

    Returns
    -------
    list
        List of tuples, where each tuple contains the node indices
        for one instance of the target motif

    Raises
    ------
    ValueError
        If sample_cut_prob is not between 0 and 1.
    """
    if approximate and not 0.0 <= sample_cut_prob <= 1.0:
        raise ValueError("sample_cut_prob must be between 0.0 and 1.0")

    collected_motifs = []
    cut_prob_vector = [sample_cut_prob] * motif_size if approximate else None

    def motif_collector_callback(graph, vertices, isoclass):
        if isoclass == target_motif_type:
            collected_motifs.append(tuple(sorted(vertices)))
        return False  # continue search

    # Find motifs with or without sampling (seeded RNG only matters for the
    # approximate, sampling-based path).
    with _seeded_igraph(seed if approximate else None):
        if approximate:
            graph.motifs_randesu(
                size=motif_size, cut_prob=cut_prob_vector, callback=motif_collector_callback
            )
        else:
            graph.motifs_randesu(size=motif_size, callback=motif_collector_callback)

    return collected_motifs
