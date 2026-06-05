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
def _idiosyncratic_pair_worker(Xcodes, f, std_baseline, pos_idx, allele_a, allele_b,
                               min_pairs, other):
    """Idiosyncratic index for one (allele_a, pos, allele_b) mutation.

    Numpy/dict reimplementation of :func:`idiosyncratic_index`'s core, giving
    identical values: keep-first dedup per background, analytic random-pair
    baseline, and NaN for degenerate (too-few-background) mutations so that
    unmeasurable mutations are excluded from — rather than diluting — the
    landscape average.
    """
    rows_a = np.flatnonzero(Xcodes[:, pos_idx] == allele_a)
    rows_b = np.flatnonzero(Xcodes[:, pos_idx] == allele_b)
    if rows_a.size == 0 or rows_b.size == 0:
        return np.nan
    bg_a = {}
    for i in rows_a:
        k = Xcodes[i, other].tobytes()
        if k not in bg_a:
            bg_a[k] = f[i]
    bg_b = {}
    for i in rows_b:
        k = Xcodes[i, other].tobytes()
        if k not in bg_b:
            bg_b[k] = f[i]
    common = bg_a.keys() & bg_b.keys()
    if len(common) < min_pairs:
        return np.nan
    eff = np.fromiter((bg_b[k] - bg_a[k] for k in common), dtype=float, count=len(common))
    return float(np.std(eff) / std_baseline)


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


def _assign_roles_for_epistasis_igraph(graph, squares):
    """Assigns roles within collected square motif instances."""
    squares_with_roles = []
    if "fitness" not in graph.vs.attributes():
        raise ValueError(
            "igraph.Graph must have a 'fitness' vertex attribute for role assignment."
        )

    for square_nodes in squares:
        if len(square_nodes) != 4:
            continue  # Should already be filtered, but double-check

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
                continue  # Skip squares not matching expected structure

            wild_type_set = square_set - set(single_mutants) - {double_mutant}
            if len(wild_type_set) != 1:
                continue  # Skip if WT cannot be uniquely identified
            wild_type = list(wild_type_set)[0]

            single_mutants.sort()  # Consistent ordering

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
            # Silently skip squares with missing data from role assignment
            continue

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


def classify_epistasis(landscape, approximate=False, sample_cut_prob=0.2):
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
    square_indices = {19, 52, 66}  # Use set for faster checking in callback

    # --- Validate Input ---
    if not hasattr(landscape, "graph") or not isinstance(landscape.graph, ig.Graph):
        raise AttributeError(
            "Input 'landscape' must have a 'graph' attribute that is an igraph.Graph object."
        )
    if "fitness" not in landscape.graph.vs.attributes():
        raise ValueError("igraph.Graph must have a 'fitness' vertex attribute.")
    if approximate and not 0.0 <= sample_cut_prob <= 1.0:
        raise ValueError("sample_cut_prob must be between 0.0 and 1.0")

    # --- Data Structures ---
    collected_square_instances = defaultdict(list)  # Stores vertex tuples for squares
    cut_prob_vector = [sample_cut_prob] * motif_size if approximate else None

    # --- Step 1 & 3 Combined (Motif Finding & Instance Collection) ---
    if approximate:
        # Run 1: Get estimated counts for mag/sign/recip calculation
        estimated_motif_counts = landscape.graph.motifs_randesu(
            size=motif_size, cut_prob=cut_prob_vector
        )

        # Define callback for collecting sampled instances
        def motif_collector_callback_approx(graph, vertices, isoclass):
            if isoclass in square_indices:
                collected_square_instances[isoclass].append(tuple(sorted(vertices)))
            return False  # Continue search

        # Run 2: Collect a *sample* of square instances
        landscape.graph.motifs_randesu(
            size=motif_size,
            cut_prob=cut_prob_vector,
            callback=motif_collector_callback_approx,
        )

        # Use estimated counts for mag/sign/recip proportions
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

    else:  # Exact calculation
        # Define callback for collecting all instances
        def motif_collector_callback_exact(graph, vertices, isoclass):
            if isoclass in square_indices:
                # Store the vertex indices, sorting is optional but good for consistency
                collected_square_instances[isoclass].append(tuple(sorted(vertices)))
            return False  # Continue search

        # Run 1: Collect all square instances
        landscape.graph.motifs_randesu(
            size=motif_size, callback=motif_collector_callback_exact
        )

        # Derive exact counts from collected instances
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
    X = data.iloc[:, : landscape.n_vars]
    f = data["fitness"]

    # Check if alleles A and B exist at the specified position
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

    # Random genotype pairs are differences of two i.i.d. draws from the fitness
    # distribution, so Var(diff) = 2*Var(f) exactly. Using this closed form
    # (rather than one noisy n_pairs-sized sample) keeps the index deterministic
    # and avoids a near-zero denominator blowing the ratio up.
    std_mutation_effect = np.std(mutation_effects)
    std_random_diff = np.sqrt(2.0) * np.std(all_fitness_values)

    idiosyncratic_val = std_mutation_effect / std_random_diff

    return _pythonize(idiosyncratic_val)


def global_idiosyncratic_index(landscape, n_jobs=-1, random_seed=None, min_pairs: int = 3):
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
    random_seed : int, optional
        Retained for backward compatibility. The index is now computed
        deterministically (analytic random-pair baseline), so this has no effect.
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
    X = data.iloc[:, : landscape.n_vars]
    f = data["fitness"].to_numpy(dtype=float)

    # Degenerate (flat) landscape: every mutation contributes 0.0, as in
    # idiosyncratic_index. Mirror that so the average is 0.0, not NaN.
    if len(f) <= 1 or np.all(f == f[0]):
        return _pythonize(0.0)
    std_baseline = float(np.sqrt(2.0) * np.std(f))

    # Code genotypes once (sorted-allele codes, matching the original sorted
    # iteration); the arrays are memmapped to worker processes by joblib.
    Xcodes = np.column_stack(
        [pd.Categorical(X[c], categories=sorted(X[c].unique())).codes for c in X.columns]
    ).astype(np.int32)

    tasks = []
    for j in range(Xcodes.shape[1]):
        other = np.delete(np.arange(Xcodes.shape[1]), j)
        alleles = np.unique(Xcodes[:, j])
        for ai in range(len(alleles)):
            for bi in range(ai + 1, len(alleles)):
                tasks.append((int(alleles[ai]), int(alleles[bi]), j, other))

    # Process-based parallelism (default loky backend) actually uses the cores,
    # unlike the previous thread backend which was GIL-bound for this work.
    values = Parallel(n_jobs=n_jobs)(
        delayed(_idiosyncratic_pair_worker)(Xcodes, f, std_baseline, j, a, b, min_pairs, other)
        for (a, b, j, other) in tasks
    )
    # Mutations with too few shared backgrounds return NaN and are excluded from
    # the average (rather than padded with 0.0, which would bias sparse
    # landscapes toward zero idiosyncrasy).
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
    landscape._check_built()  # Ensure landscape is built
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found."
            " Landscape must be built first."
        )

    node_fitnesses = []
    avg_successor_improvement = []

    nodes_with_successors = 0
    for v in landscape.graph.vs:
        current_fitness = v["fitness"]
        node_fitnesses.append(current_fitness)

        successors = v.successors()
        if successors:
            improvements = [s["fitness"] - current_fitness for s in successors]
            # Filter out non-positive improvements (should not happen with current graph def)
            positive_improvements = [imp for imp in improvements if imp > 0]
            if positive_improvements:
                avg_improvement = np.mean(positive_improvements)
                avg_successor_improvement.append(avg_improvement)
                nodes_with_successors += 1
            else:
                # Node might be LO or only have neutral/deleterious successors
                # (should not happen with graph definition, but handle defensively)
                avg_successor_improvement.append(np.nan)
        else:
            # Node is a local optimum (no successors)
            avg_successor_improvement.append(np.nan)

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
            # Add regression method using numpy's polyfit
            X = np.array(node_fitnesses).reshape(-1, 1)
            y = np.array(avg_improvement)

            # Add a constant (intercept) to the predictor matrix
            X_with_const = np.column_stack((np.ones(X.shape[0]), X))

            # Fit linear regression
            beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
            slope = beta[1]  # Slope is the coefficient for X

            # Calculate p-value for the slope
            n = len(X)
            if n <= 2:
                return slope

            # Calculate standard error of the slope
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
    landscape._check_built()  # Ensure landscape is built
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found."
            " Landscape must be built first."
        )

    node_fitnesses = []
    avg_predecessor_cost = []

    nodes_with_predecessors = 0
    for v in landscape.graph.vs:
        current_fitness = v["fitness"]
        node_fitnesses.append(current_fitness)

        predecessors = v.predecessors()
        if predecessors:
            costs = [current_fitness - p["fitness"] for p in predecessors]
            # Filter out non-positive costs (should not happen with graph def)
            positive_costs = [c for c in costs if c > 0]
            if positive_costs:
                avg_cost = np.mean(positive_costs)
                avg_predecessor_cost.append(avg_cost)
                nodes_with_predecessors += 1
            else:
                # Node might be source or only have fitter/equal predecessors
                # (should not happen with graph definition, but handle defensively)
                avg_predecessor_cost.append(np.nan)
        else:
            # Node is a source node (no predecessors)
            avg_predecessor_cost.append(np.nan)

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
            # Add regression method using numpy's polyfit
            X = np.array(node_fitnesses).reshape(-1, 1)
            y = np.array(avg_cost)

            # Add a constant (intercept) to the predictor matrix
            X_with_const = np.column_stack((np.ones(X.shape[0]), X))

            # Fit linear regression
            beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
            slope = beta[1]  # Slope is the coefficient for X

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
    X = df.iloc[:, : landscape.n_vars]

    if landscape.n_vars < 2:
        warnings.warn(
            "Gamma statistics require at least 2 variables so that fitness "
            f"effects of one mutation can be compared; this landscape has "
            f"{landscape.n_vars}. Returning NaN.",
            UserWarning,
        )
        return {"gamma": np.nan, "gamma_star": np.nan}

    # Code genotypes once (appearance-order codes, matching the original
    # df[pos].unique() iteration). Arrays are memmapped to worker processes.
    f = df["fitness"].to_numpy(dtype=float)
    Xcodes = np.column_stack([pd.factorize(X[c])[0] for c in X.columns]).astype(np.int32)
    P = Xcodes.shape[1]
    alleles = [np.unique(Xcodes[:, j]) for j in range(P)]
    position_pairs = [(p1, p2) for p1 in range(P) for p2 in range(P) if p1 != p2]

    # Process-based parallelism over ordered position pairs (default loky
    # backend uses the cores; the previous thread backend was GIL-bound).
    results = Parallel(n_jobs=n_jobs)(
        delayed(_gamma_position_pair_worker)(
            Xcodes, f, p1, p2, alleles[p1], alleles[p2], np.delete(np.arange(P), [p1, p2])
        )
        for p1, p2 in position_pairs
    )

    # Pool numerator/denominator across all ordered pairs (and, via the two
    # orderings of each pair, across both sides of every square motif) to form
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

    # Check if landscape is built
    landscape._check_built()

    if landscape.configs is None or len(landscape.configs) == 0:
        raise ValueError("Landscape has no configuration data.")

    # Validate order parameter
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

    # Extract configurations and fitness values
    X = np.vstack(landscape.configs.values)
    y = np.array(landscape.graph.vs["fitness"])

    # Build a numerically stable design matrix:
    # - boolean landscapes already have a suitable 0/1 encoding
    # - other landscape types need one-hot encoding with a reference level dropped
    if verbose:
        print(f"Encoding {X.shape[1]} variables...")

    if landscape.type == "boolean":
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

    # Handle potential numerical issues with large datasets
    try:
        if verbose:
            print(f"Fitting polynomial regression model...")
        X_poly = poly.fit_transform(X_encoded)
        model.fit(X_poly, y)
        # Avoid np.matmul here because NumPy linked against Accelerate on
        # macOS arm64 can emit spurious RuntimeWarnings for finite inputs.
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

    # Check if landscape is built
    landscape._check_built()

    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found. "
            "Landscape must be built first."
        )

    # Extract data from landscape
    data = landscape.get_data()
    X = data.iloc[:, : landscape.n_vars]  # Configuration data
    f = data["fitness"].values  # Fitness values

    # Convert configurations to string format for Walsh-Hadamard transform
    # Handle different landscape types appropriately
    if landscape.type in ["boolean"]:
        # For boolean landscapes, convert to binary strings
        X_strings = ["".join(map(str, row.astype(int))) for _, row in X.iterrows()]
    elif landscape.type in ["dna", "rna", "protein"]:
        # For sequence landscapes, use the original sequence representation
        if hasattr(landscape, "configs") and landscape.configs is not None:
            # Try to reconstruct original sequences if available
            X_strings = []
            for config_tuple in landscape.configs.values:
                # Convert encoded config back to sequence string
                if landscape.type == "dna":
                    alphabet = ["A", "C", "G", "T"]
                elif landscape.type == "rna":
                    alphabet = ["A", "C", "G", "U"]
                else:  # protein
                    alphabet = list("ACDEFGHIKLMNPQRSTVWY")

                sequence = "".join([alphabet[int(pos)] for pos in config_tuple])
                X_strings.append(sequence)
        else:
            # Fallback: treat as categorical and convert to strings
            X_strings = ["".join(map(str, row)) for _, row in X.iterrows()]
    else:
        # Map each variable's values to a single symbol so that multi-level
        # ordinal/categorical alleles occupy exactly one string position
        # (joining raw str() values splits multi-digit codes across positions).
        # Codes start at 1 so no symbol collides with "0", which marks
        # wildtype-matching positions downstream.
        codes = X.apply(lambda col: pd.factorize(col)[0] + 1).to_numpy()
        X_strings = ["".join(chr(48 + c) for c in row) for row in codes]

    # Determine wildtype (first configuration or all zeros for binary)
    if landscape.type == "boolean":
        wildtype = "0" * landscape.n_vars
    else:
        wildtype = X_strings[0]  # Use first sequence as wildtype

    wildtype_split = [c for c in wildtype]

    # Create DataFrame for sequence features
    X_df = pd.DataFrame([list(seq) for seq in X_strings])

    # One-hot encode sequence features
    enc = OneHotEncoder(
        handle_unknown="ignore", drop=np.array(wildtype_split), dtype=int
    )
    enc.fit(X_df)

    # Generate feature names: format "{original}_{position}_{mutant}" for clarity
    # e.g. "0_12_1" = position 12, mutation from 0 to 1
    one_hot_names = []
    for i, feature_name in enumerate(enc.get_feature_names_out()):
        pos = int(feature_name.split("_")[0][1:])  # Extract position
        state = feature_name.split("_")[1]  # Extract state
        one_hot_names.append(f"{wildtype_split[pos]}_{pos+1}_{state}")

    # Create one-hot encoded DataFrame
    Xoh = pd.DataFrame(enc.transform(X_df).toarray(), columns=one_hot_names)

    # Add WT column
    Xoh = pd.concat([pd.DataFrame({"WT": [1] * len(Xoh)}), Xoh], axis=1)

    # Generate interaction features with memory optimization
    Xohi = _generate_interactions(Xoh, max_order, max_cells)

    # Ensemble encode features using Walsh-Hadamard transform with chunking
    Xensemble = _ensemble_encode_features(
        X_strings, Xohi.columns, wildtype, X_df, chunk_size
    )

    # Compute coefficients with a direct least-squares solve instead of the
    # normal equations. This is more numerically stable and also avoids the
    # spurious macOS arm64 Accelerate matmul warnings seen with pinv(X^T X).
    Xensemble_values = Xensemble.to_numpy(dtype=np.float64, copy=False)
    coefficients, *_ = np.linalg.lstsq(
        Xensemble_values,
        np.asarray(f, dtype=np.float64),
        rcond=None,
    )

    # Create results dictionary with sorted coefficients
    coef_dict = {}
    for i, feature_name in enumerate(Xohi.columns):
        if feature_name == "WT":
            order = 0
        elif "-" in feature_name:
            # Multiple mutations joined with "-" (e.g. "0_10_1-0_11_1")
            order = len(feature_name.split("-"))
        else:
            order = 1  # Single mutation

        if order not in coef_dict:
            coef_dict[order] = {}

        coef_dict[order][feature_name] = coefficients[i]

    # Sort coefficients within each order
    for order in coef_dict:
        coef_dict[order] = dict(sorted(coef_dict[order].items()))

    return coef_dict


def _generate_interactions(Xoh, max_order, max_cells):
    """Generate interaction features up to max_order with memory optimization."""
    if max_order < 2:
        return copy.deepcopy(Xoh)

    # Get mutations observed
    mut_count = list(Xoh.sum(axis=0))
    pheno_mut = [
        Xoh.columns[i]
        for i in range(len(Xoh.columns))
        if mut_count[i] != 0 and Xoh.columns[i] != "WT"
    ]

    # Group mutations by position (format: original_position_mutant, e.g. "0_12_1")
    def _get_position(mut_name):
        return mut_name.split("_")[1]

    all_pos = list(set([_get_position(i) for i in pheno_mut]))
    all_pos_mut = {p: [j for j in pheno_mut if _get_position(j) == p] for p in all_pos}

    # Generate all theoretical interaction features
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

    # Flatten all features
    all_features_flat = list(itertools.chain(*list(all_features.values())))

    # Create interaction columns with memory checking
    int_list = []
    int_list_names = []
    int_order_dict_retained = {}

    for c in all_features_flat:
        # Mutations joined with "-" (e.g. "0_10_1-0_11_1" for pairwise)
        c_split = c.split("-")
        int_col = (Xoh.loc[:, c_split].sum(axis=1) == len(c_split)).astype(int)

        # Check if minimum number of observations satisfied (kept >= 0 as in original)
        if sum(int_col) >= 0:
            int_list.append(int_col)
            int_list_names.append(c)

            # Track retained features by order
            order = len(c_split)
            if order not in int_order_dict_retained:
                int_order_dict_retained[order] = 1
            else:
                int_order_dict_retained[order] += 1

        # Memory footprint check (from original code)
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

    # Concatenate interaction features
    if len(int_list) > 0:
        Xint = pd.concat(int_list, axis=1)
        Xint.columns = int_list_names
        # Reorder features to match original order
        Xint = Xint.loc[:, [i for i in all_features_flat if i in Xint.columns]]
        Xohi = pd.concat([Xoh, Xint], axis=1)
    else:
        Xohi = copy.deepcopy(Xoh)

    return Xohi


def _ensemble_encode_features(X, feature_names, wildtype, X_df, chunk_size):
    """Ensemble encode features using Walsh-Hadamard transform with chunking optimization."""

    # Wild-type mask variant sequences
    geno_list = []
    for seq in X:
        masked = "".join(x if x != y else "0" for x, y in zip(seq, wildtype))
        geno_list.append(masked)

    # Convert feature names to coefficient strings
    coef_list = [
        _coefficient_to_sequence(coef, len(wildtype)) for coef in feature_names
    ]

    # Determine number of states per position (optimized calculation)
    state_counts = X_df.apply(lambda col: col.value_counts(), axis=0)
    state_list = [(state_counts[col] > 0).sum() for col in state_counts.columns]

    # Compute Walsh-Hadamard matrices with chunking
    print("Construction time for H_matrix...")
    hmat_inv = _H_matrix_chunker(
        str_geno=geno_list,
        str_coef=coef_list,
        num_states=state_list,
        invert=True,
        chunk_size=chunk_size,
    )

    vmat_inv = _V_matrix(str_coef=coef_list, num_states=state_list, invert=True)

    # V is diagonal, so H @ V is equivalent to scaling each column of H by the
    # corresponding diagonal element. This avoids warning-prone np.matmul calls
    # on macOS arm64 Accelerate while producing the same matrix exactly.
    return pd.DataFrame(hmat_inv * np.diag(vmat_inv), columns=feature_names)


def _coefficient_to_sequence(coefficient, length):
    """Convert coefficient string to sequence representation.

    Expects format "original_position_mutant" (e.g. "0_12_1") or multiple joined
    with "-" (e.g. "0_10_1-0_11_1" for pairwise).
    """
    coefficient_seq = ["0"] * length

    if coefficient == "WT":
        return "".join(coefficient_seq)

    # Split by "-" to get individual mutations, then parse each
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
    # Check if chunking not necessary
    if len(str_geno) < chunk_size:
        return _H_matrix(str_geno, str_coef, num_states, invert)

    # Chunk processing
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

    # Convert to numeric representation (memory efficient)
    str_coef_num = [[ord(j) for j in i.replace("0", ".")] for i in str_coef]
    str_geno_num = [[ord(j) for j in i] for i in str_geno]

    # Matrix operations
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


def extradimensional_bypass_analysis(landscape, approximate=False, sample_cut_prob=0.2):
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
    bypass_lengths = []
    motifs_with_bypass = 0
    total_motifs = len(motif_19_instances)

    for motif_nodes in motif_19_instances:
        try:
            # Get fitness values for all nodes in the motif
            fitness_values = {
                node: landscape.graph.vs[node]["fitness"] for node in motif_nodes
            }

            # Find the node with highest fitness (AB)
            AB = max(fitness_values, key=fitness_values.get)

            # Find the double mutant (ab) - the node that is not a predecessor of AB
            # and is among the remaining 3 nodes
            remaining_nodes = [node for node in motif_nodes if node != AB]
            AB_predecessors = set(landscape.graph.predecessors(AB))

            # ab should be the node that is NOT a direct predecessor of AB
            ab = [node for node in remaining_nodes if node not in AB_predecessors]

            if not ab:
                # If no ab, skip this motif
                continue

            # Check if an accessible path exists from ab to AB
            try:
                # Get shortest path distance in the directed graph
                distances = landscape.graph.distances(source=ab, target=AB, mode="out")
                distance = distances[0][0]

                # If distance is finite (not inf), an extradimensional bypass exists
                if not np.isinf(distance):
                    bypass_lengths.append(distance)
                    motifs_with_bypass += 1

            except Exception as e:
                # Skip this motif if distance calculation fails
                print(
                    f"Warning: Could not calculate distance for motif {motif_nodes}: {e}"
                )
                continue

        except Exception as e:
            # Skip this motif if any error occurs during processing
            print(f"Warning: Could not process motif {motif_nodes}: {e}")
            continue

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
    graph, motif_size=4, target_motif_type=19, approximate=False, sample_cut_prob=0.2
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
    # Validate input
    if approximate and not 0.0 <= sample_cut_prob <= 1.0:
        raise ValueError("sample_cut_prob must be between 0.0 and 1.0")

    collected_motifs = []
    cut_prob_vector = [sample_cut_prob] * motif_size if approximate else None

    def motif_collector_callback(graph, vertices, isoclass):
        if isoclass == target_motif_type:
            # Store the vertex indices as a tuple
            collected_motifs.append(tuple(sorted(vertices)))
        return False  # Continue search

    # Find motifs with or without sampling
    if approximate:
        graph.motifs_randesu(
            size=motif_size, cut_prob=cut_prob_vector, callback=motif_collector_callback
        )
    else:
        graph.motifs_randesu(size=motif_size, callback=motif_collector_callback)

    return collected_motifs
