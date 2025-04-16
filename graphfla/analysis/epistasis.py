from scipy.stats import spearmanr, pearsonr, binomtest, ttest_1samp
from typing import Any, Tuple, Literal
from itertools import combinations, product
from joblib import Parallel, delayed
from collections import defaultdict

import numpy as np
import igraph as ig
import pandas as pd
import warnings


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

    return {
        "positive epistasis": perc_positive,
        "negative epistasis": perc_negative,
    }


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
        - "magnitude epistasis"
        - "sign epistasis"
        - "reciprocal sign epistasis"
        - "positive epistasis"
        - "negative epistasis"
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
        total_collected = sum(len(v) for v in collected_square_instances.values())

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
    return final_results


def idiosyncratic_index(landscape, mutation):
    """
    Calculates the idiosyncratic index for the fitness landscape proposed in [1].

    The idiosyncratic index of a specific genetic mutation quantifies the sensitivity
    of a specific mutation to idiosyncratic epistasis. It is defined as the as the
    variation in the fitness difference between genotypes that differ by the mutation,
    relative to the variation in the fitness difference between random genotypes for
    the same number of genotype pairs. We compute this for the entire fitness landscape
    by averaging it across individual mutations.

    The idiosyncratic index for a landscape varies from 0 to 1, corresponding to the
    minimum and maximum levels of idiosyncrasy, respectively.

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

    Returns
    -------
    float
        The calculated idiosyncratic index.
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
        return 0.0

    mutation_effects = df_merged["fitness_B"] - df_merged["fitness_A"]
    n_pairs = len(mutation_effects)

    if n_pairs <= 1:
        return 0.0

    std_mutation_effect = np.std(mutation_effects)
    all_fitness_values = f.values

    if len(all_fitness_values) <= 1 or np.all(
        all_fitness_values == all_fitness_values[0]
    ):
        return 0.0

    rand_f1 = np.random.choice(all_fitness_values, size=n_pairs, replace=True)
    rand_f2 = np.random.choice(all_fitness_values, size=n_pairs, replace=True)

    random_diffs = rand_f1 - rand_f2

    std_random_diff = np.std(random_diffs)

    if std_random_diff == 0:
        return 0.0

    idiosyncratic_val = std_mutation_effect / std_random_diff

    return idiosyncratic_val


def global_idiosyncratic_index(landscape, random_seed=None):
    """
    Calculates the global idiosyncratic index for the entire fitness landscape.

    This function extends the individual mutation idiosyncratic index from Lyons et al. (2020)
    to provide a global measure by averaging across all possible mutations in the landscape.
    The global index quantifies the overall sensitivity of the landscape to idiosyncratic
    epistasis.

    The index ranges from 0 to 1, with higher values indicating stronger idiosyncratic effects.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.
    random_seed : int, optional
        Seed for random number generation to ensure reproducibility.

    Returns
    -------
    dict
        A dictionary containing:
        - 'global_index': The overall idiosyncratic index (average across all mutations)
        - 'per_position': A dictionary mapping each position to its average index
        - 'mutation_counts': The number of valid mutations considered in the calculation

    References
    ----------
    .. [1] Daniel M. Lyons et al, "Idiosyncratic epistasis creates universals in mutational
       effects and evolutionary trajectories", Nat. Ecol. Evo., 2020.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    data = landscape.get_data()
    X = data.iloc[:, : landscape.n_vars]

    # Track indices for each position and overall
    position_indices = {}
    all_indices = []
    mutation_counts = 0

    # Process each position
    for pos in X.columns:
        unique_alleles = sorted(X[pos].unique())
        position_total = 0
        position_count = 0

        # Generate all unique unordered pairs of alleles (mutations)
        for i in range(len(unique_alleles)):
            for j in range(i + 1, len(unique_alleles)):
                A, B = unique_alleles[i], unique_alleles[j]

                try:
                    # Calculate idiosyncratic index for this specific mutation
                    index_A_to_B = idiosyncratic_index(landscape, (A, pos, B))

                    if not np.isnan(index_A_to_B):
                        position_total += index_A_to_B
                        all_indices.append(index_A_to_B)
                        position_count += 1
                except Exception as e:
                    # Skip mutations that cause errors
                    print(
                        f"Warning: Could not calculate index for mutation {A}->{B} at position {pos}: {e}"
                    )

        # Store average for this position
        if position_count > 0:
            position_indices[pos] = position_total / position_count
            mutation_counts += position_count
        else:
            position_indices[pos] = np.nan

    # Calculate global index
    global_index = np.mean(all_indices) if all_indices else np.nan

    return {
        "global_index": global_index,
        "per_position": position_indices,
        "mutation_counts": mutation_counts,
    }


def diminishing_returns_index(
    landscape,
    method: Literal["pearson", "spearman"] = "pearson",
) -> Tuple[float, float]:
    """Measures diminishing returns epistasis in a fitness landscape.

    Diminishing returns epistasis occurs when the fitness benefit of new
    beneficial mutations decreases as the background fitness increases. This
    function quantifies this trend by calculating the correlation between the
    fitness of each genotype (node) and the average fitness improvement
    provided by its direct successors (fitter one-mutant neighbors). A
    significant negative correlation indicates diminishing returns.

    Parameters
    ----------
    landscape : BaseLandscape
        An initialized and built fitness landscape object. The landscape graph
        must have a 'fitness' attribute for each node.
    method : {'pearson', 'spearman'}, default='pearson'
        The method used to calculate the correlation coefficient.
        'pearson' for Pearson correlation, 'spearman' for Spearman rank correlation.

    Returns
    -------
    correlation : float
        The correlation coefficient between node fitness and average successor
        fitness improvement. NaN if calculation is not possible.
    p_value : float
        The p-value associated with the correlation test. NaN if calculation
        is not possible.

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
        return np.nan, np.nan

    node_fitnesses_series = pd.Series(node_fitnesses)
    avg_improvement_series = pd.Series(avg_successor_improvement)

    mask = ~avg_improvement_series.isna()
    if mask.sum() < 2:
        warnings.warn(
            "Not enough valid data points after NaN omission to calculate correlation.",
            UserWarning,
        )
        return np.nan, np.nan
    node_fitnesses_corr = node_fitnesses_series[mask]
    avg_improvement_corr = avg_improvement_series[mask]

    if method == "pearson":
        corr_func = pearsonr
    elif method == "spearman":
        corr_func = spearmanr
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")

    try:
        correlation, p_value = corr_func(node_fitnesses_corr, avg_improvement_corr)
        return correlation, p_value
    except Exception as e:
        warnings.warn(f"Could not calculate correlation: {e}", UserWarning)
        return np.nan, np.nan


def increasing_costs_index(
    landscape,
    method: Literal["pearson", "spearman"] = "pearson",
) -> Tuple[float, float]:
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
    landscape : BaseLandscape
        An initialized and built fitness landscape object. The landscape graph
        must have a 'fitness' attribute for each node.
    method : {'pearson', 'spearman'}, default='pearson'
        The method used to calculate the correlation coefficient.
        'pearson' for Pearson correlation, 'spearman' for Spearman rank correlation.

    Returns
    -------
    correlation : float
        The correlation coefficient between node fitness and average predecessor
        fitness cost. NaN if calculation is not possible.
    p_value : float
        The p-value associated with the correlation test. NaN if calculation
        is not possible.

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
        return np.nan, np.nan

    node_fitnesses_series = pd.Series(node_fitnesses)
    avg_cost_series = pd.Series(avg_predecessor_cost)

    mask = ~avg_cost_series.isna()
    if mask.sum() < 2:
        warnings.warn(
            "Not enough valid data points after NaN omission to calculate correlation.",
            UserWarning,
        )
        return np.nan, np.nan
    node_fitnesses_corr = node_fitnesses_series[mask]
    avg_cost_corr = avg_cost_series[mask]

    if method == "pearson":
        corr_func = pearsonr
    elif method == "spearman":
        corr_func = spearmanr
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")

    try:
        correlation, p_value = corr_func(node_fitnesses_corr, avg_cost_corr)
        return correlation, p_value
    except Exception as e:
        warnings.warn(f"Could not calculate correlation: {e}", UserWarning)
        return np.nan, np.nan


def pairwise_epistasis(X, f, pos1, pos2):
    """
    Assess the pairwise epistasis effects between all unique unordered mutations at two specified positions.

    Parameters
    ----------
    X : pd.DataFrame
        The genotype matrix where each column corresponds to a genetic position.

    f : pd.Series
        The fitness values corresponding to each genotype.

    pos1 : str
        The name of the first position to assess mutations for.

    pos2 : str
        The name of the second position to assess mutations for.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing mutation pairs, median absolute epistasis effect,
        p-values, and significance flags.
    """

    def get_diff(df1, df2, f):
        f.name = "fitness"
        df1 = pd.concat([df1, f], axis=1, join="inner")
        df2 = pd.concat([df2, f], axis=1, join="inner")
        df1.set_index(0, inplace=True)
        df2.set_index(0, inplace=True)

        df_diff = pd.merge(
            df1, df2, left_index=True, right_index=True, suffixes=("_1", "_2")
        )
        df_diff.index = range(len(df_diff))
        diff = df_diff["fitness_1"] - df_diff["fitness_2"]
        return diff

    def compute_epistasis(X, f, pos1, pos2, mut1, mut2):
        _, a, A = mut1
        _, b, B = mut2

        X_AB = X[(X[pos1] == A) & (X[pos2] == B)]
        X_ab = X[(X[pos1] == a) & (X[pos2] == b)]
        X_Ab = X[(X[pos1] == A) & (X[pos2] == b)]
        X_aB = X[(X[pos1] == a) & (X[pos2] == B)]

        X_AB = pd.Series(X_AB.drop(columns=[pos1, pos2]).apply(tuple, axis=1))
        X_ab = pd.Series(X_ab.drop(columns=[pos1, pos2]).apply(tuple, axis=1))
        X_Ab = pd.Series(X_Ab.drop(columns=[pos1, pos2]).apply(tuple, axis=1))
        X_aB = pd.Series(X_aB.drop(columns=[pos1, pos2]).apply(tuple, axis=1))

        f_AB_ab = get_diff(X_AB, X_ab, f)
        f_Ab_ab = get_diff(X_Ab, X_ab, f)
        f_aB_ab = get_diff(X_aB, X_ab, f)

        diff = f_AB_ab - (f_Ab_ab + f_aB_ab)

        if diff.empty:
            cohen_d = np.nan
            ttest_p = np.nan
            mean = np.nan
        else:
            cohen_d = abs(diff).median() / f.std()
            _, ttest_p = ttest_1samp(diff, 0)
            mean = diff.mean()

        return {
            "pos1": pos1,
            "mutation1_from": a,
            "mutation1_to": A,
            "pos2": pos2,
            "mutation2_from": b,
            "mutation2_to": B,
            "cohen_d": cohen_d,
            "ttest_p": ttest_p,
            "mean_diff": mean,
        }

    unique_vals1 = sorted(X[pos1].dropna().unique())
    unique_vals2 = sorted(X[pos2].dropna().unique())

    mutations1 = [(pos1, a, b) for a, b in combinations(unique_vals1, 2)]
    mutations2 = [(pos2, c, d) for c, d in combinations(unique_vals2, 2)]

    mutation_pairs = list(product(mutations1, mutations2))

    results = [
        compute_epistasis(X, f, pos1, pos2, mut1, mut2) for mut1, mut2 in mutation_pairs
    ]

    epistasis_df = pd.DataFrame(results)

    return epistasis_df


def all_pairwise_epistasis(X, f, n_jobs=1):
    """
    Compute and aggregate epistasis effects between all unique pairs of positions in the genotype matrix using parallel execution.

    Parameters
    ----------
    X : pd.DataFrame
        The genotype matrix where each column corresponds to a genetic position.
    f : pd.Series
        The fitness values corresponding to each genotype.
    n_jobs : int, default=1
        The number of parallel jobs to run. -1 means using all available cores.

    Returns
    -------
    pd.DataFrame
        An aggregated DataFrame containing average epistasis scores for each position pair.
    """

    positions = list(X.columns)
    position_pairs = list(combinations(positions, 2))

    detailed_results = Parallel(n_jobs=n_jobs)(
        delayed(pairwise_epistasis)(X, f, pos1, pos2) for pos1, pos2 in position_pairs
    )

    all_epistasis_df = pd.concat(detailed_results, ignore_index=True)

    aggregated = (
        all_epistasis_df.groupby(["pos1", "pos2"])
        .agg(
            average_cohen_d=("cohen_d", "median"),
            average_mean_diff=("mean_diff", "median"),
            most_significant_p=("ttest_p", "min"),
            total_mutation_pairs=("ttest_p", "count"),
        )
        .reset_index()
    )

    return aggregated
