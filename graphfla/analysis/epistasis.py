from scipy.stats import spearmanr, pearsonr
from typing import Any, Tuple, Literal
from collections import defaultdict
from itertools import product, combinations
from joblib import Parallel, delayed

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


def global_idiosyncratic_index(landscape, n_jobs=-1, random_seed=None):
    """
    Calculates the global idiosyncratic index for the entire fitness landscape using parallel processing.

    This function extends the individual mutation idiosyncratic index from Lyons et al. (2020)
    to provide a global measure by averaging across all possible mutations in the landscape.
    The global index quantifies the overall sensitivity of the landscape to idiosyncratic
    epistasis.

    The index ranges from 0 to 1, with higher values indicating stronger idiosyncratic effects.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.
    n_jobs : int, optional
        Number of parallel jobs to use. Default is -1 (all available cores).
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
    from joblib import Parallel, delayed

    if random_seed is not None:
        np.random.seed(random_seed)

    data = landscape.get_data()
    X = data.iloc[:, : landscape.n_vars]

    # Generate all possible mutations to process in parallel
    mutations_to_process = []
    for pos in X.columns:
        unique_alleles = sorted(X[pos].unique())
        for i in range(len(unique_alleles)):
            for j in range(i + 1, len(unique_alleles)):
                A, B = unique_alleles[i], unique_alleles[j]
                mutations_to_process.append((A, pos, B))

    # Define a worker function to compute idiosyncratic index for each mutation
    def compute_index(mutation):
        try:
            A, pos, B = mutation
            index_value = idiosyncratic_index(landscape, mutation)
            return {"mutation": mutation, "pos": pos, "index": index_value}
        except Exception as e:
            print(
                f"Warning: Could not calculate index for mutation {A}->{B} at position {pos}: {e}"
            )
            return {"mutation": mutation, "pos": pos, "index": np.nan}

    # Execute in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_index)(mutation) for mutation in mutations_to_process
    )

    # Process results
    position_indices = defaultdict(list)
    all_indices = []

    for result in results:
        index_value = result["index"]
        pos = result["pos"]

        if not np.isnan(index_value):
            position_indices[pos].append(index_value)
            all_indices.append(index_value)

    # Calculate average per position
    position_averages = {}
    for pos, indices in position_indices.items():
        if indices:
            position_averages[pos] = np.mean(indices)
        else:
            position_averages[pos] = np.nan

    # Calculate global index
    global_index = np.mean(all_indices) if all_indices else np.nan
    mutation_counts = len(all_indices)

    return {
        "global_index": global_index,
        "per_position": position_averages,
        "mutation_counts": mutation_counts,
    }


def diminishing_returns_index(
    landscape,
    method: Literal["pearson", "spearman", "regression"] = "pearson",
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
    p_value : float
        The p-value associated with the correlation test or regression.
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
        return correlation
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
        return correlation
    except Exception as e:
        warnings.warn(f"Could not calculate correlation: {e}", UserWarning)
        return np.nan


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
    dict
        A dictionary containing:
        - 'gamma': The traditional gamma statistic value. Values close to -1 or 1 indicate
          strong epistatic interactions in magnitude, while values close to 0 indicate
          weak or no epistasis.
        - 'gamma_star': The gamma star statistic that only considers sign consistency.
          Values close to 1 indicate consistent sign epistasis across backgrounds,
          values close to -1 indicate opposing sign patterns, and values close to 0
          indicate random sign patterns.

    Notes
    -----
    - The gamma statistic measures the correlation between fitness effects of mutations
      across different genetic backgrounds, providing a measure of epistatic interactions
      in the landscape.
    - The gamma_star statistic focuses only on sign consistency, ignoring the magnitude
      of fitness effects. It indicates whether mutations tend to have consistent
      directional effects across different genetic backgrounds.
    """

    # Ensure landscape is built
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found."
            " Landscape must be built first."
        )

    # Extract data
    df = landscape.get_data()
    X = df.iloc[:, : landscape.n_vars]

    # Function to get all possible mutations for a position
    def get_all_mutations(df, pos):
        combs = list(combinations(df[pos].unique(), 2))
        return combs

    # Get all mutations for each position
    mutation_dict = {col: get_all_mutations(X, col) for col in X.columns}

    # Generate all position pairs
    positions = list(X.columns)
    position_pairs = [
        (pos1, pos2) for pos1 in positions for pos2 in positions if pos1 != pos2
    ]

    def count_differing_bits(list1, list2):
        if len(list1) != len(list2):
            raise ValueError("Lists must be of the same length.")

        total_diff_bits = 0
        for a, b in zip(list1, list2):
            xor = a ^ b
            total_diff_bits += bin(xor).count("1")

        return total_diff_bits

    # Function to process each mutation pair and compute correlation
    def process_mutation_pair(i, pos1, pos2, mutation1, mutation2):
        index_cols = list(X.drop(columns=[pos1, pos2]).columns)

        mask_pos1_0 = df[pos1] == mutation1[0]
        mask_pos1_1 = df[pos1] == mutation1[1]
        mask_pos2_0 = df[pos2] == mutation2[0]
        mask_pos2_1 = df[pos2] == mutation2[1]

        df_ab = df[mask_pos1_0 & mask_pos2_0]
        df_Ab = df[mask_pos1_1 & mask_pos2_0]
        df_aB = df[mask_pos1_0 & mask_pos2_1]
        df_AB = df[mask_pos1_1 & mask_pos2_1]

        if any(len(data) == 0 for data in [df_ab, df_Ab, df_aB, df_AB]):
            return None  # No need to compute if data frames are empty

        # Set index for the data frames
        for data in [df_ab, df_Ab, df_aB, df_AB]:
            if len(data) > 0:
                data.set_index(index_cols, inplace=True)

        # Compute fitness effects
        fit_effects_b = df_ab["fitness"] - df_Ab["fitness"]
        fit_effects_B = df_aB["fitness"] - df_AB["fitness"]

        fit_effects_b_aligned, fit_effects_B_aligned = fit_effects_b.align(
            fit_effects_B
        )

        if len(fit_effects_b_aligned) > 0:
            valid_mask = ~(
                np.isnan(fit_effects_b_aligned) | np.isnan(fit_effects_B_aligned)
            )
            if valid_mask.sum() > 1:
                # Compute traditional gamma (correlation of fitness effect values)

                fit_effects_b = fit_effects_b_aligned[valid_mask]
                fit_effects_B = fit_effects_B_aligned[valid_mask]

                gamma_value = np.corrcoef(fit_effects_b, fit_effects_B)[0, 1]

                fit_effects_b = fit_effects_b.map(
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                ).to_list()
                fit_effects_B = fit_effects_B.map(
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                ).to_list()

                # gamma_star_value = np.corrcoef(fit_effects_b, fit_effects_B)[0, 1]
                # Compute gamma_star (correlation of fitness effect signs)
                # Convert to signs: 1 for positive, -1 for negative, 0 for zero

                gamma_star_value = 1 - count_differing_bits(
                    fit_effects_b, fit_effects_B
                ) / len(fit_effects_b)

                return {"gamma": gamma_value, "gamma_star": gamma_star_value}

        return None

    # Process all mutation combinations for each position pair
    def process_position_pair(i, pos1, pos2):
        results = []
        mutations1 = mutation_dict[pos1]
        mutations2 = mutation_dict[pos2]

        for mutation1, mutation2 in product(mutations1, mutations2):
            result = process_mutation_pair(i, pos1, pos2, mutation1, mutation2)
            if result is not None:
                results.append(result)

        return results

    # Parallelize processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_position_pair)(i, pos1, pos2)
        for i, (pos1, pos2) in enumerate(position_pairs)
    )

    # Flatten results and calculate mean
    all_gamma = []
    all_gamma_star = []

    for sublist in results:
        for result in sublist:
            if result is not None:
                all_gamma.append(result["gamma"])
                all_gamma_star.append(result["gamma_star"])

    if not all_gamma:
        return {"gamma": np.nan, "gamma_star": np.nan}

    return {"gamma": np.mean(all_gamma), "gamma_star": np.mean(all_gamma_star)}


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
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    verbose : bool, optional
        Whether to print progress information. Default is False.

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
        from sklearn.pipeline import Pipeline
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

    # One-hot encode all variables
    if verbose:
        print(f"One-hot encoding {X.shape[1]} variables...")

    encoder = OneHotEncoder(sparse_output=False)
    try:
        X_encoded = encoder.fit_transform(X)
    except Exception as e:
        raise ValueError(f"Failed to one-hot encode configurations: {e}")

    if verbose:
        print(f"Encoded data shape: {X_encoded.shape}")
        print(f"Creating polynomial features of degree {order}...")

    # Create a pipeline with polynomial features and linear regression
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=order, include_bias=True)),
            ("linear", LinearRegression(n_jobs=n_jobs)),
        ]
    )

    # Handle potential numerical issues with large datasets
    try:
        if verbose:
            print(f"Fitting polynomial regression model...")
        model.fit(X_encoded, y)
        y_pred = model.predict(X_encoded)
        r2 = r2_score(y, y_pred)
    except Exception as e:
        raise RuntimeError(f"Error fitting polynomial regression model: {e}")

    if verbose:
        print(f"Order-{order} epistasis R² score: {r2:.4f}")

    return r2
