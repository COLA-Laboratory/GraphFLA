import warnings
import numpy as np
import scipy.stats as stats
from scipy.stats import cauchy  # Add explicit import for cauchy

from typing import Dict, Any


def fitness_distribution(landscape) -> Dict[str, Any]:
    """
    Calculate unitless statistics about the fitness distribution of the landscape.

    This function computes various statistics that characterize the shape and properties
    of the fitness distribution across all configurations in the landscape. The statistics
    are chosen to be unitless (scale-invariant) to allow meaningful comparisons across
    different landscapes with varying fitness scales.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the following statistics:
        - 'skewness': Measure of asymmetry of the fitness distribution
        - 'kurtosis': Measure of "tailedness" of the fitness distribution
        - 'cv': Coefficient of variation (ratio of std dev to mean)
        - 'quartile_coefficient': Interquartile range divided by median (IQR/median)
        - 'median_mean_ratio': Ratio of median to mean
        - 'relative_range': Range divided by median
        - 'cauchy_loc': Location parameter (center) of the fitted Cauchy distribution

    Raises
    ------
    RuntimeError
        If the graph is not initialized or the fitness attribute is missing.

    Notes
    -----
    - Skewness > 0 indicates right-skewed distribution (tail on right)
    - Skewness < 0 indicates left-skewed distribution (tail on left)
    - Kurtosis > 3 indicates heavy tails and peaked distribution
    - Kurtosis < 3 indicates light tails and flat distribution
    - Higher CV indicates greater relative dispersion
    - Cauchy location parameter represents the center/peak of the fitted Cauchy distribution
    """
    if landscape.graph is None:
        raise RuntimeError(
            "Graph not initialized. Cannot calculate fitness distribution statistics."
        )

    if "fitness" not in landscape.graph.vs.attributes():
        raise RuntimeError("Fitness attribute missing from graph nodes.")

    # Extract fitness values from the graph
    fitness_values = landscape.graph.vs["fitness"]
    n_samples = len(fitness_values)

    if n_samples == 0:
        warnings.warn("No fitness values found in the landscape.", RuntimeWarning)
        return {
            "skewness": np.nan,
            "kurtosis": np.nan,
            "cv": np.nan,
            "quartile_coefficient": np.nan,
            "median_mean_ratio": np.nan,
            "relative_range": np.nan,
            "cauchy_loc": np.nan,
        }

    # Calculate basic statistics
    mean = np.mean(fitness_values)
    std_dev = np.std(fitness_values, ddof=1)  # Using n-1 for sample std dev
    median = np.median(fitness_values)
    fitness_min = np.min(fitness_values)
    fitness_max = np.max(fitness_values)
    fitness_range = fitness_max - fitness_min
    q1 = np.percentile(fitness_values, 25)
    q3 = np.percentile(fitness_values, 75)
    iqr = q3 - q1

    # Calculate unitless statistics

    # Skewness: measure of asymmetry
    skewness = stats.skew(fitness_values)

    # Kurtosis: measure of "tailedness"
    # Note: scipy.stats uses Fisher's definition where normal distribution has kurtosis=0
    # Adding 3 converts to Pearson's definition where normal distribution has kurtosis=3
    kurtosis = stats.kurtosis(fitness_values) + 3

    # Coefficient of variation: std_dev / mean (unitless measure of dispersion)
    # Handle potential division by zero
    cv = np.nan if mean == 0 else std_dev / abs(mean)

    # Quartile coefficient: IQR / median (robust measure of dispersion)
    quartile_coefficient = np.nan if median == 0 else iqr / abs(median)

    # Ratio of median to mean (indicates skewness)
    median_mean_ratio = np.nan if mean == 0 else median / mean

    # Relative range: range / median (unitless measure of total spread)
    relative_range = np.nan if median == 0 else fitness_range / abs(median)

    # Fit Cauchy distribution to fitness values and extract location parameter
    try:
        loc, _ = cauchy.fit(fitness_values)
    except (ValueError, RuntimeError):
        # Handle potential fitting failures
        loc = np.nan

    return {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "cv": cv,
        "quartile_coefficient": quartile_coefficient,
        "median_mean_ratio": median_mean_ratio,
        "relative_range": relative_range,
        "cauchy_loc": loc,
    }


def distribution_fit_effects(landscape, mutation):
    """
    Calculates the distribution of fitness effects for a specific mutation
    across all possible genetic backgrounds.

    This function measures how the effect of a specific mutation varies
    depending on the genetic context (background) in which it occurs.
    It returns a list of fitness differences caused by the mutation
    in each background where both the original and mutated genotypes exist.

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
    list
        A list of fitness effect values for the mutation across different
        genetic backgrounds. Each value represents the fitness difference
        between a genotype with mutation B and the corresponding genotype
        with allele A at the specified position.

    Raises
    ------
    ValueError
        If the specified alleles don't exist at the given position.
    RuntimeError
        If the landscape has not been built.
    """
    # Check if landscape is built
    landscape._check_built()

    # Unpack mutation tuple
    A, pos, B = mutation

    # Get data from landscape
    data = landscape.get_data()
    X = data.iloc[:, : landscape.n_vars]

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

    # Filter data for genotypes with allele A or B at the specified position
    mask_A = X[pos] == A
    mask_B = X[pos] == B
    df_A = data[mask_A]
    df_B = data[mask_B]

    # Check if we have enough data
    if df_A.empty or df_B.empty:
        return []

    # Get all other positions (the genetic background)
    background_cols = [col for col in X.columns if col != pos]

    # Set the background columns as the index for both dataframes
    df_A.set_index(background_cols, inplace=True)
    df_B.set_index(background_cols, inplace=True)

    # Only keep backgrounds that exist in both dataframes by using index intersection
    common_backgrounds = df_A.index.intersection(df_B.index)

    # If no common backgrounds, return empty list
    if len(common_backgrounds) == 0:
        return []

    # Calculate the fitness effects directly using aligned indices
    df_A_common = df_A.loc[common_backgrounds]
    df_B_common = df_B.loc[common_backgrounds]

    # Compute the fitness effects (B - A)
    fitness_effects = (df_B_common["fitness"] - df_A_common["fitness"]).tolist()

    return fitness_effects
