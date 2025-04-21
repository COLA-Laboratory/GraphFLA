import warnings
import numpy as np
import scipy.stats as stats

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

    return {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "cv": cv,
        "quartile_coefficient": quartile_coefficient,
        "median_mean_ratio": median_mean_ratio,
        "relative_range": relative_range,
    }


def DFE(landscape):
    raise NotImplementedError(
        "DFE is not implemented yet. Please check the documentation for updates."
    )
