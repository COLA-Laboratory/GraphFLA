import warnings
import numpy as np
import scipy.stats as stats
from scipy.stats import cauchy

from typing import Dict, List


from ._utils import _pythonize


def fitness_distribution(landscape) -> Dict[str, float]:
    r"""Return descriptive statistics of the retained fitness distribution.

    Summarize fitness across the configurations retained in the landscape.
    The fitted Cauchy location is in fitness units; the other summaries are
    dimensionless, but need not be invariant to shifts or nonlinear transforms.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.

    Returns
    -------
    statistics : dict of str to float
        Keys are ``skewness``, ``kurtosis`` (Pearson convention, normal=3),
        ``cv`` (sample SD divided by absolute mean), ``quartile_coefficient``
        (IQR divided by absolute median), ``median_mean_ratio``, ``relative_range``
        (range divided by absolute median), and ``cauchy_loc`` (fitted location).
        Undefined summaries, including ratios with a zero denominator, are NaN.
        An empty landscape returns the same keys with NaN values.

    Raises
    ------
    RuntimeError
        If the graph is not initialized or the fitness attribute is missing.

    Notes
    -----
    Positive skewness indicates a longer right tail; negative skewness a
    longer left tail. Pearson kurtosis is 3 for a normal distribution. Cauchy
    location describes the fitted center and is not scale-invariant.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import fitness_distribution
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> round(fitness_distribution(landscape)["median_mean_ratio"], 3)
    0.857
    """
    if landscape.graph is None:
        raise RuntimeError(
            "Graph not initialized. Cannot calculate fitness distribution statistics."
        )

    if "fitness" not in landscape.graph.vs.attributes():
        raise RuntimeError("Fitness attribute missing from graph nodes.")

    fitness_values = landscape.graph.vs["fitness"]
    n_samples = len(fitness_values)

    if n_samples == 0:
        warnings.warn("No fitness values found in the landscape.", RuntimeWarning)
        return _pythonize({
            "skewness": np.nan,
            "kurtosis": np.nan,
            "cv": np.nan,
            "quartile_coefficient": np.nan,
            "median_mean_ratio": np.nan,
            "relative_range": np.nan,
            "cauchy_loc": np.nan,
        })

    mean = np.mean(fitness_values)
    std_dev = np.std(fitness_values, ddof=1)  # sample std dev (n-1)
    median = np.median(fitness_values)
    fitness_min = np.min(fitness_values)
    fitness_max = np.max(fitness_values)
    fitness_range = fitness_max - fitness_min
    q1 = np.percentile(fitness_values, 25)
    q3 = np.percentile(fitness_values, 75)
    iqr = q3 - q1

    skewness = stats.skew(fitness_values)

    # +3 converts scipy's Fisher kurtosis (normal=0) to Pearson's (normal=3)
    kurtosis = stats.kurtosis(fitness_values) + 3

    # guard division by zero
    cv = np.nan if mean == 0 else std_dev / abs(mean)
    quartile_coefficient = np.nan if median == 0 else iqr / abs(median)
    median_mean_ratio = np.nan if mean == 0 else median / mean
    relative_range = np.nan if median == 0 else fitness_range / abs(median)

    try:
        loc, _ = cauchy.fit(fitness_values)
    except (ValueError, RuntimeError):
        loc = np.nan

    return _pythonize({
        "skewness": skewness,
        "kurtosis": kurtosis,
        "cv": cv,
        "quartile_coefficient": quartile_coefficient,
        "median_mean_ratio": median_mean_ratio,
        "relative_range": relative_range,
        "cauchy_loc": loc,
    })


def fitness_effect_distribution(landscape, mutation) -> List[float]:
    r"""Return fitness effects of one mutation across matching backgrounds.

    Use each retained background in which both source and target alleles
    are observed. Effects are target fitness minus source fitness, without
    changing sign for minimization.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.
    mutation : tuple of (source, position, target)
        Allele substitution to evaluate. ``position`` is a configuration-column
        label from ``landscape.data_types``, not a positional column index.

    Returns
    -------
    effects : list of float
        Target-minus-source fitness differences, one per matching background.
        Return an empty list if no backgrounds match. A single-position
        landscape has one shared empty background.

    Raises
    ------
    ValueError
        If the specified alleles don't exist at the given position.
    graphfla.exceptions.NotBuiltError
        If the landscape has not been built.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import fitness_effect_distribution
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> fitness_effect_distribution(landscape, (0, "bit_0", 1))
    [2.0, 3.0]
    """
    landscape._check_built()

    A, pos, B = mutation

    data = landscape.get_data()
    X = data[list(landscape.data_types.keys())]

    unique_alleles = X[pos].unique()
    if A not in unique_alleles:
        raise ValueError(
            f"Original allele '{A}' not found at position '{pos}'. Available: {unique_alleles}"
        )
    if B not in unique_alleles:
        raise ValueError(
            f"New allele '{B}' not found at position '{pos}'. Available: {unique_alleles}"
        )

    mask_A = X[pos] == A
    mask_B = X[pos] == B
    df_A = data[mask_A]
    df_B = data[mask_B]

    if df_A.empty or df_B.empty:
        return []

    # genetic background = all positions except the mutated one
    background_cols = [col for col in X.columns if col != pos]

    # index by background so A and B can be aligned via intersection. A
    # single-position landscape has no background columns; every genotype then
    # shares one trivial background, represented by a constant index.
    if background_cols:
        df_A = df_A.set_index(background_cols)
        df_B = df_B.set_index(background_cols)
    else:
        df_A = df_A.set_index(np.zeros(len(df_A), dtype=np.int8))
        df_B = df_B.set_index(np.zeros(len(df_B), dtype=np.int8))
    common_backgrounds = df_A.index.intersection(df_B.index)

    if len(common_backgrounds) == 0:
        return []

    df_A_common = df_A.loc[common_backgrounds]
    df_B_common = df_B.loc[common_backgrounds]

    # fitness effect = B - A in each shared background
    fitness_effects = (df_B_common["fitness"] - df_A_common["fitness"]).tolist()

    return _pythonize(fitness_effects)
