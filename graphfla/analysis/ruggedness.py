import numpy as np
import pandas as pd

import warnings
import random

from ..algorithms import random_walk_igraph
from ..utils import autocorr_numpy
from typing import Tuple
from sklearn.linear_model import LinearRegression


def lo_ratio(landscape) -> float:
    """
    Calculate the ruggedness index of the landscape based on the ratio of the number
    of local optima to the total number of configurations.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    Returns
    -------
    float
        The ruggedness index, ranging from 0 to 1.
    """

    n_lo = landscape.n_lo
    n_configs = landscape.n_configs
    if n_configs == 0:
        return 0.0
    ruggedness = n_lo / n_configs

    return ruggedness


def autocorrelation(
    landscape, walk_length: int = 20, walk_times: int = 1000, lag: int = 1
) -> Tuple[float, float]:
    """
    A measure of landscape ruggedness. It operates by calculating the autocorrelation of
    fitness values over multiple random walks on a graph.

    Parameters:
    ----------
    walk_length : int, default=20
        The length of each random walk.

    walk_times : int, default=1000
        The number of random walks to perform.

    lag : int, default=1
        The distance lag used for calculating autocorrelation.

    Returns:
    -------
    autocorr : Tuple[float, float]
        A tuple containing the median and variance of the autocorrelation values.
    """
    corr_list = []

    for _ in range(walk_times):
        random_node = random.randrange(0, landscape.n_configs)
        logger = random_walk_igraph(
            landscape.graph, random_node, "fitness", walk_length
        )
        fitness_values = np.array(logger)[:, 2].astype(float)
        ac = autocorr_numpy(fitness_values, lag=lag)
        corr_list.append(ac)

    corr_array = np.array(corr_list)
    return np.nanmedian(corr_array), np.nanvar(corr_array)


def gradient_intensity(landscape) -> float:
    """
    Calculate the gradient intensity of the landscape using igraph. It is defined as the average absolute
    fitness difference (delta_fit) across all edges.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object with an igraph graph.

    Returns
    -------
    float
        The gradient intensity.
    """

    graph = landscape.graph  # igraph.Graph
    total_edges = graph.ecount()
    if total_edges == 0:
        return 0.0

    # Get the list of delta_fit values for all edges (default to 0 if missing)
    delta_fits = [abs(edge.attributes().get("delta_fit", 0)) for edge in graph.es]
    total_delta_fit = sum(delta_fits)

    gradient = total_delta_fit / total_edges
    return gradient


def r_s_ratio(landscape) -> float:
    """
    Calculate the roughness-to-slope (r/s) ratio of a fitness landscape.

    This metric quantifies the deviation from additivity by comparing the
    standard deviation of the residuals from a linear model fit (roughness)
    to the mean absolute additive coefficients (slope). Higher values
    indicate greater ruggedness and epistasis relative to the additive trend.

    Calculation follows definitions used in Rough Mount Fuji models and
    empirical landscape studies, e.g., [1]-[4].

    References
    ----------
    [1] I. Fragata et al., "Evolution in the light of fitness landscape
        theory," Trends Ecol. Evol., vol. 34, no. 1, pp. 69-82, Jan. 2019.
    [2] T. Aita, H. Uchiyama, T. Inaoka, M. Nakajima, T. Kokubo, and
        Y. Husimi, "Analysis of a local fitness landscape with a model of
        the rough Mount Fuji-type landscape," Biophys. Chem., vol. 88,
        no. 1-3, pp. 1-10, Dec. 2000.
    [3] A. Skwara et al., "Statistically learning the functional landscape
        of microbial communities," Nat. Ecol. Evol., vol. 7, no. 11,
        pp. 1823-1833, Nov. 2023.
    [4] C. Bank, R. T. Hietpas, J. D. Jensen, and D. N. A. Bolon, "A
        systematic survey of an intragenic epistatic landscape," Proc. Natl.
        Acad. Sci. USA, vol. 113, no. 50, pp. 14424-14429, Dec. 2016.

    Parameters
    ----------
    landscape : graphfal.landscape.Landscape
        A Landscape object instance. It must have attributes `n_vars`,
        `data_types`, and a method `get_data()` that returns a DataFrame
        containing the raw configurations and a 'fitness' column.

    Returns
    -------
    float
        The roughness-to-slope (r/s) ratio. Returns np.inf if the slope (s)
        is zero or very close to zero. Returns np.nan if the calculation fails.

    Raises
    ------
    ValueError
        If the landscape object is missing required attributes/methods or if
        data types are unsupported.
    """
    # 1. Get Data
    data = landscape.get_data()

    raw_X = data.iloc[:, : landscape.n_vars]
    fitness_values = data["fitness"].values
    data_types = landscape.data_types

    # 2. Prepare Numerical Genotype Representation for Additive Model
    numerical_X_cols = {}
    cols = list(data_types.keys())  # Maintain order

    for col in cols:
        dtype = data_types[col]
        if dtype == "boolean":
            # Convert boolean to 0/1 integer representation
            numerical_X_cols[col] = raw_X[col].astype(bool).astype(int)
        elif dtype == "categorical":
            # Use pandas Categorical codes (integer representation)
            numerical_X_cols[col] = pd.Categorical(raw_X[col]).codes
        elif dtype == "ordinal":
            # Use pandas Categorical codes (integer representation)
            numerical_X_cols[col] = pd.Categorical(raw_X[col], ordered=True).codes
        else:
            raise ValueError(
                f"Unsupported data type '{dtype}' for r/s ratio calculation in column '{col}'"
            )

    numerical_X = pd.DataFrame(numerical_X_cols, index=raw_X.index)
    X_fit = numerical_X.values  # Use numpy array for sklearn

    # Check for sufficient data
    n_samples, n_features = X_fit.shape
    if n_samples <= n_features:
        warnings.warn(
            f"Number of samples ({n_samples}) is less than or equal to "
            f"the number of features ({n_features}) after encoding. "
            "Linear regression might be underdetermined or unstable.",
            UserWarning,
        )

    # 3. Fit Additive (Linear) Model
    try:
        linear_model = LinearRegression(fit_intercept=True)
        linear_model.fit(X_fit, fitness_values)

        # 4. Calculate Slope (s)
        # Mean of absolute values of additive coefficients (betas)
        additive_coeffs = linear_model.coef_
        # potential case where n_features is 1
        if n_features == 1 and np.isscalar(additive_coeffs):
            slope_s = np.abs(additive_coeffs)
        elif n_features == 1 and isinstance(additive_coeffs, (np.ndarray, list)):
            slope_s = np.abs(additive_coeffs[0])
        elif n_features > 1:
            slope_s = np.mean(np.abs(additive_coeffs))
        else:  # n_features == 0 (should not happen if validation is correct)
            slope_s = 0

        if np.isclose(slope_s, 0):
            warnings.warn(
                "Slope 's' is zero or near zero. Landscape may be flat "
                "or purely epistatic according to the linear fit. Returning inf.",
                UserWarning,
            )
            return np.inf

        predicted_fitness = linear_model.predict(X_fit)
        residuals = fitness_values - predicted_fitness
        roughness_r = np.std(residuals, ddof=1 if n_samples > 1 else 0)

        r_s_ratio_value = roughness_r / slope_s

        return r_s_ratio_value

    except Exception as e:
        warnings.warn(
            f"Calculation of r/s ratio failed: {e}. Returning np.nan.", UserWarning
        )
        return np.nan
