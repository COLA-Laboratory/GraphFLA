import itertools
import numpy as np
import pandas as pd

from scipy.stats import qmc

def random_search(param_distributions, n_iter, evaluate):
    """
    Perform random search over the given parameter distributions.

    Parameters:
    - param_distributions: dict
        Dictionary where keys are parameter names and values are distributions or lists.
        Distributions can be from scipy.stats or custom objects with an 'rvs' method.

    - n_iter: int
        Number of iterations/samples to draw.

    - evaluate: function
        Function that takes a configuration (dict) and returns a fitness value.

    Returns:
    - results_df: pandas DataFrame
        DataFrame containing configurations and their fitness values.
    """

    results = []
    for _ in range(n_iter):
        config = {}
        for param, dist in param_distributions.items():
            if hasattr(dist, 'rvs'):
                value = dist.rvs()
            elif isinstance(dist, list):
                value = np.random.choice(dist)
            else:
                raise ValueError(f"Unsupported distribution type for parameter '{param}'. Must be a list or a distribution with an 'rvs' method.")
            config[param] = value
        fitness = evaluate(config)
        config['fitness'] = fitness
        results.append(config)
    results_df = pd.DataFrame(results)
    return results_df

def grid_search(param_grid, evaluate):
    """
    Perform grid search over the given parameter grid.

    Parameters:
    - param_grid: dict
        Dictionary where keys are parameter names and values are lists of parameter settings to try.

    - evaluate: function
        Function that takes a configuration (dict) and returns a fitness value.

    Returns:
    - results_df: pandas DataFrame
        DataFrame containing configurations and their fitness values.
    """

    for param, values in param_grid.items():
        if not isinstance(values, list):
            raise ValueError(f"Parameter '{param}' must be a list of values for grid search.")

    keys = param_grid.keys()
    combinations = list(itertools.product(*(param_grid[param] for param in keys)))

    results = []
    for combo in combinations:
        config = dict(zip(keys, combo))
        fitness = evaluate(config)
        config['fitness'] = fitness
        results.append(config)

    results_df = pd.DataFrame(results)
    return results_df

def latin_hypercube_sampling(param_distributions, n_iter, evaluate):
    """
    Perform Latin Hypercube Sampling (LHS) over the given parameter distributions.

    Parameters:
    - param_distributions: dict
        Dictionary where keys are parameter names and values are distributions or lists.
        Distributions should be from scipy.stats with a 'ppf' method or lists for categorical variables.

    - n_iter: int
        Number of iterations/samples to draw.

    - evaluate: function
        Function that takes a configuration (dict) and returns a fitness value.

    Returns:
    - results_df: pandas DataFrame
        DataFrame containing configurations and their fitness values.
    """

    continuous_params = {}
    categorical_params = {}
    for param, dist in param_distributions.items():
        if hasattr(dist, 'ppf'):
            continuous_params[param] = dist
        elif isinstance(dist, list):
            categorical_params[param] = dist
        else:
            raise ValueError(f"Unsupported distribution type for parameter '{param}'. Must be a list or a distribution with a 'ppf' method.")

    num_continuous = len(continuous_params)
    sampler = qmc.LatinHypercube(d=num_continuous, seed=None)
    sample = sampler.random(n=n_iter)  

    scaled_sample = np.empty_like(sample)
    for i, (param, dist) in enumerate(continuous_params.items()):
        scaled_sample[:, i] = dist.ppf(sample[:, i])

    results = []
    for i in range(n_iter):
        config = {}
        for j, param in enumerate(continuous_params.keys()):
            config[param] = scaled_sample[i, j]
        for param, choices in categorical_params.items():
            config[param] = np.random.choice(choices)
        # Evaluate fitness
        fitness = evaluate(config)
        config['fitness'] = fitness
        results.append(config)

    results_df = pd.DataFrame(results)
    return results_df

def sobol_sampling(param_distributions, n_iter, evaluate, scramble=True, seed=None):
    """
    Perform Sobol Sampling over the given parameter distributions.

    Parameters:
    - param_distributions: dict
        Dictionary where keys are parameter names and values are distributions or lists.
        Distributions should be from scipy.stats with a 'ppf' method or lists for categorical variables.

    - n_iter: int
        Number of iterations/samples to draw.

    - evaluate: function
        Function that takes a configuration (dict) and returns a fitness value.

    - scramble: bool, optional (default=True)
        Whether to scramble the Sobol sequence for better uniformity.

    - seed: int or None, optional (default=None)
        Seed for the random number generator (used if scramble is True).

    Returns:
    - results_df: pandas DataFrame
        DataFrame containing configurations and their fitness values.
    """

    continuous_params = {}
    categorical_params = {}
    for param, dist in param_distributions.items():
        if hasattr(dist, 'ppf'):
            continuous_params[param] = dist
        elif isinstance(dist, list):
            categorical_params[param] = dist
        else:
            raise ValueError(f"Unsupported distribution type for parameter '{param}'. Must be a list or a distribution with a 'ppf' method.")

    num_continuous = len(continuous_params)
    sampler = qmc.Sobol(d=num_continuous, scramble=scramble, seed=seed)
    sample = sampler.random(n=n_iter)  

    scaled_sample = np.empty_like(sample)
    for i, (param, dist) in enumerate(continuous_params.items()):
        scaled_sample[:, i] = dist.ppf(sample[:, i])

    results = []
    for i in range(n_iter):
        config = {}
        for j, param in enumerate(continuous_params.keys()):
            config[param] = scaled_sample[i, j]
        for param, choices in categorical_params.items():
            config[param] = np.random.choice(choices)
        fitness = evaluate(config)
        config['fitness'] = fitness
        results.append(config)

    results_df = pd.DataFrame(results)
    return results_df