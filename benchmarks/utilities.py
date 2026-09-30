"""Distance, sampling and problem-generation entry points."""

import numpy as np
from scipy.stats import uniform

from graphfla import distances, sampling, problems
from graphfla.filters import LandscapeFilter
from ._datasets import random_ordinal


class Distances:
    params = ["hamming_distance", "manhattan_distance", "mixed_distance"]
    param_names = ["method"]

    def setup(self, method):
        self.X = np.random.default_rng(0).integers(
            0, 8, size=(10000, 12), dtype=np.uint8
        )
        self.types = {i: "ordinal" if i % 2 else "categorical" for i in range(12)}
        self.call = getattr(distances, method)

    def time_distance(self, method):
        self.call(self.X, self.X[0], self.types)

    def peakmem_distance(self, method):
        self.call(self.X, self.X[0], self.types)


def objective(values):
    return sum(values.values())


class Sampling:
    params = [
        "random_search",
        "grid_search",
        "latin_hypercube_sampling",
        "sobol_sampling",
    ]
    param_names = ["method"]

    def setup(self, method):
        self.grid = {"x": list(range(16)), "y": list(range(16))}
        self.distributions = {"x": uniform(), "y": [0, 1, 2]}

    def _run(self, method):
        if method == "grid_search":
            return sampling.grid_search(self.grid, objective)
        if method == "random_search":
            np.random.seed(0)
            return sampling.random_search(self.distributions, 256, objective)
        return getattr(sampling, method)(self.distributions, 256, objective, seed=0)

    def time_sample(self, method):
        self._run(method)

    def peakmem_sample(self, method):
        self._run(method)


class Problems:
    params = [name for name in problems.__all__ if name != "OptimizationProblem"]
    param_names = ["problem"]

    def setup(self, problem):
        kwargs = (
            {"k": 2}
            if problem == "NK"
            else {"alpha": 2}
            if problem == "Max3Sat"
            else {}
        )
        self.problem = getattr(problems, problem)(n=8, seed=0, **kwargs)

    def time_evaluate(self, problem):
        self.problem.evaluate((0, 1) * 4)

    def time_get_data(self, problem):
        self.problem.get_data()

    def peakmem_get_data(self, problem):
        self.problem.get_data()


class Filtering:
    def setup(self):
        self.X, self.fitness = random_ordinal(20, 3)
        self.filter = LandscapeFilter.fitness_threshold(0)

    def time_filter_data(self):
        self.filter.filter_data(self.X, self.fitness)

    def peakmem_filter_data(self):
        self.filter.filter_data(self.X, self.fitness)
