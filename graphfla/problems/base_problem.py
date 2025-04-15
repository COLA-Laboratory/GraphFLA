import random
import pandas as pd


class OptimizationProblem:
    """
    Base class for defining optimization problems.

    This class provides a framework for representing optimization problems
    and includes methods for evaluating solutions and generating data for analysis.
    Subclasses should implement specific optimization problem behavior.

    Parameters
    ----------
    n : int
        The number of variables in the optimization problem.
    seed : int or None, optional
        Seed for the random number generator to ensure reproducibility.
        If None, the generator is initialized without a specific seed.
    """

    def __init__(self, n, seed=None):
        """
        Initialize the optimization problem with a given number of variables and seed.
        """
        if n <= 0:
            raise ValueError("Number of variables 'n' must be positive.")
        self.n = n
        self.variables = range(n)
        self.seed = seed
        # Use a dedicated RNG instance for reproducibility within the problem
        self.rng = random.Random(seed)

    def evaluate(self, config):
        """
        Evaluate the fitness of a given configuration.

        This method should be implemented by subclasses to define the
        specific evaluation criteria for the optimization problem.

        Parameters
        ----------
        config : tuple or list
            A configuration representing a potential solution.
            Using tuples is generally preferred for hashability (e.g., dictionary keys).

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_all_configs(self):
        """
        Generate all possible configurations for the problem.

        This method should be implemented by subclasses to provide the
        complete set of possible configurations for the problem.
        For high-dimensional binary problems (large n), iterating through all
        2^n configurations can be computationally infeasible.

        Returns
        -------
        iterator
            An iterator over all possible configurations.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "Subclasses should define this method to generate all configurations."
        )

    def get_data(self):
        """
        Generate a DataFrame containing configurations and their fitness values.

        Warning: This method evaluates *all* possible configurations.
        For problems with a large search space (e.g., high-dimensional binary problems),
        this can be extremely computationally expensive and memory-intensive.
        Use with caution for large 'n'.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns `config` (tuple of variables) and `fitness`.
        """
        try:
            all_configs_iter = self.get_all_configs()
            # Use a generator expression for potentially better memory efficiency if intermediate list is huge
            config_fitness_pairs = (
                (config, self.evaluate(config)) for config in all_configs_iter
            )
            # Construct DataFrame directly from the generator
            data = pd.DataFrame(config_fitness_pairs, columns=["config", "fitness"])
            # Keep config as tuple, it's generally more efficient and hashable
            # data["config"] = data["config"].apply(list) # Avoid converting to list unless necessary
            return data
        except NotImplementedError:
            print("Warning: get_all_configs is not implemented for this problem.")
            return pd.DataFrame(columns=["config", "fitness"])
        except MemoryError:
            print(
                f"Error: Generating data for n={self.n} requires too much memory. "
                "The search space is likely too large (2^{self.n})."
            )
            return pd.DataFrame(columns=["config", "fitness"])
