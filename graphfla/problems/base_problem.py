import random


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

    def _binary_string_to_config(self, s: str):
        """
        Convert a binary string to a configuration for evaluation.

        Override in subclasses that use Boolean (True/False) instead of int (0/1).

        Parameters
        ----------
        s : str
            Binary string (e.g., "0101").

        Returns
        -------
        tuple
            Configuration suitable for evaluate().
        """
        return tuple(int(c) for c in s)

    def get_data(self):
        """
        Generate configurations and their fitness values.

        Warning: This method evaluates *all* possible configurations.
        For problems with a large search space (e.g., high-dimensional binary problems),
        this can be extremely computationally expensive and memory-intensive.
        Use with caution for large 'n'.

        Returns
        -------
        tuple of (list[str], list[float])
            X: List of binary strings (e.g., ["0000000000", "0000000001", ...]).
            f: List of fitness values corresponding to each configuration.
        """
        try:
            n = self.n
            total = 1 << n  # 2**n
            X = [None] * total
            f_list = [None] * total
            for i in range(total):
                s = format(i, f"0{n}b")
                X[i] = s
                config = self._binary_string_to_config(s)
                f_list[i] = self.evaluate(config)
            return X, f_list
        except NotImplementedError:
            print("Warning: get_data is not fully implemented for this problem.")
            return [], []
        except MemoryError:
            print(
                f"Error: Generating data for n={self.n} requires too much memory. "
                f"The search space is likely too large (2^{self.n})."
            )
            return [], []
