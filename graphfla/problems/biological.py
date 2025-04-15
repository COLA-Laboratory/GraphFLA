import itertools
import math
import pandas as pd
import numpy as np

from .base_problem import OptimizationProblem


class NK(OptimizationProblem):
    """
    NK model for fitness landscapes.

    This class represents an NK landscape, a model used to study the complexity of
    adaptive landscapes based on interactions among components.

    Parameters
    ----------
    n : int
        The number of variables (genes) in the problem.
    k : int
        The number of interacting components (genes) for each variable. Must be < n.
    exponent : float, default=1.0
        An exponent used to transform the final fitness value.
    seed : int or None, optional
        Seed for the random number generator.
    """

    def __init__(self, n, k, exponent=1.0, seed=None):
        """
        Initialize the NK model with given parameters.
        """
        super().__init__(n, seed)
        if not 0 <= k < n:
            raise ValueError("k must be non-negative and less than n.")
        self.k = k
        self.exponent = float(exponent)  # Ensure exponent is float

        # Generate dependencies using the instance's RNG
        self.dependence = [
            tuple(sorted([i] + self.rng.sample(list(set(self.variables) - {i}), k)))
            for i in self.variables
        ]

        # Initialize fitness contribution table lazily during evaluation
        # Using a dictionary is efficient for sparse access needed here.
        self.values = {}

    def get_all_configs(self):
        """
        Generate all possible binary configurations for the NK model.

        Returns
        -------
        iterator
            An iterator over all binary configurations (tuples) of length `n`.
        """
        # itertools.product is memory-efficient as it yields configurations one by one.
        return itertools.product((0, 1), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the NK model.

        Parameters
        ----------
        config : tuple or list
            A binary configuration representing a potential solution.

        Returns
        -------
        float
            The fitness value of the configuration.
        """
        config = tuple(config)  # Ensure config is a tuple for consistent hashing
        if len(config) != self.n:
            raise ValueError(
                f"Configuration length {len(config)} does not match problem dimension {self.n}"
            )

        total_value = 0.0
        for i in self.variables:
            # Create the key for the fitness contribution lookup
            # This involves selecting elements based on the dependency list
            dependent_indices = self.dependence[i]
            key_elements = (i,) + tuple(config[j] for j in dependent_indices)

            # Check if the contribution for this sub-configuration is already computed
            if key_elements not in self.values:
                # Compute and store the contribution using the instance's RNG
                self.values[key_elements] = self.rng.random()

            total_value += self.values[key_elements]

        # Normalize fitness by n
        normalized_value = total_value / self.n

        # Apply exponent if it's not 1
        if self.exponent != 1.0:
            # Use max(0, ...) to avoid potential domain errors with negative bases if exponent is not an integer
            # Although normalized_value should be >= 0 here, it's safer
            normalized_value = math.pow(max(0.0, normalized_value), self.exponent)

        return normalized_value


class RoughMountFuji(OptimizationProblem):
    """
    Rough Mount Fuji (RMF) model for fitness landscapes.

    This model combines a smooth fitness component (additive) with a rugged
    (random) component, creating a landscape with controlled ruggedness.

    Parameters
    ----------
    n : int
        The number of variables in the optimization problem.
    alpha : float, default=0.5
        The ruggedness parameter that determines the balance between the smooth
        and rugged components. Must be between 0 (smoothest) and 1 (most rugged).
    seed : int or None, optional
        Seed for the random number generator.
    """

    def __init__(self, n, alpha=0.5, seed=None):
        """
        Initialize the RMF model with the given number of variables,
        ruggedness parameter, and seed.
        """
        super().__init__(n, seed)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")
        self.alpha = alpha

        # Generate smooth contributions using the instance's RNG
        # Using a list or numpy array is efficient for O(n) access later.
        self.smooth_contribution = np.array(
            [self.rng.uniform(-1.0, 1.0) for _ in range(n)]
        )

        # Initialize rugged component values lazily during evaluation
        self.random_values = {}

    def get_all_configs(self):
        """
        Generate all possible binary configurations for the RMF model.

        Returns
        -------
        iterator
            An iterator over all binary configurations (tuples) of length `n`.
        """
        return itertools.product((0, 1), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the RMF model.

        The fitness is computed as a weighted sum of the smooth and rugged components.

        Parameters
        ----------
        config : tuple or list
            A binary configuration representing a potential solution.

        Returns
        -------
        float
            The fitness value of the configuration.
        """
        config_tuple = tuple(config)  # Ensure tuple for hashing
        if len(config_tuple) != self.n:
            raise ValueError(
                f"Configuration length {len(config_tuple)} does not match problem dimension {self.n}"
            )

        # Smooth component: Efficient calculation using numpy dot product if config is array-like
        # or list comprehension/sum otherwise.
        # Convert config_tuple to numpy array for potential speedup in high-dim
        # config_array = np.array(config_tuple)
        # smooth_value = np.dot(self.smooth_contribution, config_array)
        # Using sum is often fine and avoids numpy dependency if not otherwise needed:
        smooth_value = sum(
            self.smooth_contribution[i] * config_tuple[i] for i in self.variables
        )

        # Rugged component: Lookup or generate using the instance's RNG
        if config_tuple not in self.random_values:
            self.random_values[config_tuple] = self.rng.random()
        rugged_value = self.random_values[config_tuple]

        # Combine the smooth and rugged components using alpha
        fitness = (1.0 - self.alpha) * smooth_value + self.alpha * rugged_value

        return fitness


class HoC(RoughMountFuji):
    """
    House of Cards (HoC) model for fitness landscapes.

    This model represents a purely random fitness landscape where each configuration
    is assigned a random fitness value independently of others. It is implemented
    as a special case of the Rough Mount Fuji (RMF) model where the ruggedness
    parameter alpha is 1.0.

    Parameters
    ----------
    n : int
        The number of variables in the optimization problem.
    seed : int or None, optional
        Seed for the random number generator.
    """

    def __init__(self, n, seed=None):
        """
        Initialize the HoC model with the given number of variables and seed.
        """
        # Initialize the RMF model with alpha = 1.0 (purely random) and pass the seed
        super().__init__(n, alpha=1.0, seed=seed)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the HoC model.

        The fitness is purely random for each configuration. This overrides the RMF
        evaluate slightly for clarity, although RMF with alpha=1.0 would yield
        the same result (smooth part becomes zero).

        Parameters
        ----------
        config : tuple or list
            A binary configuration representing a potential solution.

        Returns
        -------
        float
            The random fitness value of the configuration.
        """
        config_tuple = tuple(config)  # Ensure tuple for hashing
        if len(config_tuple) != self.n:
            raise ValueError(
                f"Configuration length {len(config_tuple)} does not match problem dimension {self.n}"
            )

        # Purely random fitness, generated using the RNG inherited from RMF (which uses the seed)
        if config_tuple not in self.random_values:
            # self.rng is initialized in the parent class __init__
            self.random_values[config_tuple] = self.rng.random()

        return self.random_values[config_tuple]


class Additive(OptimizationProblem):
    """
    Additive model for fitness landscapes.

    This model represents a fitness landscape where the fitness of a configuration
    is the sum of independent contributions from each variable depending on its value (0 or 1).

    Parameters
    ----------
    n : int
        The number of variables in the optimization problem.
    seed : int or None, optional
        Seed for the random number generator.
    """

    def __init__(self, n, seed=None):
        """
        Initialize the Additive model with the given number of variables and seed.
        """
        super().__init__(n, seed)
        # Assign random fitness contributions using the instance's RNG
        # Store as a list of tuples: [(contrib_for_0, contrib_for_1), ...]
        # This is efficient for access during evaluation.
        self.contributions = [
            (self.rng.random(), self.rng.random()) for _ in self.variables
        ]
        # Alternative using numpy (might be slightly faster for very large n):
        # self.contributions_np = np.array([
        #     [self.rng.random(), self.rng.random()] for _ in self.variables
        # ]) # Shape (n, 2)

    def get_all_configs(self):
        """
        Generate all possible binary configurations for the Additive model.

        Returns
        -------
        iterator
            An iterator over all binary configurations (tuples) of length `n`.
        """
        return itertools.product((0, 1), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the Additive model.

        The fitness is computed as the sum of independent contributions of each variable.

        Parameters
        ----------
        config : tuple or list
            A binary configuration representing a potential solution.

        Returns
        -------
        float
            The fitness value of the configuration.
        """
        config_tuple = tuple(config)  # Ensure tuple
        if len(config_tuple) != self.n:
            raise ValueError(
                f"Configuration length {len(config_tuple)} does not match problem dimension {self.n}"
            )

        # Efficient sum using list comprehension and direct indexing
        fitness = sum(self.contributions[i][config_tuple[i]] for i in self.variables)

        # Numpy alternative (potentially faster for very large n):
        # fitness = np.sum(self.contributions_np[np.arange(self.n), config_tuple])

        return fitness


class Eggbox(OptimizationProblem):
    """
    Eggbox model for fitness landscapes.

    This model represents a fitness landscape with regularly spaced peaks and valleys,
    resembling the structure of an egg carton. Fitness depends only on the sum of
    the elements in the configuration. This problem has no random components.

    Parameters
    ----------
    n : int
        The number of variables in the optimization problem.
    frequency : float, default=1.0
        The frequency of the peaks and troughs in the landscape. Higher frequency
        means more peaks within the range of possible sums (0 to n).
    seed : int or None, optional
        Seed for the random number generator. (Note: This problem is deterministic,
        but the parameter is included for consistency with the base class).
    """

    def __init__(self, n, frequency=1.0, seed=None):
        """
        Initialize the Eggbox model with the given number of variables and frequency.
        """
        # Pass seed to superclass, although it won't be used here.
        super().__init__(n, seed)
        if frequency <= 0:
            raise ValueError("Frequency must be positive.")
        self.frequency = float(frequency)

    def get_all_configs(self):
        """
        Generate all possible binary configurations for the Eggbox model.

        Returns
        -------
        iterator
            An iterator over all binary configurations (tuples) of length `n`.
        """
        return itertools.product((0, 1), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the Eggbox model.

        The fitness is determined based on the sum of the variables and a sine function,
        creating a periodic landscape with alternating peaks and valleys.

        Parameters
        ----------
        config : tuple or list
            A binary configuration representing a potential solution.

        Returns
        -------
        float
            The fitness value of the configuration.
        """
        # No need to convert to tuple if only sum is needed, but good practice
        config_tuple = tuple(config)
        if len(config_tuple) != self.n:
            raise ValueError(
                f"Configuration length {len(config_tuple)} does not match problem dimension {self.n}"
            )

        # Sum of the binary configuration elements (efficient)
        sum_of_elements = sum(config_tuple)

        # Fitness value using a sine function squared to ensure non-negativity
        # The argument scales with frequency and the sum
        argument = self.frequency * float(sum_of_elements) * math.pi
        fitness = math.sin(argument) ** 2
        return fitness
