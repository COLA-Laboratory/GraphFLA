import itertools
import math

from .base_problem import OptimizationProblem


class Max3Sat(OptimizationProblem):
    """
    Max-3-SAT optimization problem.

    This class represents the Max-3-SAT problem, where the goal is to find a
    Boolean variable assignment that maximizes the number of satisfied clauses
    in a formula where each clause has exactly three literals.

    Parameters
    ----------
    n : int
        The number of Boolean variables.
    alpha : float
        The clause-to-variable ratio (m/n), determining the number of clauses.
    seed : int or None, optional
        Seed for the random number generator used for clause generation.
    """

    def __init__(self, n, alpha, seed=None):
        """
        Initialize the Max-3-SAT problem with given parameters and seed.
        Generates m = floor(alpha * n) unique 3-SAT clauses randomly.
        """
        super().__init__(n, seed)
        if alpha <= 0:
            raise ValueError("Clause-to-variable ratio 'alpha' must be positive.")
        self.m = math.floor(alpha * n)  # Number of clauses
        if self.m == 0:
            print(f"Warning: alpha*n ({alpha}*{n}) resulted in 0 clauses.")
        self.alpha = alpha
        # Generate clauses using the instance's RNG for reproducibility
        self.clauses = self._generate_clauses()

    def _generate_clauses(self):
        """
        Generate a set of m unique 3-literal clauses using the instance's RNG.

        Returns
        -------
        list[tuple[tuple[int, bool]]]
            A list of clauses. Each clause is a tuple of 3 literals.
            Each literal is a tuple (variable_index, is_positive).
            Using tuples ensures clauses are hashable for the uniqueness check.
        """
        if self.n < 3:
            raise ValueError("Max-3-SAT requires at least n=3 variables.")

        clauses = set()
        attempts = 0
        max_attempts = (
            self.m * 100
        )  # Heuristic limit to prevent infinite loops if m is very large relative to possible unique clauses

        while len(clauses) < self.m and attempts < max_attempts:
            # Sample 3 distinct variable indices using the instance's RNG
            vars_indices = self.rng.sample(self.variables, 3)
            # For each variable, randomly choose True (positive) or False (negated) literal using instance's RNG
            clause_literals = tuple(
                sorted((var, self.rng.choice([True, False])) for var in vars_indices)
            )
            # Sorting makes the clause representation canonical, preventing equivalent clauses like ( (0,T),(1,F),(2,T) ) and ( (1,F),(0,T),(2,T) ) from being distinct.
            clauses.add(clause_literals)
            attempts += 1

        if len(clauses) < self.m:
            print(
                f"Warning: Could only generate {len(clauses)} unique clauses out of desired {self.m} after {max_attempts} attempts. Consider lower alpha or larger n."
            )

        return list(clauses)

    def get_all_configs(self):
        """
        Generate all possible configurations (Boolean assignments) for the Max-3-SAT problem.

        Returns
        -------
        iterator
            An iterator over all Boolean configurations (tuples of True/False) of length `n`.
        """
        # Efficiently yields configurations one by one.
        return itertools.product((True, False), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration (Boolean assignment) in the Max-3-SAT problem.
        Fitness is the number of satisfied clauses.

        Parameters
        ----------
        config : tuple or list
            A Boolean configuration (True/False values) representing a potential solution.

        Returns
        -------
        int
            The number of satisfied clauses for the given configuration.
        """
        config_tuple = tuple(config)  # Ensure tuple
        if len(config_tuple) != self.n:
            raise ValueError(
                f"Configuration length {len(config_tuple)} does not match problem dimension {self.n}"
            )
        # Check if config contains boolean values (or convertible like 0/1)
        # Optional: Add type/value check if necessary, e.g.
        # if not all(isinstance(val, bool) or val in {0, 1} for val in config_tuple):
        #    raise ValueError("Configuration must contain Boolean or 0/1 values.")

        num_satisfied = 0
        # Iterate through the pre-generated clauses
        for clause in self.clauses:
            # Check if *any* literal in the clause is satisfied by the config
            # A literal (var, is_positive) is satisfied if:
            # - is_positive is True and config[var] is True
            # - is_positive is False and config[var] is False
            # This is equivalent to checking: config[var] == is_positive
            if any(
                config_tuple[var_idx] == is_positive for var_idx, is_positive in clause
            ):
                num_satisfied += 1

        # The fitness is simply the count of satisfied clauses.
        return num_satisfied


class Knapsack(OptimizationProblem):
    """
    Knapsack optimization problem.

    This class represents the 0-1 knapsack problem, where the goal is to select a subset of items
    that maximizes total value while keeping the total weight under a capacity constraint.

    Parameters
    ----------
    n : int
        The number of items available for selection.
    capacity_ratio : float, default=0.5
        The knapsack capacity as a ratio of the sum of all item weights.
        Must be between 0.0 and 1.0.
    correlation : float, default=0.0
        Correlation between item weights and values:
        - 0.0: uncorrelated (random weights and values)
        - 1.0: strongly correlated (value = weight + constant)
        - -1.0: inversely correlated (value = capacity - weight + constant)
    seed : int or None, optional
        Seed for the random number generator.
    """

    def __init__(self, n, capacity_ratio=0.5, correlation=0.0, seed=None):
        """
        Initialize the Knapsack problem with the given parameters.
        """
        super().__init__(n, seed)
        if not 0.0 < capacity_ratio <= 1.0:
            raise ValueError("capacity_ratio must be between 0.0 and 1.0")
        if not -1.0 <= correlation <= 1.0:
            raise ValueError("correlation must be between -1.0 and 1.0")

        self.correlation = correlation

        # Generate item weights and values
        self.weights, self.values = self._generate_items()

        # Calculate capacity based on the ratio of total weight
        total_weight = sum(self.weights)
        self.capacity = int(total_weight * capacity_ratio)

    def _generate_items(self):
        """
        Generate weights and values for all items based on the correlation parameter.

        Returns
        -------
        tuple
            (weights, values) as lists of integers
        """
        # Generate weights in the range [1, 100]
        weights = [self.rng.randint(1, 100) for _ in range(self.n)]

        if abs(self.correlation) < 0.01:  # Uncorrelated
            values = [self.rng.randint(1, 100) for _ in range(self.n)]
        elif self.correlation > 0:  # Positively correlated
            # Value is proportional to weight plus some noise
            constant = 10
            values = [
                int(w + constant + self.rng.uniform(-10, 10) * (1 - self.correlation))
                for w in weights
            ]
        else:  # Negatively correlated
            # Value is inversely proportional to weight
            constant = 100
            values = [
                int(constant - w + self.rng.uniform(-10, 10) * (1 + self.correlation))
                for w in weights
            ]
            # Ensure all values are positive
            values = [max(1, v) for v in values]

        return weights, values

    def get_all_configs(self):
        """
        Generate all possible binary configurations for the Knapsack problem.

        Returns
        -------
        iterator
            An iterator over all binary configurations (tuples) of length `n`.
        """
        return itertools.product((0, 1), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the Knapsack problem.

        The fitness is the total value of selected items if the weight constraint
        is satisfied, or 0 if the constraint is violated.

        Parameters
        ----------
        config : tuple or list
            A binary configuration representing item selection (1 = selected, 0 = not selected).

        Returns
        -------
        float
            The total value of selected items if weight constraint is satisfied, 0 otherwise.
        """
        config_tuple = tuple(config)  # Ensure tuple
        if len(config_tuple) != self.n:
            raise ValueError(
                f"Configuration length {len(config_tuple)} does not match problem dimension {self.n}"
            )

        # Calculate total weight and value of selected items
        total_weight = sum(self.weights[i] * config_tuple[i] for i in self.variables)
        total_value = sum(self.values[i] * config_tuple[i] for i in self.variables)

        # Return 0 if weight constraint is violated
        if total_weight > self.capacity:
            return 0.0

        return float(total_value)


# ...existing code...


class NumberPartitioning(OptimizationProblem):
    """
    Number Partitioning optimization problem.

    This class represents the number partitioning problem, where the goal is to
    divide a set of positive integers into two subsets such that the difference
    between the sums of the two subsets is minimized.

    Parameters
    ----------
    n : int
        The number of integers to partition.
    alpha : float, default=1.0
        Control parameter defining the ratio of bit precision to number of elements (k/n).
        Higher values generate larger numbers relative to the problem size.
    seed : int or None, optional
        Seed for the random number generator.
    """

    def __init__(self, n, alpha=1.0, seed=None):
        """
        Initialize the Number Partitioning problem with the given parameters.
        """
        super().__init__(n, seed)
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        self.alpha = alpha
        # Calculate bit precision based on alpha and n
        self.bit_precision = int(alpha * n)

        # Generate the set of numbers to be partitioned
        self.numbers = self._generate_numbers()
        self.total_sum = sum(self.numbers)

    def _generate_numbers(self):
        """
        Generate a set of positive integers using alpha to determine bit precision.

        Following the literature (Mertens, 1998), generates numbers uniformly
        from [1, 2^(alpha*n) - 1].

        Returns
        -------
        list
            A list of positive integers to be partitioned.
        """
        max_value = (1 << self.bit_precision) - 1  # 2^(alpha*n) - 1
        return [self.rng.randint(1, max_value) for _ in range(self.n)]

    def get_all_configs(self):
        """
        Generate all possible binary configurations for the Number Partitioning problem.

        Returns
        -------
        iterator
            An iterator over all binary configurations (tuples) of length `n`.
        """
        return itertools.product((0, 1), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the Number Partitioning problem.

        The fitness is the negated absolute difference between the sums of the two
        partitions. Using negation ensures that higher values are better (maximization).

        Parameters
        ----------
        config : tuple or list
            A binary configuration representing the partitioning (0 = first subset, 1 = second subset).

        Returns
        -------
        float
            The negated absolute difference between the sums of the two partitions.
        """
        config_tuple = tuple(config)  # Ensure tuple
        if len(config_tuple) != self.n:
            raise ValueError(
                f"Configuration length {len(config_tuple)} does not match problem dimension {self.n}"
            )

        # Calculate sum of the first partition (where config[i] == 0)
        sum_first = sum(self.numbers[i] * (1 - config_tuple[i]) for i in self.variables)

        # Sum of second partition can be derived from total and first
        sum_second = self.total_sum - sum_first

        # Return negative difference (higher is better, 0 is perfect partition)
        return -abs(sum_first - sum_second)
