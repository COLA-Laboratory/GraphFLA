"""Random binary instances of combinatorial optimization problems."""

import math
from typing import Optional, Sequence, Union

from .base_problem import OptimizationProblem, _validate_real


class Max3Sat(OptimizationProblem):
    """Max-3-SAT problem with uniformly sampled distinct clauses.

    Parameters
    ----------
    n : int
        Number of Boolean variables. Must be at least 3.
    alpha : float
        Positive, finite clause-to-variable ratio. The number of clauses is
        ``floor(alpha * n)``, which must not exceed ``8 * comb(n, 3)``.
        Ratios giving zero clauses are allowed and yield zero fitness.
    seed : int or None, default=None
        Seed for the instance's random number generator. An integer gives
        reproducible clauses, sampled at construction. None uses
        system-provided randomness.

    Attributes
    ----------
    m : int
        Number of clauses.
    clauses : list of tuple
        Each clause contains three ``(variable_index, is_positive)`` literals
        with distinct zero-based indices in ascending order.

    Examples
    --------
    >>> from graphfla.problems import Max3Sat
    >>> problem = Max3Sat(n=3, alpha=8 / 3, seed=0)
    >>> len(problem.clauses)
    8
    >>> problem.evaluate([False, True, False])
    7
    """

    def __init__(self, n: int, alpha: float, seed: Optional[int] = None) -> None:
        super().__init__(n, seed)
        if self.n < 3:
            raise ValueError("Max3Sat requires n >= 3.")
        alpha = _validate_real(alpha, "alpha")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not math.isfinite(alpha * self.n):
            raise ValueError("alpha * n must be finite.")
        self.m = math.floor(alpha * self.n)
        if self.m > 8 * math.comb(self.n, 3):
            raise ValueError("floor(alpha * n) exceeds the number of distinct clauses.")
        self.alpha = alpha
        self.clauses = self._generate_clauses()

    def _generate_clauses(self):
        """Sample clauses without replacement, preserving the seeded random stream."""
        clauses = set()
        while len(clauses) < self.m:
            indices = self.rng.sample(self.variables, 3)
            # Canonical order makes permuted versions of one clause identical.
            clause = tuple(sorted((i, self.rng.choice([True, False])) for i in indices))
            clauses.add(clause)
        return sorted(clauses)

    def _binary_string_to_config(self, s: str):
        """Convert an enumerated binary string to a Boolean assignment."""
        return tuple(c == "1" for c in s)

    def evaluate(self, config: Union[str, Sequence[int]]) -> int:
        """Return the fitness of one configuration.

        Parameters
        ----------
        config : str or array-like of shape (n,)
            Binary string of length n, or assignments encoded as 0/1 or booleans.

        Returns
        -------
        fitness : int
            Number of satisfied clauses, between 0 and m inclusive.

        Raises
        ------
        ValueError
            If config does not contain n binary values.
        """
        config = self._validate_config(config)
        return sum(
            any(config[i] == is_positive for i, is_positive in clause)
            for clause in self.clauses
        )


class Knapsack(OptimizationProblem):
    """Random 0-1 knapsack problem with a zero penalty for infeasible selections.

    Parameters
    ----------
    n : int
        Number of items. Must be positive.
    capacity_ratio : float, default=0.5
        Capacity as a fraction of total item weight, in (0, 1]. Capacity is
        rounded down to an integer and may be zero.
    correlation : float, default=0.0
        Weight-value coupling parameter, in [-1, 1], not a target Pearson
        correlation. Values with absolute magnitude below 0.01 select
        independent weights and values; other values control the formulas below.
    seed : int or None, default=None
        Seed for the instance's random number generator. An integer gives
        reproducible items, sampled at construction. None uses
        system-provided randomness.

    Attributes
    ----------
    weights : list of int
        Item weights, sampled uniformly from 1 through 100 inclusive.
    values : list of int
        Positive item values, generated according to correlation.
    capacity : int
        Maximum allowed total weight, ``floor(capacity_ratio * sum(weights))``.

    Notes
    -----
    For |correlation| < 0.01, values are independent uniform integers from 1
    through 100. Otherwise, with w the weight, c the correlation parameter and
    u uniform on [-10, 10], values are ``int(w + 10 + (1-c)*u)`` for c > 0
    and ``max(1, int(100 - w + (1+c)*u))`` for c < 0. These are generation
    conventions, not constraints on the realized sample correlation.

    Examples
    --------
    >>> from graphfla.problems import Knapsack
    >>> problem = Knapsack(n=3, capacity_ratio=0.5, seed=0)
    >>> problem.evaluate([0, 0, 0])
    0.0
    >>> problem.evaluate([1, 1, 1])
    0.0
    >>> problem.evaluate([1, 0, 0]) == float(problem.values[0])
    True
    """

    def __init__(
        self,
        n: int,
        capacity_ratio: float = 0.5,
        correlation: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(n, seed)
        capacity_ratio = _validate_real(capacity_ratio, "capacity_ratio")
        correlation = _validate_real(correlation, "correlation")
        if not 0.0 < capacity_ratio <= 1.0:
            raise ValueError("capacity_ratio must be in (0, 1].")
        if not -1.0 <= correlation <= 1.0:
            raise ValueError("correlation must be in [-1, 1].")
        self.capacity_ratio = capacity_ratio
        self.correlation = correlation
        self.weights, self.values = self._generate_items()
        self.capacity = int(sum(self.weights) * capacity_ratio)

    def _generate_items(self):
        """Draw positive integer weights and values using the configured coupling."""
        weights = [self.rng.randint(1, 100) for _ in self.variables]
        if abs(self.correlation) < 0.01:
            values = [self.rng.randint(1, 100) for _ in self.variables]
        elif self.correlation > 0:
            values = [
                int(w + 10 + self.rng.uniform(-10, 10) * (1 - self.correlation))
                for w in weights
            ]
        else:
            values = [
                max(
                    1, int(100 - w + self.rng.uniform(-10, 10) * (1 + self.correlation))
                )
                for w in weights
            ]
        return weights, values

    def evaluate(self, config: Union[str, Sequence[int]]) -> float:
        """Return the fitness of one configuration.

        Parameters
        ----------
        config : str or array-like of shape (n,)
            Binary string of length n, or selections encoded as 0/1 or booleans.
            One selects an item.

        Returns
        -------
        fitness : float
            Total selected value if weight is at most capacity; zero otherwise.

        Raises
        ------
        ValueError
            If config does not contain n binary values.
        """
        config = self._validate_config(config)
        total_weight = sum(self.weights[i] * config[i] for i in self.variables)
        if total_weight > self.capacity:
            return 0.0
        return float(sum(self.values[i] * config[i] for i in self.variables))


class NumberPartitioning(OptimizationProblem):
    """Random integer partitioning problem expressed as fitness maximization.

    Parameters
    ----------
    n : int
        Number of integers to partition. Must be positive.
    alpha : float, default=1.0
        Positive, finite ratio of bit precision to number of elements. The bit
        precision is ``floor(alpha * n)``, which must be at least one.
    seed : int or None, default=None
        Seed for the instance's random number generator. An integer gives
        reproducible numbers, sampled at construction. None uses
        system-provided randomness.

    Attributes
    ----------
    bit_precision : int
        Number of bits used for generated integers.
    numbers : list of int
        n independent uniform integers from 1 through ``2**bit_precision - 1``
        inclusive. Repeated values are allowed.
    total_sum : int
        Sum of all generated integers.

    Examples
    --------
    >>> from graphfla.problems import NumberPartitioning
    >>> problem = NumberPartitioning(n=3, seed=0)
    >>> problem.numbers
    [7, 4, 7]
    >>> problem.evaluate([0, 1, 0])
    -10
    >>> problem.evaluate([1, 0, 1])
    -10
    """

    def __init__(self, n: int, alpha: float = 1.0, seed: Optional[int] = None) -> None:
        super().__init__(n, seed)
        alpha = _validate_real(alpha, "alpha")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not math.isfinite(alpha * self.n):
            raise ValueError("alpha * n must be finite.")
        self.alpha = alpha
        self.bit_precision = int(alpha * self.n)
        if self.bit_precision < 1:
            raise ValueError("floor(alpha * n) must be at least 1.")
        self.numbers = self._generate_numbers()
        self.total_sum = sum(self.numbers)

    def _generate_numbers(self):
        """Sample positive integers with the requested bit precision."""
        max_value = (1 << self.bit_precision) - 1
        return [self.rng.randint(1, max_value) for _ in self.variables]

    def evaluate(self, config: Union[str, Sequence[int]]) -> int:
        """Return the fitness of one configuration.

        Parameters
        ----------
        config : str or array-like of shape (n,)
            Binary string of length n, or assignments encoded as 0/1 or booleans.
            Zero selects the first subset and one selects the second.

        Returns
        -------
        fitness : int
            Negative absolute difference between subset sums. Zero is optimal;
            integer arithmetic preserves the full precision of generated values.

        Raises
        ------
        ValueError
            If config does not contain n binary values.
        """
        config = self._validate_config(config)
        sum_first = sum(self.numbers[i] * (1 - config[i]) for i in self.variables)
        sum_second = self.total_sum - sum_first
        return -abs(sum_first - sum_second)
