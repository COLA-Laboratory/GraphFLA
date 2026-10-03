"""Shared interface for binary optimization problems."""

import math
from numbers import Integral, Real
import random
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np


def _validate_real(value, name):
    """Return a finite real parameter without accepting strings or booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


class OptimizationProblem:
    """Base class for binary maximization problems.

    Subclasses implement :meth:`evaluate`; :meth:`get_data` enumerates the
    binary search space and returns inputs suitable for a BooleanLandscape.

    Parameters
    ----------
    n : int
        Number of binary variables. Must be positive.
    seed : int or None, default=None
        Seed for the instance's random number generator. An integer makes
        repeated instances reproducible; None uses system-provided randomness.

    Attributes
    ----------
    variables : range
        Zero-based variable indices, ``range(n)``.
    rng : random.Random
        Instance-local generator; global random state is not modified.

    Notes
    -----
    All concrete problems use larger fitness values for better solutions.
    NK, RoughMountFuji and HoC draw and cache random values on first access:
    reproducing a realization requires the same seed and evaluation order.
    Evaluating a cached configuration does not draw new values. Treat generated
    model attributes as read-only; construct a new instance to change the model.

    Examples
    --------
    >>> from graphfla.problems import OptimizationProblem
    >>> class CountOnes(OptimizationProblem):
    ...     def evaluate(self, config):
    ...         return sum(self._validate_config(config))
    >>> CountOnes(n=2).get_data()
    (['00', '01', '10', '11'], [0, 1, 1, 2])
    """

    def __init__(self, n: int, seed: Optional[int] = None) -> None:
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral):
            raise TypeError("n must be a positive integer.")
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
                raise TypeError("seed must be an integer or None.")
            seed = int(seed)
        self.n = int(n)
        self.variables = range(self.n)
        self.seed = seed
        self.rng = random.Random(seed)

    def _validate_config(self, config) -> Tuple[int, ...]:
        """Normalize one binary configuration before evaluation or cache updates."""
        if isinstance(config, str):
            self._validate_binary_string(config)
            return tuple(map(int, config))
        try:
            values = tuple(config)
        except TypeError as exc:
            raise ValueError(
                "config must be a one-dimensional binary sequence."
            ) from exc
        if len(values) != self.n:
            raise ValueError(f"config must have length {self.n}; got {len(values)}.")
        if any(
            not isinstance(value, (Real, np.bool_)) or value not in (0, 1)
            for value in values
        ):
            raise ValueError("config must contain only 0 and 1 (or booleans).")
        return tuple(int(value) for value in values)

    def _validate_binary_string(self, config: str) -> None:
        """Check a string without materializing its bits as Python integers."""
        if len(config) != self.n or not set(config) <= {"0", "1"}:
            raise ValueError(f"config must be a binary string of length {self.n}.")

    def evaluate(self, config: Union[str, Sequence[int]]) -> Union[int, float]:
        """Return the fitness of one configuration.

        Parameters
        ----------
        config : str or array-like of shape (n,)
            Binary string of length n, or variable values encoded as 0/1 or booleans.

        Returns
        -------
        fitness : int or float
            Objective value, with larger values indicating better solutions.
            The concrete subclass determines the scalar type.

        Raises
        ------
        NotImplementedError
            Always raised by the base class; subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement evaluate().")

    def _binary_string_to_config(self, s: str) -> Tuple[int, ...]:
        """Convert an enumerated binary string to an evaluation input."""
        return tuple(int(c) for c in s)

    def iter_data(self) -> Iterator[Tuple[str, Union[int, float]]]:
        """Yield binary configurations and fitness values one at a time.

        Yields
        ------
        config : str
            Binary string of length n, in ascending binary order.
        fitness : int or float
            Fitness of config, with the scalar type returned by :meth:`evaluate`.

        See Also
        --------
        get_data : Materialize the complete search space as two lists.

        Notes
        -----
        Iteration avoids storing the output lists and can be stopped early.
        Model-specific random-value caches still grow as configurations are
        visited, and complete enumeration still takes exponential time.

        Examples
        --------
        >>> from itertools import islice
        >>> from graphfla.problems import Eggbox
        >>> list(islice(Eggbox(n=15).iter_data(), 2))
        [('000000000000000', 0.0), ('000000000000001', 1.0)]
        """
        for i in range(1 << self.n):
            s = format(i, f"0{self.n}b")
            yield s, self.evaluate(self._binary_string_to_config(s))

    def get_data(self) -> Tuple[List[str], List[Union[int, float]]]:
        """Return all binary configurations and their fitness values.

        Returns
        -------
        X : list of str of length 2**n
            Binary strings of length n, in ascending binary order, from all
            zeros to all ones.
        f : list of int or float of length 2**n
            Fitness values aligned with X. Scalar types match :meth:`evaluate`.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement :meth:`evaluate`.
        MemoryError
            If the complete search space cannot be allocated.

        See Also
        --------
        iter_data : Yield configurations and fitness values without output lists.

        Notes
        -----
        This method materializes all 2**n configurations. Time and memory grow
        exponentially with n; use :meth:`evaluate` for selected configurations
        in larger problems. Existing random-value caches are retained.

        Examples
        --------
        >>> from graphfla.problems import Additive
        >>> problem = Additive(n=2, seed=0)
        >>> X, f = problem.get_data()
        >>> X
        ['00', '01', '10', '11']
        >>> f[1] == problem.evaluate(X[1])
        True
        """
        total = 1 << self.n
        X = [None] * total
        f_list = [None] * total
        for i, (config, fitness) in enumerate(self.iter_data()):
            X[i] = config
            f_list[i] = fitness
        return X, f_list
