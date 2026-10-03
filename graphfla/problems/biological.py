"""Binary fitness landscape models."""

import math
from numbers import Integral
from typing import Optional, Sequence, Union

import numpy as np

from .base_problem import OptimizationProblem, _validate_real


class NK(OptimizationProblem):
    """NK landscape with random interaction partners and fitness contributions.

    Parameters
    ----------
    n : int
        Number of binary variables. Must be positive.
    k : int
        Number of other variables contributing to each variable's fitness
        component. Must satisfy ``0 <= k < n``.
    exponent : float, default=1.0
        Finite power applied to the mean fitness contribution. Positive values
        preserve fitness ordering; zero gives constant fitness and negative
        values reverse ordering (undefined if the mean contribution is zero).
    seed : int or None, default=None
        Seed for the instance's random number generator. An integer gives
        reproducible values for the same evaluation order. None uses
        system-provided randomness.

    Attributes
    ----------
    dependence : list of tuple of int
        For each variable, its own index and k distinct random partner indices,
        sorted in ascending order.
    values : dict
        Cached contributions in [0, 1). Keys are internal integer encodings
        of variable/background pairs; use evaluate to access fitness values.

    Notes
    -----
    Fitness is the mean of n independent uniform contributions, raised to
    ``exponent``. Contributions are sampled on first access and then cached;
    the seed and evaluation order together determine the realized landscape.
    With k=0 and exponent=1, the model is additive. At most
    ``n * 2**(k + 1)`` contributions are cached.

    Examples
    --------
    >>> from graphfla.problems import NK
    >>> problem = NK(n=3, k=1, seed=0)
    >>> fitness = problem.evaluate([0, 1, 0])
    >>> 0.0 <= fitness < 1.0
    True
    >>> problem.evaluate([0, 1, 0]) == fitness
    True
    """

    def __init__(
        self, n: int, k: int, exponent: float = 1.0, seed: Optional[int] = None
    ) -> None:
        super().__init__(n, seed)
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
            raise TypeError("k must be an integer.")
        if not 0 <= k < self.n:
            raise ValueError("k must satisfy 0 <= k < n.")
        self.k = int(k)
        self.exponent = _validate_real(exponent, "exponent")
        self.dependence = [
            tuple(
                sorted([i] + self.rng.sample(list(set(self.variables) - {i}), self.k))
            )
            for i in self.variables
        ]
        self._dependence_masks = [
            sum(1 << j for j in indices) for indices in self.dependence
        ]
        self.values = {}

    def _binary_string_to_config(self, s: str) -> str:
        """Keep enumerated inputs compact for the encoded NK evaluator."""
        return s

    def evaluate(self, config: Union[str, Sequence[int]]) -> float:
        """Return the fitness of one configuration.

        Parameters
        ----------
        config : str or array-like of shape (n,)
            Binary string of length n, or variable values encoded as 0/1 or booleans.

        Returns
        -------
        fitness : float
            Mean fitness contribution raised to ``exponent``.

        Raises
        ------
        ValueError
            If config does not contain n binary values.
        """
        if isinstance(config, str):
            self._validate_binary_string(config)
            encoded = int(config[::-1], 2)
        else:
            config = self._validate_config(config)
            encoded = sum(value << j for j, value in enumerate(config))
        total_value = 0.0
        for i, mask in enumerate(self._dependence_masks):
            # Pack the background and focal index without allocating k-bit tuples.
            key = (encoded & mask) * self.n + i
            value = self.values.get(key)
            if value is None:
                value = self.rng.random()
                self.values[key] = value
            total_value += value
        mean_value = total_value / self.n
        return (
            math.pow(mean_value, self.exponent) if self.exponent != 1.0 else mean_value
        )


class RoughMountFuji(OptimizationProblem):
    """Weighted additive and random binary fitness landscape.

    Parameters
    ----------
    n : int
        Number of binary variables. Must be positive.
    alpha : float, default=0.5
        Weight of the random component, in [0, 1]. Zero gives an additive
        landscape and one gives a House of Cards landscape.
    seed : int or None, default=None
        Seed for the instance's random number generator. An integer gives
        reproducible values for the same evaluation order. None uses
        system-provided randomness.

    Attributes
    ----------
    smooth_contribution : ndarray of shape (n,)
        Additive coefficients drawn independently and uniformly from [-1, 1].
    random_values : dict
        Cached independent uniform values in [0, 1), keyed by configuration.

    See Also
    --------
    HoC : Purely random special case with alpha=1.
    Additive : Additive model with a contribution for each binary state.

    Notes
    -----
    Fitness is ``(1 - alpha) * sum(w[i] * config[i]) + alpha * u(config)``.
    The additive sum is not normalized by n, so alpha is a mixing weight,
    not a fraction of fitness variance. Random values are drawn on first
    access and cached; evaluation order affects the realization for a fixed seed.

    Examples
    --------
    >>> from graphfla.problems import RoughMountFuji
    >>> problem = RoughMountFuji(n=3, alpha=0.0, seed=0)
    >>> problem.evaluate([0, 0, 0])
    0.0
    >>> round(problem.evaluate([1, 0, 0]), 4)
    0.6888
    """

    def __init__(self, n: int, alpha: float = 0.5, seed: Optional[int] = None) -> None:
        super().__init__(n, seed)
        alpha = _validate_real(alpha, "alpha")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = alpha
        self.smooth_contribution = np.array(
            [self.rng.uniform(-1.0, 1.0) for _ in self.variables]
        )
        self.random_values = {}

    def evaluate(self, config: Union[str, Sequence[int]]) -> float:
        """Return the fitness of one configuration.

        Parameters
        ----------
        config : str or array-like of shape (n,)
            Binary string of length n, or variable values encoded as 0/1 or booleans.

        Returns
        -------
        fitness : float
            Weighted sum of additive and random components.

        Raises
        ------
        ValueError
            If config does not contain n binary values.
        """
        config = self._validate_config(config)
        smooth_value = sum(
            self.smooth_contribution[i] * config[i] for i in self.variables
        )
        if config not in self.random_values:
            self.random_values[config] = self.rng.random()
        return float(
            (1.0 - self.alpha) * smooth_value + self.alpha * self.random_values[config]
        )


class HoC(RoughMountFuji):
    """House of Cards landscape with independent random fitness values.

    Parameters
    ----------
    n : int
        Number of binary variables. Must be positive.
    seed : int or None, default=None
        Seed for the instance's random number generator. An integer gives
        reproducible values for the same evaluation order. None uses
        system-provided randomness.

    Attributes
    ----------
    random_values : dict
        Cached independent uniform values in [0, 1), keyed by configuration.

    See Also
    --------
    RoughMountFuji : Mixture of additive and random fitness components.

    Notes
    -----
    This is RoughMountFuji with alpha=1, including its initial random draws.
    Fitness is drawn once per configuration, not once per evaluation; the seed
    and order of first visits together determine the realization.

    Examples
    --------
    >>> from graphfla.problems import HoC
    >>> problem = HoC(n=3, seed=0)
    >>> round(problem.evaluate([0, 1, 0]), 4)
    0.2589
    >>> problem.evaluate([0, 1, 0]) == problem.evaluate((False, True, False))
    True
    """

    def __init__(self, n: int, seed: Optional[int] = None) -> None:
        super().__init__(n, alpha=1.0, seed=seed)

    def evaluate(self, config: Union[str, Sequence[int]]) -> float:
        """Return the fitness of one configuration.

        Parameters
        ----------
        config : str or array-like of shape (n,)
            Binary string of length n, or variable values encoded as 0/1 or booleans.

        Returns
        -------
        fitness : float
            Cached uniform random value in [0, 1).

        Raises
        ------
        ValueError
            If config does not contain n binary values.
        """
        config = self._validate_config(config)
        if config not in self.random_values:
            self.random_values[config] = self.rng.random()
        return self.random_values[config]


class Additive(OptimizationProblem):
    """Binary landscape with independent contributions from each variable.

    Parameters
    ----------
    n : int
        Number of binary variables. Must be positive.
    seed : int or None, default=None
        Seed for the instance's random number generator. An integer gives
        reproducible contributions, sampled at construction. None uses
        system-provided randomness.

    Attributes
    ----------
    contributions : list of tuple of float
        Two independent uniform values in [0, 1) per variable, one for each
        binary state. Fitness is their sum, without normalization by n.

    Examples
    --------
    >>> from graphfla.problems import Additive
    >>> problem = Additive(n=2, seed=0)
    >>> round(problem.evaluate([0, 1]), 4)
    1.1033
    >>> X, f = problem.get_data()
    >>> len(X), len(f)
    (4, 4)
    """

    def __init__(self, n: int, seed: Optional[int] = None) -> None:
        super().__init__(n, seed)
        self.contributions = [
            (self.rng.random(), self.rng.random()) for _ in self.variables
        ]

    def evaluate(self, config: Union[str, Sequence[int]]) -> float:
        """Return the fitness of one configuration.

        Parameters
        ----------
        config : str or array-like of shape (n,)
            Binary string of length n, or variable values encoded as 0/1 or booleans.

        Returns
        -------
        fitness : float
            Sum of the selected per-variable contributions, in [0, n).

        Raises
        ------
        ValueError
            If config does not contain n binary values.
        """
        config = self._validate_config(config)
        return sum(self.contributions[i][config[i]] for i in self.variables)


class Eggbox(OptimizationProblem):
    """Periodic binary landscape determined by the number of selected bits.

    Parameters
    ----------
    n : int
        Number of binary variables. Must be positive.
    frequency : float, default=0.5
        Positive, finite frequency in ``sin(pi * frequency * sum(config))**2``.
        A half-integer frequency gives alternating peaks and valleys; integer
        frequencies give zero fitness in exact arithmetic. Higher frequencies
        need not produce more peaks on the discrete binary space.
    seed : int or None, default=None
        Accepted for consistency with other problems; fitness is deterministic
        and independent of seed.

    Examples
    --------
    >>> from graphfla.problems import Eggbox
    >>> problem = Eggbox(n=3)
    >>> [round(problem.evaluate(x), 4) for x in ([0, 0, 0], [0, 0, 1])]
    [0.0, 1.0]
    """

    def __init__(
        self, n: int, frequency: float = 0.5, seed: Optional[int] = None
    ) -> None:
        super().__init__(n, seed)
        frequency = _validate_real(frequency, "frequency")
        if frequency <= 0:
            raise ValueError("frequency must be positive.")
        self.frequency = frequency

    def evaluate(self, config: Union[str, Sequence[int]]) -> float:
        """Return the fitness of one configuration.

        Parameters
        ----------
        config : str or array-like of shape (n,)
            Binary string of length n, or variable values encoded as 0/1 or booleans.

        Returns
        -------
        fitness : float
            Squared sine of pi times frequency times the number of ones,
            in [0, 1].

        Raises
        ------
        ValueError
            If config does not contain n binary values.
        """
        config = self._validate_config(config)
        # Reduce the phase first to preserve exact zeros and avoid large arguments.
        phase = ((self.frequency % 1.0) * sum(config)) % 1.0
        return math.sin(math.pi * phase) ** 2
