# graphfla/problems/__init__.py


"""
Classes for generating black-box optimization problems.
"""

from .base_problem import OptimizationProblem
from .biological import NK, RoughMountFuji, Eggbox, Additive, HoC
from .combinatorial import Max3Sat, NumberPartitioning, Knapsack

__all__ = [
    "OptimizationProblem",
    "NK",
    "RoughMountFuji",
    "Eggbox",
    "Additive",
    "HoC",
    "Max3Sat",
    "NumberPartitioning",
    "Knapsack",
]
