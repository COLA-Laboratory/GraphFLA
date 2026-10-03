"""Synthetic binary maximization problems with evaluation and enumeration APIs."""

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
