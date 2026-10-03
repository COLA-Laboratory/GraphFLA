---
api_grouped_classes: true
---

# Combinatorial Problems

These models are drawn from the canonical NP-hard optimization literature. Each comes with a notion of "fitness" (count of satisfied clauses, total knapsack value, etc.) and a binary configuration encoding, so they slot directly into [`BooleanLandscape`](../boolean-landscape.md).

--8<-- "problem-interface.md"

## Overview

| API | Purpose |
| --- | --- |
| [`Max3Sat`](#max-3-sat) | Random 3-SAT instance; fitness = number of satisfied clauses. |
| [`Knapsack`](#01-knapsack) | 0/1 knapsack with tunable value/weight correlation. |
| [`NumberPartitioning`](#number-partitioning) | Two-way number partitioning (Mertens generator). |

## Max-3-SAT

The Max-3-SAT problem: find a Boolean assignment of $n$ variables that satisfies the maximum number of 3-literal clauses in a random 3-CNF formula. Fitness is the number of satisfied clauses — higher is better.

At construction, $m = \lfloor \alpha n \rfloor$ unique 3-literal clauses are generated. Each clause is a length-3 tuple of `(variable_index, is_positive)` literals; clauses are canonicalized (sorted by variable) so equivalent clauses are not duplicated.

::: graphfla.problems.Max3Sat

## 0/1 Knapsack

The 0/1 Knapsack problem: select a subset of $n$ items to maximize total value subject to a capacity constraint on total weight. Item weights are drawn from $\{1, \ldots, 100\}$, and item values are generated according to the `correlation` parameter.

`evaluate(config)` returns the total value of the selected items if the weight constraint is satisfied, and `0.0` otherwise — i.e., infeasible configurations get penalized to zero fitness.

::: graphfla.problems.Knapsack

## Number Partitioning

The number-partitioning problem: divide a set of $n$ positive integers into two subsets such that the absolute difference between their sums is minimized.

`evaluate(config)` returns the **negated** absolute difference, so higher is better (perfect partitions have fitness `0.0`). The numbers themselves are drawn uniformly from $\{1, \ldots, 2^{\alpha n} - 1\}$, following the standard generator in [Mertens (1998)](https://link.aps.org/doi/10.1103/PhysRevLett.81.4281).

**Notes**

-   This problem has a well-studied phase transition in difficulty around $\alpha \approx 1$, between an easy regime (many near-optimal solutions) and a hard regime (very few good solutions).

::: graphfla.problems.NumberPartitioning

## References

-   Stephan Mertens, "Phase transition in the number partitioning problem", *Phys. Rev. Lett.* (1998).
-   David S. Johnson, "The NP-completeness column: An ongoing guide", *J. Algorithms* (1981) — for general background on the NP-hard problems above.
