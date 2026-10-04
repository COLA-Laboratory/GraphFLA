---
api_grouped_classes: true
---

# Combinatorial Problems

`graphfla.problems` also provides random instances of classic NP-hard optimization problems. Each problem encodes a solution as $n$ binary variables and defines fitness so that larger values are better. Synthetic landscapes from evolutionary biology are documented under [Biological Models](../problems.md).

--8<-- "problem-interface.md"

## Overview

| API | Purpose |
| --- | --- |
| [`Max3Sat`](#max-3-sat) | Random 3-SAT formula; fitness is the number of satisfied clauses. |
| [`Knapsack`](#01-knapsack) | 0/1 knapsack with a configurable coupling between item weights and values. |
| [`NumberPartitioning`](#number-partitioning) | Two-way partitioning of random integers, following Mertens (1998). |

## Max-3-SAT

Max-3-SAT asks for an assignment of $n$ Boolean variables that satisfies as many clauses of a 3-CNF formula as possible. Fitness is the number of satisfied clauses.

An instance contains $m = \lfloor \alpha n \rfloor$ distinct clauses. Each clause has three literals on three different variables, and each literal is negated with probability one half.

::: graphfla.problems.Max3Sat

## 0/1 Knapsack

The 0/1 knapsack problem asks for a subset of $n$ items with the largest total value whose total weight does not exceed the capacity. Item weights are integers drawn uniformly from 1 to 100, and the `correlation` parameter controls how item values are generated from the weights.

Fitness is the total value of the selected items. A selection that exceeds the capacity has fitness 0.

::: graphfla.problems.Knapsack

## Number Partitioning

The number partitioning problem asks for a division of $n$ positive integers into two subsets whose sums are as close as possible. Fitness is the negative absolute difference between the two sums, so a perfect partition has fitness 0.

Following [Mertens (1998)](https://link.aps.org/doi/10.1103/PhysRevLett.81.4281), the integers are drawn uniformly from 1 to $2^{b} - 1$, where $b = \lfloor \alpha n \rfloor$ is the number of bits per integer. The problem has a phase transition near $\alpha = 1$. For smaller $\alpha$, perfect partitions are numerous and easy to find. For larger $\alpha$, a perfect partition is unlikely to exist and the best partition is hard to find.

::: graphfla.problems.NumberPartitioning

## References

-   Stephan Mertens, "Phase transition in the number partitioning problem", *Phys. Rev. Lett.* (1998).
-   David S. Johnson, "The NP-completeness column: An ongoing guide", *J. Algorithms* (1981).
