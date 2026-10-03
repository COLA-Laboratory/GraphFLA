"""Small independent finite-difference oracle for Faure et al. (2024).

DOI: 10.1371/journal.pcbi.1012132, Eqs. (1), (9), (24). Enumerate the
complete Cartesian population, difference focal alleles against state zero,
and average over all remaining backgrounds. This deliberately does not use
the production inverse-feature formula or any GraphFLA implementation.
"""

from itertools import product

import numpy as np


def transform(arities):
    """Return configurations/terms in product order and their forward operator."""
    states = list(product(*(range(s) for s in arities)))
    matrix = np.empty((len(states), len(states)))
    for i, term in enumerate(states):
        for j, genotype in enumerate(states):
            weight = 1.0
            for state, allele, arity in zip(genotype, term, arities):
                weight *= (
                    1 / arity if allele == 0 else int(state == allele) - int(state == 0)
                )
            matrix[i, j] = weight
    return states, matrix


def regression_design(arities, observed, max_order):
    """Invert the complete forward operator, then select observations/terms."""
    states, matrix = transform(arities)
    terms = [t for t in states if sum(a != 0 for a in t) <= max_order]
    rows = [states.index(tuple(x)) for x in observed]
    columns = [states.index(t) for t in terms]
    return terms, np.linalg.inv(matrix)[np.ix_(rows, columns)]
