"""Direct substitution oracle for Ferretti et al. (2016), Eqs. (1), (11).

No GraphFLA helpers or square-grouping code are used. Every directed focal
substitution and every different-site background substitution is enumerated.
Incomplete quadruples are omitted from BOTH sums (the GraphFLA sparse-data
contract, distinct from the paper's distance-correlation estimator, Eq. (2)).
"""

import math
from collections import defaultdict


def directed_gamma(variants, fitness, *, signs=False, epsilon=0.0):
    """Return the ratio, raw sums and number of directed complete quadruples.

    Inputs should have ordinary floating-point magnitudes. Extreme-scale basic
    tests use exact analytic targets instead of this floating-point oracle.
    ``epsilon`` implements the inclusive neutral interval of paper Eq. (10);
    it is a validation option, not a proposed or implemented public API.
    """
    values = dict(zip(map(tuple, variants), map(float, fitness)))
    sites = range(len(next(iter(values))))
    alleles = [{g[j] for g in values} for j in sites]
    products, squares = [], []
    pair_sums = defaultdict(lambda: ([], []))
    for genotype, fitness_value in values.items():
        for focal in sites:
            for allele in alleles[focal] - {genotype[focal]}:
                mutant = list(genotype)
                mutant[focal] = allele
                if tuple(mutant) not in values:
                    continue
                effect = values[tuple(mutant)] - fitness_value
                for background in sites:
                    if background == focal:
                        continue
                    for alternative in alleles[background] - {genotype[background]}:
                        neighbor, double = list(genotype), mutant.copy()
                        neighbor[background] = double[background] = alternative
                        if tuple(neighbor) not in values or tuple(double) not in values:
                            continue
                        other_effect = values[tuple(double)] - values[tuple(neighbor)]
                        a, b = effect, other_effect
                        if signs:
                            a = int(a > epsilon) - int(a < -epsilon)
                            b = int(b > epsilon) - int(b < -epsilon)
                        products.append(a * b)
                        squares.append(a * a)
                        pair_products, pair_squares = pair_sums[focal, background]
                        pair_products.append(a * b)
                        pair_squares.append(a * a)
    numerator, denominator = math.fsum(products), math.fsum(squares)
    return {
        "value": numerator / denominator if denominator else math.nan,
        "numerator": numerator,
        "denominator": denominator,
        "directed_quadruples": len(products),
        "by_position_pair": {
            pair: (math.fsum(p), math.fsum(s)) for pair, (p, s) in pair_sums.items()
        },
    }
