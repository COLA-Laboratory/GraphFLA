"""Scientific metrics checked against direct equations on small landscapes."""

import warnings
from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest

from graphfla import analysis as A
from graphfla.landscape import BooleanLandscape, Landscape, ProteinLandscape


def gamma_reference(variants, fitness, signs=False):
    """Ferretti et al. (2016), eq. 3, evaluated by directed substitutions."""
    lookup = dict(zip(map(tuple, variants), fitness))
    alleles = [sorted({v[j] for v in lookup}) for j in range(len(variants[0]))]
    numerator = denominator = 0.0
    for variant, value in lookup.items():
        for focal in range(len(alleles)):
            for allele in alleles[focal]:
                if allele == variant[focal]:
                    continue
                mutant = list(variant)
                mutant[focal] = allele
                if tuple(mutant) not in lookup:
                    continue
                effect = lookup[tuple(mutant)] - value
                for background in range(len(alleles)):
                    if background == focal:
                        continue
                    for alternative in alleles[background]:
                        if alternative == variant[background]:
                            continue
                        other, double = list(variant), list(mutant)
                        other[background] = double[background] = alternative
                        if tuple(other) not in lookup or tuple(double) not in lookup:
                            continue
                        other_effect = lookup[tuple(double)] - lookup[tuple(other)]
                        a, b = (
                            (np.sign(effect), np.sign(other_effect))
                            if signs
                            else (effect, other_effect)
                        )
                        numerator += a * b
                        denominator += a * a
    return numerator / denominator if denominator else np.nan


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("n_alleles", [2, 3])
def test_gamma_matches_directed_substitution_equation(sparse, maximize, n_alleles):
    variants = list(product(range(n_alleles), repeat=3))
    if sparse:
        variants = variants[:-2]
    fitness = np.random.default_rng(27).integers(-3, 7, len(variants)).astype(float)
    sequences = ["".join("ACW"[v] for v in variant) for variant in variants]
    landscape = ProteinLandscape(maximize=maximize).build_from_data(
        sequences, fitness, verbose=False
    )
    assert A.gamma(landscape, n_jobs=1) == pytest.approx(
        gamma_reference(variants, fitness)
    )
    assert A.gamma_star(landscape, n_jobs=1) == pytest.approx(
        gamma_reference(variants, fitness, signs=True)
    )


@pytest.mark.parametrize(
    "scale,offset",
    [
        (1, 0),
        (3, 17),
        (-2, 11),
        (1e-10, 0),
        (1e10, 0),
    ],
)
def test_roughness_slope_is_invariant_to_fitness_units(scale, offset):
    # Walsh basis: residual = 3*z0*z1, slopes in 0/1 coding are 4, 8, 12.
    variants = np.array(list(product(range(2), repeat=3)))
    z = 2 * variants - 1
    fitness = 2 * z[:, 0] + 4 * z[:, 1] + 6 * z[:, 2] + 3 * z[:, 0] * z[:, 1]
    landscape = BooleanLandscape().build_from_data(
        variants, scale * fitness + offset, verbose=False
    )
    assert A.r_s_ratio(landscape) == pytest.approx(3 / 8, rel=1e-10)
    assert A.higher_order_epistasis(landscape, order=1) == pytest.approx(56 / 65)
    assert A.higher_order_epistasis(landscape, order=2) == pytest.approx(1)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_higher_order_variance_from_orthogonal_components(order):
    variants = np.array(list(product(range(2), repeat=3)))
    z = 2 * variants - 1
    fitness = z[:, 0] + 2 * z[:, 0] * z[:, 1] + 3 * np.prod(z, axis=1)
    landscape = BooleanLandscape().build_from_data(variants, fitness, verbose=False)
    assert A.higher_order_epistasis(landscape, order=order) == pytest.approx(
        sum(i * i for i in range(1, order + 1)) / 14
    )


def test_gamma_high_dimensional_fallback():
    variants = np.zeros((8, 66), dtype=int)
    variants[:, :3] = list(product(range(2), repeat=3))
    variants[:, 3:] = variants[:, 2, None]
    fitness = [0, 1, 2, 4, 3, 5, 7, 8]
    landscape = BooleanLandscape().build_from_data(variants, fitness, verbose=False)
    assert A.gamma(landscape, n_jobs=1) == pytest.approx(
        gamma_reference(variants, fitness)
    )


def test_mutation_effects_have_consistent_direction():
    X = pd.DataFrame(list(product(["A", "B"], [0, 1])), columns=["site", "background"])
    landscape = Landscape().build_from_data(
        X,
        [1, 2, 4, 7],
        data_types={"site": "categorical", "background": "boolean"},
        verbose=False,
    )
    effects = A.fitness_effect_distribution(landscape, ("A", "site", "B"))
    assert effects == [3, 5]
    summary = A.single_mutation_effects(landscape, "site").iloc[0]
    assert summary.mutation_from == "A"
    assert summary.mutation_to == "B"
    assert summary.mean_effect == pytest.approx(np.mean(effects))


def test_single_position_effect_distribution():
    landscape = ProteinLandscape().build_from_data(
        ["A", "C", "W"], [1, 4, 2], verbose=False
    )
    position = next(iter(landscape.data_types))
    assert A.fitness_effect_distribution(landscape, ("A", position, "C")) == [3]
    assert A.fitness_effect_distribution(landscape, ("C", position, "A")) == [-3]


def test_neutrality_threshold_independent_of_construction_epsilon():
    variants = np.array(list(product(range(2), repeat=3)))
    fitness = np.array([1, 1.25, 2, 3, 4, 4.25, 5, 7])
    landscape = BooleanLandscape().build_from_data(
        variants, fitness, epsilon=0.5, verbose=False
    )
    pairs = [
        (i, j)
        for i, j in combinations(range(8), 2)
        if np.count_nonzero(variants[i] != variants[j]) == 1
    ]
    for threshold in [0, 0.25, 0.5, 1, 10]:
        expected = sum(
            abs(fitness[i] - fitness[j]) <= threshold for i, j in pairs
        ) / len(pairs)
        assert A.neutrality(landscape, threshold=threshold) == pytest.approx(expected)


def test_every_metric_is_defined_on_a_fully_neutral_landscape():
    # A flat landscape has neighbours but no directed edges. Every public metric
    # must return a value or an explicit NaN rather than raising: degenerate
    # inputs are legal, and a crash here hides which statistics are undefined.
    landscape = BooleanLandscape().build_from_data(
        ["".join(bits) for bits in product("01", repeat=3)], [1.0] * 8, verbose=False
    )
    assert landscape.n_configs == 8
    assert landscape.n_edges == 0

    for name in A.list_metrics().index:
        metric = getattr(A, str(name))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metric(landscape)  # must not raise

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert A.neutrality(landscape) == 1
        assert np.isnan(A.gamma(landscape, n_jobs=1))
        assert np.isnan(A.diminishing_returns_index(landscape))
        assert np.isnan(A.increasing_costs_index(landscape))


@pytest.mark.parametrize(
    "scale,offset",
    [(1, 0), (3, 17), (-2, 11), (1e-10, 0), (1e10, 0), (1, 1e13), (1, -1e9)],
)
def test_roughness_slope_is_invariant_to_affine_fitness_rescaling(scale, offset):
    # r and s both scale by |a| and the intercept absorbs b, so r/s is invariant
    # under f -> a*f + b. Guards against a degeneracy tolerance that tracks
    # either the fitness magnitude or its offset.
    variants = np.array(list(product(range(2), repeat=3)))
    z = 2 * variants - 1
    fitness = 2 * z[:, 0] + 4 * z[:, 1] + 6 * z[:, 2] + 3 * z[:, 0] * z[:, 1]
    landscape = BooleanLandscape().build_from_data(
        variants, scale * fitness + offset, verbose=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert A.r_s_ratio(landscape) == pytest.approx(3 / 8, rel=1e-9)


def test_roughness_slope_is_nan_for_constant_fitness():
    # r = s = 0, so the ratio is undefined -- distinct from a purely epistatic
    # landscape (r > 0, s = 0), which is inf.
    variants = np.array(list(product(range(2), repeat=3)))
    landscape = BooleanLandscape().build_from_data(
        variants, np.ones(8), verbose=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(A.r_s_ratio(landscape))


@pytest.mark.parametrize("filter_mode", ["both", "any"])
def test_component_filter_keeps_a_neutrally_connected_landscape(filter_mode):
    # Neutral pairs carry no directed edge, so a plateau looks like singleton
    # components to a directed connectivity check. Largest-component filtering
    # must not shred it down to one genotype.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        landscape = BooleanLandscape().build_from_data(
            ["00", "01", "10", "11"], [1] * 4,
            tau=0, filter_mode=filter_mode, verbose=False,
        )
        assert landscape.n_configs == 4
        assert landscape.n_edges == 0
        assert A.neutrality(landscape) == 1


@pytest.mark.parametrize("maximize", [True, False])
def test_functional_filter_severs_below_threshold_neutral_pairs(maximize):
    # A neutral pair is the undirected counterpart of a directed edge, so tau
    # must sever it on the same rule. Otherwise a below-threshold plateau stays
    # connected and wins largest-component selection over the functional region.
    sign = 1 if maximize else -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        landscape = BooleanLandscape(maximize=maximize).build_from_data(
            ["0000", "0001", "0010", "1111", "1110"],
            [0, 0, 0, 2 * sign, 3 * sign],
            tau=1 * sign,
            filter_mode="both",
            verbose=False,
        )
    assert landscape.n_configs == 2
    assert sorted(landscape.graph.vs["fitness"]) == sorted([2 * sign, 3 * sign])


def test_roughness_slope_is_nan_for_constant_non_representable_fitness():
    # np.std of identical 0.1 values is ~1.4e-17, not 0. Constant fitness must be
    # detected by exact equality or this falls through and returns inf.
    landscape = ProteinLandscape().build_from_data(["A", "C", "W"], [0.1] * 3, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(A.r_s_ratio(landscape))


@pytest.mark.parametrize(
    "fitness,expected",
    [
        ([0, 1e155, 2e155, 3e155], 0.0),        # additive at a scale where std overflows
        ([0, 1e-200, 1e-200, 0], float("inf")),  # purely epistatic where std underflows
    ],
)
def test_roughness_slope_survives_extreme_fitness_scales(fitness, expected):
    # The degeneracy test uses the fitness range, not the standard deviation:
    # squaring the deviations overflows near 1e155 and underflows near 1e-200 on
    # inputs that are still finite.
    landscape = BooleanLandscape().build_from_data(
        ["00", "01", "10", "11"], fitness, verbose=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert A.r_s_ratio(landscape) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("maximize", [True, False])
def test_neutral_pair_filter_preserves_integer_fitness_dtype(maximize):
    # Casting fitness to float64 rounds integers above 2**53 and would let a
    # below-threshold plateau survive the functional filter.
    offset = 2**53
    fitness = (
        [offset + 3, offset + 3, offset + 3, offset + 5, offset + 6]
        if maximize
        else [offset + 1, offset + 1, offset + 1, offset - 2, offset - 3]
    )
    tau = offset + 4 if maximize else offset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        landscape = BooleanLandscape(maximize=maximize).build_from_data(
            ["0000", "0001", "0010", "1111", "1110"],
            fitness, tau=tau, filter_mode="both", verbose=False,
        )
    assert landscape.n_configs == 2
    assert sorted(landscape.graph.vs["fitness"]) == sorted(fitness[3:])
