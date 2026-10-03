"""Public contracts and small independent oracles for binary problems."""

from itertools import islice, product
import math
import random

import numpy as np
import pytest

from graphfla import problems


CASES = [
    (problems.NK, {"k": 2}, float),
    (problems.RoughMountFuji, {}, float),
    (problems.HoC, {}, float),
    (problems.Additive, {}, float),
    (problems.Eggbox, {}, float),
    (problems.Max3Sat, {"alpha": 2}, int),
    (problems.Knapsack, {}, float),
    (problems.NumberPartitioning, {}, int),
]


@pytest.mark.parametrize(
    "cls,kwargs,result_type", CASES, ids=lambda x: getattr(x, "__name__", None)
)
def test_problem_encodings_enumeration_and_random_state(cls, kwargs, result_type):
    before = random.getstate()
    numpy_before = np.random.get_state()
    problem = cls(n=3, seed=np.int64(0), **kwargs)
    X, f = problem.get_data()
    assert X == ["".join(map(str, bits)) for bits in product((0, 1), repeat=3)]
    assert all(type(value) is result_type for value in f)
    for i, bits in enumerate(product((0, 1), repeat=3)):
        for config in (
            bits,
            list(bits),
            X[i],
            np.array(bits),
            np.array(bits, dtype=bool),
            np.array(bits, dtype=float),
        ):
            assert problem.evaluate(config) == f[i]
    assert list(problem.iter_data()) == list(zip(X, f))
    assert cls(n=3, seed=0, **kwargs).get_data() == (X, f)
    assert random.getstate() == before
    for old, new in zip(numpy_before, np.random.get_state()):
        np.testing.assert_equal(old, new)


@pytest.mark.parametrize("cls,kwargs,_", CASES)
@pytest.mark.parametrize(
    "config",
    [
        None,
        1,
        [],
        [0, 1],
        [0, 1, 2],
        [-1, 0, 1],
        [0, 0.5, 1],
        [0, np.nan, 1],
        [0, np.inf, 1],
        [[0], [1], [0]],
        ["0", "1", "0"],
        "01",
        "0x1",
        [0j, 1, 0],
    ],
)
def test_invalid_configuration_does_not_consume_randomness(cls, kwargs, _, config):
    problem = cls(n=3, seed=0, **kwargs)
    state = problem.rng.getstate()
    with pytest.raises(ValueError, match="config"):
        problem.evaluate(config)
    assert problem.rng.getstate() == state


@pytest.mark.parametrize(
    "n,error",
    [
        (True, TypeError),
        (3.0, TypeError),
        ("3", TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_dimension_validation(n, error):
    with pytest.raises(error, match="n"):
        problems.OptimizationProblem(n)


@pytest.mark.parametrize("seed", [True, 1.5, "seed", np.random.default_rng(0)])
def test_invalid_seed(seed):
    with pytest.raises(TypeError, match="seed"):
        problems.Additive(3, seed=seed)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"k": True}, TypeError),
        ({"k": 1.0}, TypeError),
        ({"k": -1}, ValueError),
        ({"k": 3}, ValueError),
    ],
)
def test_nk_invalid_k(kwargs, error):
    with pytest.raises(error, match="k"):
        problems.NK(3, **kwargs)


@pytest.mark.parametrize(
    "cls,kwargs,name",
    [
        (problems.NK, {"k": 1}, "exponent"),
        (problems.RoughMountFuji, {}, "alpha"),
        (problems.Eggbox, {}, "frequency"),
        (problems.Max3Sat, {}, "alpha"),
        (problems.Knapsack, {}, "capacity_ratio"),
        (problems.Knapsack, {}, "correlation"),
        (problems.NumberPartitioning, {}, "alpha"),
    ],
)
@pytest.mark.parametrize(
    "value,error",
    [(np.nan, ValueError), (np.inf, ValueError), ("0.5", TypeError), (True, TypeError)],
)
def test_real_parameter_validation(cls, kwargs, name, value, error):
    with pytest.raises(error, match=name):
        cls(3, **kwargs, **{name: value})


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (problems.Max3Sat, {"n": 2, "alpha": 1}),
        (problems.Max3Sat, {"n": 3, "alpha": 3}),
        (problems.Max3Sat, {"n": 3, "alpha": -1}),
        (problems.NumberPartitioning, {"n": 3, "alpha": 0.1}),
        (problems.NumberPartitioning, {"n": 3, "alpha": 0}),
        (problems.Eggbox, {"n": 3, "frequency": 0}),
        (problems.RoughMountFuji, {"n": 3, "alpha": 1.1}),
        (problems.Knapsack, {"n": 3, "capacity_ratio": 0}),
        (problems.Knapsack, {"n": 3, "correlation": -1.1}),
        (problems.Max3Sat, {"n": 3, "alpha": 1e308}),
        (problems.NumberPartitioning, {"n": 3, "alpha": 1e308}),
    ],
)
def test_impossible_instances_fail_early(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_get_data_propagates_errors(capsys):
    with pytest.raises(NotImplementedError):
        problems.OptimizationProblem(2).get_data()

    class AllocationFailure(problems.OptimizationProblem):
        def evaluate(self, config):
            raise MemoryError("test allocation failure")

    with pytest.raises(MemoryError, match="test allocation"):
        AllocationFailure(2).get_data()
    assert capsys.readouterr().out == ""


def test_iter_data_is_lazy_and_resumable():
    problem = problems.NK(30, 2, seed=0)
    iterator = problem.iter_data()
    assert not problem.values
    prefix = list(islice(iterator, 3))
    assert [x for x, _ in prefix] == [format(i, "030b") for i in range(3)]
    assert len(problem.values) <= 3 * 30
    assert next(iterator)[0] == format(3, "030b")
    reference = problems.NK(30, 2, seed=0)
    assert prefix == list(islice(reference.iter_data(), 3))


@pytest.mark.parametrize(
    "n,k,exponent", [(1, 0, 1.0), (4, 0, 2.0), (4, 2, 1.0), (4, 3, -1.0), (72, 2, 1.0)]
)
def test_nk_matches_independent_contribution_table(n, k, exponent):
    problem = problems.NK(n, k, exponent=exponent, seed=3)
    # The oracle names backgrounds by strings, independently of the packed cache.
    rng = random.Random()
    rng.setstate(problem.rng.getstate())
    table = {}
    configs = list(islice(product((0, 1), repeat=n), 16))
    for config in configs[::-1] + configs:
        contributions = []
        for i, sites in enumerate(problem.dependence):
            key = (i, "".join(str(config[j]) for j in sites))
            if key not in table:
                table[key] = rng.random()
            contributions.append(table[key])
        assert problem.evaluate(config) == pytest.approx(
            (sum(contributions) / n) ** exponent, rel=2e-15
        )
    assert len(problem.values) == len(table)
    assert problem.rng.getstate() == rng.getstate()


def test_additive_and_rmf_equations_and_hoc_equivalence():
    additive = problems.Additive(3, seed=0)
    rmf = problems.RoughMountFuji(3, alpha=0.4, seed=0)
    hoc = problems.HoC(3, seed=0)
    pure_random = problems.RoughMountFuji(3, alpha=1, seed=0)
    rng = random.Random()
    rng.setstate(rmf.rng.getstate())
    for bits in product((0, 1), repeat=3):
        assert additive.evaluate(bits) == sum(
            c[b] for c, b in zip(additive.contributions, bits)
        )
        expected = (
            0.6 * sum(c * b for c, b in zip(rmf.smooth_contribution, bits))
            + 0.4 * rng.random()
        )
        assert rmf.evaluate(bits) == expected
        assert hoc.evaluate(bits) == pure_random.evaluate(bits)


def test_eggbox_default_alternates_and_frequency_remains_explicit():
    for bits in product((0, 1), repeat=5):
        assert problems.Eggbox(5).evaluate(bits) == float(sum(bits) % 2)
        assert problems.Eggbox(5, frequency=1).evaluate(bits) == 0.0
        actual = problems.Eggbox(5, frequency=0.13).evaluate(bits)
        assert actual == pytest.approx(math.sin(math.pi * 0.13 * sum(bits)) ** 2)
    assert problems.Eggbox(3, frequency=1e308).evaluate([1, 1, 1]) == 0.0


def test_max3sat_full_and_empty_clause_sets(capsys):
    full = problems.Max3Sat(3, alpha=8 / 3, seed=0)
    expected = {tuple(enumerate(signs)) for signs in product((False, True), repeat=3)}
    assert set(full.clauses) == expected
    assert full.m == len(full.clauses) == 8
    assert full.get_data()[1] == [7] * 8
    assert problems.Max3Sat(3, alpha=0.1).get_data()[1] == [0] * 8
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("correlation", [-1, -0.4, 0, 0.005, 0.4, 1])
def test_knapsack_generated_items_and_independent_subset_objective(correlation):
    problem = problems.Knapsack(4, correlation=correlation, seed=0)
    assert problem.capacity_ratio == 0.5
    assert all(w > 0 and v > 0 for w, v in zip(problem.weights, problem.values))
    if correlation == 1:
        assert problem.values == [w + 10 for w in problem.weights]
    elif correlation == -1:
        assert problem.values == [max(1, 100 - w) for w in problem.weights]
    for bits in product((0, 1), repeat=4):
        selected = [i for i, bit in enumerate(bits) if bit]
        weight = sum(problem.weights[i] for i in selected)
        expected = (
            sum(problem.values[i] for i in selected)
            if weight <= problem.capacity
            else 0
        )
        assert problem.evaluate(bits) == expected


def test_partitioning_preserves_large_integer_arithmetic():
    problem = problems.NumberPartitioning(3, alpha=30, seed=0)
    assert all(type(value) is int and 1 <= value < 2**90 for value in problem.numbers)
    for bits in product((0, 1), repeat=3):
        left = sum(v for v, bit in zip(problem.numbers, bits) if not bit)
        right = sum(v for v, bit in zip(problem.numbers, bits) if bit)
        fitness = problem.evaluate(bits)
        assert type(fitness) is int
        assert fitness == -abs(left - right)
