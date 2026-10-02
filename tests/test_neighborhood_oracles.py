"""Neighborhood kernels checked against exhaustive pair enumeration."""

from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest

from graphfla._neighbors import (
    BooleanNeighborGenerator,
    DefaultNeighborGenerator,
    OrdinalNeighborGenerator,
    SequenceNeighborGenerator,
    build_edges,
)
from graphfla.landscape import (
    BooleanLandscape,
    DNALandscape,
    Landscape,
    OrdinalLandscape,
    ProteinLandscape,
    RNALandscape,
    SequenceLandscape,
)


def reference_pairs(variants, fitness, types, maximize, epsilon, n_edit=1):
    """Enumerate pairs directly, without GraphFLA encoders or generators."""
    edges, neutral = {}, set()
    bound = np.nextafter(epsilon, np.inf) if epsilon else 0
    for i, j in combinations(range(len(variants)), 2):
        distance = sum(
            abs(int(a) - int(b)) if kind == "ordinal" else a != b
            for a, b, kind in zip(variants[i], variants[j], types)
        )
        if not 0 < distance <= n_edit:
            continue
        delta = float(fitness[j]) - float(fitness[i])
        if abs(delta) <= bound:
            neutral.add((i, j))
        else:
            edge = (i, j) if (delta > 0) == maximize else (j, i)
            edges[edge] = abs(delta)
    return edges, neutral


def assert_pairs(result, expected):
    edges = {tuple(pair): delta for pair, delta in zip(result.edges, result.delta_fits)}
    assert len(edges) == len(result.edges), "duplicate edges"
    assert edges == pytest.approx(expected[0])
    assert {tuple(sorted(pair)) for pair in result.neutral_pairs} == expected[1]
    assert len(result.neutral_pairs) == len(expected[1])


@pytest.mark.parametrize("strategy", ["active", "pairwise", "broadcast", "auto"])
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("epsilon", [0, 0.5])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    "kind,cards",
    [
        ("boolean", (2, 2, 2)),
        ("sequence", (4, 4)),
        ("categorical", (3, 2)),
        ("ordinal", (4, 3)),
        ("mixed", (2, 3, 4)),
    ],
)
def test_kernel_against_pair_enumeration(
    strategy, maximize, epsilon, sparse, kind, cards
):
    variants = np.array(list(product(*(range(n) for n in cards))))
    order = np.random.default_rng(42).permutation(len(variants))
    variants = variants[order[::2] if sparse else order]
    fitness = np.array([0, 0.5, 0.75, 2, -1, 2] * len(variants))[: len(variants)]
    types = [kind] * len(cards)
    if kind == "mixed":
        types = ["boolean", "categorical", "ordinal"]
    generators = {
        "boolean": BooleanNeighborGenerator(),
        "sequence": SequenceNeighborGenerator(max(cards)),
        "categorical": DefaultNeighborGenerator(),
        "ordinal": OrdinalNeighborGenerator(),
        "mixed": DefaultNeighborGenerator(),
    }
    metadata = {i: {"type": t, "max": cards[i] - 1} for i, t in enumerate(types)}
    # Explicit pairwise/broadcast kernels define Hamming neighborhoods.
    effective_types = (
        types if strategy in {"active", "auto"} else ["categorical"] * len(cards)
    )
    result = build_edges(
        configs=None,
        configs_array=variants,
        config_dict=metadata,
        fitness=fitness,
        n_configs=len(variants),
        n_vars=len(cards),
        n_edit=1,
        strategy=strategy,
        epsilon=epsilon,
        maximize=maximize,
        verbose=False,
        neighbor_generator=generators[kind].generate,
    )
    assert_pairs(
        result, reference_pairs(variants, fitness, effective_types, maximize, epsilon)
    )


@pytest.mark.parametrize("strategy", ["pairwise", "broadcast", "auto"])
@pytest.mark.parametrize("n_edit", [2, 3, 5])
def test_multiple_edits(strategy, n_edit):
    variants = np.array(list(product(range(2), repeat=3)))
    fitness = np.array([0, 3, 1, 5, 4, 0, 2, 3])
    result = build_edges(
        configs=None,
        configs_array=variants,
        config_dict={i: {"type": "boolean", "max": 1} for i in range(3)},
        fitness=fitness,
        n_configs=8,
        n_vars=3,
        n_edit=n_edit,
        strategy=strategy,
        epsilon=0,
        maximize=True,
        verbose=False,
        neighbor_generator=BooleanNeighborGenerator().generate,
    )
    assert_pairs(
        result, reference_pairs(variants, fitness, ["boolean"] * 3, True, 0, n_edit)
    )


@pytest.mark.parametrize(
    "kind,width,card",
    [
        ("boolean", 70, 2),
        ("sequence", 36, 4),
        ("sequence", 13, 4),
        ("ordinal", 70, 2),
        ("ordinal", 4, 1000),
    ],
)
def test_sparse_large_key_spaces(kind, width, card):
    variants = np.zeros((5, width), dtype=np.int64)
    variants[1, 0] = variants[2, 1] = 1
    variants[3, :2] = 1
    variants[4, -1] = card - 1
    fitness = np.arange(5, dtype=float)
    generator = {
        "boolean": BooleanNeighborGenerator(),
        "sequence": SequenceNeighborGenerator(card),
        "ordinal": OrdinalNeighborGenerator(),
    }[kind]
    result = build_edges(
        configs=None,
        configs_array=variants,
        config_dict={i: {"type": kind, "max": card - 1} for i in range(width)},
        fitness=fitness,
        n_configs=5,
        n_vars=width,
        n_edit=1,
        strategy="active",
        epsilon=0,
        maximize=False,
        verbose=False,
        neighbor_generator=generator.generate,
    )
    assert_pairs(result, reference_pairs(variants, fitness, [kind] * width, False, 0))


@pytest.mark.parametrize(
    "cls,alphabet",
    [
        (BooleanLandscape, "01"),
        (DNALandscape, "ACT"),
        (RNALandscape, "ACU"),
        (ProteinLandscape, "ACW"),
        (SequenceLandscape, "XYZ"),
    ],
)
def test_specialized_classes_keep_shuffled_fitness_aligned(cls, alphabet):
    sequences = ["".join(v) for v in product(alphabet, repeat=2)][::-1]
    fitness = np.arange(len(sequences), dtype=float)
    landscape = cls().build_from_data(sequences, fitness, verbose=False)
    expected, _ = reference_pairs(sequences, fitness, ["categorical"] * 2, True, 0)
    assert set(landscape.graph.get_edgelist()) == set(expected)
    assert landscape.graph.vs["fitness"] == list(fitness)


@pytest.mark.parametrize("mixed", [False, True])
def test_automatic_strategy_respects_ordered_categories(mixed):
    X = pd.DataFrame(
        list(product(["low", "mid", "high"], [False, True])), columns=["dose", "allele"]
    )
    X["dose"] = pd.Categorical(X.dose, categories=["low", "mid", "high"], ordered=True)
    fitness = [0, 4, 1, 5, 2, 6]
    cls = Landscape if mixed else OrdinalLandscape
    kwargs = {"data_types": {"dose": "ordinal", "allele": "boolean"}} if mixed else {}
    landscape = cls().build_from_data(
        X, fitness, verbose=False, neighborhood_strategy="auto", **kwargs
    )
    variants = list(product(range(3), range(2)))
    expected, _ = reference_pairs(variants, fitness, ["ordinal", "boolean"], True, 0)
    assert set(landscape.graph.get_edgelist()) == set(expected)


@pytest.mark.parametrize("epsilon", [0, 0.1])
def test_flat_landscape_keeps_neutral_neighbors(epsilon):
    landscape = BooleanLandscape().build_from_data(
        ["00", "01", "10", "11"], [1] * 4, epsilon=epsilon, verbose=False
    )
    assert landscape.n_configs == 4
    assert landscape.n_edges == 0
    from graphfla.analysis import neutrality

    assert neutrality(landscape) == 1
