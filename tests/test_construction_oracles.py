"""Independent pair enumeration for construction and its optimized backends."""

from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from graphfla._neighbors import build_edges
from graphfla._neighbors import _kernels
from graphfla._neighbors.generators import (
    BooleanNeighborGenerator,
    DefaultNeighborGenerator,
    OrdinalNeighborGenerator,
    SequenceNeighborGenerator,
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


def enumerate_pairs(rows, fitness, *, maximize=True, epsilon=0, radius=1, ordinal=()):
    """Enumerate pairs using scalar distances, independent of GraphFLA helpers."""
    edges, neutral = {}, set()
    for i, j in combinations(range(len(rows)), 2):
        distance = sum(
            abs(int(a) - int(b)) if k in ordinal else int(a != b)
            for k, (a, b) in enumerate(zip(rows[i], rows[j]))
        )
        if not 0 < distance <= radius:
            continue
        delta = float(fitness[j]) - float(fitness[i])
        if abs(delta) <= epsilon:
            neutral.add((i, j))
        else:
            endpoints = (i, j) if (delta > 0) == maximize else (j, i)
            edges[endpoints] = abs(delta)
    return edges, neutral


def assert_pairs(edges, weights, neutral, expected):
    actual = {tuple(map(int, edge)): float(w) for edge, w in zip(edges, weights)}
    assert len(actual) == len(edges) == len(weights)
    assert actual == expected[0]
    assert {tuple(sorted(p)) for p in neutral} == expected[1]
    assert len(neutral) == len(expected[1])


KINDS = ["boolean", "dna", "rna", "protein", "sequence", "ordinal", "mixed"]


@pytest.mark.parametrize(
    "kind,strategy",
    [
        (kind, strategy)
        for kind in KINDS
        for strategy in ("auto", "active", "pairwise", "broadcast")
        if kind not in {"mixed", "ordinal"} or strategy in {"auto", "active"}
    ],
)
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("epsilon", [0, 0.5])
def test_public_construction_matches_pair_oracle(kind, maximize, epsilon, strategy):
    base = 2 if kind == "boolean" else 3
    rows = np.array(list(product(range(base), repeat=3)))
    rng = np.random.default_rng(42)
    keep = rng.permutation(len(rows))[: len(rows) - 2]
    rows = rows[keep]
    fitness = (np.arange(len(rows)) % 7) / 2
    ordinal = ()
    kwargs = {}
    if kind == "boolean":
        landscape, X = BooleanLandscape(maximize=maximize), rows
    elif kind in {"ordinal", "mixed"}:
        X = pd.DataFrame(rows, columns=["a", "b", "c"])
        if kind == "ordinal":
            landscape, ordinal = OrdinalLandscape(maximize=maximize), (0, 1, 2)
        else:
            landscape, ordinal = Landscape(maximize=maximize), (1,)
            kwargs["data_types"] = dict(a="categorical", b="ordinal", c="categorical")
    else:
        cls = dict(
            dna=DNALandscape,
            rna=RNALandscape,
            protein=ProteinLandscape,
            sequence=SequenceLandscape,
        )[kind]
        alphabet = "ACU" if kind == "rna" else "ACG"
        landscape = cls(maximize=maximize)
        X = ["".join(alphabet[c] for c in row) for row in rows]
    landscape.build_from_data(
        X,
        fitness,
        epsilon=epsilon,
        neighborhood_strategy=strategy,
        verbose=False,
        **kwargs,
    )
    assert landscape.n_configs == len(rows)
    neutral = {
        (min(u, v), max(u, v))
        for u, neighbors in (landscape._neutral_neighbors or {}).items()
        for v in neighbors
    }
    assert_pairs(
        landscape.graph.get_edgelist(),
        landscape.graph.es["delta_fit"],
        neutral,
        enumerate_pairs(
            rows, fitness, maximize=maximize, epsilon=epsilon, ordinal=ordinal
        ),
    )


def kernel_result(rows, fitness, generator, strategy="active", **kwargs):
    info = {
        j: {"type": "categorical", "max": int(rows[:, j].max())}
        for j in range(rows.shape[1])
    }
    return build_edges(
        configs=None,
        configs_array=rows,
        config_dict=info,
        data=pd.DataFrame({"fitness": fitness}),
        n_configs=len(rows),
        n_vars=rows.shape[1],
        n_edit=kwargs.pop("radius", 1),
        strategy=strategy,
        epsilon=kwargs.pop("epsilon", 0),
        maximize=kwargs.pop("maximize", True),
        verbose=False,
        neighbor_generator=generator.generate,
    )


@pytest.mark.parametrize("strategy", ["pairwise", "broadcast", "auto"])
@pytest.mark.parametrize("radius", [1, 2, 3])
@settings(max_examples=20, deadline=None, derandomize=True)
@given(sample=st.lists(st.integers(0, 26), min_size=2, max_size=20, unique=True))
def test_sparse_hamming_neighborhoods(sample, strategy, radius):
    rows = np.array(list(product(range(3), repeat=3)), dtype=np.uint8)[sample]
    fitness = np.arange(len(rows), dtype=float) % 4
    result = kernel_result(
        rows, fitness, SequenceNeighborGenerator(3), strategy, radius=radius
    )
    assert_pairs(
        result.edges,
        result.delta_fits,
        result.neutral_pairs,
        enumerate_pairs(rows, fitness, radius=radius),
    )


@pytest.mark.parametrize(
    "backend", ["lookup", "searchsorted", "overflow", "ordinal", "wide", "custom"]
)
def test_active_backends_match_brute_force(backend, monkeypatch):
    rng = np.random.default_rng(6)
    rows = rng.integers(0, 2, size=(12, 70 if backend == "overflow" else 6))
    rows = np.concatenate([rows, rows ^ np.eye(1, rows.shape[1], dtype=int)])
    rows = np.unique(rows, axis=0)
    generator = BooleanNeighborGenerator()
    ordinal = ()
    if backend == "searchsorted":
        monkeypatch.setattr(_kernels, "_LUT_MAX_CELLS", 0)
    elif backend == "lookup":
        monkeypatch.setattr(_kernels, "_BYTEMAP_CHUNK_CANDIDATES", 3)
    elif backend == "ordinal":
        rows = np.array(list(product(range(3), repeat=3)))
        generator, ordinal = OrdinalNeighborGenerator(), (0, 1, 2)
    elif backend == "wide":
        rows = np.array([[0, 0], [0, 256], [257, 256], [257, 0]])
        generator = DefaultNeighborGenerator()
    elif backend == "custom":

        class OnePosition(BooleanNeighborGenerator):
            def generate(self, config, config_dict, n_edit=1):
                return [(1 - config[0], *config[1:])]

        generator = OnePosition()
    fitness = rng.integers(0, 5, size=len(rows)).astype(float)
    result = kernel_result(rows, fitness, generator)
    if backend == "custom":
        expected = enumerate_pairs(rows, fitness)
        expected = (
            {
                e: w
                for e, w in expected[0].items()
                if np.array_equal(rows[e[0], 1:], rows[e[1], 1:])
            },
            {e for e in expected[1] if np.array_equal(rows[e[0], 1:], rows[e[1], 1:])},
        )
    else:
        expected = enumerate_pairs(rows, fitness, ordinal=ordinal)
    assert_pairs(result.edges, result.delta_fits, result.neutral_pairs, expected)


@pytest.mark.parametrize("strategy", ["active", "pairwise", "broadcast"])
def test_empty_neighborhood_kernel(strategy):
    rows = np.array([[0, 0, 0], [1, 1, 1]])
    result = kernel_result(rows, [0, 1], BooleanNeighborGenerator(), strategy)
    assert result.edges.shape == (0, 2)
    assert result.delta_fits.shape == (0,)
    assert result.neutral_pairs == []


def test_fingerprint_collisions_do_not_duplicate_or_invent_edges(monkeypatch):
    class CollidingRandom:
        def integers(self, low, high, *, size, **kwargs):
            return np.zeros(size, dtype=np.int64)

    monkeypatch.setattr(
        _kernels.np.random, "default_rng", lambda seed: CollidingRandom()
    )
    rows = np.array(list(product(range(3), repeat=3)), dtype=np.uint8)
    fitness = np.arange(len(rows), dtype=float)
    result = kernel_result(rows, fitness, SequenceNeighborGenerator(3), "pairwise")
    assert_pairs(
        result.edges,
        result.delta_fits,
        result.neutral_pairs,
        enumerate_pairs(rows, fitness),
    )


@pytest.mark.parametrize("strategy", ["active", "pairwise", "broadcast"])
def test_epsilon_accepts_one_ulp_roundoff_only(strategy):
    rows = np.array([[0], [1], [2]])
    fitness = [
        0.0,
        np.nextafter(0.5, np.inf),
        np.nextafter(np.nextafter(0.5, np.inf), np.inf),
    ]
    result = kernel_result(
        rows, fitness, SequenceNeighborGenerator(3), strategy, epsilon=0.5
    )
    assert set(map(tuple, result.edges)) == {(0, 2)}
    assert set(result.neutral_pairs) == {(0, 1), (1, 2)}
