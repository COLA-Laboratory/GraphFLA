"""Construction input, filtering, extension and serialization contracts."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from graphfla.landscape import (
    BooleanLandscape,
    DNALandscape,
    Landscape,
    OrdinalLandscape,
    ProteinLandscape,
    RNALandscape,
    SequenceLandscape,
)


@pytest.mark.parametrize(
    "cls,alphabet",
    [
        (DNALandscape, "AC"),
        (RNALandscape, "AU"),
        (ProteinLandscape, "AW"),
        (SequenceLandscape, "AC"),
    ],
)
@pytest.mark.parametrize("form", ["list", "series", "array", "dataframe"])
def test_genetic_background_preserves_topology(cls, alphabet, form):
    core = ["".join(p) for p in product(alphabet, repeat=3)]
    full = [alphabet[0] * 500 + s[0] + alphabet[1] * 400 + s[1:] for s in core]
    X = {
        "list": lambda: full,
        "series": lambda: pd.Series(full, index=range(10, 18)),
        "array": lambda: np.array(full),
        "dataframe": lambda: pd.DataFrame([list(s) for s in full]),
    }[form]()
    before = X.copy() if hasattr(X, "copy") else list(X)
    fitness = np.arange(8, dtype=float)
    actual = cls().build_from_data(X, fitness, verbose=False)
    expected = cls().build_from_data(core, fitness, verbose=False)
    assert actual.n_vars == 3
    assert actual._configs_array.shape == (8, 3)
    assert actual.graph.get_edgelist() == expected.graph.get_edgelist()
    assert actual.graph.es["delta_fit"] == expected.graph.es["delta_fit"]
    columns = [
        c for c in actual.graph.vs.attributes() if c.startswith("pos_") or c.isdigit()
    ]
    assert len(columns) == 903
    assert ["".join(row) for row in zip(*(actual.graph.vs[c] for c in columns))] == full
    if isinstance(X, pd.DataFrame):
        pd.testing.assert_frame_equal(X, before)
    elif isinstance(X, pd.Series):
        pd.testing.assert_series_equal(X, before)
    else:
        np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize(
    "cls,values",
    [(BooleanLandscape, ["00", "01", "11"]), (DNALandscape, ["AA", "AC", "CC"])],
)
def test_nondefault_series_index_is_positional(cls, values):
    ls = cls().build_from_data(
        pd.Series(values, index=[7, 4, 8]),
        pd.Series([0.0, 1.0, 2.0], index=[8, 7, 4]),
        verbose=False,
    )
    assert ls.graph.vs["fitness"] == [0, 1, 2]
    assert set(ls.graph.get_edgelist()) == {(0, 1), (1, 2)}


@pytest.mark.parametrize(
    "values",
    [
        ["00", "0", "111"],
        [[0, 0], [1.9, 0]],
        np.array([[0.0, 0.0], [1.9, 0.0]]),
        pd.DataFrame([[0.0, 0.0], [1.9, 0.0]]),
    ],
)
def test_malformed_boolean_input_rejected(values):
    with pytest.raises((ValueError, TypeError)):
        BooleanLandscape().build_from_data(
            values, list(range(len(values))), verbose=False
        )


@pytest.mark.parametrize(
    "cls,X",
    [
        (BooleanLandscape, ["00", "01", "10", "11"]),
        (DNALandscape, ["AA", "AC", "CA", "CC"]),
    ],
)
def test_fully_neutral_connected_landscape(cls, X):
    ls = cls().build_from_data(X, [1.0] * 4, verbose=False)
    assert ls.shape == (4, 0)
    assert ls.n_plateau == 1
    assert list(ls.plateaus.values()) == [[0, 1, 2, 3]]
    assert ls.n_lo == 1 and ls.n_lo_members == 4


@pytest.mark.parametrize(
    "param,value",
    [
        ("epsilon", np.nan),
        ("epsilon", np.inf),
        ("epsilon", -1),
        ("n_edit", 1.5),
        ("n_edit", 0),
        ("tau", np.nan),
        ("tau", np.inf),
    ],
)
def test_invalid_build_parameters_fail_before_mutation(param, value):
    ls = BooleanLandscape()
    with pytest.raises((ValueError, TypeError)):
        ls.build_from_data(["00", "01"], [0.0, 1.0], verbose=False, **{param: value})
    assert not ls._is_built and ls.graph is None


@pytest.mark.parametrize(
    "cls,X",
    [
        (BooleanLandscape, ["00", "01"]),
        (DNALandscape, ["AA", "AC"]),
        (OrdinalLandscape, pd.DataFrame({"x": [0, 1]})),
    ],
)
@pytest.mark.parametrize("fitness", [[0.0], [0.0, np.nan], [0.0, np.inf]])
def test_invalid_fitness_is_rejected(cls, X, fitness):
    with pytest.raises(ValueError):
        cls().build_from_data(X, fitness, verbose=False)


@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("mode", ["any", "both"])
def test_functional_threshold_boundary_and_remapping(maximize, mode):
    X = ["000", "001", "011", "111", "110"]
    fitness = np.array([-2.0, 0.0, 2.0, -1.0, -3.0])
    if not maximize:
        fitness = -fitness
    ls = BooleanLandscape(maximize=maximize).build_from_data(
        X,
        fitness,
        tau=0,
        filter_mode=mode,
        verbose=False,
    )
    expected = [1, 2] if mode == "any" else [0, 1, 2, 3]
    np.testing.assert_array_equal(ls.graph.vs["fitness"], fitness[expected])
    assert list(ls.configs.index) == list(range(len(expected)))
    expected_strings = [X[i] for i in expected]
    columns = [ls.graph.vs[f"bit_{i}"] for i in range(3)]
    assert ["".join(map(str, row)) for row in zip(*columns)] == expected_strings
    assert ls.n_edges == len(expected) - 1


def test_duplicate_rows_keep_first_fitness_without_mutating_input():
    X = pd.DataFrame({"x": [0, 1, 0, 2], "fixed": [7] * 4}, index=[8, 2, 1, 4])
    f = pd.Series([0.0, 1.0, 99.0, 2.0], index=[1, 2, 3, 4])
    before_X, before_f = X.copy(), f.copy()
    ls = OrdinalLandscape().build_from_data(X, f, verbose=False)
    assert ls.n_configs == 3 and ls.n_vars == 1
    assert ls.graph.vs["fitness"] == [0.0, 1.0, 2.0]
    assert ls.graph.vs["fixed"] == [7] * 3
    assert list(ls.configs.index) == [0, 1, 2]
    pd.testing.assert_frame_equal(X, before_X)
    pd.testing.assert_series_equal(f, before_f)


@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("epsilon", [0.0, 0.5])
def test_graphml_roundtrip_preserves_construction(tmp_path, maximize, epsilon):
    ls = DNALandscape(maximize=maximize).build_from_data(
        ["AA", "AC", "CA", "CC"],
        [0.0, 0.5, 1.0, 2.0],
        epsilon=epsilon,
        verbose=False,
    )
    path = str(tmp_path / "landscape.graphml")
    ls.to_graph(path)
    restored = Landscape.build_from_graph(path, verbose=False)
    assert restored.maximize == maximize
    assert restored.epsilon == epsilon
    assert restored.kind == "dna"
    assert restored.graph.get_edgelist() == ls.graph.get_edgelist()
    assert restored.graph.es["delta_fit"] == ls.graph.es["delta_fit"]
    assert restored.graph.vs["fitness"] == ls.graph.vs["fitness"]
    assert restored.n_lo == ls.n_lo
    assert restored._neutral_neighbors == ls._neutral_neighbors


@pytest.mark.parametrize(
    "alphabet,values",
    [(None, ["aa", "ac", "ca", "cc"]), (["Α", "Β"], ["ΑΑ", "ΑΒ", "ΒΑ", "ΒΒ"])],
)
def test_general_sequence_alphabet(alphabet, values):
    ls = SequenceLandscape(alphabet=alphabet).build_from_data(
        values, [0.0, 1.0, 2.0, 3.0], verbose=False
    )
    assert ls.shape == (4, 4)
    assert set(ls.graph.get_edgelist()) == {(0, 1), (0, 2), (1, 3), (2, 3)}


def test_ordered_categorical_levels_are_not_compacted_when_absent():
    X = pd.DataFrame(
        {
            "x": pd.Categorical(
                ["small", "large"],
                categories=["small", "medium", "large"],
                ordered=True,
            )
        }
    )
    with pytest.raises(ValueError, match="no edges"):
        OrdinalLandscape().build_from_data(X, [0.0, 1.0], verbose=False)


def test_input_permutation_preserves_labeled_graph():
    X = ["AA", "AC", "CA", "CC"]
    fitness = np.array([0.0, 0.25, 1.0, 2.0])

    def edges(order):
        ls = DNALandscape().build_from_data(
            [X[i] for i in order], fitness[order], verbose=False
        )
        return {(X[order[u]], X[order[v]]) for u, v in ls.graph.get_edgelist()}

    assert edges([0, 1, 2, 3]) == edges([3, 1, 0, 2])


def test_streaming_and_batched_attributes_match(monkeypatch):
    from graphfla.landscape import _build

    X = pd.DataFrame({"level": [0, 1, 2], "constant": [9, 9, 9]})
    landscapes = []
    for budget in (0, 1024**2):
        monkeypatch.setattr(_build, "_ATTRIBUTE_BUFFER_BYTES", budget)
        landscapes.append(
            OrdinalLandscape().build_from_data(X, [0.0, 1.0, 2.0], verbose=False)
        )
    first, second = landscapes
    assert first.graph.get_edgelist() == second.graph.get_edgelist()
    assert first.graph.es["delta_fit"] == second.graph.es["delta_fit"]
    pd.testing.assert_frame_equal(first.get_data(), second.get_data())


def test_auto_strategy_honors_registered_neighbor_generator():
    from graphfla._neighbors.generators import BooleanNeighborGenerator

    class FirstBitOnly(BooleanNeighborGenerator):
        def generate(self, config, config_dict, n_edit=1):
            return [(1 - config[0], *config[1:])]

    ls = BooleanLandscape()
    ls.register_neighbor_generator("boolean", FirstBitOnly())
    ls.build_from_data(["00", "01", "10", "11"], [0.0, 1.0, 2.0, 3.0], verbose=False)
    assert set(ls.graph.get_edgelist()) == {(0, 2), (1, 3)}


@pytest.mark.parametrize("missing", [False, True])
def test_wide_duplicate_detection_preserves_full_row_identity(missing):
    from graphfla._data._validation import _drop_duplicates

    X = pd.DataFrame({f"fixed_{j}": [1, 1, 1, 1] for j in range(24)})
    X["variable"] = [0, 0, 0, 1]
    if missing:
        X["fixed_0"] = [np.nan, 1, np.nan, 1]
    fitness = pd.Series([1.0, 2.0, 3.0, 4.0])
    expected = [0, 1, 3] if missing else [0, 3]
    actual_X, actual_f = _drop_duplicates(X, fitness)
    pd.testing.assert_frame_equal(actual_X, X.iloc[expected])
    pd.testing.assert_series_equal(actual_f, fitness.iloc[expected])


def test_all_constant_wide_rows_keep_first_occurrence():
    from graphfla._data._validation import _drop_duplicates

    X = pd.DataFrame(np.ones((4, 20)))
    fitness = pd.Series([1.0, 2.0, 3.0, 4.0])
    actual_X, actual_f = _drop_duplicates(X, fitness)
    pd.testing.assert_frame_equal(actual_X, X.iloc[:1])
    pd.testing.assert_series_equal(actual_f, fitness.iloc[:1])
