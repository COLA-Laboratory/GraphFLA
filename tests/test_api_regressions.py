"""Regression guards for public API behaviour and previously-untested modules.

Covers the builder pattern (methods return ``self``), seeded-RNG
reproducibility, instance-level strategy registries, pickling (the Modal /
ProteinGym pipeline depends on it), input validation, the ``LandscapeFilter``
``contains`` guard, the LON module (zero prior coverage), and the sampling
seed contract. These would have caught the API regressions introduced this
cycle; today they are the only coverage for several whole files.
"""

import pickle
from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from graphfla.landscape import Landscape, BooleanLandscape, DNALandscape
from graphfla._data import BooleanHandler
from graphfla.distances import hamming_distance
from graphfla.algorithms import random_walk, SearchCache
from graphfla.filters import LandscapeFilter
from graphfla.sampling import latin_hypercube_sampling, sobol_sampling
from graphfla.analysis import (
    autocorrelation,
    classify_epistasis,
    global_idiosyncratic_index,
    mean_path_lengths_go,
)

from _landscapes import onemax, hoc_landscape, from_map, TWO_PEAK_3CUBE


# ----------------------------------------------------------------------
# Builder pattern: build_from_data and determine_* return self
# ----------------------------------------------------------------------


def test_build_from_data_returns_self():
    X = [np.array(c) for c in product([0, 1], repeat=3)]
    f = [float(sum(c)) for c in X]
    ls = BooleanLandscape()
    assert ls.build_from_data(X, f, verbose=False) is ls


def test_determine_methods_return_self():
    ls = onemax(3)
    # The optima determiners return self for chaining.
    assert ls.determine_local_optima() is ls
    assert ls.determine_global_optimum() is ls


def test_cached_analysis_properties_compute_lazily():
    """The canonical cached properties compute on first access (no build flags)
    and cache via the guard flags."""
    ls = onemax(3)  # built without any calculate_* flag
    assert not ls._basin_calculated and not ls._distance_calculated
    # Hamming distance to the unique all-ones optimum = number of zeros.
    assert sorted(ls.dist_to_go) == [0, 1, 1, 1, 2, 2, 2, 3]
    assert ls._distance_calculated  # cached after first access
    assert len(ls.basins) == ls.n_configs and ls._basin_calculated
    assert len(ls.neighbor_fitness) == ls.n_configs and ls._neighbor_fit_calculated
    assert len(ls.accessible_paths) == ls.n_configs and ls._path_calculated


def test_count_properties_single_source_from_graph():
    """n_configs / n_edges / shape derive from the graph (single source of truth)."""
    ls = onemax(4)
    assert ls.n_configs == ls.graph.vcount() == 16
    assert ls.n_edges == ls.graph.ecount()
    assert ls.shape == (ls.n_configs, ls.n_edges)
    # They are read-only properties now (no stored field to drift).
    with pytest.raises(AttributeError):
        ls.n_configs = 99
    with pytest.raises(AttributeError):
        ls.n_edges = 99


def test_ordinal_default_strategy_and_hamming_warning():
    """Ordinal landscapes default to the type-correct 'active' neighbourhood and
    warn when a Hamming strategy (pairwise/broadcast) is requested (fix 3 + 5)."""
    from graphfla.landscape import OrdinalLandscape

    assert OrdinalLandscape._default_neighborhood_strategy == "active"
    X = pd.DataFrame({"a": [0, 1, 2, 0, 1, 2], "b": [0, 0, 0, 1, 1, 1]})
    f = [0.1, 0.5, 0.9, 0.2, 0.6, 1.0]
    # Default (None) resolves to 'active' for ordinal — builds without warning.
    OrdinalLandscape().build_from_data(X, f, verbose=False)
    # Explicit Hamming strategy on ordinal data warns.
    with pytest.warns(UserWarning, match="ordinal"):
        OrdinalLandscape().build_from_data(
            X, f, neighborhood_strategy="pairwise", verbose=False
        )


# ----------------------------------------------------------------------
# Seeded-RNG reproducibility
# ----------------------------------------------------------------------


def test_random_walk_seed_reproducible():
    ls = onemax(5)
    cache = SearchCache(ls.graph)
    w1 = random_walk(cache, 0, "fitness", 60, seed=42)
    w2 = random_walk(cache, 0, "fitness", 60, seed=42)
    w3 = random_walk(cache, 0, "fitness", 60, seed=7)
    assert np.array_equal(w1, w2)        # same seed -> identical walk
    assert not np.array_equal(w1, w3)    # different seed -> different walk


def test_autocorrelation_seed_reproducible():
    ls = onemax(5)
    a1 = autocorrelation(ls, walk_length=20, walk_times=200, seed=123)
    a2 = autocorrelation(ls, walk_length=20, walk_times=200, seed=123)
    assert a1 == a2


def test_mean_path_lengths_go_seed_reproducible():
    ls = onemax(7)  # 128 nodes, so sampling actually subsets
    m1 = mean_path_lengths_go(ls, n_samples=30, seed=5)
    m2 = mean_path_lengths_go(ls, n_samples=30, seed=5)
    m3 = mean_path_lengths_go(ls, n_samples=30, seed=99)
    assert m1 == m2
    assert m1 != m3


def test_classify_epistasis_approximate_seed_reproducible():
    ls = onemax(5)
    r1 = classify_epistasis(ls, approximate=True, seed=11)
    r2 = classify_epistasis(ls, approximate=True, seed=11)
    assert r1 == r2


def test_global_idiosyncratic_seed_reproducible():
    ls = hoc_landscape(6, seed=2)
    assert global_idiosyncratic_index(ls, n_jobs=1, seed=3) == global_idiosyncratic_index(
        ls, n_jobs=1, seed=3
    )


# ----------------------------------------------------------------------
# Instance-level strategy registries (no global/cross-instance leakage)
# ----------------------------------------------------------------------


def test_register_handler_is_instance_isolated():
    a = Landscape(type="default")
    b = Landscape(type="default")
    a.register_input_handler("custom_xyz", BooleanHandler())
    assert "custom_xyz" in a._input_handlers
    assert "custom_xyz" not in b._input_handlers
    assert "custom_xyz" not in Landscape._input_handlers  # class registry intact


# ----------------------------------------------------------------------
# Pickling (Modal / ProteinGym pipeline requirement)
# ----------------------------------------------------------------------


def test_boolean_landscape_pickle_roundtrip():
    ls = onemax(4)
    ls2 = pickle.loads(pickle.dumps(ls))
    assert ls2.n_lo == ls.n_lo
    assert ls2.go_index == ls.go_index
    assert ls2.n_configs == ls.n_configs
    assert ls2.graph.ecount() == ls.graph.ecount()


def test_sequence_landscape_pickle_roundtrip():
    # SequenceLandscape keys its handler by id(alphabet); the handler must
    # travel with the pickled instance (the registry is per-instance).
    seqs = ["".join(p) for p in product("ACGT", repeat=2)]
    f = [float(i) for i in range(len(seqs))]
    ls = DNALandscape()
    ls.build_from_data(seqs, f, verbose=False)
    ls2 = pickle.loads(pickle.dumps(ls))
    assert ls2.n_configs == ls.n_configs
    assert ls2.graph.ecount() == ls.graph.ecount()
    assert ls2.type in ls2._input_handlers
    assert ls2.type in ls2._neighbor_generators


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------


def test_unknown_type_raises_valueerror():
    with pytest.raises(ValueError):
        Landscape(type="banana")


# ----------------------------------------------------------------------
# LandscapeFilter 'contains' guard
# ----------------------------------------------------------------------


def test_landscapefilter_contains_typeerror_on_numeric_column():
    df = pd.DataFrame({"x": [1, 2, 3], "fitness": [0.1, 0.2, 0.3]})
    filt = LandscapeFilter([{"column": "x", "operation": "contains", "value": "1"}])
    with pytest.raises(TypeError):
        filt.apply(df)


def test_landscapefilter_contains_on_string_column():
    df = pd.DataFrame({"seq": ["AAA", "ACA", "GGG"], "fitness": [1.0, 2.0, 3.0]})
    filt = LandscapeFilter([{"column": "seq", "operation": "contains", "value": "C"}])
    out = filt.apply(df)
    assert list(out["seq"]) == ["ACA"]


# ----------------------------------------------------------------------
# Local Optima Network (previously zero coverage)
# ----------------------------------------------------------------------


def test_lon_node_count_matches_optima():
    ls = from_map(TWO_PEAK_3CUBE, 3)
    lon = ls.get_lon(mlon=False, min_edge_freq=0, verbose=False)
    assert lon.vcount() == ls.n_lo == 2
    assert set(lon.vs["name"]) == set(ls._peak_index)


def test_mlon_keeps_only_non_worsening_edges():
    ls = from_map(TWO_PEAK_3CUBE, 3)
    lon = ls.get_lon(mlon=True, min_edge_freq=0, verbose=False)
    fit = dict(zip(lon.vs["name"], lon.vs["fitness"]))
    for e in lon.es:
        src = lon.vs[e.source]["name"]
        tgt = lon.vs[e.target]["name"]
        assert fit[tgt] >= fit[src]  # M-LON drops worsening transitions


def test_lon_min_edge_freq_boundary():
    # Each peak has C(3,2)=3 two-edit neighbours in the other basin -> escape
    # frequency 3. The mask is `lo_adj <= min_edge_freq -> 0` (strictly greater
    # survives): freq 3 is KEPT at threshold 2 but DROPPED at threshold 3.
    ls = from_map(TWO_PEAK_3CUBE, 3)
    keep = ls.get_lon(mlon=False, min_edge_freq=2, verbose=False)
    drop = ls.get_lon(mlon=False, min_edge_freq=3, verbose=False)
    assert keep.ecount() > drop.ecount()


# ----------------------------------------------------------------------
# Sampling seed contract
# ----------------------------------------------------------------------


def test_lhs_seed_reproducible_continuous():
    dists = {"x": scipy_stats.uniform(0, 1), "y": scipy_stats.norm(0, 1)}
    evaluate = lambda c: c["x"] + c["y"]
    df1 = latin_hypercube_sampling(dists, 20, evaluate, seed=4)
    df2 = latin_hypercube_sampling(dists, 20, evaluate, seed=4)
    pd.testing.assert_frame_equal(df1[["x", "y"]], df2[["x", "y"]])


def test_lhs_seed_reproducible_categorical():
    # Categorical columns must also honour seed= (regression for the unseeded
    # np.random.choice bug). Two same-seed runs are identical end-to-end.
    dists = {"x": scipy_stats.uniform(0, 1), "cat": ["a", "b", "c", "d"]}
    evaluate = lambda c: 1.0
    df1 = latin_hypercube_sampling(dists, 40, evaluate, seed=4)
    df2 = latin_hypercube_sampling(dists, 40, evaluate, seed=4)
    pd.testing.assert_frame_equal(df1, df2)


def test_sobol_seed_reproducible_categorical():
    dists = {"x": scipy_stats.uniform(0, 1), "cat": ["a", "b", "c", "d"]}
    evaluate = lambda c: 1.0
    df1 = sobol_sampling(dists, 16, evaluate, seed=4)
    df2 = sobol_sampling(dists, 16, evaluate, seed=4)
    pd.testing.assert_frame_equal(df1, df2)


# ----------------------------------------------------------------------
# Plotting smoke (previously zero coverage): builds without raising
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def rugged_landscape():
    from _landscapes import nk_landscape

    return nk_landscape(6, 3, seed=1)


def test_plotting_smoke_all_single_arg_draws(rugged_landscape):
    # graphfla.plotting relies on optional extras (matplotlib + seaborn) that
    # are not core install_requires; skip cleanly if the module can't import.
    pytest.importorskip("graphfla.plotting")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from graphfla.plotting import (
        draw_diminishing_return,
        draw_fitness_distance_corr,
        draw_neighbor_fit_corr,
        draw_adaptive_walk,
        draw_global_epistasis,
        draw_fitness_distribution,
        draw_basin_fit_corr,
    )

    draws = [
        lambda ls: draw_diminishing_return(ls),
        lambda ls: draw_fitness_distance_corr(ls),
        lambda ls: draw_neighbor_fit_corr(ls),
        lambda ls: draw_adaptive_walk(ls, n_walks=5),
        lambda ls: draw_global_epistasis(ls),
        lambda ls: draw_fitness_distribution(ls),
        lambda ls: draw_basin_fit_corr(ls),
    ]
    for draw in draws:
        draw(rugged_landscape)  # must not raise
        plt.close("all")
