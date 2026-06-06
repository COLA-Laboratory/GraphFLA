"""Correctness anchors for landscape *construction*.

These pin the structural invariants of the directed improving-edge graph on
hand-computable landscapes: edge set and direction, the neighbourhood
semantics for boolean / sequence / ordinal / categorical / mixed encodings,
the epsilon neutral-vs-improving boundary, local-optima detection, basins,
and the minimisation regime. A silently-wrong neighbourhood or edge direction
corrupts every downstream metric, so these are the foundation.
"""

import numpy as np
import pandas as pd
import pytest

from graphfla.landscape import Landscape, OrdinalLandscape
from graphfla.problems import NK
from graphfla._neighbors import (
    BooleanNeighborGenerator,
    SequenceNeighborGenerator,
    OrdinalNeighborGenerator,
    DefaultNeighborGenerator,
)

from _landscapes import onemax, from_map, TWO_PEAK_3CUBE


# ----------------------------------------------------------------------
# Complete-cube invariants
# ----------------------------------------------------------------------


def test_onemax_cube_construction_invariants():
    # A complete boolean n-cube has n * 2^(n-1) Hamming-1 edges; with strict
    # monotone fitness every edge is improving and there is a single sink.
    ls = onemax(3)
    assert ls.n_configs == 8
    assert ls.graph.ecount() == 3 * 2 ** 2  # 12 directed improving edges
    assert ls.n_lo == 1
    assert ls.go_index == 7  # all-ones corner
    assert ls._has_plateaus is False


def test_edge_direction_and_delta_fit_uphill():
    ls = onemax(4)
    fit = ls.graph.vs["fitness"]
    for e in ls.graph.es:
        # Every directed edge points from lower to higher fitness (maximise),
        assert fit[e.target] > fit[e.source]
        # and delta_fit records the (positive) fitness gap.
        assert e["delta_fit"] == pytest.approx(abs(fit[e.target] - fit[e.source]))


def test_minimize_flips_optimum_and_edge_direction():
    # Same OneMax data, minimising: the optimum becomes the all-zeros corner
    # and edges point downhill toward it.
    ls = onemax(3, maximize=False)
    assert ls.go_index == 0  # all-zeros
    assert ls.n_lo == 1
    fit = ls.graph.vs["fitness"]
    for e in ls.graph.es:
        assert fit[e.target] < fit[e.source]


@pytest.mark.parametrize("strategy", ["active", "pairwise", "broadcast", "auto"])
def test_neighborhood_strategies_agree(strategy):
    # All neighbourhood strategies must produce the identical edge set on a
    # complete boolean cube; a bug in any one path would diverge here.
    ls = onemax(4, neighborhood_strategy=strategy)
    assert sorted(ls.graph.get_edgelist()) == sorted(onemax(4).graph.get_edgelist())
    assert ls.graph.ecount() == 4 * 2 ** 3
    assert ls.n_lo == 1
    assert ls.go_index == 15


# ----------------------------------------------------------------------
# Local optima (the classic out_degree == 0 branch) and basins
# ----------------------------------------------------------------------


def test_local_optima_outdegree_branch_additive():
    # Additive cube has no plateaus -> exercises the non-plateau LO branch
    # (the Papkou-514 algorithm: out_degree == 0). Exactly one sink.
    ls = onemax(3)
    assert ls._has_plateaus is False
    assert ls.n_lo == 1
    assert ls.lo_index == [7]
    assert sum(1 for d in ls.graph.outdegree() if d == 0) == 1
    assert ls.graph.vs[7]["is_lo"] is True


def test_two_peak_optima_and_greedy_basins():
    ls = from_map(TWO_PEAK_3CUBE, 3)
    assert ls.n_lo == 2
    assert ls.lo_index == [0, 7]
    assert ls.go_index == 0  # f(000)=10 is the global max
    # Greedy (best-improvement) basins partition the 8 nodes 4/4.
    ls.basins  # compute basins lazily (populates basin_index / size_basin_greedy)
    basin_index = ls.graph.vs["basin_index"]
    assert set(basin_index) == {0, 7}
    assert ls.graph.vs[0]["size_basin_greedy"] == 4
    assert ls.graph.vs[7]["size_basin_greedy"] == 4
    sizes = {b: basin_index.count(b) for b in set(basin_index)}
    assert sum(sizes.values()) == 8  # greedy basins are a partition


# ----------------------------------------------------------------------
# Neighbour generators (unit level — exact neighbour sets)
# ----------------------------------------------------------------------


def test_boolean_neighbor_generator_single_flips():
    nb = BooleanNeighborGenerator().generate((0, 1, 0), {}, n_edit=1)
    assert set(nb) == {(1, 1, 0), (0, 0, 0), (0, 1, 1)}
    with pytest.raises(ValueError):
        BooleanNeighborGenerator().generate((0, 1), {}, n_edit=2)


def test_sequence_neighbor_generator_substitutions():
    # Alphabet size 4 at each of 2 positions -> 3 substitutions per position.
    nb = SequenceNeighborGenerator(4).generate((0, 0), {}, n_edit=1)
    assert set(nb) == {(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)}
    assert len(nb) == 6


def test_ordinal_neighbor_generator_pm1_only():
    # Manhattan-1 steps only: boundary 0 has no -1, boundary max has no +1,
    # and a +2 jump (0->2) never appears.
    cd = {0: {"max": 2}, 1: {"max": 2}, 2: {"max": 2}}
    nb = OrdinalNeighborGenerator().generate((0, 1, 2), cd, n_edit=1)
    assert set(nb) == {(1, 1, 2), (0, 0, 2), (0, 2, 2), (0, 1, 1)}
    assert (0, 2, 2) not in {(0, 1, 2)}  # sanity: 0->2 on locus 0 absent
    assert all(abs(np.array(n)[0] - 0) <= 1 for n in nb)


def test_default_neighbor_generator_mixed_types():
    # boolean (1 flip) + categorical(max=2, current=2 -> 2 others)
    # + ordinal(current=1 interior -> 2 steps) == 5 neighbours.
    cd = {
        0: {"type": "boolean", "max": 1},
        1: {"type": "categorical", "max": 2},
        2: {"type": "ordinal", "max": 2},
    }
    nb = DefaultNeighborGenerator().generate((1, 2, 1), cd, n_edit=1)
    assert set(nb) == {(0, 2, 1), (1, 0, 1), (1, 1, 1), (1, 2, 0), (1, 2, 2)}
    assert len(nb) == 5


# ----------------------------------------------------------------------
# Neighbourhood semantics at the landscape level: ordinal vs categorical
# ----------------------------------------------------------------------


def test_ordinal_landscape_is_manhattan_not_hamming():
    # 3-level single ordinal variable: only adjacent levels connect.
    X = pd.DataFrame({"d": [0, 1, 2]})
    ls = OrdinalLandscape()
    ls.build_from_data(X, [0.0, 1.0, 2.0], verbose=False)
    assert ls.graph.ecount() == 2  # 0->1, 1->2 only
    # level 0 and level 2 are NOT neighbours (Manhattan distance 2)
    assert ls.graph.get_eid(0, 2, directed=False, error=False) == -1


def test_categorical_connects_every_other_value():
    # Same data as categorical: every distinct value is a neighbour, so
    # level 0 and level 2 ARE connected -- the defining ordinal/categorical
    # distinction. Uses the 'active' strategy to exercise the generator.
    X = pd.DataFrame({"d": [0, 1, 2]})
    ls = Landscape(type="default")
    ls.build_from_data(
        X,
        [0.0, 1.0, 2.0],
        data_types={"d": "categorical"},
        neighborhood_strategy="active",
        verbose=False,
    )
    assert ls.graph.ecount() == 3  # 0->1, 0->2, 1->2
    assert ls.graph.get_eid(0, 2, directed=False, error=False) != -1


# ----------------------------------------------------------------------
# Epsilon neutral-vs-improving boundary
# ----------------------------------------------------------------------


def test_epsilon_neutral_boundary():
    # Pair (00, 01) differs by exactly 0.5 in fitness.
    fmap = {(0, 0): 0.0, (0, 1): 0.5, (1, 0): 1.0, (1, 1): 1.6}

    # epsilon = 0: the 0.5 gap is an *improving* edge; no plateau.
    strict = from_map(fmap, 2, epsilon=0.0)
    assert strict._has_plateaus is False
    assert strict.graph.get_eid(0, 1, directed=False, error=False) != -1

    # epsilon = 0.5: |Δf| <= epsilon -> the pair is neutral (no directed edge)
    # and the two nodes share a plateau.
    smooth = from_map(fmap, 2, epsilon=0.5)
    assert smooth.graph.get_eid(0, 1, directed=False, error=False) == -1
    assert smooth._has_plateaus is True
    assert smooth._node_to_plateau[0] == smooth._node_to_plateau[1] >= 0


# ----------------------------------------------------------------------
# Problem generators (the anchors every metric test is built on)
# ----------------------------------------------------------------------


def test_nk_k0_is_purely_additive():
    # K=0 -> each locus depends only on itself; flips superpose exactly.
    prob = NK(5, 0, seed=0)
    assert prob.dependence == [(0,), (1,), (2,), (3,), (4,)]
    base = (0, 0, 0, 0, 0)
    f0 = prob.evaluate(base)
    single = []
    for i in range(5):
        c = list(base)
        c[i] = 1
        single.append(prob.evaluate(tuple(c)) - f0)
    full = prob.evaluate((1, 1, 1, 1, 1)) - f0
    assert sum(single) == pytest.approx(full, abs=1e-9)


def test_nk_kpos_has_interactions():
    prob = NK(5, 2, seed=0)
    assert any(len(d) > 1 for d in prob.dependence)


def test_get_data_contract():
    prob = NK(3, 1, seed=0)
    X, f = prob.get_data()
    assert X == ["000", "001", "010", "011", "100", "101", "110", "111"]
    assert len(X) == 8 == len(f)
