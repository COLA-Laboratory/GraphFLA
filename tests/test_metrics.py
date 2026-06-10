"""Correctness anchors for the analysis metrics.

Each test pins a metric to a value that is analytically known for a controlled
landscape, so a silently-wrong implementation fails CI. These are the guards
the smoke tests in ``test_all.py`` lack: the three value-bugs fixed during
verification (gamma/gamma*, autocorrelation, idiosyncratic-index dilution) all
returned plausible floats and would pass a type check, but fail the anchors
below.

Anchor key (additive == OneMax unless noted):
  gamma=1, gamma*(mag/sign/recip)=+1/0/-1, classify additive=all-magnitude,
  idiosyncratic additive=0 / HoC~1, autocorr=exact regular-graph rho,
  r/s additive=0, FDC OneMax=-1, neighbor_fit_corr additive=+1,
  higher-order additive order-1 R^2=1, Walsh additive order>=2 = 0,
  evol_enhance additive=1, GO-accessibility=1, mean path length OneMax(6)=3.
"""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from graphfla.analysis import (
    gamma_statistic,
    gamma_star,
    classify_epistasis,
    global_idiosyncratic_index,
    autocorrelation,
    r_s_ratio,
    fitness_distance_corr,
    neighbor_fit_corr,
    higher_order_epistasis,
    walsh_hadamard_coefficient,
    fitness_distribution,
    neutrality,
    evol_enhance_mutations,
    global_optima_accessibility,
    mean_path_lengths_go,
    lo_ratio,
    gradient_intensity,
)

from _landscapes import (
    onemax,
    hoc_landscape,
    additive_landscape,
    from_map,
    MAGNITUDE_SQUARE,
    SIGN_SQUARE,
    RECIPROCAL_SQUARE,
)


# ----------------------------------------------------------------------
# Gamma / gamma* (Ferretti 2016) -- regression for fixed bug #1
# ----------------------------------------------------------------------


def test_gamma_additive_is_one():
    # Purely additive: a mutation's effect is identical on every background,
    # so the (non-centered) correlation of effects is exactly 1.
    assert gamma_statistic(onemax(5), n_jobs=1) == pytest.approx(1.0, abs=1e-6)


def test_gamma_hoc_near_zero():
    # House-of-Cards: effects are uncorrelated across backgrounds -> ~0.
    assert abs(gamma_statistic(hoc_landscape(6, seed=3), n_jobs=1)) < 0.5


def test_gamma_star_magnitude_sign_reciprocal():
    # The discriminating anchor: only the corrected (non-capped) implementation
    # can produce -1. On a single 2-locus square there is one locus pair, so
    # the value is exact with no background contamination.
    assert gamma_star(from_map(MAGNITUDE_SQUARE, 2), n_jobs=1) == pytest.approx(1.0, abs=1e-6)
    assert gamma_star(from_map(SIGN_SQUARE, 2), n_jobs=1) == pytest.approx(0.0, abs=1e-6)
    assert gamma_star(from_map(RECIPROCAL_SQUARE, 2), n_jobs=1) == pytest.approx(-1.0, abs=1e-6)


# ----------------------------------------------------------------------
# classify_epistasis -- motif mapping & counting
# ----------------------------------------------------------------------


def test_classify_epistasis_pure_squares():
    for square, key in [
        (MAGNITUDE_SQUARE, "magnitude epistasis"),
        (SIGN_SQUARE, "sign epistasis"),
        (RECIPROCAL_SQUARE, "reciprocal sign epistasis"),
    ]:
        res = classify_epistasis(from_map(square, 2))
        assert res[key] == pytest.approx(1.0)


def test_classify_epistasis_additive_all_magnitude():
    res = classify_epistasis(onemax(4))
    assert res["magnitude epistasis"] == pytest.approx(1.0)
    assert res["sign epistasis"] == pytest.approx(0.0)
    assert res["reciprocal sign epistasis"] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Idiosyncratic index (Lyons 2020) -- regression for fixed bug #2
# ----------------------------------------------------------------------


def test_global_idiosyncratic_additive_zero():
    ls, _ = additive_landscape(6, seed=2)
    assert global_idiosyncratic_index(ls, n_jobs=1) == pytest.approx(0.0, abs=0.02)


def test_global_idiosyncratic_hoc_near_one():
    val = global_idiosyncratic_index(hoc_landscape(6, seed=3), n_jobs=1)
    assert 0.7 < val < 1.3


# ----------------------------------------------------------------------
# Autocorrelation (Weinberger 1990) -- regression for fixed bug #3
# ----------------------------------------------------------------------


def test_autocorrelation_matches_exact_edge_rho():
    # On a regular graph (the OneMax hypercube, degree n) the stationary walk
    # is uniform, so the lag-1 autocorrelation has a closed form. The pooled
    # estimator must converge to it; the old per-walk-centering bug drove it
    # ~28% low, far outside this tolerance.
    ls = onemax(6)
    fit = np.asarray(ls.graph.vs["fitness"], dtype=float)
    mu = fit.mean()
    degree = 6
    directed_pairs = set()
    for u, v in ls.graph.get_edgelist():
        directed_pairs.add((u, v))
        directed_pairs.add((v, u))
    num = sum((fit[u] - mu) * (fit[v] - mu) for u, v in directed_pairs)
    rho_exact = num / (degree * np.sum((fit - mu) ** 2))

    rho_est = autocorrelation(ls, walk_length=40, walk_times=4000, seed=0)
    assert rho_est == pytest.approx(rho_exact, abs=0.05)


def test_autocorrelation_additive_high_hoc_low():
    assert autocorrelation(onemax(6), walk_length=40, walk_times=2000, seed=1) > 0.4
    assert abs(autocorrelation(hoc_landscape(6, seed=3), walk_length=40, walk_times=2000, seed=1)) < 0.25


# ----------------------------------------------------------------------
# r/s ratio (Szendro 2013)
# ----------------------------------------------------------------------


def test_r_s_ratio_additive_zero_hoc_large():
    assert r_s_ratio(onemax(4)) < 1e-9  # additive fit is exact -> RMSE 0
    assert r_s_ratio(hoc_landscape(4, seed=1)) > 0.5


# ----------------------------------------------------------------------
# Fitness-distance correlation (Jones & Forrest 1995)
# ----------------------------------------------------------------------


def test_fitness_distance_corr_onemax_is_minus_one():
    # OneMax: fitness = n - distance_to_optimum exactly, so the (Spearman)
    # correlation of distance vs fitness is exactly -1. The *sign* is the
    # load-bearing property; a distance/direction flip would invert it.
    ls = onemax(5)
    assert fitness_distance_corr(ls) == pytest.approx(-1.0, abs=1e-9)


# ----------------------------------------------------------------------
# Neighbor-fitness correlation
# ----------------------------------------------------------------------


def test_neighbor_fit_corr_additive_is_one():
    # For additive f, mean-neighbour-fitness is an exactly linear function of
    # f (the quadratic terms cancel), so Pearson correlation is exactly +1.
    ls = onemax(5)
    assert neighbor_fit_corr(ls, method="pearson") == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------------
# Higher-order epistasis & Walsh-Hadamard (additive decomposition)
# ----------------------------------------------------------------------


def test_higher_order_epistasis_additive_order1_is_one():
    # Additive landscape is fully explained by main (order-1) effects.
    assert higher_order_epistasis(onemax(4), order=1) == pytest.approx(1.0, abs=1e-6)
    # HoC is not explainable by order-1 alone.
    assert higher_order_epistasis(hoc_landscape(4, seed=1), order=1) < 0.99


def test_walsh_additive_higher_orders_are_zero():
    coeffs = walsh_hadamard_coefficient(onemax(5), max_order=2)
    order2 = coeffs.get(2, {})
    assert order2  # there are pairwise terms to check
    assert max(abs(v) for v in order2.values()) < 1e-6
    # ...while the additive (order-1) effects are non-trivial.
    assert max(abs(v) for v in coeffs[1].values()) > 1e-2


# ----------------------------------------------------------------------
# Fitness distribution (formula faithfulness, incl. Pearson +3 kurtosis)
# ----------------------------------------------------------------------


def test_fitness_distribution_matches_reference_formulas():
    ls, _ = additive_landscape(6, seed=4)
    fit = np.asarray(ls.graph.vs["fitness"], dtype=float)
    res = fitness_distribution(ls)
    # Kurtosis must be Pearson's (normal == 3), i.e. scipy Fisher + 3.
    assert res["kurtosis"] == pytest.approx(scipy_stats.kurtosis(fit) + 3.0, abs=1e-9)
    assert res["skewness"] == pytest.approx(scipy_stats.skew(fit), abs=1e-9)
    assert res["cv"] == pytest.approx(np.std(fit, ddof=1) / abs(fit.mean()), abs=1e-9)


# ----------------------------------------------------------------------
# Neutrality
# ----------------------------------------------------------------------


def test_neutrality_onemax_is_zero():
    # Every OneMax neighbour differs in fitness by exactly 1 -> no neutrality.
    assert neutrality(onemax(4)) == 0.0


def test_neutrality_detects_ties():
    # A 3-cube with a shell of equal-fitness genotypes has genuine neutrality.
    fmap = {
        (0, 0, 0): 0.0,
        (1, 0, 0): 1.0, (0, 1, 0): 1.0, (0, 0, 1): 1.0,
        (1, 1, 0): 1.0, (1, 0, 1): 1.0, (0, 1, 1): 1.0,
        (1, 1, 1): 2.0,
    }
    assert neutrality(from_map(fmap, 3)) > 0.3


# ----------------------------------------------------------------------
# Evolvability-enhancing mutations (Wagner 2023)
# ----------------------------------------------------------------------


def test_evol_enhance_additive_is_one():
    # Additive: every improving edge also increases mean neighbour fitness.
    assert evol_enhance_mutations(onemax(4)) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Navigability
# ----------------------------------------------------------------------


def test_global_optima_accessibility_onemax_is_one():
    # Single peak == global optimum: every node reaches it via improving moves.
    assert global_optima_accessibility(onemax(5)) == pytest.approx(1.0)


def test_mean_path_lengths_go_onemax_is_n_over_2():
    # OneMax(6): improving path length to the all-ones corner equals the number
    # of zeros; averaged over all nodes that is n/2 = 3.
    assert mean_path_lengths_go(onemax(6)) == pytest.approx(3.0)


# ----------------------------------------------------------------------
# Ruggedness scalars with exact small-cube values
# ----------------------------------------------------------------------


def test_lo_ratio_single_peak():
    assert lo_ratio(onemax(4)) == pytest.approx(1.0 / 16.0)


def test_gradient_intensity_exact():
    # 2-cube with fitness [0,1,2,3]: mean|delta_fit| == mean(fitness) == 1.5.
    ls = from_map({(0, 0): 0.0, (0, 1): 1.0, (1, 0): 2.0, (1, 1): 3.0}, 2)
    assert gradient_intensity(ls) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# single / all_mutation_effects -- vectorised impl vs an independent
# brute-force reference (nested-loop background matching). These metrics
# had no correctness anchor before the vectorisation.
# ----------------------------------------------------------------------
import warnings as _warnings
from itertools import combinations as _combinations

from graphfla.analysis import single_mutation_effects, all_mutation_effects


def _ref_mutation_effects(landscape, position, test_type="positive"):
    """Independent reference: group genotypes by their background tuple (all other
    positions) and pair alleles, mirroring the original semantics directly."""
    data = landscape.get_data()
    X = data[list(landscape.data_types.keys())]
    f = data["fitness"]
    f_std = f.std()
    bg_cols = [c for c in X.columns if c != position]
    vals = sorted(X[position].dropna().unique())
    rows = []
    for A, B in _combinations(vals, 2):
        bg_a, bg_b = {}, {}
        for idx in X.index:
            bg = tuple(X.loc[idx, c] for c in bg_cols)
            v = X.loc[idx, position]
            if v == A and bg not in bg_a:
                bg_a[bg] = f.loc[idx]
            elif v == B and bg not in bg_b:
                bg_b[bg] = f.loc[idx]
        diff = np.array([bg_a[k] - bg_b[k] for k in bg_a if k in bg_b], dtype=float)
        n = diff.size
        if n == 0:
            rows.append((A, B, np.nan, np.nan, np.nan, False))
            continue
        succ = int((diff > 0).sum()) if test_type == "positive" else int((diff < 0).sum())
        r = scipy_stats.binomtest(succ, n, 0.5, alternative="greater")
        rows.append((A, B, float(np.median(np.abs(diff))) / f_std,
                     float(diff.mean()), r.pvalue, r.pvalue < 0.05))
    return rows


def _assert_effects_match(df, ref):
    assert len(df) == len(ref)
    for row, (A, B, med, mean, pval, sig) in zip(df.to_dict("records"), ref):
        assert row["mutation_from"] == A and row["mutation_to"] == B
        assert row["median_abs_effect"] == pytest.approx(med, nan_ok=True, rel=1e-9)
        assert row["mean_effect"] == pytest.approx(mean, nan_ok=True, rel=1e-9)
        assert row["p_value"] == pytest.approx(pval, nan_ok=True, rel=1e-9)
        assert bool(row["significant"]) is bool(sig)


def test_single_mutation_effects_matches_reference():
    ls = hoc_landscape(4, seed=7)
    X = ls.get_data()[list(ls.data_types.keys())]
    for pos in X.columns:
        for tt in ("positive", "negative"):
            _assert_effects_match(
                single_mutation_effects(ls, pos, test_type=tt),
                _ref_mutation_effects(ls, pos, tt),
            )


def test_all_mutation_effects_matches_reference():
    ls = hoc_landscape(4, seed=11)
    X = ls.get_data()[list(ls.data_types.keys())]
    ref = []
    for pos in X.columns:
        ref.extend(_ref_mutation_effects(ls, pos, "positive"))
    _assert_effects_match(
        all_mutation_effects(ls, test_type="positive").reset_index(drop=True), ref
    )


def test_mutation_effects_multiallele():
    # A position with 3 levels exercises the combinations(>1 pair) path.
    from graphfla.landscape import OrdinalLandscape
    import pandas as pd

    rng = np.random.default_rng(3)
    combos = [(a, b) for a in range(3) for b in range(2)]
    Xdf = pd.DataFrame(combos, columns=["p0", "p1"])
    fser = pd.Series(rng.normal(size=len(combos)))
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        ls = OrdinalLandscape(maximize=True).build_from_data(Xdf, fser, verbose=False)
    pos = list(ls.data_types.keys())[0]
    _assert_effects_match(
        single_mutation_effects(ls, pos, test_type="positive"),
        _ref_mutation_effects(ls, pos, "positive"),
    )
