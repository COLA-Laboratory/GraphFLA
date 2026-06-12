"""Tests for analysis.profile() and the classify/bypass auto sample_cut_prob."""

import numpy as np
import pandas as pd
import pytest

from graphfla import analysis as A
from graphfla.analysis import profile, list_metrics
from graphfla.analysis.epistasis.motifs import (
    _auto_cut_prob,
    _motif_cost,
    _resolve_cut_prob,
)

from _landscapes import hoc_landscape, nk_landscape, onemax


# --------------------------------------------------------------------------- #
# profile() -- output shape & contents
# --------------------------------------------------------------------------- #
def test_profile_returns_float_series_with_expected_metrics():
    s = profile(nk_landscape(6, 2, seed=0), seed=0)
    assert isinstance(s, pd.Series)
    assert s.dtype == float
    for col in (
        "gamma", "fdc", "neutrality", "evolvability_enhancing_mutations",
        "fitness.skewness", "epistasis.magnitude", "bypass.proportion",
    ):
        assert col in s.index


def test_profile_columns_match_registry():
    s = profile(onemax(5), seed=0)
    expected = set()
    for cols in list_metrics()["columns"]:
        expected.update(c.strip() for c in cols.split(","))
    assert set(s.index) == expected


# --------------------------------------------------------------------------- #
# profile() -- selection semantics (groups XOR include, exclude composes)
# --------------------------------------------------------------------------- #
def test_profile_groups_restrict_to_group():
    s = profile(nk_landscape(6, 2, seed=1), groups="epistasis", seed=0)
    lm = list_metrics()
    epi = set()
    for cols in lm[lm["group"] == "epistasis"]["columns"]:
        epi.update(c.strip() for c in cols.split(","))
    assert set(s.index) == epi


def test_profile_include_is_exact_set():
    s = profile(onemax(5), include=["fdc", "gamma"], seed=0)
    assert list(s.index) == ["fdc", "gamma"]


def test_profile_exclude_composes_with_groups():
    s = profile(
        nk_landscape(6, 2, seed=2),
        groups="epistasis",
        exclude=["classify_epistasis", "extradimensional_bypass"],
        seed=0,
    )
    assert "gamma" in s.index
    assert "epistasis.magnitude" not in s.index
    assert "bypass.proportion" not in s.index


def test_profile_groups_and_include_conflict():
    with pytest.raises(ValueError):
        profile(onemax(4), groups="epistasis", include=["fdc"])


@pytest.mark.parametrize("kw", [{"groups": "nope"}, {"include": ["nope"]}, {"exclude": ["nope"]}])
def test_profile_unknown_names_raise(kw):
    with pytest.raises(ValueError):
        profile(onemax(4), **kw)


# --------------------------------------------------------------------------- #
# profile() -- params, error handling, structure, multi-landscape
# --------------------------------------------------------------------------- #
def test_profile_params_override_changes_value():
    ls = hoc_landscape(6, seed=3)
    base = profile(ls, include=["neutrality"])["neutrality"]
    wide = profile(ls, include=["neutrality"], params={"neutrality": {"threshold": 1e9}})
    assert wide["neutrality"] != base  # a huge threshold makes ~everything neutral


def test_profile_on_error_warn_isolates_failure():
    with pytest.warns(UserWarning):
        s = profile(
            onemax(5), include=["gamma", "fdc"],
            params={"gamma": {"n_jobs": "bad"}}, on_error="warn",
        )
    assert np.isnan(s["gamma"])
    assert np.isfinite(s["fdc"])


def test_profile_on_error_raise_propagates():
    with pytest.raises(Exception):
        profile(
            onemax(5), include=["gamma"],
            params={"gamma": {"n_jobs": "bad"}}, on_error="raise",
        )


def test_profile_bad_on_error_rejected():
    with pytest.raises(ValueError):
        profile(onemax(4), on_error="boom")


def test_profile_multiple_landscapes_dataframe():
    lss = [onemax(4), hoc_landscape(4, seed=1), nk_landscape(4, 1, seed=2)]
    df = profile(lss, index=["a", "b", "c"], include=["fdc", "gamma"], seed=0)
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["a", "b", "c"]
    assert list(df.columns) == ["fdc", "gamma"]


def test_profile_include_structure():
    ls = onemax(5)
    s = profile(ls, include=["fdc"], include_structure=True)
    assert s["structure.n_configs"] == ls.n_configs


def test_list_metrics_shape():
    lm = list_metrics()
    assert len(lm) == 22
    assert {"group", "kind", "columns", "n_jobs", "seed", "time_budget"}.issubset(lm.columns)
    assert lm.loc["classify_epistasis", "time_budget"]
    assert lm.loc["autocorrelation", "seed"]


# --------------------------------------------------------------------------- #
# classify_epistasis / extradimensional_bypass -- sample_cut_prob redesign
# --------------------------------------------------------------------------- #
def test_classify_auto_small_equals_exact():
    ls = nk_landscape(6, 2, seed=0)  # tiny -> auto resolves to exact
    assert A.classify_epistasis(ls) == A.classify_epistasis(ls, sample_cut_prob=0)


def test_classify_explicit_cut_prob_reproducible():
    ls = onemax(6)
    a = A.classify_epistasis(ls, sample_cut_prob=0.5, seed=11)
    b = A.classify_epistasis(ls, sample_cut_prob=0.5, seed=11)
    assert a == b


@pytest.mark.parametrize("bad", [1.5, -0.1, "bogus"])
def test_classify_invalid_cut_prob(bad):
    with pytest.raises(ValueError):
        A.classify_epistasis(onemax(4), sample_cut_prob=bad)


def test_bypass_auto_small_equals_exact():
    ls = nk_landscape(6, 2, seed=1)
    a = A.extradimensional_bypass(ls)                       # auto -> exact on small
    b = A.extradimensional_bypass(ls, sample_cut_prob=0)    # explicit exact
    # compare the deterministic fields (average_bypass_length may be NaN != NaN)
    assert a.bypass_proportion == b.bypass_proportion
    assert a.total_motifs == b.total_motifs
    assert a.motifs_with_bypass == b.motifs_with_bypass


# --------------------------------------------------------------------------- #
# the auto-cutoff ladder (tested on stubs so no large landscape build is needed)
# --------------------------------------------------------------------------- #
class _StubGraph:
    def __init__(self, n, e):
        self._n, self._e = n, e

    def vcount(self):
        return self._n

    def ecount(self):
        return self._e


class _StubLS:
    def __init__(self, n, e):
        self.graph = _StubGraph(n, e)


def _ls_with_cost(P):
    # n=2 => P = 2*e**2 / 2 = e**2
    return _StubLS(2, int(round(P ** 0.5)))


@pytest.mark.parametrize(
    "P,expected",
    [(1e6, 0.0), (5e6, 0.1), (1e7, 0.25), (5e7, 0.5), (2e8, 0.75)],
)
def test_auto_cut_prob_ladder(P, expected):
    ls = _ls_with_cost(P)
    assert abs(_motif_cost(ls) - P) / P < 0.01
    assert _auto_cut_prob(ls, 15.0) == expected


def test_auto_cut_prob_floor_warns_when_too_large():
    ls = _ls_with_cost(1e9)  # beyond even 0.75's 15s budget
    with pytest.warns(UserWarning):
        assert _auto_cut_prob(ls, 15.0) == 0.75


def test_auto_cut_prob_scales_with_time_budget():
    ls = _ls_with_cost(1e7)
    assert _auto_cut_prob(ls, 15.0) == 0.25  # tight budget -> more pruning
    assert _auto_cut_prob(ls, 60.0) == 0.0   # generous budget -> exact


def test_resolve_cut_prob_modes():
    ls = _ls_with_cost(1e6)  # exact band
    assert _resolve_cut_prob(ls, "auto", 15.0) is None
    assert _resolve_cut_prob(ls, 0, 15.0) is None
    assert _resolve_cut_prob(ls, None, 15.0) is None
    assert _resolve_cut_prob(ls, 0.5, 15.0) == 0.5
