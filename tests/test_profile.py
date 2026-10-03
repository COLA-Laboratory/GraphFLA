"""Profile selection, dictionaries, and automatic motif sampling."""

import inspect
import importlib

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


def test_profile_defaults_cover_the_documented_portfolio():
    result = profile(nk_landscape(6, 2, seed=0), seed=0, progress=False)
    table = list_metrics()
    expected = [c.strip() for names in table["columns"] for c in names.split(",")]
    assert isinstance(result, pd.Series) and result.dtype == float
    assert result.index.tolist() == expected
    assert len(table) == 21
    # Keep the self-contained choices in the docstring in sync with the registry.
    doc = inspect.getdoc(profile)
    assert all(f"``{name}``" in doc for name in table.index)
    assert all(f'``"{group}"``' in doc for group in table.group.unique())


def test_profile_selects_mixed_groups_and_metrics_once_in_input_order():
    result = profile(onemax(4), metrics=["fdc", "ruggedness", "fdc"], seed=0)
    assert result.index.tolist() == [
        "fdc",
        "local_optima_ratio",
        "gradient_intensity",
        "autocorrelation",
        "r_s_ratio",
    ]
    assert profile(onemax(3), metrics="fdc").index.tolist() == ["fdc"]
    assert profile(onemax(3), metrics=[]).empty


def test_profile_group_expands_dictionary_output_fields():
    result = profile(onemax(4), metrics="epistasis", seed=0)
    table = list_metrics().query("group == 'epistasis'")
    expected = [c.strip() for names in table["columns"] for c in names.split(",")]
    assert result.index.tolist() == expected
    assert result["epistasis.magnitude"] == 1.0


@pytest.mark.parametrize(
    "name",
    [
        "nope",
        "epistasis.magnitude",
        "walsh_hadamard",
        "evolvability_enhancing_mutations",
    ],
)
def test_profile_unknown_choices_are_rejected(name):
    with pytest.raises(ValueError, match="Unknown metric or group"):
        profile(onemax(3), metrics=name)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include": ["fdc"]},
        {"exclude": ["fdc"]},
        {"groups": "epistasis"},
        {"index": ["name"]},
        {"include_structure": True},
        {"time_budget": 2},
        {"on_error": "raise"},
    ],
)
def test_profile_has_no_obsolete_keyword_entry_points(kwargs):
    with pytest.raises(TypeError, match="unexpected keyword"):
        profile(onemax(3), **kwargs)


def test_profile_metric_params_override_values_and_shared_controls():
    landscape = hoc_landscape(5, seed=3)
    result = profile(
        landscape,
        metrics=["neutrality", "autocorrelation"],
        seed=1,
        params={
            "neutrality": {"threshold": 1e9},
            "autocorrelation": {"seed": 7, "walk_times": 10},
        },
    )
    assert result["neutrality"] == 1.0
    assert result["autocorrelation"] == A.autocorrelation(
        landscape, seed=7, walk_times=10
    )
    exact = profile(
        landscape,
        metrics="classify_epistasis",
        params={"classify_epistasis": {"sample_cut_prob": 0}},
    )
    assert (
        exact["epistasis.sign"]
        == A.classify_epistasis(landscape, sample_cut_prob=0)["sign"]
    )


@pytest.mark.parametrize(
    "params,error",
    [
        ([], TypeError),
        ({"fdc": []}, TypeError),
        ({"missing": {}}, ValueError),
        ({"evolvability_enhancing_mutations": {}}, ValueError),
        ({"evolvability_enhancing_fraction": {"epsilon": 0}}, ValueError),
        ({"fdc": {"typo": 2}}, ValueError),
    ],
)
def test_profile_settings_fail_early(params, error):
    with pytest.raises(error):
        profile(onemax(3), metrics="fdc", params=params)


def test_profile_warns_and_continues_after_a_metric_failure():
    with pytest.warns(UserWarning, match="metric 'gamma' failed"):
        result = profile(
            onemax(4), metrics=["gamma", "fdc"], params={"gamma": {"n_jobs": "bad"}}
        )
    assert np.isnan(result["gamma"]) and np.isfinite(result["fdc"])


def test_profile_multiple_and_empty_landscapes_have_stable_columns():
    landscapes = [onemax(3), hoc_landscape(3, seed=1)]
    result = profile(landscapes, metrics=["fdc", "gamma"], seed=0)
    assert isinstance(result, pd.DataFrame)
    assert result.index.tolist() == [0, 1]
    assert result.columns.tolist() == ["fdc", "gamma"]
    pd.testing.assert_series_equal(
        result.iloc[0],
        profile(landscapes[0], metrics=["fdc", "gamma"], seed=0),
        check_names=False,
    )
    empty = profile([], metrics=["fdc", "gamma"])
    assert empty.empty and empty.columns.tolist() == ["fdc", "gamma"]


def test_progress_does_not_change_results_or_landscape_verbosity(capsys):
    landscape = onemax(3)
    original_verbose = landscape.verbose
    quiet = profile(landscape, metrics=["fdc", "gamma"], progress=False)
    shown = profile(landscape, metrics=["fdc", "gamma"], progress=True)
    pd.testing.assert_series_equal(quiet, shown)
    assert landscape.verbose == original_verbose
    assert capsys.readouterr().out == ""


def test_epistasis_results_are_plain_dictionaries():
    landscape = onemax(3)
    classification = A.classify_epistasis(landscape, sample_cut_prob=0)
    assert type(classification) is dict
    assert set(classification) == {
        "magnitude",
        "sign",
        "reciprocal_sign",
        "positive",
        "negative",
    }
    assert all(type(value) is float for value in classification.values())
    bypass = A.extradimensional_bypass(landscape, sample_cut_prob=0)
    assert type(bypass) is dict
    assert set(bypass) == {
        "bypass_proportion",
        "average_bypass_length",
        "total_motifs",
        "motifs_with_bypass",
    }
    assert type(bypass["total_motifs"]) is int
    assert np.isnan(bypass["average_bypass_length"])


def test_removed_results_and_ee_wrapper_are_absent():
    for module_name, names in [
        (
            "graphfla.analysis",
            [
                "EpistasisClassification",
                "ExtradimensionalBypass",
                "evolvability_enhancing_mutations",
                "higher_order_epistasis",
            ],
        ),
        (
            "graphfla.analysis.epistasis",
            ["EpistasisClassification", "ExtradimensionalBypass"],
        ),
        (
            "graphfla.analysis.epistasis.motifs",
            ["EpistasisClassification", "ExtradimensionalBypass"],
        ),
        ("graphfla.analysis.robustness", ["evolvability_enhancing_mutations"]),
        ("graphfla.algorithms", ["WalkResult"]),
        ("graphfla.algorithms.walk", ["WalkResult"]),
    ]:
        module = importlib.import_module(module_name)
        assert all(not hasattr(module, name) for name in names)


def test_profile_ee_uses_current_parameters():
    landscape = hoc_landscape(4, seed=2)
    name = "evolvability_enhancing_fraction"
    options = {"fdr": 0.9, "effect_type": "beneficial"}
    result = profile(landscape, metrics=name, params={name: options})
    assert result[name] == A.evolvability_enhancing_fraction(landscape, **options)


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
    a = A.extradimensional_bypass(ls)  # auto -> exact on small
    b = A.extradimensional_bypass(ls, sample_cut_prob=0)  # explicit exact
    # compare the deterministic fields (average_bypass_length may be NaN != NaN)
    assert a["bypass_proportion"] == b["bypass_proportion"]
    assert a["total_motifs"] == b["total_motifs"]
    assert a["motifs_with_bypass"] == b["motifs_with_bypass"]


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
    return _StubLS(2, int(round(P**0.5)))


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
    assert _auto_cut_prob(ls, 60.0) == 0.0  # generous budget -> exact


def test_resolve_cut_prob_modes():
    ls = _ls_with_cost(1e6)  # exact band
    assert _resolve_cut_prob(ls, "auto", 15.0) is None
    assert _resolve_cut_prob(ls, 0, 15.0) is None
    assert _resolve_cut_prob(ls, None, 15.0) is None
    assert _resolve_cut_prob(ls, 0.5, 15.0) == 0.5
