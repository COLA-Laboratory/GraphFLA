"""Bounded, deterministic analysis-only inputs (construction is untimed)."""

from itertools import product

import numpy as np
import pandas as pd

from graphfla.landscape import BooleanLandscape, Landscape, OrdinalLandscape


EE_CASES = [
    "boolean-6",
    "boolean-10",
    "categorical-4x5",
    "categorical-8x3",
    "sparse-boolean-12",
    "ordinal-8x3",
]
TREND_CASES = [
    "boolean-6", "boolean-14", "categorical-4x5", "categorical-64x2",
    "sparse-boolean-12", "ordinal-8x3",
]
SMALL_CASES = ["boolean-6", "categorical-3x3"]
IDIOSYNCRASY_CASES = [
    "boolean-6",
    "boolean-10",
    "categorical-4x5",
    "categorical-16x2",
    "sparse-boolean-12",
    "long-boolean-72",
]
GAMMA_CASES = [
    "boolean-6",
    "boolean-10",
    "categorical-4x5",
    "categorical-16x2",
    "sparse-boolean-12",
    "long-boolean-72",
]


RS_CASES = [
    "boolean-6", "boolean-14", "categorical-4x5", "categorical-64x2",
    "sparse-boolean-12", "long-boolean-72", "ordinal-8x3", "mixed-4x3",
]

WALSH_CASES = [
    "boolean-6", "boolean-10", "categorical-3x3", "categorical-4x4",
    "sparse-boolean-12", "ordinal-4x3", "mixed-4x3",
]


def cases_for_metric(metric):
    if metric in {"diminishing_returns_index", "increasing_costs_index"}:
        return TREND_CASES
    if metric == "ee":
        return EE_CASES
    if metric in {"idiosyncratic_index", "global_idiosyncratic_index"}:
        return IDIOSYNCRASY_CASES
    if metric in {"gamma", "gamma_star"}:
        return GAMMA_CASES
    if metric == "r_s_ratio":
        return RS_CASES
    if metric in {"walsh_hadamard", "higher_order_epistasis"}:
        return WALSH_CASES
    return SMALL_CASES


def build_case(name):
    rng = np.random.RandomState(23)
    if name == "long-boolean-72":
        # 70 adjacent backgrounds x 8 focal configurations, not 2**72 inputs.
        backgrounds = np.arange(69)[None, :] < np.arange(70)[:, None]
        focal = np.asarray(list(product([0, 1], repeat=3)))
        X = np.column_stack(
            [np.tile(focal, (70, 1)), np.repeat(backgrounds, 8, axis=0)]
        )
        cls, kwargs = BooleanLandscape, {}
    elif name.startswith("sparse-"):
        X = np.asarray(list(product([0, 1], repeat=12)))
        X = X[np.sort(rng.choice(len(X), 1024, replace=False))]
        cls, kwargs = BooleanLandscape, {}
    elif name.startswith("boolean-"):
        n = int(name.split("-")[1])
        X = np.asarray(list(product([0, 1], repeat=n)))
        cls, kwargs = BooleanLandscape, {}
    else:
        kind, dimensions = name.split("-")
        arity, n = map(int, dimensions.split("x"))
        X = np.asarray(list(product(range(arity), repeat=n)))
        cls = OrdinalLandscape if kind == "ordinal" else Landscape
        kwargs = (
            {}
            if kind == "ordinal"
            else {"data_types": {str(i): "categorical" for i in range(n)}}
        )
    if name == "mixed-4x3":
        X[:, -1] %= 2
        X = np.unique(X, axis=0)
        kwargs = {"data_types": {"mode": "categorical", "level": "ordinal", "on": "boolean"}}
    # Interaction and nonmonotonic terms prevent an all-additive timing fixture.
    f = X.sum(axis=1) + 2 * X[:, 0] * X[:, -1] + rng.normal(size=len(X))
    if kwargs:
        X = pd.DataFrame(X, columns=list(kwargs["data_types"]))
    return cls().build_from_data(
        X, f, verbose=False, neighborhood_strategy="active", **kwargs
    )
