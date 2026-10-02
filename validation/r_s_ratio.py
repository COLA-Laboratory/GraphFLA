"""Offline r/s reproductions; published numbers and coding checks stay separate."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from graphfla import analysis
from graphfla.landscape import BooleanLandscape
from validation.gamma import load_input
from validation.oracles.roughness import additive_reference, rna_design
from validation.testing import input_paths


def boolean_input(path, name):
    if name == "chou":
        frame = pd.read_csv(path, float_precision="round_trip")
        X = frame[[f"pos{i}" for i in range(1, 5)]].to_numpy()
        y = frame.fitness.to_numpy()
    else:
        X, y = load_input(path)
        X = np.asarray(X)
    landscape = BooleanLandscape().build_from_data(X, y, verbose=False)
    return X, y, landscape, additive_reference(X, y)


def kuo_input(path):
    frame = pd.read_csv(path, float_precision="round_trip")
    sequences = frame.sequences.to_numpy()
    y = frame.fitness.to_numpy()
    results = {}
    for reference in ("A", "U"):
        design = rna_design(sequences, reference)
        independent = additive_reference(design, y)
        # The statistic uses observations, not edges. This table view preserves
        # all 197,890 rows without building millions of unused graph edges.
        # Recode labels explicitly so the public drop-first convention selects
        # the specified reference. This is not an RNA-specific package default.
        order = reference + "".join(s for s in "ACGU" if s != reference)
        codes = {state: str(i) for i, state in enumerate(order)}
        data = pd.DataFrame([list(s) for s in sequences]).replace(codes)
        kinds = dict.fromkeys(data.columns, "categorical")
        data["fitness"] = y
        view = SimpleNamespace(get_data=lambda: data, data_types=kinds)
        actual = analysis.r_s_ratio(view)
        results[reference] = {"independent": independent, "public": actual}
        if reference == "U":
            # Algebraic replay of Generate_raw_data.ipynb, not execution of
            # downloaded code. Its empirical estimator uses Ridge(alpha=1).
            ridge = Ridge(alpha=1).fit(design, y)
            residual = y - ridge.intercept_ - np.einsum("ij,j->i", design, ridge.coef_)
            results[reference]["author_ridge"] = float(
                np.sqrt(np.mean(residual**2)) / np.mean(abs(ridge.coef_))
            )
    return frame, results


def main():
    report = {}
    for name, case_id in [
        ("chou", "szendro.chou.rs.v1"),
        ("csi", "ferretti.csi.rs.v1"),
    ]:
        X, _, landscape, ref = boolean_input(input_paths(case_id)[0], name)
        report[name] = {
            "rows": len(X),
            "ratio": analysis.r_s_ratio(landscape),
            "independent": ref["ratio"],
        }
    frame, results = kuo_input(input_paths("song.kuo.rs.ols.v1")[0])
    report["kuo"] = {
        "rows": len(frame),
        "references": {
            key: {
                "ratio": value["public"],
                "independent": value["independent"]["ratio"],
                "author_ridge": value.get("author_ridge"),
            }
            for key, value in results.items()
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
