"""Offline gamma reproduction on three pinned 32-genotype empirical tables.

Run ``python -m validation.gamma``. Printed results and independently evaluated
equations are deliberately distinct. See GAMMA_REVIEW.md for input limitations.
"""

import json
import numpy as np
import pandas as pd

from graphfla import analysis
from graphfla.landscape import BooleanLandscape
from validation.oracles.gamma import directed_gamma
from validation.testing import input_paths


CASES = {
    "csi": "ferretti.csi.equations.v1",
    "csi2008": "ferretti.csi2008.equations.v1",
    "tem": "ferretti.tem.equations.v1",
}


def load_input(path):
    frame = pd.read_csv(path, dtype={"genotype": str}, float_precision="round_trip")
    if "genotype" in frame:
        variants = [tuple(map(int, g)) for g in frame.genotype]
        fitness = np.log(frame.W.to_numpy())
    else:
        variants = list(map(tuple, frame.iloc[:, :5].to_numpy(dtype=int)))
        fitness = np.log(frame.MIC.to_numpy())
    return variants, fitness


def reproduce(path):
    variants, fitness = load_input(path)
    reference = {
        name: directed_gamma(variants, fitness, signs=signs)
        for name, signs in [("gamma", False), ("gamma_star", True)]
    }
    landscape = BooleanLandscape().build_from_data(variants, fitness, verbose=False)
    actual = {name: getattr(analysis, name)(landscape, n_jobs=1) for name in reference}
    return variants, fitness, landscape, reference, actual


def main():
    report = {}
    for name, case_id in CASES.items():
        (path,) = input_paths(case_id)
        variants, _, landscape, reference, actual = reproduce(path)
        printed = {"gamma": 0.85, "gamma_star": 0.59} if name == "tem" else {
            "gamma": 0.33, "gamma_star": 0.25
        }
        report[name] = {
            "rows": len(variants), "retained_rows": landscape.n_configs,
            "graphfla": actual,
            "independent": {k: {kk: vv for kk, vv in v.items()
                                 if kk != "by_position_pair"} for k, v in reference.items()},
            "figure4c": printed,
            "matches_printed_precision": {k: abs(actual[k]-v) <= 0.005
                                           for k, v in printed.items()},
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
