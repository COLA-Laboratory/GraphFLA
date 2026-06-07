#!/usr/bin/env python
"""Behaviour-identity verification for the construction-optimization campaign.

Builds each of the 8 benchmark landscapes and emits a canonical signature
(graph edges+delta, per-vertex fitness, local optima, global optimum, plateaus,
and PageRank) so the FINAL branch can be diffed against ORIGINAL main and proven
to construct IDENTICAL landscapes. Self-contained (no bench package import) so it
can be copied out and run after `git checkout main`.

  python bench/verify_identical.py --out /tmp/sig_X.json
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

MAIN = "/Users/arwen/Documents/GitHub/GraphFLA"
BIO = os.path.join(MAIN, "data", "BioSequence")
WREOS = os.path.join(MAIN, "data", "Materials", "WReOs", "simplex.csv")
HPO = os.path.join(MAIN, "bench", "_localdata", "HPO_44136.csv")

DATASETS = [
    ("GB1", "protein", os.path.join(BIO, "Wu2016_GB1.csv"), ["sequences"], "fitness"),
    ("TrpB3I", "protein", os.path.join(BIO, "Johnston2024_TrpB3I.csv"), ["sequences"], "fitness"),
    ("Papkou", "dna", os.path.join(BIO, "Papkou2023_DHFR_RAW.csv"), ["seq"], "fitness"),
    ("Westmann", "dna", os.path.join(BIO, "Westmann2024.csv"), ["sequences"], "fitness"),
    ("CR9114", "boolean", os.path.join(BIO, "Phillips2021_CR9114_h1.csv"), ["sequences"], "fitness"),
    ("CR6261", "boolean", os.path.join(BIO, "Phillips2021_CR6261_h1.csv"), ["sequences"], "fitness"),
    ("WReOs", "ordinal", WREOS, ["W", "Re"], "TOPSIS_Ci"),
    ("HPO", "ordinal", HPO, ["learning_rate", "subsample", "max_depth",
                             "min_child_weight", "n_estimators"], "mean_r2"),
]


def _h(arr):
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def load(ds):
    name, typ, path, xcols, fcol = ds
    if typ == "ordinal":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, dtype={xcols[0]: str})
    df = df.dropna(subset=[fcol] + xcols).reset_index(drop=True)
    X = df[xcols] if typ == "ordinal" else df[xcols[0]]
    return X, df[fcol]


def cls_for(typ):
    from graphfla.landscape.protein import ProteinLandscape
    from graphfla.landscape.dna import DNALandscape
    from graphfla.landscape.boolean import BooleanLandscape
    from graphfla.landscape.ordinal import OrdinalLandscape
    return {"protein": ProteinLandscape, "dna": DNALandscape,
            "boolean": BooleanLandscape, "ordinal": OrdinalLandscape}[typ]


def signature(ls):
    g = ls.graph
    n_e = g.ecount()
    if n_e:
        el = np.asarray(g.get_edgelist(), dtype=np.int64)
        delta = np.asarray(g.es["delta_fit"], dtype=np.float64)
        order = np.lexsort((el[:, 1], el[:, 0]))
        edge_hash = _h(el[order])
        delta_hash = _h(np.round(delta[order], 9))
    else:
        edge_hash = delta_hash = "empty"
    fit = np.round(np.asarray(g.vs["fitness"], dtype=np.float64), 9)
    # local optima (sorted), global optimum
    lo = ls.lo_index
    lo_arr = np.sort(np.asarray(lo, dtype=np.int64)) if lo is not None else np.zeros(0, np.int64)
    # plateaus: canonical = sorted tuple of sorted member tuples (label-independent)
    plat = ls.plateaus or {}
    members = sorted(tuple(sorted(int(x) for x in v)) for v in plat.values())
    # pagerank via get_data (works on eager-main AND lazy-final), rounded
    try:
        gd = ls.get_data()
        pr = gd["pagerank"].to_numpy(dtype=np.float64) if "pagerank" in gd.columns else None
        pr_hash = _h(np.round(pr, 7)) if pr is not None else "none"
        gd_cols = sorted(map(str, gd.columns))
    except Exception as e:
        pr_hash = f"ERR:{e}"
        gd_cols = []
    return dict(
        vcount=g.vcount(), ecount=n_e,
        edge_hash=edge_hash, delta_hash=delta_hash, fit_hash=_h(fit),
        n_lo=ls.n_lo, lo_hash=_h(lo_arr), go_index=ls.go_index,
        n_plateau=ls.n_plateau,
        plat_hash=hashlib.sha1(repr(members).encode()).hexdigest()[:16],
        pagerank_hash=pr_hash, get_data_cols=gd_cols,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import graphfla
    sigs = {"graphfla_file": graphfla.__file__, "datasets": {}}
    for ds in DATASETS:
        name, typ = ds[0], ds[1]
        try:
            X, f = load(ds)
            ls = cls_for(typ)(maximize=True).build_from_data(X, f, verbose=False)
            sigs["datasets"][name] = signature(ls)
            print(f"  {name}: vcount={sigs['datasets'][name]['vcount']} "
                  f"ecount={sigs['datasets'][name]['ecount']} OK", flush=True)
        except Exception as e:
            sigs["datasets"][name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {name}: ERROR {e}", flush=True)
    with open(args.out, "w") as fh:
        json.dump(sigs, fh, indent=2)
    print("WROTE " + args.out)


if __name__ == "__main__":
    main()
