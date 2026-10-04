"""Reproduce the five homepage examples in a scientific Python environment.

This is an explicit data-preparation step, never part of a documentation build.
All models and sampling use seed 42; the HPO grid is single-threaded and bounded.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "examples"
DATA = ROOT / "tutorials/datasets/data"


def prepare():
    import igraph as ig
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import ParameterGrid, train_test_split
    from graphfla import analysis
    from graphfla.landscape import Landscape, ProteinLandscape

    OUT.mkdir(exist_ok=True)
    protein_path = ROOT / "data/BioSequence/Wu2016_GB1.csv"
    protein = pd.read_csv(protein_path)
    protein = protein.loc[protein.sequences.str.fullmatch("[CV][DE][AG][AV]"), ["sequences", "fitness"]]
    assert len(protein) == 16
    chemistry = pd.read_csv(DATA / "suzuki.csv", keep_default_na=False)
    materials = pd.read_csv(DATA / "wreos.csv")
    software = pd.read_csv(DATA / "llvm.csv")
    flags = list(software.columns[:-1])
    software = software.loc[(software[flags[6:]] == 0).all(axis=1)]
    assert len(software) == 64

    X, y = load_diabetes(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)
    rows = []
    grid = {"max_depth": [2, 4, 6, 10], "min_samples_leaf": [1, 2, 4, 8], "max_features": [0.5, 0.75, 1.0]}
    for params in ParameterGrid(grid):
        model = RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=1, **params)
        model.fit(train_X, train_y)
        rows.append({**params, "rmse": float(np.sqrt(np.mean((model.predict(test_X) - test_y) ** 2)))})
    hpo = pd.DataFrame(rows)

    definitions = [
        ("protein", protein, ["sequences"], "fitness", True, None, protein_path),
        ("chemistry", chemistry, ["ligand", "base", "solvent"], "response_uv_pct", True, "categorical", DATA / "suzuki.csv"),
        ("materials", materials, ["W", "Re"], "H1000_HV", True, "ordinal", DATA / "wreos.csv"),
        ("software", software, flags[:6], "compile_time_raw", False, "boolean", DATA / "llvm.csv"),
        ("hpo", hpo, ["max_depth", "min_samples_leaf", "max_features"], "rmse", False, "ordinal", None),
    ]
    report = {"seed": 42, "python": platform.python_version(), "sklearn": sklearn.__version__,
              "hpo": {"dataset": "sklearn.datasets.load_diabetes", "training_rows": len(train_y),
                      "validation_rows": len(test_y), "n_estimators": 40, "grid": grid}, "scenarios": {}}
    for key, frame, columns, outcome, maximize, kind, source in definitions:
        table = frame[columns + (["Os"] if key == "materials" else []) + [outcome]].reset_index(drop=True)
        table.to_csv(OUT / (key + ".csv"), index=False)
        if key == "protein":
            landscape = ProteinLandscape().build_from_data(frame.sequences, frame[outcome], verbose=False)
        else:
            landscape = Landscape(maximize=maximize).build_from_data(
                frame[columns], frame[outcome], data_types={c: kind for c in columns}, verbose=False)
        g = landscape.graph
        assert g.vcount() == len(table), (key, g.vcount(), len(table))
        values = np.asarray(g.vs["fitness"], dtype=float)
        span = float(np.ptp(values))
        threshold = span * 0.01
        neutral = analysis.neutrality(landscape, threshold=threshold)
        roughness = (analysis.r_s_ratio(landscape) if key == "protein" else
                     analysis.autocorrelation(landscape, walk_length=20, walk_times=200, seed=42))
        interactions = analysis.classify_epistasis(landscape, sample_cut_prob=0, seed=42)
        # Validate displayed neutrality against an independent pair calculation.
        pairs = {tuple(sorted(e)) for e in g.get_edgelist()}
        for a, adjacent in (getattr(landscape, "_neutral_neighbors", None) or {}).items():
            pairs.update(tuple(sorted((a, b))) for b in adjacent)
        oracle = sum(abs(values[a] - values[b]) <= threshold for a, b in pairs) / len(pairs)
        assert abs(neutral - oracle) < 1e-12

        # Draw a connected, representative induced subgraph, never invented links.
        undirected = ig.Graph(n=g.vcount(), edges=sorted(pairs), directed=False)
        best = int(np.argmax(values) if maximize else np.argmin(values))
        chosen = sorted(undirected.bfs(best)[0][:32])
        small = undirected.induced_subgraph(chosen)
        initial = [[math.cos(2 * math.pi * i / len(chosen)), math.sin(2 * math.pi * i / len(chosen))]
                   for i in range(len(chosen))]
        ig.set_random_number_generator(random.Random(42))
        coords = np.asarray(small.layout_fruchterman_reingold(seed=initial, niter=600).coords)
        ig.set_random_number_generator(None)
        coords -= coords.mean(axis=0)
        coords = coords / max(np.linalg.norm(coords, axis=1)) * 0.42 + 0.5
        quality = ((values - values.min()) / span if maximize else (values.max() - values) / span)
        nodes = [{"id": int(i), "x": float(pos[0]), "y": float(pos[1]),
                  "value": float(values[i]), "quality": float(quality[i]), "best": i == best}
                 for i, pos in zip(chosen, coords)]
        preview = list(dict.fromkeys([best] + chosen))[:7]
        records = table.iloc[preview].to_dict(orient="records")
        report["scenarios"][key] = {
            "columns": columns, "outcome": outcome, "maximize": maximize,
            "count": len(table), "edges": len(pairs), "local_optima": int(landscape.n_lo),
            "roughness": float(roughness), "neutrality": float(neutral), "threshold": threshold,
            "interactions": {k: float(interactions[k]) for k in ("magnitude", "sign", "reciprocal_sign")},
            "nodes": nodes, "edges_shown": small.get_edgelist(), "preview": records,
            "source": str(source.relative_to(ROOT)) if source else "generated random-forest validation grid",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
            "csv_sha256": hashlib.sha256((OUT / (key + ".csv")).read_bytes()).hexdigest(),
        }
        print(key, {k: report["scenarios"][key][k] for k in ("count", "local_optima", "roughness", "neutrality")}, flush=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Example preparation exceeded 60 seconds")))
    signal.alarm(60)
    prepare()
