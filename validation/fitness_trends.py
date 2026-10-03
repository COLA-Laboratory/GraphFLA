"""Study adapters and independent fits for returns/costs validation.

No import-time data loading. The adapters do not infer missing measurements,
reconstruct a full genotype space or assert that per-mutation fits are DRI/ICI.
"""

from types import SimpleNamespace

import igraph as ig
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


def johnson_input(path):
    frame = pd.read_csv(path, float_precision="round_trip")
    return [
        (edge, group)
        for edge, group in frame.groupby("Edge", sort=True)
        if len(group) >= 50
    ]


def papkou_input(fitness_path, edges_path, *, clip):
    frame = pd.read_csv(fitness_path, float_precision="round_trip")
    edges = pd.read_csv(edges_path, sep=" ", header=None, names=["source", "target"])
    names = pd.Index(pd.unique(edges.to_numpy().ravel()))
    fitness = frame.set_index("sequence").fitness.reindex(names).to_numpy()
    if not np.isfinite(fitness).all():
        raise ValueError("Author graph contains a vertex without measured fitness")
    if clip:
        fitness = np.maximum(fitness, -0.507774)
    source, target = names.get_indexer(edges.source), names.get_indexer(edges.target)
    graph = ig.Graph(
        n=len(names), edges=np.column_stack([source, target]), directed=True
    )
    graph.vs["fitness"] = fitness
    graph.vs["name"] = names.tolist()
    view = SimpleNamespace(graph=graph, maximize=True, _check_built=lambda: None)
    return view, fitness[source], fitness[target]


def independent_trends(source, target):
    gain = target - source
    return {
        f"{kind}_{method}": float(
            linregress(x, gain).slope
            if method == "regression"
            else (pearsonr if method == "pearson" else spearmanr)(x, gain).statistic
        )
        for kind, x in [("returns", source), ("costs", target)]
        for method in ["pearson", "spearman", "regression"]
    }
