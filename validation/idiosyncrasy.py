"""Offline Lyons (2020) reproduction, separate from GraphFLA's implementation.

Run ``python -m validation.idiosyncrasy`` from the repository root.
The reference enumerates sequence neighbors directly, rather than using the
metric's encoded-background grouping. No author notebook is executed.
"""

from itertools import permutations
from pathlib import Path
import hashlib
import json
import tempfile

import igraph as ig
import numpy as np
import pandas as pd

from graphfla.analysis import global_idiosyncratic_index
from graphfla.analysis.epistasis.idiosyncrasy import (
    _idiosyncratic_data,
    _idiosyncratic_position_worker,
    _idiosyncratic_ratio,
)
from graphfla.landscape import DNALandscape


FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/literature/lyons2020"


def load_trna():
    provenance = json.loads((FIXTURE / "provenance.json").read_text())
    path = FIXTURE / "trna.csv.gz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    frame = pd.read_csv(path, float_precision="round_trip")
    assert len(frame) == 28530 and frame.sequence.is_unique
    assert frame.sequence.str.len().eq(72).all()
    assert (frame.fitness > 0).all()
    return frame


def enumerate_trna(frame):
    """Direct sequence substitution oracle; also construct true Hamming-1 edges."""
    seqs = frame.sequence.tolist()
    fitness = frame.fitness.to_numpy()
    lookup = {s: i for i, s in enumerate(seqs)}
    effects, edges = {}, []
    # Authors' position labels are Python string offsets 1..69 in this input.
    for pos in range(1, 70):
        for a, b in permutations("ACGT", 2):
            values = []
            for i, seq in enumerate(seqs):
                if seq[pos] != a:
                    continue
                other = seq[:pos] + b + seq[pos + 1 :]
                j = lookup.get(other)
                if j is not None:
                    values.append(fitness[j] - fitness[i])
                    if fitness[i] < fitness[j]:
                        edges.append((i, j))
            effects[pos, a, b] = np.asarray(values)
    assert len(effects) == 828
    assert min(map(len, effects.values())) == 3
    return effects, edges


def load_full_population(frame, edges, path):
    """Use the public GraphML import to keep isolates in the reference pool.

    This is a validation adapter, not a change to the construction contract.
    The graph has genuine improving Hamming-1 edges and all viable vertices.
    """
    graph = ig.Graph(n=len(frame), edges=edges, directed=True)
    graph.vs["fitness"] = frame.fitness.tolist()
    columns = [f"site_{p}" for p in range(1, 70)]
    for p, column in enumerate(columns, 1):
        graph.vs[column] = frame.sequence.str[p].tolist()
    graph["data_types_data"] = repr(dict.fromkeys(columns, "categorical"))
    graph["maximize"] = True
    graph["epsilon"] = "0"
    graph.write_graphml(str(path))
    landscape = DNALandscape.build_from_graph(str(path), verbose=False)
    assert landscape.n_configs == len(frame)
    np.testing.assert_allclose(
        landscape.graph.vs["fitness"], frame.fitness, rtol=1e-14, atol=1e-15
    )
    return landscape


def reference_ratios(effects, pool, seed=None, author_seeds=False):
    """Independent literal implementation of the paper's matched-size control."""
    rng = np.random.RandomState(seed)
    values = []
    for effect in effects.values():
        n = len(effect)
        if author_seeds:
            rng = np.random.RandomState(n**2 + 3)
        pair_indices = rng.choice(len(pool), size=(n, 2), replace=True)
        null = [pool[j] - pool[i] for i, j in pair_indices]
        values.append(float(np.std(effect) / np.std(null)))
    return np.asarray(values)


def reproduce():
    frame = load_trna()
    effects, edges = enumerate_trna(frame)
    pool = frame.fitness.to_numpy()
    author = reference_ratios(effects, pool, author_seeds=True)
    example = effects[10, "G", "A"]
    example_pairs = np.random.RandomState(4033).choice(pool, (len(example), 2))
    example_control = example_pairs[:, 1] - example_pairs[:, 0]
    global_example_pairs = np.random.RandomState(len(example) ** 2 + 3).choice(
        pool, (len(example), 2)
    )
    global_example_control = global_example_pairs[:, 1] - global_example_pairs[:, 0]
    oracle_seed0 = float(reference_ratios(effects, pool, seed=0).mean())
    with tempfile.TemporaryDirectory() as tmp:
        landscape = load_full_population(frame, edges, Path(tmp) / "trna.graphml")
        _, codes, f, _ = _idiosyncratic_data(landscape)
        summaries = [
            row
            for j in range(codes.shape[1])
            for row in _idiosyncratic_position_worker(codes, f, j)
        ]
        kernel_author = np.asarray(
            [
                _idiosyncratic_ratio(sd, n, f, np.random.RandomState(n**2 + 3))
                for _, _, sd, n in summaries
            ]
        )
        np.testing.assert_allclose(kernel_author, author, rtol=1e-12, atol=1e-12)
        actual = global_idiosyncratic_index(landscape, n_jobs=1, seed=0)
        parallel = global_idiosyncratic_index(landscape, n_jobs=2, seed=0)
        assert actual == parallel
        np.testing.assert_allclose(actual, oracle_seed0, rtol=1e-12, atol=1e-12)
        isolates = int(sum(d == 0 for d in landscape.graph.degree()))
    return {
        "paper_targets": {"mean": 0.612, "sem": 0.005, "example": 0.49},
        "population": {
            "viable_genotypes": len(frame),
            "isolates": isolates,
            "directed_mutations": len(effects),
        },
        "independent_author_procedure": {
            "mean": float(author.mean()),
            "sem": float(author.std() / np.sqrt(len(author))),
            "example_n": len(example),
            "example_effect_sd": float(example.std()),
            "example_control_sd": float(example_control.std()),
            "example_index": float(example.std() / example_control.std()),
            "example_index_under_global_notebook_policy": float(
                example.std() / global_example_control.std()
            ),
        },
        "production_kernel_with_author_seeds": {
            "mean": float(kernel_author.mean()),
            "max_abs_per_mutation_error": float(np.max(np.abs(kernel_author - author))),
        },
        "public_global_seed0": {
            "independent_oracle": oracle_seed0,
            "serial": actual,
            "parallel": parallel,
            "note": "Different random stream from the paper; compare to the same-stream oracle.",
        },
        "old_analytic_full_population": float(
            np.mean([v.std() / (np.sqrt(2) * pool.std()) for v in effects.values()])
        ),
    }


if __name__ == "__main__":
    print(json.dumps(reproduce(), indent=2))
