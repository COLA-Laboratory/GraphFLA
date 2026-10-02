"""Offline Lyons (2020) reproduction, separate from GraphFLA's implementation.

Run ``python -m validation.idiosyncrasy`` from the repository root.
The reference enumerates sequence neighbors directly, rather than using the
metric's encoded-background grouping. No author notebook is executed.
"""

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
from validation.oracles.idiosyncrasy import sequence_effects, control_ratios


FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/literature/lyons2020"


def load_trna(path=None):
    provenance = json.loads((FIXTURE / "provenance.json").read_text())
    path = FIXTURE / "trna.csv.gz" if path is None else Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    frame = pd.read_csv(path, float_precision="round_trip")
    return frame


def enumerate_trna(frame):
    """Direct sequence substitution oracle; also construct true Hamming-1 edges."""
    return sequence_effects(
        frame.sequence.tolist(), frame.fitness.to_numpy(), range(1, 70)
    )


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
    return landscape


def reference_ratios(effects, pool, seed=None, author_seeds=False):
    """Independent literal implementation of the paper's matched-size control."""
    return np.asarray(
        list(
            control_ratios(effects, pool, seed=seed, author_seeds=author_seeds).values()
        )
    )


def reproduce(path=None, *, details=False):
    frame = load_trna(path)
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
        _, codes, f, labels = _idiosyncratic_data(landscape)
        summaries = [
            (j + 1, *row)
            for j in range(codes.shape[1])
            for row in _idiosyncratic_position_worker(codes, f, j)
        ]
        keys = [
            (pos, labels[pos - 1][a], labels[pos - 1][b])
            for pos, a, b, _, _ in summaries
        ]
        counts = np.asarray([n for _, _, _, _, n in summaries])
        effect_sds = np.asarray([sd for _, _, _, sd, _ in summaries])
        kernel_author = np.asarray(
            [
                _idiosyncratic_ratio(sd, n, f, np.random.RandomState(n**2 + 3))
                for _, _, _, sd, n in summaries
            ]
        )
        actual = global_idiosyncratic_index(landscape, n_jobs=1, seed=0)
        parallel = global_idiosyncratic_index(landscape, n_jobs=2, seed=0)
        isolates = int(sum(d == 0 for d in landscape.graph.degree()))
    report = {
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
            "sem": float(kernel_author.std() / np.sqrt(len(kernel_author))),
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

    if details:
        return {
            "report": report,
            "frame": frame,
            "loaded_fitness": f,
            "effect_keys": list(effects),
            "reference_counts": np.asarray([len(v) for v in effects.values()]),
            "reference_sds": np.asarray([np.std(v) for v in effects.values()]),
            "reference_ratios": author,
            "production_keys": keys,
            "production_counts": counts,
            "production_sds": effect_sds,
            "production_ratios": kernel_author,
        }
    return report


if __name__ == "__main__":
    print(json.dumps(reproduce(), indent=2))
