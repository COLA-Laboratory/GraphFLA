"""Prepare the empirical protein-landscape evolution benchmark.

Run sequentially with ``.venv-ci39/bin/python`` from the repository root.
The homepage data is a compact summary; seed-level runs and provenance stay in
this local directory. No landscape rows are imputed or removed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import tempfile
import time
import warnings

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np

ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "graphfla").is_dir() and (parent / "data/BioSequence").is_dir()
)
LOCAL = ROOT / ".codex-local/insights/evolution"
PUBLIC_DIR = ROOT / "docs/home/insights/evolution"
PUBLIC = PUBLIC_DIR / "data.json"
SEED_BASE = 20261004
TOTAL_BUDGET = 480
INITIAL_BATCH = 96
ROUND_BATCH = 96
ROUNDS = 4
N_REPEATS = 10
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

PAPER_PHOQ = {
    "title": "Pervasive degeneracy and epistasis in a protein-protein interface",
    "journal": "Science",
    "year": 2015,
    "doi": "10.1126/science.1257360",
    "url": "https://doi.org/10.1126/science.1257360",
}
PAPER_GB1 = {
    "title": "Adaptation in protein fitness landscapes is facilitated by indirect paths",
    "journal": "eLife",
    "year": 2016,
    "doi": "10.7554/eLife.16965",
    "url": "https://doi.org/10.7554/eLife.16965",
}
PAPER_JALAL = {
    "title": "Diversification of DNA-Binding Specificity by Permissive and Specificity-Switching Mutations in the ParB/Noc Protein Family",
    "journal": "Cell Reports",
    "year": 2020,
    "doi": "10.1016/j.celrep.2020.107928",
    "url": "https://doi.org/10.1016/j.celrep.2020.107928",
}
PAPER_TU = {
    "title": "An ultra-high-throughput method for measuring biomolecular activities",
    "journal": "bioRxiv",
    "year": 2024,
    "doi": "10.1101/2022.03.09.483646",
    "url": "https://doi.org/10.1101/2022.03.09.483646",
}
PAPER_LITE = {
    "title": "Uncovering the basis of protein-protein interaction specificity with a combinatorially complete library",
    "journal": "eLife",
    "year": 2020,
    "doi": "10.7554/eLife.60924",
    "url": "https://doi.org/10.7554/eLife.60924",
}
PAPER_JOHNSTON = {
    "title": "A combinatorially complete epistatic fitness landscape in an enzyme active site",
    "journal": "Proceedings of the National Academy of Sciences",
    "year": 2024,
    "doi": "10.1073/pnas.2400439121",
    "url": "https://doi.org/10.1073/pnas.2400439121",
}

DATASETS = [
    {
        "id": "Podgornaia2015_PhoQ",
        "label": "PhoQ protein-protein interface",
        "path": "data/BioSequence/Podgornaia2015_PhoQ.csv",
        "publication": PAPER_PHOQ,
        "system": "PhoQ kinase signaling domain",
    },
    {
        "id": "Wu2016_GB1",
        "label": "Protein G B1 domain (GB1)",
        "path": "data/BioSequence/Wu2016_GB1.csv",
        "publication": PAPER_GB1,
        "system": "Protein G B1 domain",
    },
    {
        "id": "Jalal2020_NBS",
        "label": "Noc DNA-binding specificity landscape",
        "path": "data/BioSequence/Jalal2020_NBS.csv",
        "publication": PAPER_JALAL,
        "system": "Noc protein variants; NBS is the assayed DNA-binding site",
    },
    {
        "id": "Jalal2020_parS",
        "label": "ParB DNA-binding specificity landscape",
        "path": "data/BioSequence/Jalal2020_parS.csv",
        "publication": PAPER_JALAL,
        "system": "ParB protein variants; parS is the assayed DNA-binding site",
    },
    {
        "id": "Tu2022_TEV",
        "label": "TEV protease activity landscape",
        "path": "data/BioSequence/Tu2022_TEV.csv",
        "publication": PAPER_TU,
        "system": "Tobacco etch virus protease",
    },
    {
        "id": "Tu2022_T7",
        "label": "T7 RNA polymerase activity (three-site library)",
        "path": "data/BioSequence/Tu2022_T7.csv",
        "publication": PAPER_TU,
        "system": "T7 RNA polymerase activity on the T3 promoter",
        "caveat": "The local table contains 6,725 measured genotypes and retains its supplied scores. Optimization outcomes use within-table percentiles; higher T7 activity is the maximization objective.",
    },
    {
        "id": "Lite2020_ParD2",
        "label": "ParD3 variants assayed with ParE2",
        "path": "data/BioSequence/Lite2020_ParD2.csv",
        "publication": PAPER_LITE,
        "system": "ParD3 antitoxin protein variants assayed against ParE2",
    },
    {
        "id": "Lite2020_ParD3",
        "label": "ParD3 variants assayed with ParE3",
        "path": "data/BioSequence/Lite2020_ParD3.csv",
        "publication": PAPER_LITE,
        "system": "ParD3 antitoxin protein variants assayed against ParE3",
    },
    {
        "id": "Johnston2024_TrpB4",
        "label": "TrpB active site (four variable residues)",
        "path": "data/BioSequence/Johnston2024_TrpB4.csv",
        "publication": PAPER_JOHNSTON,
        "system": "Tryptophan synthase beta-subunit active site",
    },
    {
        "id": "Johnston2024_TrpB3A",
        "label": "TrpB active site (three-site library A)",
        "path": "data/BioSequence/Johnston2024_TrpB3A.csv",
        "publication": PAPER_JOHNSTON,
        "system": "Tryptophan synthase beta-subunit active site",
    },
    {
        "id": "Johnston2024_TrpB3B",
        "label": "TrpB active site (three-site library B)",
        "path": "data/BioSequence/Johnston2024_TrpB3B.csv",
        "publication": PAPER_JOHNSTON,
        "system": "Tryptophan synthase beta-subunit active site",
    },
    {
        "id": "Johnston2024_TrpB3C",
        "label": "TrpB active site (three-site library C)",
        "path": "data/BioSequence/Johnston2024_TrpB3C.csv",
        "publication": PAPER_JOHNSTON,
        "system": "Tryptophan synthase beta-subunit active site",
    },
    {
        "id": "Johnston2024_TrpB3D",
        "label": "TrpB active site (three-site library D)",
        "path": "data/BioSequence/Johnston2024_TrpB3D.csv",
        "publication": PAPER_JOHNSTON,
        "system": "Tryptophan synthase beta-subunit active site",
    },
    {
        "id": "Johnston2024_TrpB3E",
        "label": "TrpB active site (three-site library E)",
        "path": "data/BioSequence/Johnston2024_TrpB3E.csv",
        "publication": PAPER_JOHNSTON,
        "system": "Tryptophan synthase beta-subunit active site",
    },
    {
        "id": "Johnston2024_TrpB3F",
        "label": "TrpB active site (three-site library F)",
        "path": "data/BioSequence/Johnston2024_TrpB3F.csv",
        "publication": PAPER_JOHNSTON,
        "system": "Tryptophan synthase beta-subunit active site",
    },
    {
        "id": "Johnston2024_TrpB3G",
        "label": "TrpB active site (three-site library G)",
        "path": "data/BioSequence/Johnston2024_TrpB3G.csv",
        "publication": PAPER_JOHNSTON,
        "system": "Tryptophan synthase beta-subunit active site",
    },
    {
        "id": "Johnston2024_TrpB3H",
        "label": "TrpB active site (three-site library H)",
        "path": "data/BioSequence/Johnston2024_TrpB3H.csv",
        "publication": PAPER_JOHNSTON,
        "system": "Tryptophan synthase beta-subunit active site",
    },
    {
        "id": "Johnston2024_TrpB3I",
        "label": "TrpB active site (three-site library I)",
        "path": "data/BioSequence/Johnston2024_TrpB3I.csv",
        "publication": PAPER_JOHNSTON,
        "system": "Tryptophan synthase beta-subunit active site",
    },
]

FEATURES = [
    {"key": "epistasis.magnitude", "label": "Magnitude epistasis"},
    {"key": "epistasis.sign", "label": "Sign epistasis"},
    {"key": "epistasis.reciprocal_sign", "label": "Reciprocal sign epistasis"},
    {"key": "diminishing_returns_index", "label": "Diminishing returns index"},
    {"key": "increasing_costs_index", "label": "Increasing costs index"},
    {"key": "global_idiosyncratic_index", "label": "Global idiosyncratic index"},
    {"key": "r_s_ratio", "label": "Roughness-to-slope ratio"},
    {"key": "gamma", "label": "Gamma"},
    {"key": "gamma_star", "label": "Gamma-star"},
    {"key": "local_optima_ratio", "label": "Local-optima ratio"},
    {"key": "autocorrelation", "label": "Autocorrelation"},
    {"key": "fdc", "label": "Fitness-distance correlation"},
    {"key": "evolvability_enhancing_fraction", "label": "Evolvability-enhancing mutations"},
    {"key": "global_optima_accessibility", "label": "Global-optimum accessibility"},
]

OUTCOMES = [
    {"key": "rf_greedy", "label": "RF greedy", "metric": "Best fitness percentile"},
    {"key": "rf_ucb", "label": "RF uncertainty-guided", "metric": "Best fitness percentile"},
    {"key": "random", "label": "Random search", "metric": "Best fitness percentile"},
    {"key": "de_greedy", "label": "Greedy directed evolution", "metric": "Best fitness percentile"},
]

def canonical(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def _clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def _write_public(records):
    payload = {
        "schema_version": 1,
        "status": "partial" if len(records) < len(DATASETS) else "complete",
        "x_label": "Landscape feature",
        "y_label": "Best fitness percentile",
        "y_format": "percent",
        "features": FEATURES,
        "outcomes": OUTCOMES,
        "records": records,
        "provenance": {
            "kind": "GraphFLA directed-evolution benchmark on empirical protein amino-acid landscapes",
            "landscape_count": len(records),
            "feature_seed": SEED_BASE,
            "simulation_seeds": [SEED_BASE + i for i in range(N_REPEATS)],
            "simulation_repeats": N_REPEATS,
            "outcome_aggregation": "Arithmetic mean of the 10 seed-level best-fitness percentiles for each landscape and strategy.",
            "query_budget_cap": TOTAL_BUDGET,
            "exact_budget_strategies": ["rf_greedy", "rf_ucb", "random"],
            "early_stopping_strategy": "de_greedy",
            "initial_random_measurements": INITIAL_BATCH,
            "active_learning_rounds": ROUNDS,
            "active_learning_batch_size": ROUND_BATCH,
            "strategy_protocols": {
                "rf_greedy": "96 shared random observations, then four batches of 96 unmeasured genotypes ranked by a 32-tree RandomForestRegressor predictive mean.",
                "rf_ucb": "Same 96+4x96 schedule and forest; rank by standardized predictive mean plus one between-tree standard deviation.",
                "random": "480 unique uniformly random observed genotypes; shares the initial 96 with both RF strategies.",
                "de_greedy": "Starts at the first shared initial genotype, assays all available unmeasured one-site neighbors, moves to the best improving neighbor, and stops at a local optimum or the 480-query cap. Actual counts are in simulation_runs.csv.",
            },
            "random_forest": {
                "n_estimators": 32,
                "max_features": "sqrt",
                "max_depth": 10,
                "min_samples_leaf": 1,
                "bootstrap": True,
                "n_jobs": 1,
                "target_standardization": "mean and population SD of currently observed labels at each round",
                "uncertainty": "population SD of individual-tree predictions",
            },
            "fitness_rank": "fraction of observed genotypes with fitness <= best fitness measured; ties share the same empirical percentile",
            "objective": "maximize the source-file fitness column without rescaling; ranks are computed against all observed genotypes in that landscape",
            "source_population": "Every row in each listed BioSequence CSV is retained; unmeasured sequence combinations are not imputed or removed.",
            "features_source": "GraphFLA landscape analysis on the full retained one-edit graph with epsilon=0; parameter settings are in README.md.",
        },
    }
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_clean_json(payload), indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".data.", suffix=".tmp", dir=PUBLIC.parent)
    os.fchmod(fd, 0o644)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, PUBLIC)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _pearson_from_direct_sums(sx, sy, sxx, syy, sxy, n):
    if n < 2:
        return float("nan")
    covariance = sxy - sx * sy / n
    variance_x = sxx - sx * sx / n
    variance_y = syy - sy * sy / n
    if variance_x <= 0 or variance_y <= 0:
        return float("nan")
    return float(np.clip(covariance / math.sqrt(variance_x * variance_y), -1.0, 1.0))


def stable_trend_oracle(landscape, block_size=100000):
    """Two-pass-equivalent raw-sum Pearson oracle for pooled edge trends."""
    from graphfla.analysis.epistasis._fitness_trends import _oriented_fitness

    graph = landscape.graph
    raw = np.asarray(graph.vs["fitness"], dtype=float)
    active = np.asarray(graph.degree()) > 0
    oriented = np.zeros(len(raw), dtype=float)
    oriented[active] = _oriented_fitness(raw[active], landscape.maximize)
    sums = {
        "diminishing_returns_index": [[], [], [], [], []],
        "increasing_costs_index": [[], [], [], [], []],
    }
    edge_count = 0
    for start in range(0, graph.ecount(), block_size):
        stop = min(start + block_size, graph.ecount())
        endpoints = np.asarray(
            [edge.tuple for edge in graph.es[start:stop]], dtype=np.intp
        )
        source, target = endpoints.T
        effect = oriented[target] - oriented[source]
        selected = effect > 0
        if not selected.any():
            continue
        source, target, effect = source[selected], target[selected], effect[selected]
        x_values = {
            "diminishing_returns_index": oriented[source],
            "increasing_costs_index": oriented[target],
        }
        for key, x in x_values.items():
            values = sums[key]
            values[0].append(float(x.sum(dtype=np.float64)))
            values[1].append(float(effect.sum(dtype=np.float64)))
            values[2].append(float(np.dot(x, x)))
            values[3].append(float(np.dot(effect, effect)))
            values[4].append(float(np.dot(x, effect)))
        edge_count += len(effect)
    results = {}
    for key, totals in sums.items():
        sx, sy, sxx, syy, sxy = (math.fsum(part) for part in totals)
        results[key] = _pearson_from_direct_sums(sx, sy, sxx, syy, sxy, edge_count)
    return results


def bounded_ee_fraction(landscape, fdr=0.01, edge_block_size=100000):
    """Compute GraphFLA's EE fraction without its all-pairs DataFrame.

    Strict one-site neighbors occur once as directed improving graph edges;
    exact neutral neighbors are held separately on the landscape and are also
    included, matching the public implementation. The public implementation
    duplicates each pair in the opposite direction before BH correction and
    DataFrame creation. The two-sided p-values are identical in both
    directions, and duplicating every hypothesis leaves its BH-adjusted value
    unchanged. We therefore calculate one p-value per undirected pair, then
    count both directional EE predicates over the unchanged 2E denominator.
    """
    from graphfla.analysis._evolvability import (
        _bh_adjusted_pvalues,
        _ee_pvalues,
        _validate_fdr,
    )

    _validate_fdr(fdr)
    graph = landscape.graph
    if not graph.is_directed():
        raise ValueError("A directed improving graph is required.")
    data = landscape.get_data()
    if landscape.data_types is None:
        raise ValueError("Configuration columns are required.")
    X = data[list(landscape.data_types)]
    if X.isna().any().any() or X.duplicated().any():
        raise ValueError("Configurations must be unique and nonmissing.")
    configs = X.to_numpy()
    fitness = data["fitness"].to_numpy(dtype=float)
    if not landscape.maximize:
        fitness = -fitness
    if len(fitness):
        fitness = fitness - fitness[0]
    n_nodes, n_sites = configs.shape
    moments = np.full((n_nodes, n_sites, 3), np.nan, dtype=np.float64)

    neutral_neighbors = getattr(landscape, "_neutral_neighbors", None) or {}
    # The full neighborhood is required even though the fitness graph is
    # directed; merge its incoming/outgoing neighbors with retained exact
    # neutral pairs without making a Python set of every edge.
    for node in range(n_nodes):
        neighbors = np.asarray(graph.neighbors(node, mode="all"), dtype=np.intp)
        neutral = np.asarray(neutral_neighbors.get(node, ()), dtype=np.intp)
        if len(neutral):
            neighbors = np.concatenate((neighbors, neutral))
        if not len(neighbors):
            continue
        neighbors.sort()
        changed = configs[neighbors] != configs[node]
        changes_per_neighbor = changed.sum(axis=1)
        if np.any(changes_per_neighbor != 1):
            raise ValueError("EE mutations require one-site neighbor pairs.")
        changed_site = changed.argmax(axis=1)
        neighbor_fitness = fitness[neighbors]
        for focal in range(n_sites):
            keep = changed_site != focal
            count = int(keep.sum())
            if count:
                values = neighbor_fitness[keep]
                mean = float(values.mean())
                moments[node, focal] = (
                    mean,
                    float(np.mean((values - mean) ** 2)),
                    count,
                )

    directed_edge_count = graph.ecount()
    neutral_pair_count = sum(
        1 for source, targets in neutral_neighbors.items()
        for target in targets if source < target
    )
    pair_count = directed_edge_count + neutral_pair_count
    if not pair_count:
        return float("nan"), {"edge_pairs": 0, "testable_pairs": 0}
    pairs = np.empty((pair_count, 2), dtype=np.intp)
    for index, edge in enumerate(graph.es):
        pairs[index, 0] = edge.source
        pairs[index, 1] = edge.target
    if index + 1 != directed_edge_count:
        raise RuntimeError("Could not extract every improving edge.")
    neutral_flags = np.zeros(pair_count, dtype=bool)
    neutral_index = directed_edge_count
    for source, targets in neutral_neighbors.items():
        for target in targets:
            if source < target:
                pairs[neutral_index] = (source, target)
                neutral_flags[neutral_index] = True
                neutral_index += 1
    if neutral_index != pair_count:
        raise RuntimeError("Could not extract every retained neutral pair.")

    p_effect = np.full(pair_count, np.nan, dtype=float)
    p_zero = np.full(pair_count, np.nan, dtype=float)
    ee_beneficial = np.zeros(pair_count, dtype=bool)
    ee_deleterious = np.zeros(pair_count, dtype=bool)
    ee_neutral = np.zeros(pair_count, dtype=bool)
    for start in range(0, pair_count, edge_block_size):
        stop = min(start + edge_block_size, pair_count)
        block = pairs[start:stop]
        source, target = block[:, 0], block[:, 1]
        changed = configs[source] != configs[target]
        changes_per_edge = changed.sum(axis=1)
        if np.any(changes_per_edge != 1):
            raise ValueError("EE mutations require one-site neighbor pairs.")
        position = changed.argmax(axis=1)
        left = moments[source, position]
        right = moments[target, position]
        delta_fitness = fitness[target] - fitness[source]
        delta_neighbor = right[:, 0] - left[:, 0]
        variance = left[:, 1] + right[:, 1]
        n = np.minimum(left[:, 2], right[:, 2])
        p_effect[start:stop] = _ee_pvalues(
            delta_neighbor - delta_fitness, variance, n
        )
        p_zero[start:stop] = _ee_pvalues(delta_neighbor, variance, n)
        scale = np.maximum.reduce((
            np.abs(left[:, 0]), np.abs(right[:, 0]),
            np.abs(fitness[source]), np.abs(fitness[target]),
        ))
        roundoff = 8 * np.finfo(float).eps * scale
        ee_beneficial[start:stop] = (
            delta_neighbor - np.maximum(0.0, delta_fitness) > roundoff
        )
        ee_deleterious[start:stop] = (
            -delta_neighbor - np.maximum(0.0, -delta_fitness) > roundoff
        )
        ee_neutral[start:stop] = np.abs(delta_neighbor) > roundoff

    ee_beneficial &= ~neutral_flags
    ee_deleterious &= ~neutral_flags
    q_effect = _bh_adjusted_pvalues(p_effect)
    q_zero = _bh_adjusted_pvalues(p_zero)
    testable = np.isfinite(p_effect) | np.isfinite(p_zero)
    if not testable.any():
        return float("nan"), {"edge_pairs": pair_count, "testable_pairs": 0}
    count = np.count_nonzero((q_effect <= fdr) & ee_beneficial)
    count += np.count_nonzero((q_zero <= fdr) & ee_deleterious)
    count += np.count_nonzero((q_zero <= fdr) & ee_neutral & neutral_flags)
    return float(count / (2 * pair_count)), {
        "improving_edge_pairs": directed_edge_count,
        "neutral_neighbor_pairs": neutral_pair_count,
        "ordered_pair_denominator": 2 * pair_count,
        "testable_pairs": int(testable.sum()),
        "implementation": "blockwise GraphFLA private EE kernels; exact full population; symmetric two-orientation BH family",
    }


def _population(meta):
    import numpy as np
    import pandas as pd

    source = ROOT / meta["path"]
    raw = source.read_bytes()
    frame = pd.read_csv(source)
    if "sequences" not in frame.columns or "fitness" not in frame.columns:
        raise ValueError("Expected source columns 'sequences' and 'fitness'.")
    if frame.empty or frame["sequences"].isna().any() or frame["fitness"].isna().any():
        raise ValueError("Source population contains missing sequences or fitness values.")
    if frame["sequences"].duplicated().any():
        raise ValueError("Source contains duplicate genotypes; refusing to aggregate silently.")
    sequences = frame["sequences"].astype(str).tolist()
    length_set = {len(s) for s in sequences}
    if len(length_set) != 1 or next(iter(length_set)) not in (3, 4):
        raise ValueError("Expected a three- or four-site sequence library.")
    if not all(set(s) <= set(ALPHABET) for s in sequences):
        raise ValueError("Source contains noncanonical amino acids or symbols.")
    site_alleles = [sorted({s[j] for s in sequences}) for j in range(len(sequences[0]))]
    if any(len(a) != 20 or set(a) != set(ALPHABET) for a in site_alleles):
        raise ValueError("Expected all 20 canonical amino-acid states at every site.")
    fitness = frame["fitness"].to_numpy(dtype=np.float64)
    if not np.isfinite(fitness).all():
        raise ValueError("Source fitness must be finite.")
    expected = 20 ** len(sequences[0])
    return {
        "meta": meta,
        "frame": frame,
        "sequences": sequences,
        "fitness": fitness,
        "n": len(frame),
        "n_sites": len(sequences[0]),
        "expected": expected,
        "coverage": len(frame) / expected,
        "site_alleles": site_alleles,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_size": len(raw),
    }


def compute_features(pop):
    import numpy as np
    from graphfla import analysis
    from graphfla.landscape import ProteinLandscape
    from graphfla.analysis.epistasis.motifs import _resolve_cut_prob

    meta = pop["meta"]
    sequences = pop["sequences"]
    fitness = pop["fitness"]
    seed = SEED_BASE
    t0 = time.perf_counter()
    landscape = ProteinLandscape().build_from_data(
        sequences, fitness, epsilon=0, n_edit=1, verbose=False
    )
    build_seconds = time.perf_counter() - t0
    if landscape.n_configs != pop["n"]:
        raise RuntimeError("GraphFLA changed the observed genotype population.")
    if landscape.n_vars != pop["n_sites"]:
        raise RuntimeError("GraphFLA did not retain all variable sites.")

    motif_cut = _resolve_cut_prob(landscape, "auto", 15.0)
    values, reasons, timings = {}, {}, {}

    def run(key, fn):
        start = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = fn()
            values[key] = canonical(value)
            if caught:
                counts = Counter(str(w.message) for w in caught)
                reasons[key] = {
                    "warning_count": len(caught),
                    "warnings": [
                        {"message": message, "count": count}
                        for message, count in counts.items()
                    ],
                }
        except Exception as exc:  # retain other independently computed features
            values[key] = None
            reasons[key] = {"error": "{}: {}".format(type(exc).__name__, exc)}
        timings[key] = time.perf_counter() - start

    motifs = run(
        "epistasis",
        lambda: analysis.classify_epistasis(
            landscape, sample_cut_prob="auto", seed=seed, time_budget=15.0
        ),
    )
    # run() stores a dict for this one multi-value metric.
    motif_result = values.pop("epistasis", None)
    if isinstance(motif_result, dict):
        values["epistasis.magnitude"] = canonical(motif_result.get("magnitude"))
        values["epistasis.sign"] = canonical(motif_result.get("sign"))
        values["epistasis.reciprocal_sign"] = canonical(
            motif_result.get("reciprocal_sign")
        )
    else:
        detail = reasons.get("epistasis", "No motif result was returned.")
        for key in (
            "epistasis.magnitude",
            "epistasis.sign",
            "epistasis.reciprocal_sign",
        ):
            values[key] = None
            reasons[key] = detail

    run("diminishing_returns_index", lambda: analysis.diminishing_returns_index(landscape))
    run("increasing_costs_index", lambda: analysis.increasing_costs_index(landscape))
    trend_oracle = stable_trend_oracle(landscape)
    trend_checks = {}
    for key, oracle_value in trend_oracle.items():
        api_value = values.get(key)
        difference = (
            abs(api_value - oracle_value)
            if api_value is not None and oracle_value is not None
            else None
        )
        verified = difference is not None and difference <= 1e-8
        trend_checks[key] = {
            "api_value": api_value,
            "stable_direct_sum_oracle": oracle_value,
            "absolute_difference": difference,
            "verified_within_1e-8": verified,
        }
        # Preserve the exact pooled-Pearson result if the API's streaming
        # accumulator lost precision on this unusually large edge population.
        if oracle_value is not None and (not verified):
            values[key] = canonical(oracle_value)
            reasons[key] = {
                "note": "Public result differed from the stable direct-sum Pearson oracle; the oracle value is used.",
                "check": trend_checks[key],
            }
    run(
        "global_idiosyncratic_index",
        lambda: analysis.global_idiosyncratic_index(
            landscape, n_jobs=1, seed=seed, min_pairs=3
        ),
    )
    run("r_s_ratio", lambda: analysis.r_s_ratio(landscape))
    run("gamma", lambda: analysis.gamma(landscape, n_jobs=1))
    run("gamma_star", lambda: analysis.gamma_star(landscape, n_jobs=1))
    run("local_optima_ratio", lambda: analysis.local_optima_ratio(landscape))
    run(
        "autocorrelation",
        lambda: analysis.autocorrelation(
            landscape, walk_length=20, walk_times=1000, lag=1, seed=seed
        ),
    )
    run("fdc", lambda: analysis.fdc(landscape, method="spearman"))
    ee_start = time.perf_counter()
    try:
        ee_value, ee_details = bounded_ee_fraction(landscape, fdr=0.01)
        values["evolvability_enhancing_fraction"] = canonical(ee_value)
        if ee_details["testable_pairs"] == 0:
            reasons["evolvability_enhancing_fraction"] = {
                "reason": "No testable ordered neighbor pairs."
            }
    except Exception as exc:
        ee_details = {"error": "{}: {}".format(type(exc).__name__, exc)}
        values["evolvability_enhancing_fraction"] = None
        reasons["evolvability_enhancing_fraction"] = ee_details
    timings["evolvability_enhancing_fraction"] = time.perf_counter() - ee_start
    run(
        "global_optima_accessibility",
        lambda: analysis.global_optima_accessibility(landscape),
    )

    graph_facts = {
        "n_nodes": int(landscape.graph.vcount()),
        "n_directed_edges": int(landscape.graph.ecount()),
        "n_local_optima": int(landscape.n_lo),
        "global_optimum_fitness": canonical(landscape.go["fitness"]),
        "build_seconds": build_seconds,
        "feature_seconds": timings,
        "resolved_motif_sample_cut_prob": 0.0 if motif_cut is None else float(motif_cut),
        "feature_seed": seed,
        "dri_ici_stable_oracle": trend_checks,
        "ee_implementation": {
            "method": "blockwise exact-equivalent preparation helper; GraphFLA private EE p-value/BH kernels; full graph; fdr=0.01; no neutrality or experimental-error input",
            **ee_details,
        },
    }
    landscape = None
    gc.collect()

    return values, reasons, graph_facts


def verify_bounded_ee_equivalence(ids=("Johnston2024_TrpB3A", "Lite2020_ParD2")):
    """Compare the bounded EE fraction with the public API on small full files."""
    import numpy as np
    from graphfla import analysis
    from graphfla.landscape import ProteinLandscape

    results = []
    for dataset_id in ids:
        meta = next(d for d in DATASETS if d["id"] == dataset_id)
        pop = _population(meta)
        landscape = ProteinLandscape().build_from_data(
            pop["sequences"], pop["fitness"], epsilon=0, n_edit=1, verbose=False
        )
        bounded, details = bounded_ee_fraction(landscape, fdr=0.01)
        public = analysis.evolvability_enhancing_fraction(
            landscape, fdr=0.01, effect_type="all"
        )
        difference = abs(float(public) - float(bounded))
        if not np.isclose(public, bounded, rtol=0, atol=1e-12):
            raise AssertionError(
                "Bounded EE helper differs from the public API on {}: {} vs {}".format(
                    dataset_id, bounded, public
                )
            )
        results.append({
            "landscape": dataset_id,
            "n_configs": pop["n"],
            "public_api": float(public),
            "bounded_helper": float(bounded),
            "absolute_difference": difference,
            **details,
        })
        del landscape, pop
        gc.collect()
    output = {
        "status": "verified",
        "comparison": "Public GraphFLA `evolvability_enhancing_fraction` versus preparation-only blockwise equivalent on two full observed 3-site landscapes.",
        "result": results,
    }
    (LOCAL / "ee_equivalence.json").write_text(
        json.dumps(_clean_json(output), indent=2, allow_nan=False) + "\n"
    )
    print("EE_EQUIVALENCE", json.dumps(output, allow_nan=False), flush=True)


def _one_hot(sequences):
    import numpy as np

    aa = {c: i for i, c in enumerate(ALPHABET)}
    n, length = len(sequences), len(sequences[0])
    encoded = np.zeros((n, length, len(ALPHABET)), dtype=np.uint8)
    rows = np.arange(n)[:, None]
    sites = np.arange(length)[None, :]
    codes = np.fromiter((aa[c] for seq in sequences for c in seq), dtype=np.int8)
    encoded[rows, sites, codes.reshape(n, length)] = 1
    return encoded.reshape(n, length * len(ALPHABET))


def _percentile(sorted_fitness, best):
    import numpy as np

    return float(np.searchsorted(sorted_fitness, best, side="right") / len(sorted_fitness))


def _fit_predict(pop, X, observed, random_state):
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    y = pop["fitness"]
    y_seen = y[observed]
    sd = float(y_seen.std())
    if not math.isfinite(sd) or sd == 0:
        y_model = np.zeros_like(y_seen)
    else:
        y_model = (y_seen - float(y_seen.mean())) / sd
    model = RandomForestRegressor(
        n_estimators=32,
        max_features="sqrt",
        max_depth=10,
        min_samples_leaf=1,
        bootstrap=True,
        n_jobs=1,
        random_state=random_state,
    )
    model.fit(X[observed], y_model)
    n = len(y)
    total = np.zeros(n, dtype=np.float64)
    total_sq = np.zeros(n, dtype=np.float64)
    for tree in model.estimators_:
        prediction = tree.predict(X)
        total += prediction
        total_sq += prediction * prediction
    mean = total / len(model.estimators_)
    variance = np.maximum(total_sq / len(model.estimators_) - mean * mean, 0.0)
    std = np.sqrt(variance)
    return mean, std


def simulate(pop):
    import numpy as np

    sequences = pop["sequences"]
    fitness = pop["fitness"]
    n = len(fitness)
    if n <= TOTAL_BUDGET:
        raise ValueError("Landscape must contain more than the fixed 480-query budget.")
    X = _one_hot(sequences)
    sorted_fitness = np.sort(fitness)
    seq_to_idx = {seq: i for i, seq in enumerate(sequences)}
    rows = []
    aa = ALPHABET

    for repeat in range(N_REPEATS):
        seed = SEED_BASE + repeat
        rng = np.random.default_rng(seed)
        initial = rng.choice(n, size=INITIAL_BATCH, replace=False)
        initial_set = set(map(int, initial))

        # All three batch methods share their initial random library for paired
        # comparisons. Each query is a unique, actually observed genotype.
        for strategy in ("rf_greedy", "rf_ucb", "random"):
            measured = list(map(int, initial))
            measured_set = set(initial_set)
            for round_index in range(ROUNDS):
                if strategy == "random":
                    candidates = np.fromiter(
                        (i for i in range(n) if i not in measured_set),
                        dtype=np.int32,
                    )
                    batch = rng.choice(candidates, size=ROUND_BATCH, replace=False)
                else:
                    mean, std = _fit_predict(
                        pop, X, np.asarray(measured, dtype=np.int32),
                        random_state=seed * 100 + round_index,
                    )
                    acquisition = mean if strategy == "rf_greedy" else mean + std
                    acquisition[np.asarray(measured, dtype=np.int32)] = -np.inf
                    # Stable ordering makes ties deterministic and repeatable.
                    ranked = np.lexsort((np.arange(n), -acquisition))
                    batch = ranked[:ROUND_BATCH]
                measured.extend(map(int, batch))
                measured_set.update(map(int, batch))
            best = float(fitness[np.asarray(measured, dtype=np.int32)].max())
            rows.append({
                "landscape": pop["meta"]["id"],
                "strategy": strategy,
                "seed": seed,
                "best_fitness": best,
                "best_fitness_percentile": _percentile(sorted_fitness, best),
                "n_unique_measured": len(measured_set),
            })

        # Greedy single-mutant walk: evaluate observed one-site neighbors of the
        # current best until a local optimum or the same 480-query cap.
        current = int(initial[0])
        measured = [current]
        measured_set = {current}
        while len(measured_set) < TOTAL_BUDGET:
            sequence = sequences[current]
            neighbors = set()
            for pos in range(len(sequence)):
                for allele in aa:
                    if allele == sequence[pos]:
                        continue
                    candidate = sequence[:pos] + allele + sequence[pos + 1 :]
                    index = seq_to_idx.get(candidate)
                    if index is not None and index not in measured_set:
                        neighbors.add(index)
            if not neighbors:
                break
            ordered = np.asarray(sorted(neighbors), dtype=np.int32)
            left = TOTAL_BUDGET - len(measured_set)
            if len(ordered) > left:
                ordered = rng.choice(ordered, size=left, replace=False)
            measured.extend(map(int, ordered))
            measured_set.update(map(int, ordered))
            current = max(
                (current, *map(int, ordered)), key=lambda i: float(fitness[i])
            )
            if all(fitness[i] <= fitness[current] for i in ordered):
                # Continue only if the selected current point changed and has
                # an unvisited neighborhood; the next loop checks that.
                pass
        best = float(fitness[np.asarray(measured, dtype=np.int32)].max())
        rows.append({
            "landscape": pop["meta"]["id"],
            "strategy": "de_greedy",
            "seed": seed,
            "best_fitness": best,
            "best_fitness_percentile": _percentile(sorted_fitness, best),
            "n_unique_measured": len(measured_set),
        })
    return rows


def _record(pop, values, reasons, graph_facts, outcomes=None, outcome_reasons=None):
    meta = pop["meta"]
    source_url = "https://github.com/COLA-Laboratory/GraphFLA/blob/main/" + meta["path"]
    out_values = {}
    if outcomes:
        for item in OUTCOMES:
            out_values[item["key"]] = outcomes.get(item["key"])
    else:
        out_values = {item["key"]: None for item in OUTCOMES}
    unavailable = {
        key: reasons.get(key, "No result was returned.")
        for key, value in values.items()
        if value is None
    }
    unavailable.update({
        key: (outcome_reasons or {}).get(key, "No simulation result was produced.")
        for key, value in out_values.items()
        if value is None
    })
    record = {
        "id": meta["id"],
        "label": meta["label"],
        "features": {f["key"]: values.get(f["key"]) for f in FEATURES},
        "outcomes": out_values,
        "publication": meta["publication"],
        "source_url": source_url,
        "data_caveat": meta.get("caveat"),
    }
    if unavailable:
        record["unavailable"] = unavailable
    return record


def _load_existing():
    if PUBLIC.exists():
        try:
            payload = json.loads(PUBLIC.read_text())
            return {r["id"]: r for r in payload.get("records", [])}
        except Exception:
            pass
    return {}


def export_compact():
    """Atomically refresh the compact public data and full seed-level CSV."""
    import pandas as pd

    simulation_path = LOCAL / "simulation_runs.csv"
    runs = pd.read_csv(simulation_path) if simulation_path.exists() else pd.DataFrame()
    records = []
    for meta in DATASETS:
        detail_path = LOCAL / (meta["id"] + ".json")
        if not detail_path.exists():
            continue
        details = json.loads(detail_path.read_text())
        strategy_runs = (
            runs[runs["landscape"] == meta["id"]]
            if not runs.empty else pd.DataFrame()
        )
        outcome_values = {}
        reasons = {}
        for item in OUTCOMES:
            if strategy_runs.empty or item["key"] not in set(strategy_runs["strategy"]):
                outcome_values[item["key"]] = None
                reasons[item["key"]] = "No completed seed-level simulation is available."
            else:
                values = strategy_runs.loc[
                    strategy_runs["strategy"] == item["key"],
                    "best_fitness_percentile",
                ].astype(float)
                outcome_values[item["key"]] = float(values.mean())
        record_pop = {
            "meta": meta,
            "n": details["population"]["n_configs"],
            "expected": details["population"]["expected"],
            "coverage": details["population"]["coverage"],
            "n_sites": details["population"]["sites"],
            "source_sha256": details["population"]["source_sha256"],
        }
        records.append(_record(
            record_pop,
            details["features"],
            details.get("feature_reasons", {}),
            details.get("graph", {}),
            outcomes=outcome_values,
            outcome_reasons=reasons,
        ))
    _write_public(records)

    if simulation_path.exists():
        PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".simulation_runs.", suffix=".tmp", dir=PUBLIC_DIR
        )
        os.fchmod(fd, 0o644)
        os.close(fd)
        try:
            shutil.copyfile(simulation_path, temporary)
            os.replace(temporary, PUBLIC_DIR / "simulation_runs.csv")
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    export_source_manifest()
    print("EXPORTED", len(records), "records", flush=True)


def export_source_manifest():
    """Publish a relative-path source and population ledger for all records."""
    audit_path = ROOT / "docs/home/insights/references/protein_landscape_sources.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {"datasets": {}}
    entries = []
    for meta in DATASETS:
        details_path = LOCAL / (meta["id"] + ".json")
        details = json.loads(details_path.read_text()) if details_path.exists() else {}
        population = details.get("population", {})
        source = audit.get("datasets", {}).get(meta["id"], {})
        scale_note = source.get("scale_note")
        if meta["id"] in {"Tu2022_TEV", "Tu2022_T7"}:
            scale_note = (
                "Local score calibration is unresolved. Source-file values are retained without transformation; optimization outcomes use within-table percentiles."
            )
        entries.append({
            "id": meta["id"],
            "label": meta["label"],
            "local_file": meta["path"],
            "source_url": "https://github.com/COLA-Laboratory/GraphFLA/blob/main/" + meta["path"],
            "source_sha256": population.get("source_sha256"),
            "molecule": source.get("molecule", "protein"),
            "protein": source.get("protein", meta["system"]),
            "assay": source.get("assay", meta["system"]),
            "direction": source.get("direction", "maximize"),
            "direction_basis": source.get("direction_basis"),
            "scale_note": scale_note,
            "n_sites": population.get("sites"),
            "site_alleles": population.get("site_alleles"),
            "observed_rows": population.get("n_configs"),
            "expected_rows": population.get("expected"),
            "coverage_fraction": population.get("coverage"),
            "population_handling": "All source rows retained; missing combinations are not imputed; no score transformation.",
            "fitness_column": "fitness",
            "citation": source.get("citation", meta["publication"]),
            "data_caveat": meta.get("caveat"),
        })
    manifest = {
        "schema_version": 1,
        "landscape_count": len(entries),
        "collection": "Protein amino-acid landscapes selected from the repository's empirical BioSequence files.",
        "simulation_provenance": {
            "kind": "GraphFLA directed-evolution benchmark on empirical protein amino-acid landscapes.",
            "outcome": "Arithmetic mean over 10 seed-level best-fitness percentiles, each computed within that landscape's observed source population.",
            "query_budget_cap": TOTAL_BUDGET,
            "exact_budget_strategies": ["rf_greedy", "rf_ucb", "random"],
            "greedy_de_budget": "At most 480 distinct variants; can stop at a local optimum.",
            "seed_range": [SEED_BASE, SEED_BASE + N_REPEATS - 1],
            "initial_measurements": INITIAL_BATCH,
            "active_learning_rounds": ROUNDS,
            "active_learning_batch_size": ROUND_BATCH,
            "strategy_protocols": {
                "rf_greedy": "Shared 96 random initial observations, then four 96-variant batches ranked by a 32-tree random-forest predictive mean.",
                "rf_ucb": "Same schedule and forest, ranked by standardized predictive mean plus one between-tree standard deviation.",
                "random": "480 unique uniformly sampled genotypes with the same initial 96 as the RF strategies.",
                "de_greedy": "Greedy best-improving walk over measured one-site neighbors; stops at a local optimum or query cap.",
            },
            "fitness_rank": "Fraction of observed genotypes with fitness less than or equal to the best measured fitness; ties share a percentile.",
            "source_population": "Every listed CSV row is retained; missing genotype combinations are not imputed.",
            "feature_protocol": {
                "graph": "One-edit landscape, epsilon=0.",
                "epistasis_classification": "sample_cut_prob=auto, seed=20261004, time_budget=15 seconds.",
                "global_idiosyncratic_index": "n_jobs=1, seed=20261004, min_pairs=3.",
                "autocorrelation": "walk_length=20, walk_times=1000, lag=1, seed=20261004.",
                "fitness_distance_correlation": "Spearman.",
                "evolvability_enhancing_fraction": "fdr=0.01, effect_type=all, without neutrality or experimental-error inputs.",
            },
        },
        "datasets": entries,
    }
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    destination = PUBLIC_DIR / "sources.json"
    fd, temporary = tempfile.mkstemp(prefix=".sources.", suffix=".tmp", dir=PUBLIC_DIR)
    os.fchmod(fd, 0o644)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(_clean_json(manifest), indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _run(ids):
    import pandas as pd

    records = _load_existing()
    simulation_path = LOCAL / "simulation_runs.csv"
    run_cache = {}
    if simulation_path.exists():
        existing = pd.read_csv(simulation_path)
        for key, group in existing.groupby("landscape"):
            run_cache[key] = group.to_dict(orient="records")

    for meta in DATASETS:
        if meta["id"] not in ids:
            continue
        print("START", meta["id"], flush=True)
        pop = _population(meta)
        values, reasons, graph_facts = compute_features(pop)
        print(
            "FEATURES",
            meta["id"],
            json.dumps(_clean_json(values), allow_nan=False),
            "nodes",
            pop["n"],
            "coverage",
            round(pop["coverage"], 6),
            "build_seconds",
            round(graph_facts["build_seconds"], 2),
            "motif_cut",
            graph_facts["resolved_motif_sample_cut_prob"],
            flush=True,
        )
        pending_reasons = {
            item["key"]: "Simulation pending"
            for item in OUTCOMES
        }
        pending = _record(
            pop,
            values,
            reasons,
            graph_facts,
            outcomes=None,
            outcome_reasons=pending_reasons,
        )
        records[meta["id"]] = pending
        ordered = [r for m in DATASETS if (r := records.get(m["id"])) is not None]
        _write_public(ordered)
        outcome_reasons = {}
        if meta["id"] in run_cache:
            runs = run_cache[meta["id"]]
        else:
            t0 = time.perf_counter()
            try:
                runs = simulate(pop)
                run_cache[meta["id"]] = runs
                all_runs = []
                if simulation_path.exists():
                    all_runs = pd.read_csv(simulation_path).to_dict(orient="records")
                    all_runs = [r for r in all_runs if r["landscape"] != meta["id"]]
                all_runs.extend(runs)
                pd.DataFrame(all_runs).to_csv(simulation_path, index=False)
                print(
                    "SIMULATED",
                    meta["id"],
                    "seconds",
                    round(time.perf_counter() - t0, 2),
                    "rows",
                    len(runs),
                    flush=True,
                )
            except Exception as exc:
                outcome_reasons = {
                    key: "{}: {}".format(type(exc).__name__, exc)
                    for key in (o["key"] for o in OUTCOMES)
                }
                runs = []
        outcome_values = {}
        for item in OUTCOMES:
            values_for_strategy = [
                float(r["best_fitness_percentile"])
                for r in runs
                if r["strategy"] == item["key"]
            ]
            outcome_values[item["key"]] = (
                float(sum(values_for_strategy) / len(values_for_strategy))
                if values_for_strategy
                else None
            )
        rec = _record(
            pop,
            values,
            reasons,
            graph_facts,
            outcomes=outcome_values,
            outcome_reasons=outcome_reasons,
        )
        records[meta["id"]] = rec
        ordered = [r for m in DATASETS if (r := records.get(m["id"])) is not None]
        _write_public(ordered)
        details = {
            "landscape": meta["id"],
            "population": {
                "n_configs": pop["n"],
                "expected": pop["expected"],
                "coverage": pop["coverage"],
                "sites": pop["n_sites"],
                "site_alleles": pop["site_alleles"],
                "source_sha256": pop["source_sha256"],
                "source_size": pop["source_size"],
            },
            "features": values,
            "feature_reasons": reasons,
            "graph": graph_facts,
            "outcomes": outcome_values,
        }
        (LOCAL / (meta["id"] + ".json")).write_text(
            json.dumps(_clean_json(details), indent=2, allow_nan=False) + "\n"
        )
        del pop, values, reasons, graph_facts, rec
        gc.collect()
        print("DONE", meta["id"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument("--verify-ee", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.verify_ee:
        verify_bounded_ee_equivalence()
        return
    if args.export_only:
        export_compact()
        return
    ids = set(args.ids) if args.ids else {d["id"] for d in DATASETS}
    unknown = ids - {d["id"] for d in DATASETS}
    if unknown:
        raise SystemExit("Unknown dataset IDs: " + ", ".join(sorted(unknown)))
    _run(ids)
    export_compact()
    print("max_rss", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)


if __name__ == "__main__":
    main()
