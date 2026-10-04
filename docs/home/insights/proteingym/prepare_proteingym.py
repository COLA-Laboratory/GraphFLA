#!/usr/bin/env python3
"""Prepare the ProteinGym × GraphFLA landing-page dataset.

Run from the repository root with the scientific environment, for example::

    ./.venv-ci39/bin/python docs/home/insights/proteingym/prepare_proteingym.py \
        --limit 3
    ./.venv-ci39/bin/python docs/home/insights/proteingym/prepare_proteingym.py

The first command is useful for a quick preview. The full run processes each
eligible assay in a separate, single-threaded subprocess, with at most two
assays active at once. It never samples or
filters assay variants to make a landscape smaller. The only sampling is the
documented GraphFLA motif and random-walk estimators, both with a fixed seed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CACHE = ROOT / ".codex-local" / "insights" / "proteingym"
RAW = CACHE / "raw"
METRIC_CACHE = CACHE / "metrics"
OUTPUT = HERE / "data.json"
REPORT = HERE / "source_report.md"
MANIFEST = HERE / "provenance.json"

PROTEINGYM_VERSION = "v1.3"
ZENODO_RECORD = "https://zenodo.org/records/15293562"
DATA_ARCHIVE = RAW / "DMS_ProteinGym_substitutions.zip"
REFERENCE_CSV = RAW / "DMS_substitutions.csv"
PERFORMANCE_CSV = RAW / "DMS_substitutions_Spearman_DMS_level.csv"
CITATIONS_JSON = ROOT / "docs" / "home" / "insights" / "references" / "proteingym_dms_citations.json"
EE_EQUIVALENCE_JSON = HERE / "ee_validation.json"
SPARSE_GAMMA_PATH = HERE / "sparse_gamma.py"
BOUNDED_EE_PATH = HERE / "bounded_ee_fast.py"
GAMMA_VALIDATION_PATH = HERE / "gamma_validation.json"
FDC_VALIDATION_PATH = HERE / "fdc_validation.json"

REPO_COMMIT = "144fe22b07dfaeec2b366f2346203a9838a55b4c"
DATA_URL = f"{ZENODO_RECORD}/files/DMS_ProteinGym_substitutions.zip?download=1"
REFERENCE_URL = f"{ZENODO_RECORD}/files/DMS_substitutions.csv?download=1"
PERFORMANCE_PATH = (
    "benchmarks/DMS_zero_shot/substitutions/Spearman/"
    "DMS_substitutions_Spearman_DMS_level.csv"
)
PERFORMANCE_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/"
    f"{REPO_COMMIT}/{PERFORMANCE_PATH}"
)

# SHA-256 values of the official v1.3 data archive/reference file and the
# repository performance table pinned above.
EXPECTED_SHA256 = {
    DATA_ARCHIVE.name: "3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921",
    REFERENCE_CSV.name: "a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308",
    PERFORMANCE_CSV.name: "f432423b87f79ac9778dfac86e3d95be041d246bc618dc0a406b35b0b7466437",
}

MUTATION_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
PERFORMANCE_METADATA_COLUMNS = {
    "Number of Mutants",
    "Selection Type",
    "UniProt ID",
    "MSA_Neff_L_category",
    "Taxon",
}
BASE_SEED = 20261004
MOTIF_TIME_BUDGET_SEC = 15.0
DEFAULT_METRIC_TIMEOUT_SEC = 300
DEFAULT_ASSAY_TIMEOUT_SEC = 1800
DEFAULT_MAX_RSS_MB = 2048

FEATURES = [
    {"key": "epistasis.magnitude", "label": "Magnitude epistasis"},
    {"key": "epistasis.sign", "label": "Sign epistasis"},
    {"key": "epistasis.reciprocal_sign", "label": "Reciprocal sign epistasis"},
    {"key": "diminishing_returns_index", "label": "Diminishing returns index"},
    {"key": "increasing_costs_index", "label": "Increasing costs index"},
    {"key": "global_idiosyncratic_index", "label": "Global idiosyncratic index"},
    {"key": "r_s_ratio", "label": "Roughness-to-slope ratio"},
    {"key": "gamma", "label": "Gamma"},
    {"key": "gamma_star", "label": "Gamma star"},
    {"key": "local_optima_ratio", "label": "Local optima ratio"},
    {"key": "autocorrelation", "label": "Fitness autocorrelation"},
    {"key": "fdc", "label": "Fitness-distance correlation"},
    {
        "key": "evolvability_enhancing_fraction",
        "label": "Evolvability-enhancing mutation fraction",
    },
    {
        "key": "global_optima_accessibility",
        "label": "Global optimum accessibility",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(path: Path, url: str) -> None:
    """Download one pinned source file and refuse hash mismatches."""
    from urllib.request import urlopen

    path.parent.mkdir(parents=True, exist_ok=True)
    expected = EXPECTED_SHA256[path.name]
    if path.exists() and _sha256(path) == expected:
        return
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    try:
        with urlopen(url, timeout=60) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
        actual = _sha256(partial)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}"
            )
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def _fetch_sources() -> None:
    _download(DATA_ARCHIVE, DATA_URL)
    _download(REFERENCE_CSV, REFERENCE_URL)
    _download(PERFORMANCE_CSV, PERFORMANCE_URL)


def _parse_mutant(label: str) -> list[tuple[str, int, str]]:
    changes = []
    for token in str(label).split(":"):
        match = MUTATION_RE.fullmatch(token.strip())
        if match is None:
            raise ValueError(f"Unparseable substitution label {label!r}")
        wt, position, mutant = match.groups()
        changes.append((wt.upper(), int(position), mutant.upper()))
    if not changes:
        raise ValueError(f"Empty substitution label {label!r}")
    return changes


def _assay_members(archive: zipfile.ZipFile) -> dict[str, str]:
    members = {
        Path(name).stem: name
        for name in archive.namelist()
        if name.endswith(".csv") and "/" in name
    }
    if len(members) != 217:
        raise ValueError(f"Expected 217 substitution assay files; found {len(members)}")
    return members


def _read_reference() -> dict[str, dict[str, str]]:
    with REFERENCE_CSV.open(newline="", encoding="utf-8") as stream:
        rows = {row["DMS_id"]: row for row in csv.DictReader(stream)}
    return rows


def _scan_eligibility(
    archive: zipfile.ZipFile,
    members: dict[str, str],
    reference: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Use all assay rows and mutation labels to apply the strict mean cutoff."""
    if set(members) != set(reference):
        raise ValueError("Assay ZIP IDs and v1.3 reference IDs do not match exactly")
    result = {}
    total_variants = 0
    for dms_id in sorted(members):
        n_rows = 0
        n_mutations = 0
        with archive.open(members[dms_id]) as binary:
            stream = __import__("io").TextIOWrapper(binary, encoding="utf-8")
            for row in csv.DictReader(stream):
                mutations = _parse_mutant(row.get("mutant", ""))
                n_rows += 1
                n_mutations += len(mutations)
        if n_rows == 0:
            raise ValueError(f"Assay {dms_id} contains no measurement rows")
        result[dms_id] = {
            "n_variants": n_rows,
            "mean_mutations": n_mutations / n_rows,
            "eligible": n_mutations / n_rows > 1.5,
        }
        total_variants += n_rows
    return result, {"n_assays": len(result), "n_variants": total_variants}


def _read_performance() -> tuple[list[dict[str, str]], dict[str, dict[str, float | None]]]:
    with PERFORMANCE_CSV.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if "DMS ID" not in headers:
            raise ValueError("Pinned ProteinGym score table has no 'DMS ID' column")
        models = [
            col for col in headers
            if col != "DMS ID" and col not in PERFORMANCE_METADATA_COLUMNS
        ]
        rows: dict[str, dict[str, float | None]] = {}
        for row in reader:
            dms_id = row["DMS ID"]
            if dms_id in rows:
                raise ValueError(f"Duplicate per-DMS score row for {dms_id}")
            scores = {}
            for model in models:
                raw = row.get(model, "")
                try:
                    scores[model] = float(raw) if raw not in (None, "") else None
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric zero-shot Spearman value for {dms_id}/{model}: {raw!r}"
                    ) from exc
            rows[dms_id] = scores
    if len(rows) != 217:
        raise ValueError(f"Expected 217 per-DMS score rows; found {len(rows)}")
    if len(models) != 97:
        raise ValueError(f"Expected 97 zero-shot model score columns; found {len(models)}")
    return [{"key": _slugify(model), "label": model} for model in models], rows


def _slugify(label: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return value or "model"


def _validate_and_reduce(
    archive: zipfile.ZipFile,
    member: str,
    target_seq: str,
) -> tuple[list[str], list[float], list[int], dict[str, Any]]:
    """Validate every label/sequence/score, then retain only mutable positions."""
    import numpy as np
    import pandas as pd

    dms_df = pd.read_csv(
        archive.open(member),
        usecols=["mutant", "mutated_sequence", "DMS_score"],
        dtype={"mutant": "string", "mutated_sequence": "string", "DMS_score": "float64"},
    )
    if dms_df.empty or dms_df["DMS_score"].isna().any():
        raise ValueError("DMS_score has missing values or the assay is empty")
    if not np.isfinite(dms_df["DMS_score"].to_numpy()).all():
        raise ValueError("DMS_score contains non-finite values")

    parsed = [_parse_mutant(label) for label in dms_df["mutant"].astype(str)]
    positions = sorted({pos for changes in parsed for _, pos, _ in changes})
    reduced: list[str] = []
    target = target_seq.upper()
    nonstandard = set()
    for label, changes, seq_value in zip(
        dms_df["mutant"].astype(str), parsed, dms_df["mutated_sequence"].astype(str)
    ):
        if len(seq_value) != len(target):
            raise ValueError(
                f"{label}: mutated_sequence length {len(seq_value)} != target length {len(target)}"
            )
        expected = list(target)
        seen_positions = set()
        for wt, pos, mutant in changes:
            if pos < 1 or pos > len(target):
                raise ValueError(f"{label}: position {pos} is outside the target sequence")
            if target[pos - 1] != wt:
                raise ValueError(
                    f"{label}: WT label {wt} disagrees with target residue "
                    f"{target[pos - 1]} at position {pos}"
                )
            if mutant == wt:
                raise ValueError(f"{label}: a substitution must change the residue")
            if pos in seen_positions:
                raise ValueError(f"{label}: position {pos} occurs more than once")
            seen_positions.add(pos)
            expected[pos - 1] = mutant
            if mutant not in STANDARD_AA:
                nonstandard.add(mutant)
        if "".join(expected) != seq_value.upper():
            raise ValueError(f"{label}: mutation labels do not reproduce mutated_sequence")
        reduced.append("".join(seq_value[pos - 1].upper() for pos in positions))

    if nonstandard:
        raise ValueError(f"Non-standard mutant symbols require review: {sorted(nonstandard)}")
    if len(set(reduced)) != len(reduced):
        raise ValueError("Repeated genotype rows need explicit replicate handling; none were filtered")
    return (
        reduced,
        dms_df["DMS_score"].to_list(),
        positions,
        {
            "n_variants": len(reduced),
            "n_mutated_positions": len(positions),
            "original_sequence_length": len(target),
            "variable_sequence_length": len(positions),
            "invariant_positions_removed": len(target) - len(positions),
            "fitness_min": float(dms_df["DMS_score"].min()),
            "fitness_max": float(dms_df["DMS_score"].max()),
            "validation": "all mutation labels match the WT sequence and reproduce the measured sequence",
        },
    )


def _metric_result(value: Any) -> float | None:
    if isinstance(value, dict):
        raise TypeError("expected scalar metric result")
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _bounded_fdc(landscape, method: str = "spearman", chunk_rows: int = 8192):
    """Compute exact Hamming FDC from retained codes in bounded row chunks.

    This preparation-only path follows GraphFLA's nearest-global-optimum
    convention while avoiding tuple/DataFrame copies and the dense stack of
    distances created when many global optima tie.
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    if method not in ("spearman", "pearson"):
        raise ValueError("FDC method must be 'spearman' or 'pearson'.")
    if landscape._configs_array is None:
        raise ValueError("Exact chunked FDC requires retained configuration codes.")
    configs = np.asarray(landscape._configs_array)
    fitness = np.asarray(landscape.graph.vs["fitness"], dtype=np.float64)
    if configs.ndim != 2 or configs.shape[0] != len(fitness):
        raise ValueError("Landscape configuration codes and fitness do not align.")
    best = float(fitness.max()) if landscape.maximize else float(fitness.min())
    optima = np.flatnonzero(fitness == best)
    if not len(optima):
        raise ValueError("Landscape has no global optimum.")
    distances = np.full(len(fitness), np.iinfo(np.int32).max, dtype=np.int32)
    chunks = 0
    for optimum in optima:
        target = configs[int(optimum)]
        for start in range(0, len(fitness), chunk_rows):
            stop = min(start + chunk_rows, len(fitness))
            block = np.count_nonzero(configs[start:stop] != target, axis=1)
            distances[start:stop] = np.minimum(distances[start:stop], block)
            chunks += 1
    correlation = (
        spearmanr(distances, fitness).statistic
        if method == "spearman"
        else pearsonr(distances, fitness).statistic
    )
    return float(correlation), {
        "implementation": "chunked exact Hamming distance to nearest maximizing global optimum",
        "n_configs": int(len(fitness)),
        "n_global_optima": int(len(optima)),
        "chunk_rows": int(chunk_rows),
        "chunks_processed": int(chunks),
    }


def _bounded_ee_fraction(landscape, fdr: float = 0.01):
    """Call the bundled exact EE helper using GraphFLA's sparse moment kernel."""
    from bounded_ee_fast import bounded_ee_fraction_fast

    def report(stage, elapsed, details):
        print(
            f"[EE] {stage}: {elapsed:.1f}s {details}",
            flush=True,
        )

    return bounded_ee_fraction_fast(
        landscape,
        fdr=fdr,
        progress_callback=report,
    )

def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(path)


def _timed(callable_, seconds: int) -> Any:
    if not hasattr(signal, "setitimer"):
        return callable_()

    def timeout(_signum, _frame):
        raise TimeoutError(f"metric exceeded {seconds}s limit")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return callable_()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _worker(
    dms_id: str,
    metric_timeout_sec: int,
    motif_cut_prob_override: float | None = None,
) -> None:
    """Build one complete landscape and compute the requested metrics serially."""
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    import numpy as np
    import graphfla.analysis as analysis
    from graphfla.analysis.epistasis.motifs import _resolve_cut_prob
    from graphfla.landscape import ProteinLandscape

    references = _read_reference()
    with zipfile.ZipFile(DATA_ARCHIVE) as archive:
        members = _assay_members(archive)
        target_seq = references[dms_id]["target_seq"]
        genotypes, scores, positions, data_meta = _validate_and_reduce(
            archive, members[dms_id], target_seq
        )

    print(f"[{dms_id}] validated {len(genotypes):,} measured variants; "
          f"{len(positions)} mutated positions", flush=True)
    landscape = ProteinLandscape(maximize=True)
    landscape.build_from_data(
        genotypes,
        scores,
        epsilon=0,
        n_edit=1,
        verbose=False,
    )
    landscape_meta = {
        "n_configs": int(landscape.n_configs),
        "n_edges": int(landscape.graph.ecount()),
        "n_local_optima": int(landscape.n_lo),
        "n_mutated_positions": len(positions),
    }
    print(f"[{dms_id}] landscape {landscape_meta['n_configs']:,} nodes, "
          f"{landscape_meta['n_edges']:,} edges", flush=True)

    cache_path = METRIC_CACHE / f"{dms_id}.json"
    previous = {}
    if cache_path.exists():
        try:
            previous = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("status") in ("failed", "partial") and not previous.get("landscape"):
            previous = {}
    feature_values: dict[str, float | None] = previous.get("features", {})
    failures: dict[str, str] = previous.get("feature_failures", {})
    params: dict[str, Any] = previous.get("parameters", {})

    def persist() -> None:
        payload = {
            "id": dms_id,
            **data_meta,
            "landscape": landscape_meta,
            "features": feature_values,
            "feature_failures": failures,
            "parameters": params,
            "resource_audit": previous.get("resource_audit", {}),
            "status": "complete" if set(feature_values) >= {f["key"] for f in FEATURES} else "partial",
        }
        _atomic_json(cache_path, payload)

    def run_feature(key: str, fn, description: str, **parameter_values) -> None:
        if key in feature_values or key in failures:
            return
        print(f"[{dms_id}] {description}", flush=True)
        params[key] = parameter_values
        caught = []
        try:
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                value = _timed(fn, metric_timeout_sec)
                caught = [f"{w.category.__name__}: {w.message}" for w in seen]
            result = _metric_result(value)
            feature_values[key] = result
            if result is None:
                failures[key] = "; ".join(caught) or "GraphFLA returned a non-finite value for this landscape"
        except BaseException as exc:
            feature_values[key] = None
            failures[key] = f"{type(exc).__name__}: {exc}"
        finally:
            persist()

    # A fixed seed and GraphFLA's automatic motif sampling ladder provide a
    # reproducible result while keeping the documented target runtime bounded.
    motif_request = "auto" if motif_cut_prob_override is None else float(motif_cut_prob_override)
    resolved_cut = _resolve_cut_prob(
        landscape, motif_request, time_budget=MOTIF_TIME_BUDGET_SEC
    )
    motif_result = None
    if not all(k in feature_values or k in failures for k in (
        "epistasis.magnitude", "epistasis.sign", "epistasis.reciprocal_sign"
    )):
        motif_params = {
            "requested_cut": motif_request,
            "sample_cut_prob": resolved_cut,
            "seed": BASE_SEED,
            "time_budget": MOTIF_TIME_BUDGET_SEC,
        }
        params["classify_epistasis"] = motif_params
        print(f"[{dms_id}] classify_epistasis (cut={resolved_cut})", flush=True)
        try:
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                motif_result = _timed(
                    lambda: analysis.classify_epistasis(
                        landscape,
                        sample_cut_prob=resolved_cut,
                        seed=BASE_SEED,
                        time_budget=MOTIF_TIME_BUDGET_SEC,
                    ),
                    metric_timeout_sec,
                )
            for component in ("magnitude", "sign", "reciprocal_sign"):
                key = f"epistasis.{component}"
                value = _metric_result(motif_result.get(component))
                feature_values[key] = value
                if value is None:
                    failures[key] = "; ".join(
                        f"{w.category.__name__}: {w.message}" for w in seen
                    ) or "GraphFLA returned a non-finite value for this landscape"
            persist()
        except BaseException as exc:
            for component in ("magnitude", "sign", "reciprocal_sign"):
                key = f"epistasis.{component}"
                feature_values[key] = None
                failures[key] = f"{type(exc).__name__}: {exc}"
            persist()

    run_feature(
        "diminishing_returns_index",
        lambda: analysis.diminishing_returns_index(landscape),
        "diminishing_returns_index",
        method="pearson",
    )
    run_feature(
        "increasing_costs_index",
        lambda: analysis.increasing_costs_index(landscape),
        "increasing_costs_index",
        method="pearson",
    )
    run_feature(
        "global_idiosyncratic_index",
        lambda: analysis.global_idiosyncratic_index(
            landscape, n_jobs=1, seed=BASE_SEED, min_pairs=3
        ),
        "global_idiosyncratic_index",
        n_jobs=1,
        seed=BASE_SEED,
        min_pairs=3,
    )
    run_feature(
        "r_s_ratio",
        lambda: analysis.r_s_ratio(landscape),
        "r_s_ratio",
    )
    gamma_keys = ("gamma", "gamma_star")
    if not all(key in feature_values or key in failures for key in gamma_keys):
        from sparse_gamma import sparse_gamma_statistics

        print(f"[{dms_id}] gamma and gamma_star (exact sparse square traversal)", flush=True)
        gamma_params = {
            "implementation": "bundled sparse_gamma_statistics",
            "n_jobs": 1,
            "computed_jointly": True,
            "row_subsampling": False,
            "kernel": "GraphFLA _gamma_grid_moments and _merge_gamma_contributions",
        }
        params["gamma"] = dict(gamma_params)
        params["gamma_star"] = dict(gamma_params)
        try:
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                gamma_started = time.monotonic()
                gamma_values = _timed(
                    lambda: sparse_gamma_statistics(landscape, return_diagnostics=True),
                    metric_timeout_sec,
                )
            elapsed_sec = time.monotonic() - gamma_started
            diagnostics = gamma_values.pop("diagnostics", {})
            for key in gamma_keys:
                params[key]["elapsed_sec"] = round(elapsed_sec, 6)
                params[key]["diagnostics"] = diagnostics
            warning_reason = "; ".join(
                f"{w.category.__name__}: {w.message}" for w in seen
            )
            for key in gamma_keys:
                value = _metric_result(gamma_values.get(key))
                feature_values[key] = value
                if value is None:
                    failures[key] = warning_reason or "GraphFLA returned a non-finite value for this landscape"
        except BaseException as exc:
            for key in gamma_keys:
                feature_values[key] = None
                failures[key] = f"{type(exc).__name__}: {exc}"
        persist()
    run_feature(
        "local_optima_ratio",
        lambda: analysis.local_optima_ratio(landscape),
        "local_optima_ratio",
    )
    run_feature(
        "autocorrelation",
        lambda: analysis.autocorrelation(
            landscape, walk_length=20, walk_times=1000, lag=1, seed=BASE_SEED
        ),
        "autocorrelation",
        walk_length=20,
        walk_times=1000,
        lag=1,
        seed=BASE_SEED,
    )
    fdc_parameters = {"method": "spearman"}
    if landscape.n_configs >= 50000:
        fdc_parameters.update(
            implementation="chunked exact nearest-optimum Hamming distance",
            chunk_rows=8192,
        )

        def compute_bounded_fdc():
            value, details = _bounded_fdc(landscape, method="spearman")
            params["fdc"].update(details)
            return value

        run_feature(
            "fdc",
            compute_bounded_fdc,
            "fdc (bounded exact nearest-optimum distances)",
            **fdc_parameters,
        )
    else:
        run_feature(
            "fdc",
            lambda: analysis.fdc(landscape, method="spearman"),
            "fdc",
            **fdc_parameters,
        )
    ee_details = {}
    if landscape.n_configs >= 50000:
        def compute_bounded_ee():
            value, details = _bounded_ee_fraction(landscape, fdr=0.01)
            ee_details.update(details)
            return value

        run_feature(
            "evolvability_enhancing_fraction",
            compute_bounded_ee,
            "evolvability_enhancing_fraction (degree-bucketed exact path)",
            fdr=0.01,
            effect_type="all",
            implementation="degree-bucketed exact GraphFLA nonfocal-moment kernel",
        )
        if ee_details:
            params["evolvability_enhancing_fraction"].update(ee_details)
            persist()
    else:
        run_feature(
            "evolvability_enhancing_fraction",
            lambda: analysis.evolvability_enhancing_fraction(
                landscape, fdr=0.01, effect_type="all"
            ),
            "evolvability_enhancing_fraction",
            fdr=0.01,
            effect_type="all",
            implementation="public GraphFLA API",
        )
    run_feature(
        "global_optima_accessibility",
        lambda: analysis.global_optima_accessibility(landscape),
        "global_optima_accessibility",
    )
    print(f"[{dms_id}] worker finished", flush=True)


def _read_citations() -> dict[str, dict[str, Any]]:
    if not CITATIONS_JSON.exists():
        return {}
    payload = json.loads(CITATIONS_JSON.read_text(encoding="utf-8"))
    return payload.get("citations", payload)


def _build_records(
    summaries: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, str]],
    performance: dict[str, dict[str, float | None]],
    citations: dict[str, dict[str, Any]],
    selected_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    records = []
    missing_reasons: dict[str, dict[str, str]] = {}
    for dms_id in selected_ids:
        cached_path = METRIC_CACHE / f"{dms_id}.json"
        cached = json.loads(cached_path.read_text(encoding="utf-8")) if cached_path.exists() else {}
        features = {item["key"]: None for item in FEATURES}
        features.update(cached.get("features", {}))
        assay = reference[dms_id]
        citation = citations.get(dms_id, {})
        if "publication" in citation and isinstance(citation["publication"], dict):
            publication = citation["publication"]
            extra_citation = {k: v for k, v in citation.items() if k != "publication"}
        elif any(key in citation for key in ("title", "journal", "year", "doi", "url")):
            publication = citation
            extra_citation = citation
        else:
            publication = {
                "title": assay.get("title") or None,
                "journal": None,
                "year": int(assay["year"]) if assay.get("year", "").isdigit() else None,
                "doi": assay.get("jo") or None,
                "url": f"https://doi.org/{assay['jo']}" if assay.get("jo") else None,
            }
            extra_citation = {}
        if not publication.get("title") or not publication.get("year") or not publication.get("doi"):
            missing_reasons.setdefault(dms_id, {})["publication"] = (
                "citation metadata not available from the verified citation map"
            )
        for feature, reason in cached.get("feature_failures", {}).items():
            missing_reasons.setdefault(dms_id, {})[feature] = reason
        for feature in features:
            if features[feature] is None and feature not in missing_reasons.get(dms_id, {}):
                missing_reasons.setdefault(dms_id, {})[feature] = (
                    "feature computation has not completed"
                    if cached.get("status") not in ("complete", "failed")
                    else "GraphFLA returned an undefined value"
                )
        if not missing_reasons.get(dms_id):
            missing_reasons.pop(dms_id, None)
        record = {
            "id": dms_id,
            "label": dms_id,
            "protein": assay.get("UniProt_ID") or dms_id.split("_")[0],
            "mean_mutations": summaries[dms_id]["mean_mutations"],
            "n_variants": summaries[dms_id]["n_variants"],
            "n_graph_configs": cached.get("landscape", {}).get("n_configs"),
            "n_isolated_removed": (
                max(0, summaries[dms_id]["n_variants"] - cached["landscape"]["n_configs"])
                if cached.get("landscape", {}).get("n_configs") is not None
                else None
            ),
            "original_sequence_length": cached.get("original_sequence_length"),
            "variable_sequence_length": cached.get("variable_sequence_length"),
            "features": features,
            "models": {
                model: _finite_or_none(score)
                for model, score in performance[dms_id].items()
            },
            "publication": {
                "title": publication.get("title"),
                "journal": publication.get("journal"),
                "year": publication.get("year"),
                "doi": publication.get("doi"),
                "url": publication.get("url"),
            },
            "source_url": DATA_URL,
        }
        if extra_citation:
            record["proteinGym_reference_doi"] = extra_citation.get("proteinGym_reference_doi")
        records.append(record)
    return records, missing_reasons


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _write_artifacts(
    summaries: dict[str, dict[str, Any]],
    summary_totals: dict[str, int],
    reference: dict[str, dict[str, str]],
    model_meta: list[dict[str, str]],
    performance: dict[str, dict[str, float | None]],
    selected_ids: list[str],
) -> None:
    citations = _read_citations()
    records, missing_reasons = _build_records(
        summaries, reference, performance, citations, selected_ids
    )
    all_attempted = all(
        (METRIC_CACHE / f"{dms_id}.json").exists()
        and json.loads((METRIC_CACHE / f"{dms_id}.json").read_text(encoding="utf-8")).get("status") in ("complete", "failed")
        for dms_id in selected_ids
    )
    is_partial = not all_attempted or len(selected_ids) != 36
    completed_status = (
        "partial" if is_partial
        else "completed_with_missing_metrics" if missing_reasons
        else "complete"
    )
    payload = {
        "metadata": {
            "status": completed_status,
            "dataset_release": PROTEINGYM_VERSION,
            "eligible_assays": len(selected_ids),
            "eligible_assays_total": 36,
            "default_feature": "r_s_ratio",
            "default_model": "ESM-1b",
        },
        "partial": is_partial,
        "features": FEATURES,
        "models": model_meta,
        "records": records,
        "provenance": {
            "status": completed_status,
            "dataset": "ProteinGym DMS substitution benchmark",
            "dataset_release": PROTEINGYM_VERSION,
            "dataset_release_record": ZENODO_RECORD,
            "reference_rows": len(reference),
            "assay_archive_files": summary_totals["n_assays"],
            "assay_archive_variants": summary_totals["n_variants"],
            "eligible_assays": len([row for row in summaries.values() if row["eligible"]]),
            "eligibility_rule": "unfiltered mean number of substitution labels per measured variant > 1.5; indels are excluded by the substitution-only benchmark",
            "eligible_variants": sum(summaries[dms_id]["n_variants"] for dms_id in selected_ids),
            "selected_dms_ids": selected_ids,
            "missing_published_scores": {
                dms_id: {
                    model: "The pinned official ProteinGym per-DMS Spearman table has no score in this cell."
                    for model, value in performance[dms_id].items()
                    if value is None
                }
                for dms_id in selected_ids
                if any(value is None for value in performance[dms_id].values())
            },
            "source_files": {
                DATA_ARCHIVE.name: {
                    "url": DATA_URL,
                    "sha256": _sha256(DATA_ARCHIVE),
                },
                REFERENCE_CSV.name: {
                    "url": REFERENCE_URL,
                    "sha256": _sha256(REFERENCE_CSV),
                },
                PERFORMANCE_CSV.name: {
                    "url": PERFORMANCE_URL,
                    "repository_commit": REPO_COMMIT,
                    "sha256": _sha256(PERFORMANCE_CSV),
                    "scope": "published per-DMS zero-shot Spearman correlations; no models were trained or scored here",
                },
                CITATIONS_JSON.name: {
                    "path": str(CITATIONS_JSON.relative_to(ROOT)),
                    "sha256": _sha256(CITATIONS_JSON) if CITATIONS_JSON.exists() else None,
                    "scope": "verified per-assay peer-reviewed publication metadata",
                },
            },
            "graphfla": {
                "repository_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "landscape": "ProteinLandscape over the union of observed mutated positions; DMS_score; maximize=True; n_edit=1; epsilon=0",
                "isolate_handling": "GraphFLA's standard construction removes configurations with no observed one-edit neighbors after eligibility and input validation; per-record n_graph_configs and n_isolated_removed are retained",
                "ee_equivalence_validation": {
                    "path": str(EE_EQUIVALENCE_JSON.relative_to(ROOT)),
                    "sha256": _sha256(EE_EQUIVALENCE_JSON) if EE_EQUIVALENCE_JSON.exists() else None,
                    "status": "optimized helper matches public EE and prior exact helper on two full landscapes, including a neutral pair",
                },
                "sparse_gamma_validation": {
                    "path": str(GAMMA_VALIDATION_PATH.relative_to(ROOT)),
                    "sha256": _sha256(GAMMA_VALIDATION_PATH) if GAMMA_VALIDATION_PATH.exists() else None,
                    "status": "bundled exact sparse traversal compared with GraphFLA's joint public API on full ProteinGym assays",
                },
                "bounded_fdc_validation": {
                    "path": str(FDC_VALIDATION_PATH.relative_to(ROOT)),
                    "sha256": _sha256(FDC_VALIDATION_PATH) if FDC_VALIDATION_PATH.exists() else None,
                    "status": "exact match with GraphFLA public FDC for single and tied global optima",
                },
                "preparation_code": {
                    "script": {
                        "path": str(Path(__file__).resolve().relative_to(ROOT)),
                        "sha256": _sha256(Path(__file__).resolve()),
                    },
                    "sparse_gamma_helper": {
                        "path": str(SPARSE_GAMMA_PATH.relative_to(ROOT)),
                        "sha256": _sha256(SPARSE_GAMMA_PATH) if SPARSE_GAMMA_PATH.exists() else None,
                    },
                    "bounded_ee_helper": {
                        "path": str(BOUNDED_EE_PATH.relative_to(ROOT)),
                        "sha256": _sha256(BOUNDED_EE_PATH) if BOUNDED_EE_PATH.exists() else None,
                    },
                },
                "metric_failures": missing_reasons,
                "parameters_by_dms": {
                    dms_id: json.loads((METRIC_CACHE / f"{dms_id}.json").read_text(encoding="utf-8")).get("parameters", {})
                    for dms_id in selected_ids
                    if (METRIC_CACHE / f"{dms_id}.json").exists()
                },
                "resource_audit_by_dms": {
                    dms_id: json.loads((METRIC_CACHE / f"{dms_id}.json").read_text(encoding="utf-8")).get("resource_audit", {})
                    for dms_id in selected_ids
                    if (METRIC_CACHE / f"{dms_id}.json").exists()
                    and json.loads((METRIC_CACHE / f"{dms_id}.json").read_text(encoding="utf-8")).get("resource_audit")
                },
                "sampling": {
                    "classify_epistasis": {
                        "seed": BASE_SEED,
                        "time_budget_sec": MOTIF_TIME_BUDGET_SEC,
                        "requested_cut_by_dms": {
                            dms_id: json.loads((METRIC_CACHE / f"{dms_id}.json").read_text(encoding="utf-8")).get("parameters", {}).get("classify_epistasis", {}).get("requested_cut")
                            for dms_id in selected_ids
                            if (METRIC_CACHE / f"{dms_id}.json").exists()
                        },
                        "resolved_cut_probability_by_dms": {
                            dms_id: json.loads((METRIC_CACHE / f"{dms_id}.json").read_text(encoding="utf-8")).get("parameters", {}).get("classify_epistasis", {}).get("sample_cut_prob")
                            for dms_id in selected_ids
                            if (METRIC_CACHE / f"{dms_id}.json").exists()
                        },
                    },
                    "global_idiosyncratic_index_seed": BASE_SEED,
                    "autocorrelation_seed": BASE_SEED,
                },
            },
            "citation_fields": "publication is the verified peer-reviewed assay paper; proteinGym_reference_doi preserves the benchmark's DOI if it differs",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT, payload)
    _atomic_json(MANIFEST, payload["provenance"])

    sizes = "\n".join(
        f"- `{path.name}` — {path.stat().st_size:,} bytes; SHA-256 `{_sha256(path)}`"
        for path in (DATA_ARCHIVE, REFERENCE_CSV, PERFORMANCE_CSV, CITATIONS_JSON)
        if path.exists()
    )
    feature_lines = "\n".join(
        f"- `{item['key']}`: {item['label']}"
        for item in FEATURES
    )
    status_counts = {}
    for dms_id in selected_ids:
        cached_path = METRIC_CACHE / f"{dms_id}.json"
        status = json.loads(cached_path.read_text(encoding="utf-8")).get("status", "pending") if cached_path.exists() else "pending"
        status_counts[status] = status_counts.get(status, 0) + 1
    REPORT.write_text(
        f"""# ProteinGym data preparation report

Dataset: ProteinGym DMS substitution benchmark, {PROTEINGYM_VERSION}.
Official record: {ZENODO_RECORD} (v1.3, published 2025-04-27).

## Source files

{sizes}

The model outcomes come from the official zero-shot per-DMS Spearman table at
ProteinGym repository commit `{REPO_COMMIT}`. The current 103-column CSV has
97 model columns; the remaining columns are the assay key and benchmark metadata.
This process uses those published correlations directly. It performs no model
training or inference and does not download the 1.9 GB model-score archive.

## Eligibility and input validation

All 217 substitution assay CSVs were scanned using every measured row. Eligibility
is based on the arithmetic mean number of colon-separated substitution tokens in
`mutant`, strictly greater than 1.5. No variant rows were filtered to calculate
that mean. The v1.3 archive contains {summary_totals['n_variants']:,} rows; the
selected {len(selected_ids)} assays contain
{sum(summaries[dms_id]['n_variants'] for dms_id in selected_ids):,} complete measured
rows. Every selected row was checked against the official target sequence: the
WT letter and position match, the labels reproduce `mutated_sequence`, and the
fitness value is finite. Every selected genotype is unique. The 36-assay cohort
is recomputed from the current v1.3 release and is not copied from an earlier
paper table with rounded mutation-depth summaries.

## GraphFLA method

Each assay is passed to `ProteinLandscape` with all measured rows after removing
only positions that are invariant throughout that assay. The original target
length and retained variable length are saved per assay. GraphFLA's standard
construction then removes configurations with no observed one-edit neighbor;
both the raw `n_variants` and resulting `n_graph_configs` are saved. Eligibility
is always computed before construction from every assay row. The landscape uses
processed `DMS_score` with maximization, one-edit neighbors, and `epsilon=0`.
No assay-level row is sampled or removed by this preparation script. The feature
set is:

{feature_lines}

`classify_epistasis` uses reproducible GraphFLA motif sampling with a fixed seed;
the resolved cutoff is recorded per assay. Gamma and gamma-star use the bundled
exact sparse traversal with GraphFLA's existing pooled moment kernels. Its
equivalence checks on two full ProteinGym assays are in
`gamma_validation.json`; no genotype rows are subsampled. For large landscapes,
FDC uses an exact chunked Hamming calculation to the nearest global optimum;
the single- and tied-optimum checks are in `fdc_validation.json`.
`global_idiosyncratic_index` uses one worker, `min_pairs=3`, and the fixed seed.
`autocorrelation` uses 1,000 random walks of at most 20 visited states, lag 1,
and the same fixed seed. Other scalar settings are recorded in the JSON
provenance. For landscapes with at least 50,000 retained graph configurations,
the EE fraction uses a bundled degree-bucketed implementation of GraphFLA's
nonfocal-moment and p-value/BH kernels; it was checked against the public API
and the prior exact helper on two full landscapes with zero numerical
difference. The neutrality metric is excluded.

Per-assay feature status: {status_counts}. Missing/non-finite values are JSON
`null`; explanations are in `provenance.graphfla.metric_failures`.

## Reproduction

From the repository root, run:

```bash
./.venv-ci39/bin/python docs/home/insights/proteingym/prepare_proteingym.py
```

The script verifies source SHA-256 values, scans and validates the official data,
and processes at most two assays concurrently in separate single-threaded
subprocesses. A worker is stopped above the configured resident-memory ceiling
or assay timeout;
individual metric limits and incomplete metric reasons are recorded in the output.
Use `--limit 3` for an explicitly partial preview.
    """,
        encoding="utf-8",
    )


def _monitor_worker(
    dms_id: str,
    metric_timeout_sec: int,
    assay_timeout_sec: int,
    max_rss_mb: int,
    motif_cut_prob_override: float | None = None,
) -> tuple[str | None, str | None]:
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        dms_id,
        "--metric-timeout-sec",
        str(metric_timeout_sec),
    ]
    if motif_cut_prob_override is not None:
        cmd.extend(["--motif-cut-prob", str(motif_cut_prob_override)])
    process = subprocess.Popen(cmd, cwd=ROOT, env=env)
    started = time.monotonic()
    failure = None
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed > assay_timeout_sec:
            failure = f"assay worker exceeded {assay_timeout_sec}s wall-clock limit"
            process.kill()
            break
        try:
            rss_kb = int(
                subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(process.pid)],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except (ValueError, subprocess.CalledProcessError):
            rss_kb = 0
        if rss_kb > max_rss_mb * 1024:
            failure = f"assay worker exceeded {max_rss_mb} MiB resident-memory ceiling"
            process.kill()
            break
        time.sleep(1)
    return_code = process.wait()
    if return_code and failure is None:
        failure = f"assay worker exited with status {return_code}"
    return failure, None


def _write_worker_failure(dms_id: str, reason: str) -> None:
    path = METRIC_CACHE / f"{dms_id}.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    features = {item["key"]: None for item in FEATURES}
    features.update(existing.get("features", {}))
    failures = existing.get("feature_failures", {})
    for key in features:
        if key not in existing.get("features", {}):
            failures[key] = reason
    existing.update({"id": dms_id, "features": features, "feature_failures": failures, "status": "failed"})
    _atomic_json(path, existing)


def _run(args: argparse.Namespace) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    METRIC_CACHE.mkdir(parents=True, exist_ok=True)
    _fetch_sources()
    reference = _read_reference()
    with zipfile.ZipFile(DATA_ARCHIVE) as archive:
        members = _assay_members(archive)
        summaries, totals = _scan_eligibility(archive, members, reference)
    model_meta, performance = _read_performance()
    if set(performance) != set(reference):
        raise ValueError("Current per-DMS score IDs do not match v1.3 reference IDs exactly")
    eligible_ids = sorted(dms_id for dms_id, row in summaries.items() if row["eligible"])
    if len(eligible_ids) != 36:
        raise ValueError(f"Expected 36 eligible assays at mean >1.5; found {len(eligible_ids)}")
    selected_ids = eligible_ids if args.limit is None else eligible_ids[: args.limit]
    if args.only:
        unknown = set(args.only) - set(eligible_ids)
        if unknown:
            raise ValueError(f"Requested IDs are not eligible: {sorted(unknown)}")
        selected_ids = list(args.only)
    output_ids = eligible_ids if args.keep_full_cohort else selected_ids
    if args.assemble_only:
        _write_artifacts(
            summaries, totals, reference, model_meta, performance, output_ids
        )
        print(f"Assembled cached records into {OUTPUT}", flush=True)
        return
    if args.retry_failed:
        coupled = (
            {"epistasis.magnitude", "epistasis.sign", "epistasis.reciprocal_sign"},
            {"gamma", "gamma_star"},
        )
        for dms_id in selected_ids:
            cache_path = METRIC_CACHE / f"{dms_id}.json"
            if not cache_path.exists():
                continue
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("status") == "failed":
                print(f"[{dms_id}] retrying failed worker/build attempt", flush=True)
                cache_path.unlink()
                continue
            failures = cached.get("feature_failures", {})
            retry = {
                key for key, reason in failures.items()
                if any(token in str(reason).lower() for token in (
                    "timeouterror", "memoryerror", "memory limit", "resource ceiling",
                    "resident-memory ceiling", "rss ceiling",
                ))
            }
            for group in coupled:
                if retry & group:
                    retry.update(group)
            if retry:
                print(f"[{dms_id}] retrying timed-out/resource-limited metrics: {sorted(retry)}", flush=True)
                for key in retry:
                    cached.get("features", {}).pop(key, None)
                    failures.pop(key, None)
                _atomic_json(cache_path, cached)
    print(
        f"ProteinGym {PROTEINGYM_VERSION}: {len(eligible_ids)} eligible assays; "
        f"processing {len(selected_ids)} with at most two workers; {len(model_meta)} zero-shot models",
        flush=True,
    )
    pending = []
    for dms_id in selected_ids:
        cache_path = METRIC_CACHE / f"{dms_id}.json"
        is_complete = False
        if cache_path.exists() and not args.force:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                is_complete = (
                    cached.get("status") == "complete"
                    and set(cached.get("features", {})) >= {feature["key"] for feature in FEATURES}
                )
            except (OSError, json.JSONDecodeError):
                pass
        if not is_complete:
            pending.append(dms_id)
    if args.force:
        for dms_id in selected_ids:
            path = METRIC_CACHE / f"{dms_id}.json"
            if path.exists():
                path.unlink()

    print(f"Running at most two assay workers concurrently; each is capped at {args.max_rss_mb} MiB RSS.", flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_ids = {
            executor.submit(
                _monitor_worker,
                dms_id,
                args.metric_timeout_sec,
                args.assay_timeout_sec,
                args.max_rss_mb,
                args.motif_cut_prob,
            ): dms_id
            for dms_id in pending
        }
        done = 0
        for future in as_completed(future_ids):
            dms_id = future_ids[future]
            done += 1
            failure, _ = future.result()
            if failure:
                print(f"[{dms_id}] {failure}", flush=True)
                _write_worker_failure(dms_id, failure)
            print(f"Completed worker {done}/{len(pending)}: {dms_id}", flush=True)
            _write_artifacts(
                summaries, totals, reference, model_meta, performance, output_ids
            )
    _write_artifacts(summaries, totals, reference, model_meta, performance, output_ids)
    print(f"\nWrote {OUTPUT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="process the first N eligible assays (partial preview)")
    parser.add_argument("--only", action="append", help="process one eligible DMS_id; repeatable")
    parser.add_argument("--keep-full-cohort", action="store_true", help="keep all 36 records in public outputs during targeted retry runs")
    parser.add_argument("--force", action="store_true", help="recompute assay feature caches")
    parser.add_argument("--assemble-only", action="store_true", help="rewrite JSON and reports from verified cached results without launching workers")
    parser.add_argument("--retry-failed", action="store_true", help="retry failed builds and metrics with timeout/resource-limit reasons only")
    parser.add_argument("--metric-timeout-sec", type=int, default=DEFAULT_METRIC_TIMEOUT_SEC)
    parser.add_argument("--assay-timeout-sec", type=int, default=DEFAULT_ASSAY_TIMEOUT_SEC)
    parser.add_argument("--max-rss-mb", type=int, default=DEFAULT_MAX_RSS_MB)
    parser.add_argument("--motif-cut-prob", type=float, help="explicit reproducible classify_epistasis sampling probability for bounded retry runs")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--worker-only", help="run one guarded assay worker and update only its local metric cache")
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker, args.metric_timeout_sec, args.motif_cut_prob)
    elif args.worker_only:
        CACHE.mkdir(parents=True, exist_ok=True)
        METRIC_CACHE.mkdir(parents=True, exist_ok=True)
        failure, _ = _monitor_worker(
            args.worker_only,
            args.metric_timeout_sec,
            args.assay_timeout_sec,
            args.max_rss_mb,
            args.motif_cut_prob,
        )
        if failure:
            print(f"[{args.worker_only}] {failure}", flush=True)
            _write_worker_failure(args.worker_only, failure)
    else:
        _run(args)


if __name__ == "__main__":
    main()
