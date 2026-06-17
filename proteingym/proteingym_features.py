"""Annotate GraphFLA landscape features for ProteinGym DMS substitution datasets.

This module turns each ProteinGym deep-mutational-scanning (DMS) substitution
assay into a GraphFLA fitness landscape and computes a panel of landscape
features (ruggedness, navigability, epistasis, ...). It then merges those
features with ProteinGym's zero-shot and supervised model-performance tables
into a single CSV (one row per DMS).

Landscape construction follows two conventions required for meaningful
epistasis analysis:

* Only the *mutated* positions are kept. Each genotype is the string of amino
  acids at the union of positions mutated anywhere in the assay; invariant
  positions are dropped.
* Assays whose mean number of mutations is <= ``--min-avg-mutations`` (default
  1.0 -- i.e. single-site saturation) are removed, because epistasis cannot be
  derived from them.

The same functions power both the public command-line interface here and the
Modal-based large-scale runner; keep this file free of any Modal dependency.

Example
-------
    python proteingym_features.py \
        --dms-dir   /path/to/ProteinGym/DMS_ProteinGym_substitutions \
        --reference /path/to/ProteinGym/reference_files/DMS_substitutions.csv \
        --zero-shot /path/to/DMS_substitutions_Spearman_DMS_level.csv \
        --supervised /path/to/DMS_substitutions_Spearman_DMS_level.csv \
        --output    proteingym_graphfla_table.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
_MUT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")

# ---------------------------------------------------------------------------
# Feature panel
# ---------------------------------------------------------------------------
# The whole landscape feature panel is produced by ``analysis.profile`` -- one
# call runs every whole-landscape metric and forwards ``n_jobs`` to those that
# accept it. ``feature_names`` reports the metric names, split by whether they
# use multiple cores (which the large-scale runner uses to schedule work).


def feature_names(which: str = "all") -> List[str]:
    """Panel metric names ('single', 'multi', or 'all', by core usage)."""
    import graphfla.analysis as A

    lm = A.list_metrics()
    if which == "multi":
        lm = lm[lm["n_jobs"]]
    elif which == "single":
        lm = lm[~lm["n_jobs"]]
    return list(lm.index)


# ---------------------------------------------------------------------------
# Parsing & landscape construction
# ---------------------------------------------------------------------------
def parse_mutations(mutant: str) -> List[Tuple[str, int, str]]:
    """Parse a ProteinGym mutant string like 'A1P:D2N' into (wt, pos, mut)."""
    subs = []
    for token in str(mutant).split(":"):
        token = token.strip()
        m = _MUT_RE.match(token)
        if not m:
            raise ValueError(f"unparseable mutation token: {token!r}")
        subs.append((m.group(1).upper(), int(m.group(2)), m.group(3).upper()))
    return subs


def infer_target_seq(dms_df: pd.DataFrame) -> Optional[str]:
    """Reconstruct the wild-type sequence without a reference file.

    Takes the first row that carries a ``mutated_sequence`` and reverts that
    row's own substitutions (the ``wt`` char of each ``wt{pos}mut`` token),
    recovering the full-length wild-type -- positions not mutated in that row are
    already wild-type. Returns ``None`` if no usable ``mutated_sequence`` exists,
    in which case the caller falls back to mutated-position parsing only.
    """
    if "mutated_sequence" not in dms_df.columns:
        return None
    usable = dms_df.dropna(subset=["mutated_sequence"])
    if usable.empty:
        return None
    row = usable.iloc[0]
    seq = list(str(row["mutated_sequence"]))
    for wt, pos, _mut in parse_mutations(row["mutant"]):
        if 1 <= pos <= len(seq):
            seq[pos - 1] = wt
    return "".join(seq)


def dataset_meta(dms_df: pd.DataFrame, target_seq: str) -> dict:
    """Compute mutation/sequence meta statistics directly from the DMS table."""
    counts = np.array([len(parse_mutations(m)) for m in dms_df["mutant"]])
    positions = sorted({p for m in dms_df["mutant"] for (_, p, _) in parse_mutations(m)})
    return {
        "seq_len": len(target_seq) if isinstance(target_seq, str) else np.nan,
        "n_mutated_positions": len(positions),
        "n_variants_raw": int(len(dms_df)),
        "avg_n_mutations": float(np.mean(counts)),
        "median_n_mutations": float(np.median(counts)),
        "max_n_mutations": int(np.max(counts)),
        "frac_single": float(np.mean(counts == 1)),
        "frac_multi": float(np.mean(counts > 1)),
        "fitness_mean": float(np.mean(dms_df["DMS_score"])),
        "fitness_std": float(np.std(dms_df["DMS_score"])),
        "fitness_min": float(np.min(dms_df["DMS_score"])),
        "fitness_max": float(np.max(dms_df["DMS_score"])),
    }


def reduce_to_mutated_positions(
    dms_df: pd.DataFrame, target_seq: str
) -> Tuple[List[str], np.ndarray, List[int], dict]:
    """Reduce each variant to the amino acids at the union of mutated positions.

    Returns (reduced_sequences, fitness, mutated_positions, info). Drops rows
    with missing scores, duplicate genotypes, or non-standard amino acids.
    """
    df = dms_df.dropna(subset=["DMS_score"]).copy()
    positions = sorted({p for m in df["mutant"] for (_, p, _) in parse_mutations(m)})
    info = {"n_dropped_nan": int(len(dms_df) - len(df))}

    have_seq = "mutated_sequence" in df.columns and df["mutated_sequence"].notna().all()
    L = len(target_seq) if isinstance(target_seq, str) else None

    def reduced_for(row) -> Optional[str]:
        seq = row.get("mutated_sequence") if have_seq else None
        if isinstance(seq, str) and (L is None or len(seq) == L):
            try:
                return "".join(seq[p - 1] for p in positions)
            except IndexError:
                return None
        # Fallback: apply this row's mutations to the wild-type sequence.
        if not isinstance(target_seq, str):
            return None
        chars = list(target_seq)
        for _wt, p, mut in parse_mutations(row["mutant"]):
            if 1 <= p <= len(chars):
                chars[p - 1] = mut
            else:
                return None
        return "".join(chars[p - 1] for p in positions)

    reduced = df.apply(reduced_for, axis=1)
    scores = df["DMS_score"].to_numpy(dtype=float)

    keep = reduced.notna().to_numpy()
    reduced = reduced[keep]
    scores = scores[keep]
    info["n_dropped_unmappable"] = int((~keep).sum())

    # Drop genotypes with non-standard amino acids (outside the 20-letter set).
    valid = np.array([set(s) <= PROTEIN_ALPHABET for s in reduced])
    info["n_dropped_nonstandard"] = int((~valid).sum())
    reduced = reduced[valid]
    scores = scores[valid]

    # Deduplicate identical reduced genotypes (keep first occurrence).
    seqs: List[str] = []
    keep_scores: List[float] = []
    seen = set()
    for s, sc in zip(reduced.tolist(), scores.tolist()):
        if s in seen:
            continue
        seen.add(s)
        seqs.append(s)
        keep_scores.append(sc)
    info["n_dropped_duplicate"] = int(len(reduced) - len(seqs))
    info["n_variants"] = len(seqs)
    return seqs, np.array(keep_scores, dtype=float), positions, info


def build_landscape(reduced_seqs: List[str], scores: np.ndarray, maximize: bool = True):
    """Build a ProteinLandscape over the reduced genotypes.

    Basins, accessible paths, distances and neighbour-fitness are computed
    lazily on first access by the analysis layer, so the build only constructs
    the graph -- no explicit ``calculate_*`` flags are needed.
    """
    from graphfla.landscape import ProteinLandscape

    ls = ProteinLandscape(maximize=maximize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls.build_from_data(list(reduced_seqs), list(scores), verbose=False)
    return ls


def reregister_sequence_type(ls) -> None:
    """No-op retained for backward compatibility (e.g. the Modal runner).

    Earlier landscapes keyed their sequence handler/generator under a
    process-specific type id that did not survive pickling, so callers had to
    re-register it after unpickling. The current landscape stores input handlers
    and neighbour generators as instance-level registries that pickle correctly
    -- a built landscape round-trips through ``pickle`` intact -- so no
    re-registration is needed. This shim keeps existing callers working.
    """
    return None


def landscape_meta(ls) -> dict:
    """Structural meta extracted from a built landscape."""
    go_fit = None
    if getattr(ls, "go", None) is not None:
        go_fit = float(ls.go.get("fitness")) if "fitness" in ls.go else None
    return {
        "n_configs": int(ls.n_configs) if ls.n_configs is not None else None,
        "n_edges": int(ls.graph.ecount()) if ls.graph is not None else None,
        "n_lo": int(ls.n_lo) if ls.n_lo is not None else None,
        "n_lo_members": int(ls.n_lo_members) if getattr(ls, "n_lo_members", None) is not None else None,
        "n_plateau": int(ls.n_plateau) if getattr(ls, "n_plateau", None) is not None else None,
        "go_fitness": go_fit,
    }


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def compute_features(
    ls,
    which: str = "all",
    n_jobs: int = 1,
    skip: Optional[set] = None,
    report: Optional[Callable[[str], None]] = None,
) -> Dict[str, float]:
    """Compute the landscape feature panel via :func:`graphfla.analysis.profile`.

    Returns a flat ``{column: value}`` mapping (structured metrics flatten to
    dotted columns, e.g. ``epistasis.magnitude``). A single failing metric is
    recorded as NaN, never aborting the panel. Exact motif census is forced
    (``sample_cut_prob=0``): on these sparse Hamming-1 DMS graphs it is fast and
    ESU sampling noticeably distorts the motif counts.

    The cheap single-core metrics are computed in one batch; each expensive
    multi-core metric (``gamma``, ``gamma_star``, ...) is then computed on its
    own so ``report`` -- an optional ``callable(stage_message)`` -- can announce
    the step currently in flight. Per-metric calls are free relative to the
    metric cost and give live, per-step progress on large assays.
    """
    import graphfla.analysis as A

    skip = set(skip or ())
    lm = A.list_metrics()
    is_multi = lm["n_jobs"]
    names = [m for m in lm.index if m not in skip]
    if which == "multi":
        names = [m for m in names if is_multi[m]]
    elif which == "single":
        names = [m for m in names if not is_multi[m]]
    single = [m for m in names if not is_multi[m]]
    multi = [m for m in names if is_multi[m]]

    params = {
        "classify_epistasis": {"sample_cut_prob": 0},
        "extradimensional_bypass": {"sample_cut_prob": 0},
    }
    out: Dict[str, float] = {}

    def _run(include: List[str], stage: str) -> None:
        if not include:
            return
        if report is not None:
            report(stage)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = A.profile(
                ls, n_jobs=n_jobs, params=params, include=include, on_error="warn",
            )
        out.update({k: (float(v) if pd.notna(v) else float("nan")) for k, v in s.items()})

    _run(single, f"computing {len(single)} core features")
    for m in multi:
        _run([m], f"computing {m} (slow)")
    return out


def flatten_results(results: Dict[str, float]) -> Dict[str, float]:
    """Pass-through: :func:`compute_features` already returns a flat column map."""
    return dict(results)


# ---------------------------------------------------------------------------
# End-to-end per-dataset processing (used by both CLI and Modal)
# ---------------------------------------------------------------------------
def process_dataset(
    dms_df: pd.DataFrame,
    target_seq: str,
    which: str = "all",
    n_jobs: int = 1,
    min_avg_mutations: float = 1.0,
    skip: Optional[set] = None,
    report: Optional[Callable[[str], None]] = None,
) -> dict:
    """Build + annotate one DMS dataset; returns a record with meta + features.

    The record's ``status`` is one of: 'ok', 'skipped_too_sparse',
    'too_few_variants', 'build_failed'. ``report`` -- an optional
    ``callable(stage_message)`` -- is invoked at each stage (scan / reduce /
    build / per-metric) so a caller can surface live progress.
    """
    def _say(stage: str) -> None:
        if report is not None:
            report(stage)

    _say("scanning variants")
    meta = dataset_meta(dms_df, target_seq)
    record = {"meta": meta}

    if meta["avg_n_mutations"] <= min_avg_mutations:
        record["status"] = "skipped_too_sparse"
        return record

    _say("reducing to mutated positions")
    seqs, scores, positions, info = reduce_to_mutated_positions(dms_df, target_seq)
    meta.update(info)
    if len(seqs) < 3:
        record["status"] = "too_few_variants"
        return record

    try:
        _say(f"building landscape · {len(seqs)} configs")
        ls = build_landscape(seqs, scores)
    except Exception as exc:  # noqa: BLE001
        record["status"] = "build_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    meta.update(landscape_meta(ls))
    record["features"] = compute_features(
        ls, which=which, n_jobs=n_jobs, skip=skip, report=report
    )
    record["status"] = "ok"
    return record


# ---------------------------------------------------------------------------
# ProteinGym I/O and table assembly
# ---------------------------------------------------------------------------
def load_reference(path: str) -> pd.DataFrame:
    """Load reference_files/DMS_substitutions.csv indexed by DMS_id."""
    ref = pd.read_csv(path)
    return ref.set_index("DMS_id")


def load_perf_table(path: str, prefix: str) -> pd.DataFrame:
    """Load a DMS-level performance CSV indexed by DMS id, prefixing columns."""
    df = pd.read_csv(path)
    id_col = next(
        (c for c in ("DMS_id", "DMS id", "Assay_id", "assay_id") if c in df.columns),
        df.columns[0],
    )
    df = df.set_index(id_col)
    df = df.select_dtypes(include=[np.number])
    df.columns = [f"{prefix}{c}" for c in df.columns]
    return df


def assemble_table(
    records: Dict[str, dict],
    zero_shot: Optional[pd.DataFrame] = None,
    supervised: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Assemble the merged feature + performance table from per-DMS records."""
    rows = []
    for dms_id, rec in records.items():
        if rec.get("status") != "ok":
            continue
        row = {"DMS_id": dms_id}
        row.update(rec.get("meta", {}))
        row.update(flatten_results(rec.get("features", {})))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).set_index("DMS_id")
    if zero_shot is not None:
        table = table.join(zero_shot, how="left")
    if supervised is not None:
        table = table.join(supervised, how="left")
    return table.reset_index()


# ---------------------------------------------------------------------------
# Command-line interface (simple, resumable loop; no Modal)
# ---------------------------------------------------------------------------
def _eligible_dms(dms_dir: str, reference: pd.DataFrame) -> List[str]:
    files = {f[:-4] for f in os.listdir(dms_dir) if f.endswith(".csv")}
    return sorted(files & set(reference.index.astype(str)))


def _report_dataset(console, i: int, n: int, dms_id: str, rec: dict,
                    dt: float, cached: bool) -> None:
    """Print a compact, styled per-dataset summary (renders above the bar).

    Three lines: a header (id, position in the queue, status, timing), the DMS
    data summary, and -- when a landscape was built -- cheap build-time landscape
    statistics. Uses parentheses (not brackets) around numeric ranges so the
    values are never mistaken for rich markup tags.
    """
    meta = rec.get("meta", {})
    status = rec.get("status", "?")
    tag = {
        "ok": "[green]✓ ok[/]",
        "skipped_too_sparse": "[yellow]⊘ too sparse[/]",
        "too_few_variants": "[yellow]⊘ too few variants[/]",
        "build_failed": "[red]✗ build failed[/]",
    }.get(status, status)
    timing = "[dim](cached)[/]" if cached else f"[dim]{dt:.1f}s[/]"
    console.print(
        f"[bold cyan]{dms_id}[/] [dim]· {i}/{n} · {n - i} left[/]  {tag}  {timing}"
    )

    if meta:
        dms = []
        if pd.notna(meta.get("n_variants_raw")):
            dms.append(f"{int(meta['n_variants_raw'])} variants")
        if pd.notna(meta.get("avg_n_mutations")):
            dms.append(f"⟨mut⟩={meta['avg_n_mutations']:.2f}")
        if pd.notna(meta.get("max_n_mutations")):
            dms.append(f"max={int(meta['max_n_mutations'])}")
        if pd.notna(meta.get("n_mutated_positions")):
            dms.append(f"{int(meta['n_mutated_positions'])} mut-pos")
        if pd.notna(meta.get("seq_len")):
            dms.append(f"len={int(meta['seq_len'])}")
        if pd.notna(meta.get("fitness_min")) and pd.notna(meta.get("fitness_max")):
            dms.append(f"fit∈({meta['fitness_min']:.2f}, {meta['fitness_max']:.2f})")
        if dms:
            console.print("    [dim]DMS      [/] " + " · ".join(dms))

    if status == "ok" and pd.notna(meta.get("n_configs")):
        ls = [f"{int(meta['n_configs'])} configs"]
        if pd.notna(meta.get("n_edges")):
            ls.append(f"{int(meta['n_edges'])} edges")
        if pd.notna(meta.get("n_lo")):
            ls.append(f"{int(meta['n_lo'])} optima")
        if pd.notna(meta.get("n_plateau")):
            ls.append(f"{int(meta['n_plateau'])} plateaus")
        if pd.notna(meta.get("go_fitness")):
            ls.append(f"GO={meta['go_fitness']:.3g}")
        console.print("    [dim]landscape[/] " + " · ".join(ls))
    elif status == "build_failed" and rec.get("error"):
        console.print(f"    [red]{rec['error']}[/]")


def run_cli(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,
        TimeElapsedColumn,
    )

    console = Console()
    reference = load_reference(args.reference) if args.reference else None
    os.makedirs(args.cache_dir, exist_ok=True)

    if reference is not None:
        dms_ids = _eligible_dms(args.dms_dir, reference)
        source = f"reference {os.path.basename(args.reference)}"
    else:
        dms_ids = sorted(f[:-4] for f in os.listdir(args.dms_dir) if f.endswith(".csv"))
        source = "all CSVs in --dms-dir (wild-type inferred; no --reference)"
    if args.limit:
        dms_ids = dms_ids[: args.limit]
    skip = {s.strip() for s in args.skip_features.split(",") if s.strip()}

    console.rule("[bold]ProteinGym × GraphFLA — landscape feature annotation[/]")
    console.print(
        f"datasets : [bold]{len(dms_ids)}[/]   source: {source}\n"
        f"cache    : [dim]{args.cache_dir}[/]   "
        f"filter   : drop assays with mean #mutations ≤ [bold]{args.min_avg_mutations}[/]"
        + (f"   skip: {sorted(skip)}" if skip else "")
    )

    records: Dict[str, dict] = {}
    tally: Dict[str, int] = {}
    with Progress(
        SpinnerColumn(),  # animates via the refresh thread -- stays "alive" during blocking work
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=12,
    ) as progress:
        task = progress.add_task("[cyan]starting…[/]", total=len(dms_ids))
        for i, dms_id in enumerate(dms_ids, 1):
            short = dms_id if len(dms_id) <= 28 else dms_id[:27] + "…"

            def report(stage: str, _s: str = short) -> None:
                progress.update(task, description=f"[cyan]{_s}[/] [dim]·[/] {stage}")

            report("queued")
            cache_path = os.path.join(args.cache_dir, f"{dms_id}.json")
            cached = os.path.exists(cache_path) and not args.force
            t0 = time.time()
            if cached:
                with open(cache_path) as fh:
                    rec = json.load(fh)
            else:
                report("reading CSV")
                dms_df = pd.read_csv(os.path.join(args.dms_dir, f"{dms_id}.csv"))
                target_seq = (
                    reference.loc[dms_id, "target_seq"]
                    if reference is not None else infer_target_seq(dms_df)
                )
                rec = process_dataset(
                    dms_df, target_seq, which="all", n_jobs=args.n_jobs,
                    min_avg_mutations=args.min_avg_mutations, skip=skip,
                    report=report,
                )
                with open(cache_path, "w") as fh:
                    json.dump(rec, fh, default=_json_default)
            records[dms_id] = rec
            status = rec.get("status", "build_failed")
            tally[status] = tally.get(status, 0) + 1
            _report_dataset(progress.console, i, len(dms_ids), dms_id, rec,
                            time.time() - t0, cached)
            progress.advance(task)

    zs = load_perf_table(args.zero_shot, "zs_") if args.zero_shot else None
    sup = load_perf_table(args.supervised, "sup_") if args.supervised else None
    table = assemble_table(records, zero_shot=zs, supervised=sup)
    table.to_csv(args.output, index=False)

    console.rule("[bold green]done[/]")
    console.print("   ·   ".join([
        f"[green]{tally.get('ok', 0)} ok[/]",
        f"[yellow]{tally.get('skipped_too_sparse', 0)} too-sparse[/]",
        f"[yellow]{tally.get('too_few_variants', 0)} too-few-variants[/]",
        f"[red]{tally.get('build_failed', 0)} build-failed[/]",
    ]))
    console.print(
        f"wrote [bold]{len(table)}[/] rows × [bold]{table.shape[1]}[/] columns "
        f"→ [bold]{args.output}[/]"
    )


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dms-dir", required=True, help="Folder of per-DMS substitution CSVs.")
    p.add_argument(
        "--reference", default=None,
        help="ProteinGym reference_files/DMS_substitutions.csv -- per-assay "
        "METADATA supplying the wild-type 'target_seq' (and the canonical list "
        "of assay ids). OPTIONAL: if omitted, the wild-type is reconstructed "
        "from each CSV's 'mutated_sequence' column and every CSV in --dms-dir is "
        "processed. This is NOT a model-performance file -- those are the "
        "separate --zero-shot / --supervised tables.",
    )
    p.add_argument("--zero-shot", default=None, help="Zero-shot DMS-level Spearman CSV.")
    p.add_argument("--supervised", default=None, help="Supervised DMS-level Spearman CSV.")
    p.add_argument("--output", default="proteingym_graphfla_table.csv", help="Merged output CSV.")
    p.add_argument(
        "--min-avg-mutations", type=float, default=1.0,
        help="Drop DMS assays whose MEAN number of mutations per variant is "
        "<= this value (too sparse to derive epistasis). Default 1.0 drops "
        "single-mutant saturation assays; raise it (e.g. 2.0) to require more "
        "multi-mutant coverage.",
    )
    p.add_argument("--cache-dir", default="proteingym_graphfla_cache", help="Per-DMS JSON cache (resumable).")
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU cores for multi-core features.")
    p.add_argument("--limit", type=int, default=0, help="Process at most this many datasets (0 = all).")
    p.add_argument(
        "--skip-features", default="",
        help="Comma-separated metric names to skip (e.g. 'gamma,gamma_star' "
        "which are the most expensive). See feature_names() / analysis.list_metrics().",
    )
    p.add_argument("--force", action="store_true", help="Recompute even if cached.")
    return p


if __name__ == "__main__":
    run_cli(build_arg_parser().parse_args())
