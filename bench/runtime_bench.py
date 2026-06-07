#!/usr/bin/env python
"""Local single-process construction benchmark for GraphFLA.

Modeled on the user's runtime.py demo (load CSV -> build_from_data) but keeps
the full measurement harness:

  * CONSTRUCTION ONLY (no analysis), all 8 datasets in ONE process.
  * 5 reps each (override with --reps). Each CSV is read once, reused per rep.
  * Per-module runtime from the always-on @timeit stdout + perf_counter total.
  * Peak / stable RSS via a measurement-only sampler thread (psutil). The
    construction code itself stays strictly single-threaded; threads are pinned
    to 1 so timings are clean and comparable.
  * Any build exceeding ~2x its baseline is KILLED (signal.SIGALRM) so a
    runaway variant fails fast.
  * Streams per-dataset / per-rep progress; writes aggregated JSON (--out).

Run:  python bench/runtime_bench.py [--only NAME[,NAME]] [--reps 5]
                                    [--out PATH] [--label baseline]

To benchmark a variant: apply its diff to THIS checkout (graphfla is editable-
installed here), run this script, then revert. Behaviour guard (pytest) is run
separately.
"""
import os

# Pin threads BEFORE importing numpy so every build is single-threaded and
# comparable (the campaign forbids parallel/threaded optimisation).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import gc
import io
import json
import re
import signal
import statistics
import threading
import time

import sys

import psutil
import pandas as pd

# Prefer THIS checkout's graphfla (e.g. an agent's worktree variant) over the
# editable-installed copy, so concurrent worktree benchmarks test their own code.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphfla.landscape.protein import ProteinLandscape
from graphfla.landscape.dna import DNALandscape
from graphfla.landscape.boolean import BooleanLandscape
from graphfla.landscape.ordinal import OrdinalLandscape

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_BIO = os.path.join(REPO, "data", "BioSequence")
DATA_WREOS = os.path.join(REPO, "data", "Materials", "WReOs")
HPO_LOCAL = os.path.join(REPO, "bench", "_localdata", "HPO_44136.csv")
DMS_DIR = "/Users/arwen/Documents/GitHub/proteingym_run/dms_data/DMS_ProteinGym_substitutions"

CLS = {"protein": ProteinLandscape, "dna": DNALandscape,
       "boolean": BooleanLandscape, "ordinal": OrdinalLandscape}

DATASETS = [
    dict(name="GB1_protein_large", type="protein",
         path=os.path.join(DATA_BIO, "Wu2016_GB1.csv"),
         xcols=["sequences"], fcol="fitness"),
    dict(name="TrpB3I_protein_small", type="protein",
         path=os.path.join(DATA_BIO, "Johnston2024_TrpB3I.csv"),
         xcols=["sequences"], fcol="fitness"),
    dict(name="Papkou_dna_large", type="dna",
         path=os.path.join(DATA_BIO, "Papkou2023_DHFR_RAW.csv"),
         xcols=["seq"], fcol="fitness"),
    dict(name="Westmann_dna_small", type="dna",
         path=os.path.join(DATA_BIO, "Westmann2024.csv"),
         xcols=["sequences"], fcol="fitness"),
    dict(name="CR9114h1_boolean_large", type="boolean",
         path=os.path.join(DATA_BIO, "Phillips2021_CR9114_h1.csv"),
         xcols=["sequences"], fcol="fitness"),
    dict(name="CR6261h1_boolean_small", type="boolean",
         path=os.path.join(DATA_BIO, "Phillips2021_CR6261_h1.csv"),
         xcols=["sequences"], fcol="fitness"),
    dict(name="WReOs_ordinal_small", type="ordinal",
         path=os.path.join(DATA_WREOS, "simplex.csv"),
         xcols=["W", "Re"], fcol="TOPSIS_Ci"),
    dict(name="HPO_ordinal_large", type="ordinal",
         path=HPO_LOCAL,
         xcols=["learning_rate", "subsample", "max_depth",
                "min_child_weight", "n_estimators"], fcol="mean_r2"),
    # --- SPARSE high-dim ProteinGym DMS (reduce-to-mutated-positions loader) ---
    # cover the strategies sparse data triggers: pairwise / broadcast / active.
    dict(name="UBE4B_sparse_S", type="protein", loader="proteingym",
         path=os.path.join(DMS_DIR, "UBE4B_HUMAN_Tsuboyama_2023_3L1X.csv")),
    dict(name="D7PM05_sparse_M", type="protein", loader="proteingym",
         path=os.path.join(DMS_DIR, "D7PM05_CLYGR_Somermeyer_2022.csv")),
    dict(name="GFP_sparse_L_broadcast", type="protein", loader="proteingym",
         path=os.path.join(DMS_DIR, "GFP_AEQVI_Sarkisyan_2016.csv")),
    dict(name="PHOT_sparse_L_active", type="protein", loader="proteingym",
         path=os.path.join(DMS_DIR, "PHOT_CHLRE_Chen_2023.csv")),
    dict(name="HIS7_sparse_XL_active", type="protein", loader="proteingym",
         path=os.path.join(DMS_DIR, "HIS7_YEAST_Pokusaeva_2019.csv")),
]
SPARSE = ["UBE4B_sparse_S", "D7PM05_sparse_M", "GFP_sparse_L_broadcast",
          "PHOT_sparse_L_active", "HIS7_sparse_XL_active"]

# Current-branch (post A1+A2) local baseline total_s per dataset -> per-rep kill
# timeout. Update when the branch baseline shifts after merging winners.
BASELINE_S = {
    "GB1_protein_large": 2.06, "TrpB3I_protein_small": 0.10,
    "Papkou_dna_large": 1.66, "Westmann_dna_small": 0.55,
    "CR9114h1_boolean_large": 0.29, "CR6261h1_boolean_small": 0.02,
    "WReOs_ordinal_small": 0.01, "HPO_ordinal_large": 1.38,
    # sparse ProteinGym (generous; on current code some hit the Python-loop slow path)
    "UBE4B_sparse_S": 3, "D7PM05_sparse_M": 90, "GFP_sparse_L_broadcast": 150,
    "PHOT_sparse_L_active": 150, "HIS7_sparse_XL_active": 600,
}

TIMEIT_RE = re.compile(r"Method (\w+) executed in ([\d.eE+-]+) seconds\.")
_PME = psutil.Process()


def _rss_mb():
    return _PME.memory_info().rss / 1024 ** 2


class _Sampler(threading.Thread):
    """Measurement-only: poll RSS every 5 ms, remember the peak."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = _PME.memory_info().rss
        self._stop = threading.Event()

    def run(self):
        ev = self._stop
        while not ev.is_set():
            try:
                r = _PME.memory_info().rss
                if r > self.peak:
                    self.peak = r
            except Exception:
                pass
            ev.wait(0.005)

    def stop(self):
        self._stop.set()
        self.join(timeout=1.0)
        return self.peak / 1024 ** 2


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return dict(
        mean=round(statistics.mean(vals), 4),
        std=round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        min=round(min(vals), 4),
        max=round(max(vals), 4),
        vals=[round(v, 4) for v in vals],
    )


def _section(ds, n_variants, extra):
    lines = [
        "",
        "┌" + "─" * 58 + "┐",
        f"  {ds['name']}  [{ds['type']}]",
        "└" + "─" * 58 + "┘",
        f"  Variants:  {n_variants:,}",
    ]
    for k, v in extra.items():
        lines.append(f"  {k}: {v}")
    print("\n".join(lines), flush=True)


_MUT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")
_PROT = set("ACDEFGHIKLMNPQRSTVWY")


def _load_proteingym(path):
    """Reduce a ProteinGym DMS assay to a sparse high-dim protein landscape.

    X = sub-sequence over the UNION of mutated positions; f = DMS_score; drop
    rows with NaN score / length mismatch / non-standard AA / duplicate reduced
    genotype (mirrors examples/proteingym/proteingym_features.
    reduce_to_mutated_positions via the mutated_sequence column).
    """
    df = pd.read_csv(path, usecols=["mutant", "mutated_sequence", "DMS_score"])
    df = df.dropna(subset=["DMS_score"]).reset_index(drop=True)
    positions = sorted({
        int(m.group(2))
        for s in df["mutant"].astype(str) for tok in s.split(":")
        for m in (_MUT_RE.match(tok.strip()),) if m
    })
    seqs = df["mutated_sequence"].to_numpy()
    scores = df["DMS_score"].to_numpy(dtype=float)
    L = len(seqs[0]) if len(seqs) else 0
    red, sc, seen = [], [], set()
    for s, score in zip(seqs, scores):
        if not isinstance(s, str) or len(s) != L:
            continue
        g = "".join(s[p - 1] for p in positions)
        if not (set(g) <= _PROT) or g in seen:
            continue
        seen.add(g)
        red.append(g)
        sc.append(score)
    return pd.Series(red), pd.Series(sc, dtype=float)


def load_xy(ds):
    if ds.get("loader") == "proteingym":
        return _load_proteingym(ds["path"])
    xcols = ds["xcols"]
    if ds["type"] == "ordinal":
        df = pd.read_csv(ds["path"])
    else:
        # bitstrings like "0000000001" must stay str, not parse to int
        df = pd.read_csv(ds["path"], dtype={xcols[0]: str})
    df = df.dropna(subset=[ds["fcol"]] + xcols).reset_index(drop=True)
    X = df[xcols] if ds["type"] == "ordinal" else df[xcols[0]]
    return X, df[ds["fcol"]]


def bench_dataset(ds, reps):
    base = BASELINE_S.get(ds["name"], 60.0)
    # ~3x baseline: headroom for contention when agents run concurrently; true
    # algorithmic blowups (the thing we want to kill fast) are >>3x.
    kill_s = max(10.0, 3.0 * base)
    X, f = load_xy(ds)
    cls = CLS[ds["type"]]
    dtypes = ds.get("data_types")
    strat = ds.get("strategy")
    _section(ds, len(X), {"kill/rep": f">{kill_s:.0f}s", "strategy": strat or "auto/default"})

    per_rep = []
    for rep in range(reps):
        gc.collect()
        Xr = X.copy()
        fr = f.copy()
        gc.collect()
        rss0 = _rss_mb()
        sampler = _Sampler()
        sampler.start()
        buf = io.StringIO()
        signal.setitimer(signal.ITIMER_REAL, kill_s)
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(buf):
                ls = cls(maximize=True).build_from_data(
                    Xr, fr, data_types=dtypes,
                    neighborhood_strategy=strat, verbose=False)
            total_s = time.perf_counter() - t0
            signal.setitimer(signal.ITIMER_REAL, 0)
        except _Timeout:
            signal.setitimer(signal.ITIMER_REAL, 0)
            sampler.stop()
            el = time.perf_counter() - t0
            print(f"  ✗ rep {rep + 1}/{reps}: TIMEOUT {el:.0f}s "
                  f"(>{kill_s:.0f}s ~2x baseline) — KILLED", flush=True)
            return dict(name=ds["name"], type=ds["type"],
                        error=f"TIMEOUT >{kill_s:.0f}s (>2x baseline {base}s)")
        except Exception as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            sampler.stop()
            print(f"  ✗ rep {rep + 1}/{reps}: FAILED {type(e).__name__}: {e}",
                  flush=True)
            return dict(name=ds["name"], type=ds["type"],
                        error=f"build failed: {type(e).__name__}: {e}")
        peak = sampler.stop()
        rss1 = _rss_mb()
        modules = {m.group(1): float(m.group(2))
                   for m in TIMEIT_RE.finditer(buf.getvalue())}
        try:
            n_configs = ls.graph.vcount()
            n_edges = ls.graph.ecount()
        except Exception:
            n_configs = getattr(ls, "_n_configs", None)
            n_edges = getattr(ls, "_n_edges", None)
        per_rep.append(dict(total_s=total_s, modules=modules,
                            n_configs=n_configs, n_edges=n_edges,
                            peak_rss_mb=round(peak, 1),
                            rss_postbuild_mb=round(rss1, 1),
                            construct_delta_mb=round(peak - rss0, 1)))
        print(f"  • rep {rep + 1}/{reps}: {total_s:.3f}s  peak {peak:.0f}MB  "
              f"Δ{peak - rss0:.0f}MB  edges {n_edges}", flush=True)
        del ls, Xr, fr
        gc.collect()
    del X, f
    gc.collect()

    mk = set()
    for r in per_rep:
        mk.update(r["modules"].keys())
    d = dict(
        name=ds["name"], type=ds["type"],
        n_configs=per_rep[-1]["n_configs"], n_edges=per_rep[-1]["n_edges"],
        total_s=_agg([r["total_s"] for r in per_rep]),
        peak_rss_mb=_agg([r["peak_rss_mb"] for r in per_rep]),
        construct_delta_mb=_agg([r["construct_delta_mb"] for r in per_rep]),
        rss_postbuild_mb=_agg([r["rss_postbuild_mb"] for r in per_rep]),
        modules={k: _agg([r["modules"].get(k) for r in per_rep])
                 for k in sorted(mk)})
    t = d["total_s"]
    print(f"  ✓ {ds['name']}: mean {t['mean']}s ±{t['std']} "
          f"(base {base}s)  peak {d['peak_rss_mb']['mean']}MB", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    signal.signal(signal.SIGALRM, _on_alarm)
    datasets = DATASETS
    if args.only:
        wanted = set(args.only.split(","))
        datasets = [d for d in DATASETS if d["name"] in wanted]

    print(f"[bench:{args.label}] {len(datasets)} dataset(s) x {args.reps} reps "
          f"(local, single process, threads=1)", flush=True)
    results = [bench_dataset(ds, args.reps) for ds in datasets]
    out = dict(label=args.label, reps=args.reps, datasets=results)

    path = args.out or f"/tmp/graphfla_bench_{args.label}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n=== BENCH_JSON_START ===")
    print(json.dumps(out, indent=2))
    print("=== BENCH_JSON_END ===")
    print("WROTE " + path, flush=True)


if __name__ == "__main__":
    main()
