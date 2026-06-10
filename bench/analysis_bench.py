#!/usr/bin/env python
"""Runtime benchmark for GraphFLA landscape-analysis functions.

Probes the wall-time of every analysis metric (EXCEPT the fitness-distribution
ones, which are thin wrappers over already-optimised scipy/numpy) across a tier
of landscapes that grows from tiny synthetic NK cubes up to the large empirical
assays, so we can find the hot functions and optimise them.

Design (mirrors bench/runtime_bench.py + bench/algo_bench.py):
  * Threads pinned to 1 BEFORE importing numpy -> single-process, comparable.
    The joblib/loky parallel metrics (gamma, idiosyncratic, mutation-effects)
    still fan out to `--n-jobs` worker PROCESSES; pinning only stops nested BLAS
    threads, which is what we want when probing single- vs multi-core.
  * Each landscape is BUILT ONCE (construction excluded) and reused.
  * BABY-SITTING: every (function, rep) runs under a per-rep SIGALRM kill so a
    single pathological call (exponential motif enumeration, a regression on a
    huge one-hot design) can never hang the run -- it is recorded as `timeout`
    and we move on.  Run small synthetic first, scale up via --datasets/--funcs.
  * Lazy landscape prerequisites (basins / dist_to_go / neighbor_fitness /
    accessible_paths) are first-class benchmarked items.  Dependent wrappers are
    timed with their prerequisite PRE-WARMED (so their row is the marginal cost);
    the prerequisite probes RESET the cache before EVERY rep so they are always
    measured cold and reproducibly.
  * Memory is not the target this round, but a 5 ms RSS sampler records peak so
    an optimisation that regresses memory is visible.

Run (safe first step -- tiny synthetic only):
    python bench/analysis_bench.py
Scale up explicitly:
    python bench/analysis_bench.py --datasets cr9114,hpo --funcs cheap,prereq,mid
    python bench/analysis_bench.py --datasets gb1 --funcs gamma,gidi --n-jobs 1,8
    python bench/analysis_bench.py --list          # show datasets / funcs / groups
"""
import os

# Pin nested-BLAS threads BEFORE numpy import (probe single- vs multi-core cleanly).
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
import sys
import threading
import time
import warnings

import numpy as np
import pandas as pd
import psutil

# Prefer THIS checkout's graphfla over any editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphfla.landscape.protein import ProteinLandscape
from graphfla.landscape.dna import DNALandscape
from graphfla.landscape.boolean import BooleanLandscape
from graphfla.landscape.ordinal import OrdinalLandscape
from graphfla.analysis import (
    lo_ratio, autocorrelation, r_s_ratio, gradient_intensity,
    fitness_distance_corr, fitness_flattening_index, basin_fit_corr,
    neighbor_fit_corr,
    global_optima_accessibility, local_optima_accessibility,
    mean_path_lengths_go, mean_dist_lo, mean_dist_go,
    neutrality, single_mutation_effects, all_mutation_effects,
    evol_enhance_mutations,
    classify_epistasis, idiosyncratic_index, global_idiosyncratic_index,
    diminishing_returns_index, increasing_costs_index,
    gamma_statistic, walsh_hadamard_coefficient,
    extradimensional_bypass_analysis, higher_order_epistasis,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_BIO = os.path.join(REPO, "data", "BioSequence")
HPO_LOCAL = os.path.join(REPO, "bench", "_localdata", "HPO_44136.csv")
DMS_DIR = "/Users/arwen/Documents/GitHub/proteingym_run/dms_data/DMS_ProteinGym_substitutions"

CLS = {"protein": ProteinLandscape, "dna": DNALandscape,
       "boolean": BooleanLandscape, "ordinal": OrdinalLandscape}

# Fixed seeds -> stable timings & reproducible stochastic metrics (motif sampling,
# random-walk autocorrelation).  Correctness itself is guarded by pytest, not here.
SEED = 20240608

# Landscapes, ordered small -> large for convenient staged baby-sitting.
DATASETS = {
    "nk10": dict(kind="nk", N=10, K=2, type="boolean"),    # 1,024 nodes
    "nk12": dict(kind="nk", N=12, K=2, type="boolean"),    # 4,096
    "nk14": dict(kind="nk", N=14, K=2, type="boolean"),    # 16,384
    "ube4b": dict(kind="pg", type="protein",
                  path=os.path.join(DMS_DIR, "UBE4B_HUMAN_Tsuboyama_2023_3L1X.csv")),
    "wreos": dict(kind="csv", type="ordinal",
                  path=os.path.join(REPO, "data", "Materials", "WReOs", "simplex.csv"),
                  xcols=["W", "Re"], fcol="TOPSIS_Ci"),
    "cr9114": dict(kind="csv", type="boolean",
                   path=os.path.join(DATA_BIO, "Phillips2021_CR9114_h1.csv"),
                   xcol="sequences", fcol="fitness"),
    "hpo": dict(kind="csv", type="ordinal", path=HPO_LOCAL,
                xcols=["learning_rate", "subsample", "max_depth",
                       "min_child_weight", "n_estimators"], fcol="mean_r2"),
    "gb1": dict(kind="csv", type="protein",
                path=os.path.join(DATA_BIO, "Wu2016_GB1.csv"),
                xcol="sequences", fcol="fitness"),
    "papkou": dict(kind="csv", type="dna",
                   path=os.path.join(DATA_BIO, "Papkou2023_DHFR_RAW.csv"),
                   xcol="seq", fcol="fitness"),
    "d7pm05": dict(kind="pg", type="protein",   # 234 positions -> highest-dimensional
                   path=os.path.join(DMS_DIR, "D7PM05_CLYGR_Somermeyer_2022.csv")),
}

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


class _Timeout(BaseException):
    """Subclasses BaseException so the metrics' own broad ``except Exception``
    handlers (classify_epistasis, extradimensional_bypass, mean_path_lengths)
    cannot swallow the kill — otherwise a per-rep timeout is silently ignored
    and the call runs to completion."""
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


@contextlib.contextmanager
def _quiet():
    """Silence the metrics' incidental warnings / progress prints during timing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            yield


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return dict(
        mean=round(statistics.mean(vals), 5),
        std=round(statistics.pstdev(vals), 5) if len(vals) > 1 else 0.0,
        min=round(min(vals), 5),
        max=round(max(vals), 5),
    )


def _summ(v):
    """Compact, JSON-safe summary of a metric's return (sanity, not correctness)."""
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            f = float(v)
            return round(f, 5) if f == f else "nan"
        if isinstance(v, dict):
            if v and all(isinstance(k, int) for k in v):   # WH: order -> {feat: coef}
                return {f"order_{k}": len(v[k]) for k in sorted(v)}
            return {str(k): _summ(x) for k, x in list(v.items())[:8]}
        if hasattr(v, "shape"):                            # ndarray / DataFrame
            return f"<{type(v).__name__}{tuple(v.shape)}>"
        if isinstance(v, (list, tuple)):
            return f"<{type(v).__name__}[{len(v)}]>"
        return str(v)[:40]
    except Exception:
        return "?"


# ----------------------------- NK synthetic --------------------------------

def _nk_boolean(N, K=2, seed=0):
    """Classic Kauffman NK model on the full boolean cube (2**N genotypes).

    Locus i interacts with the K adjacent loci (cyclic); each (locus, K+1-bit
    neighbourhood) draws a U(0,1) contribution; fitness = mean over loci. Tunable,
    realistic ruggedness so the epistasis/motif metrics see non-degenerate input.
    Bit p (0 = leftmost) of genotype g matches the format(g, '0Nb') string layout.
    """
    rng = np.random.default_rng(seed)
    tables = rng.random((N, 1 << (K + 1)))
    g = np.arange(1 << N, dtype=np.int64)
    bits = ((g[:, None] >> np.arange(N - 1, -1, -1)[None, :]) & 1).astype(np.int64)
    f = np.zeros(len(g))
    for i in range(N):
        key = np.zeros(len(g), dtype=np.int64)
        for b in range(K + 1):
            key = (key << 1) | bits[:, (i + b) % N]
        f += tables[i, key]
    f /= N
    X = pd.Series([format(int(v), f"0{N}b") for v in g])
    return X, pd.Series(f)


# ----------------------------- loaders -------------------------------------

_MUT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")
_PROT = set("ACDEFGHIKLMNPQRSTVWY")


def _load_proteingym(path):
    """Reduce a ProteinGym DMS assay to a sparse high-dim protein landscape.

    X = sub-sequence over the UNION of mutated positions; f = DMS_score; drop NaN
    score / length mismatch / non-standard AA / duplicate reduced genotype
    (mirrors bench/runtime_bench.py)."""
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
        gtype = "".join(s[p - 1] for p in positions)
        if not (set(gtype) <= _PROT) or gtype in seen:
            continue
        seen.add(gtype)
        red.append(gtype)
        sc.append(score)
    return pd.Series(red), pd.Series(sc, dtype=float)


def load_xy(ds):
    kind = ds["kind"]
    if kind == "nk":
        return _nk_boolean(ds["N"], ds["K"], seed=SEED)
    if kind == "pg":
        return _load_proteingym(ds["path"])
    if ds["type"] == "ordinal":
        xcols = ds["xcols"]
        df = pd.read_csv(ds["path"], usecols=[*xcols, ds["fcol"]])
        df = df.dropna(subset=[ds["fcol"], *xcols]).reset_index(drop=True)
        return df[xcols], df[ds["fcol"]]
    xcol = ds["xcol"]
    df = pd.read_csv(ds["path"], dtype={xcol: str})
    df = df.dropna(subset=[ds["fcol"], xcol]).reset_index(drop=True)
    return df[xcol], df[ds["fcol"]]


# ----------------------- lazy-prerequisite control -------------------------
# (vertex attrs, edge attrs, calculated-flag) each lazy property populates.
_PREREQ = {
    "basins": (["size_basin_greedy", "basin_index"], [], "_basin_calculated"),
    "accessible_paths": (["size_basin_accessible"], [], "_path_calculated"),
    "dist_to_go": (["dist_go"], [], "_distance_calculated"),
    "neighbor_fitness": (["mean_neighbor_fit"], ["delta_mean_neighbor_fit"],
                         "_neighbor_fit_calculated"),
}


def reset_prereq(ls, name):
    """Clear a lazy prerequisite so the next access recomputes it (cold)."""
    vattrs, eattrs, flag = _PREREQ[name]
    for a in vattrs:
        try:
            if a in ls.graph.vs.attributes():
                del ls.graph.vs[a]
        except Exception:
            pass
    for a in eattrs:
        try:
            if a in ls.graph.es.attributes():
                del ls.graph.es[a]
        except Exception:
            pass
    setattr(ls, flag, False)


def warm_prereq(ls, name):
    getattr(ls, name)   # property access computes + caches


# ----------------------------- per-dataset ctx -----------------------------

def make_ctx(ls):
    """Representative args (mutation / position / a local optimum) + an encoded-
    width estimate used to guard the memory-explosive regression metrics."""
    cols = list(ls.data_types.keys())
    data = ls.get_data()
    pos = cols[0]
    vc = data[pos].value_counts()
    A = vc.index[0]
    B = vc.index[1] if len(vc) > 1 else vc.index[0]
    lo0 = ls.lo_index[0] if getattr(ls, "lo_index", None) else int(ls.go_index)
    width = 0
    for c in cols:
        if ls.data_types[c] == "boolean":
            width += 1
        else:
            width += max(0, int(data[c].nunique()) - 1)
    return dict(pos=pos, mutation=(A, pos, B), lo0=int(lo0),
                go=int(ls.go_index), width=width, n=ls.n_configs)


# ----------------------------- function registry ---------------------------
# body(ls, ctx, nj) -> metric value.  warm: prereqs computed once (untimed).
# cold: prereqs reset before EVERY rep (measured).  parallel: honour --n-jobs.
# guard(ctx) -> reason string to skip (memory blow-up), or None.

def _guard_order2(max_cells):
    def g(ctx):
        w = ctx["width"]
        cells = (w * (w - 1) // 2) * ctx["n"]
        return f"wide(order2_cells~{cells:.0e})" if cells > max_cells else None
    return g


def _guard_linear(max_cells):
    def g(ctx):
        cells = ctx["width"] * ctx["n"]
        return f"wide(cells~{cells:.0e})" if cells > max_cells else None
    return g


def _make_funcs(max_cells):
    return [
        # ---- cheap structural / correlation wrappers ----
        dict(key="lo_ratio", group="cheap",
             body=lambda ls, c, nj: lo_ratio(ls)),
        dict(key="gradient_intensity", group="cheap",
             body=lambda ls, c, nj: gradient_intensity(ls)),
        dict(key="global_optima_accessibility", group="cheap",
             body=lambda ls, c, nj: global_optima_accessibility(ls)),
        dict(key="local_optima_accessibility", group="cheap",
             body=lambda ls, c, nj: local_optima_accessibility(ls, c["lo0"])),
        dict(key="mean_path_lengths_go", group="cheap",
             body=lambda ls, c, nj: mean_path_lengths_go(ls)),
        dict(key="mean_dist_lo", group="cheap",
             body=lambda ls, c, nj: mean_dist_lo(ls, c["lo0"])),
        dict(key="fitness_distance_corr", group="cheap", warm=["dist_to_go"],
             body=lambda ls, c, nj: fitness_distance_corr(ls)),
        dict(key="basin_fit_corr", group="cheap", warm=["basins"],
             body=lambda ls, c, nj: basin_fit_corr(ls)),
        dict(key="neighbor_fit_corr", group="cheap", warm=["neighbor_fitness"],
             body=lambda ls, c, nj: neighbor_fit_corr(ls)),
        dict(key="evol_enhance_mutations", group="cheap", warm=["neighbor_fitness"],
             body=lambda ls, c, nj: evol_enhance_mutations(ls)),
        dict(key="mean_dist_go", group="cheap", cold=["dist_to_go"],
             body=lambda ls, c, nj: mean_dist_go(ls)),
        # ---- heavy lazy prerequisites (measured cold, every rep) ----
        dict(key="prereq_basins", group="prereq", cold=["basins"],
             body=lambda ls, c, nj: ls.basins),
        dict(key="prereq_dist_to_go", group="prereq", cold=["dist_to_go"],
             body=lambda ls, c, nj: ls.dist_to_go),
        dict(key="prereq_neighbor_fitness", group="prereq", cold=["neighbor_fitness"],
             body=lambda ls, c, nj: ls.neighbor_fitness),
        dict(key="prereq_accessible_paths", group="prereq", cold=["accessible_paths"],
             body=lambda ls, c, nj: ls.accessible_paths),
        # ---- mid-cost single-thread metrics ----
        dict(key="diminishing_returns_index", group="mid",
             body=lambda ls, c, nj: diminishing_returns_index(ls)),
        dict(key="increasing_costs_index", group="mid",
             body=lambda ls, c, nj: increasing_costs_index(ls)),
        dict(key="neutrality", group="mid",
             body=lambda ls, c, nj: neutrality(ls)),
        dict(key="autocorrelation", group="mid",
             body=lambda ls, c, nj: autocorrelation(ls, seed=SEED)),
        dict(key="r_s_ratio", group="mid", guard=_guard_linear(max_cells),
             body=lambda ls, c, nj: r_s_ratio(ls)),
        dict(key="fitness_flattening_index", group="mid",
             body=lambda ls, c, nj: fitness_flattening_index(ls)),
        dict(key="idiosyncratic_one", group="mid",
             body=lambda ls, c, nj: idiosyncratic_index(ls, c["mutation"])),
        dict(key="higher_order_epistasis", group="mid", guard=_guard_order2(max_cells),
             body=lambda ls, c, nj: higher_order_epistasis(ls, order=2)),
        dict(key="walsh_hadamard_coefficient", group="mid", guard=_guard_order2(max_cells),
             body=lambda ls, c, nj: walsh_hadamard_coefficient(ls, max_order=2)),
        # ---- mutation-effects (joblib-parallel over pairs/positions) ----
        dict(key="single_mutation_effects", group="mid", parallel=True,
             body=lambda ls, c, nj: single_mutation_effects(ls, c["pos"], n_jobs=nj)),
        dict(key="all_mutation_effects", group="mid", parallel=True,
             body=lambda ls, c, nj: all_mutation_effects(ls, n_jobs=nj)),
        # ---- motif metrics (approximate to avoid exponential enumeration) ----
        dict(key="classify_epistasis", group="motif",
             body=lambda ls, c, nj: classify_epistasis(
                 ls, approximate=True, sample_cut_prob=APPROX_CUT, seed=SEED)),
        dict(key="extradimensional_bypass", group="motif",
             body=lambda ls, c, nj: extradimensional_bypass_analysis(
                 ls, approximate=True, sample_cut_prob=APPROX_CUT, seed=SEED)),
        # ---- the user-flagged multi-core epistasis metrics (sweep n_jobs) ----
        dict(key="gamma", group="epi", parallel=True,
             body=lambda ls, c, nj: gamma_statistic(ls, n_jobs=nj)),
        dict(key="gidi", group="epi", parallel=True,
             body=lambda ls, c, nj: global_idiosyncratic_index(ls, n_jobs=nj)),
    ]


APPROX_CUT = 0.5   # set from CLI in main()


def bench_func(ls, ctx, f, reps, timeout_s, nj):
    guard = f.get("guard")
    if guard is not None:
        reason = guard(ctx)
        if reason:
            return dict(status="skip:" + reason, time_s=None, value=None,
                        peak_rss_mb=None, delta_mb=None, n_jobs=nj)
    for w in f.get("warm", []):
        with _quiet():
            warm_prereq(ls, w)
    cold = f.get("cold", [])
    times, status, val = [], "ok", None
    sampler = _Sampler()
    rss0 = _rss_mb()
    sampler.start()
    for _ in range(reps):
        for cname in cold:
            reset_prereq(ls, cname)
        gc.collect()
        t0 = time.perf_counter()
        # Repeating interval (re-fires every timeout_s) as a backstop in case one
        # alarm lands in a spot where it cannot propagate immediately.
        signal.setitimer(signal.ITIMER_REAL, timeout_s, timeout_s)
        try:
            with _quiet():
                val = f["body"](ls, ctx, nj)
            times.append(time.perf_counter() - t0)
        except _Timeout:
            status = "timeout"
            break
        except Exception as e:
            status = "error:%s:%s" % (type(e).__name__, str(e)[:120])
            break
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    peak = sampler.stop()
    return dict(status=status, time_s=_agg(times), value=_summ(val),
                peak_rss_mb=round(peak, 1), delta_mb=round(peak - rss0, 1),
                n_jobs=nj)


def bench_dataset(dname, funcs, n_jobs_list, reps, timeout_s):
    ds = DATASETS[dname]
    print(f"\n{'=' * 74}\n  {dname}  [{ds['type']}]\n{'=' * 74}", flush=True)
    X, f = load_xy(ds)
    gc.collect()
    t0 = time.perf_counter()
    ls = CLS[ds["type"]](maximize=True).build_from_data(X, f, verbose=False)
    build_s = time.perf_counter() - t0
    n, e = ls.graph.vcount(), ls.graph.ecount()
    print(f"  built {n:,} nodes / {e:,} edges in {build_s:.2f}s", flush=True)
    del X, f
    gc.collect()
    ctx = make_ctx(ls)

    out = {"n_configs": n, "n_edges": e, "n_vars": ls.n_vars,
           "width": ctx["width"], "build_s": round(build_s, 3), "funcs": {}}
    for fn in funcs:
        njs = n_jobs_list if fn.get("parallel") else [None]
        for nj in njs:
            r = bench_func(ls, ctx, fn, reps, timeout_s, nj)
            key = fn["key"] + (f"@{nj}" if nj is not None else "")
            out["funcs"][key] = r
            t = r["time_s"]
            tstr = f"{t['mean']:.4f}±{t['std']:.4f}s" if t else "--"
            dmb = r.get("delta_mb")
            dstr = f"peakΔ{dmb:.0f}MB" if dmb is not None else ""
            print(f"  {key:<34} {r['status']:<10} {tstr:>16}  "
                  f"{dstr:>10}  ={r['value']}", flush=True)
    del ls
    gc.collect()
    return out


def main():
    global APPROX_CUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="nk10,nk12,nk14",
                    help="comma names or 'all' (default: tiny synthetic NK)")
    ap.add_argument("--funcs", default="all",
                    help="comma func keys, group names (cheap/prereq/mid/motif/epi), or 'all'")
    ap.add_argument("--n-jobs", default="1", dest="n_jobs",
                    help="comma list for parallel metrics, e.g. 1,8 (cap 8)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=120.0, help="per-rep kill (s)")
    ap.add_argument("--approx-cut", type=float, default=0.5, dest="approx_cut",
                    help="sample_cut_prob for motif metrics")
    ap.add_argument("--max-cells", type=float, default=5e8, dest="max_cells",
                    help="skip regression metrics whose design exceeds this many cells")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--out", default="")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    APPROX_CUT = args.approx_cut
    all_funcs = _make_funcs(args.max_cells)
    groups = sorted({fn["group"] for fn in all_funcs})

    if args.list:
        print("datasets:", ", ".join(DATASETS))
        print("groups  :", ", ".join(groups))
        print("funcs   :", ", ".join(fn["key"] for fn in all_funcs))
        return

    signal.signal(signal.SIGALRM, _on_alarm)
    dnames = list(DATASETS) if args.datasets == "all" else args.datasets.split(",")
    n_jobs_list = [min(8, int(x)) for x in args.n_jobs.split(",") if x]

    if args.funcs == "all":
        funcs = all_funcs
    else:
        want = set(args.funcs.split(","))
        funcs = [fn for fn in all_funcs if fn["key"] in want or fn["group"] in want]
    if not funcs:
        print("no functions selected", file=sys.stderr)
        sys.exit(2)

    print(f"[analysis_bench:{args.label}] datasets={dnames} "
          f"funcs={len(funcs)} n_jobs={n_jobs_list} reps={args.reps} "
          f"timeout={args.timeout}s cut={APPROX_CUT}", flush=True)

    results = {}
    for dname in dnames:
        if dname not in DATASETS:
            print(f"  ! unknown dataset {dname}, skipping", flush=True)
            continue
        results[dname] = bench_dataset(dname, funcs, n_jobs_list, args.reps, args.timeout)

    out = dict(label=args.label, n_jobs=n_jobs_list, reps=args.reps,
               timeout_s=args.timeout, approx_cut=APPROX_CUT, datasets=results)
    path = args.out or f"/tmp/graphfla_analysis_{args.label}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nWROTE " + path, flush=True)


if __name__ == "__main__":
    main()
