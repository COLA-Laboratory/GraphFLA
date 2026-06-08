#!/usr/bin/env python
"""Batch benchmark for GraphFLA trajectory algorithms.

Probes runtime + memory of the three trajectory "optimizers" on three large
landscapes (GB1 protein, Papkou-RAW DNA, CR9114 boolean antibody), running
many independent initialisations:

  * local_search   (best- and first-improvement)
  * hill_climb     (best- and first-improvement)
  * random_walk

local_search / hill_climb are very cheap per call, so they run at large N
(default 10k and 100k) to stay above timing noise; random_walk is ~100x more
expensive per call so it runs at smaller N (default 1k and 10k).

Design (mirrors bench/runtime_bench.py):
  * Threads pinned to 1 BEFORE importing numpy -> single-process, comparable.
  * Each landscape is BUILT ONCE (construction time excluded) and reused.
  * Start nodes are a fixed, seeded sample (we store the seed, not the nodes).
  * Each workload runs through a `prep()` step (timed separately as "build")
    then the per-node loop ("loop"); end-to-end "total" = build + loop. For the
    optimised code `prep` builds a once-per-batch SearchCache (fitness + CSR
    adjacency); for the original code it is a no-op, so build~0.
  * RNG reseeded before every pass -> identical work per rep, reproducible
    first-improvement / random_walk trajectories.
  * BEHAVIOUR GUARD: every trajectory is folded into ONE streaming sha1 digest
    per (dataset, workload, N) -- trajectories are never stored, so disk/RAM
    stay tiny. `--check REF.json` asserts the digests equal a saved baseline.

Run:
  python bench/algo_bench.py [--only NAME] [--n-climb 10000,100000]
                             [--n-walk 1000,10000] [--reps 5] [--out P]
  python bench/algo_bench.py --check /tmp/graphfla_algo_original.json
"""
import os

# Pin threads BEFORE numpy import (campaign forbids parallel/threaded opt).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import gc
import hashlib
import json
import random
import signal
import statistics
import struct
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil

# Prefer THIS checkout's graphfla over any editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphfla.landscape.protein import ProteinLandscape
from graphfla.landscape.dna import DNALandscape
from graphfla.landscape.boolean import BooleanLandscape
from graphfla.algorithms import local_search, hill_climb, random_walk

try:  # present only after the optimisation lands; benchmark works either way
    from graphfla.algorithms import SearchCache
except ImportError:
    SearchCache = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_BIO = os.path.join(REPO, "data", "BioSequence")

CLS = {"protein": ProteinLandscape, "dna": DNALandscape, "boolean": BooleanLandscape}

DATASETS = [
    dict(name="GB1_protein_large", type="protein",
         path=os.path.join(DATA_BIO, "Wu2016_GB1.csv"), xcol="sequences", fcol="fitness"),
    dict(name="Papkou_dna_large", type="dna",
         path=os.path.join(DATA_BIO, "Papkou2023_DHFR_RAW.csv"), xcol="seq", fcol="fitness"),
    dict(name="CR9114h1_boolean_large", type="boolean",
         path=os.path.join(DATA_BIO, "Phillips2021_CR9114_h1.csv"), xcol="sequences", fcol="fitness"),
]

# Fixed seeds -> reproducible start nodes and reproducible stochastic trajectories.
NODE_SEED = 12345
GLOBAL_SEED = 20240608    # seeds the global `random` used by first-improvement
WALK_SEED = 99            # master RNG that hands a per-walk seed to random_walk
WALK_LEN = 100

# Switch: True once graphfla.algorithms exposes the cache-based API.
USE_CACHE = SearchCache is not None

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
        mean=round(statistics.mean(vals), 5),
        std=round(statistics.pstdev(vals), 5) if len(vals) > 1 else 0.0,
        min=round(min(vals), 5),
        max=round(max(vals), 5),
    )


def load_xy(ds):
    # bitstrings / sequences must stay str, not parse to int
    df = pd.read_csv(ds["path"], dtype={ds["xcol"]: str})
    df = df.dropna(subset=[ds["fcol"], ds["xcol"]]).reset_index(drop=True)
    return df[ds["xcol"]], df[ds["fcol"]]


def pick_nodes(n_configs, N):
    rng = np.random.default_rng(NODE_SEED)
    if N <= n_configs:
        return rng.choice(n_configs, size=N, replace=False).astype(np.int64)
    return rng.integers(0, n_configs, size=N, dtype=np.int64)


# ----------------------------- prep (per-batch) ----------------------------
# `prep(landscape, tier)` returns the object the workloads operate on, and is
# timed as the one-time "build" cost. With the cache API it builds a fresh
# SearchCache and materialises the adjacency the tier needs; without it, it
# just returns the bare graph (build ~0).

def make_prep(landscape, tier):
    g = landscape.graph
    if USE_CACHE:
        return lambda: SearchCache(g)   # builds the hoisted fitness vector (O(V))
    return lambda: g


# ----------------------------- workloads -----------------------------------
# Each takes (obj, nodes, method, fold). `obj` is a graph (original API) or a
# SearchCache (optimised API). When `fold` is a hash we are in the IDENTITY
# pass (capture full trajectories); when None we are TIMING the hot path.

def wl_local_search(obj, nodes, method, fold=None):
    if fold is not None:
        out = np.empty(len(nodes), dtype=np.int64)
        if USE_CACHE:
            for k in range(len(nodes)):
                nxt = local_search(obj, int(nodes[k]), method)
                out[k] = -1 if nxt is None else nxt
        else:
            for k in range(len(nodes)):
                nxt = local_search(obj, int(nodes[k]), "delta_fit", method)
                out[k] = -1 if nxt is None else nxt
        fold.update(out.tobytes())
    elif USE_CACHE:
        for k in range(len(nodes)):
            local_search(obj, int(nodes[k]), method)
    else:
        for k in range(len(nodes)):
            local_search(obj, int(nodes[k]), "delta_fit", method)


def wl_hill_climb(obj, nodes, method, fold=None):
    if fold is not None:
        for k in range(len(nodes)):
            if USE_CACHE:
                lo, steps, trace = hill_climb(obj, int(nodes[k]), return_trace=True, search_method=method)
            else:
                lo, steps, trace = hill_climb(obj, int(nodes[k]), "delta_fit", return_trace=True, search_method=method)
            fold.update(struct.pack("<qq", int(lo), int(steps)))
            fold.update(np.asarray(trace, dtype=np.int64).tobytes())
    elif USE_CACHE:
        for k in range(len(nodes)):
            hill_climb(obj, int(nodes[k]), search_method=method)
    else:
        for k in range(len(nodes)):
            hill_climb(obj, int(nodes[k]), "delta_fit", search_method=method)


def wl_random_walk(obj, nodes, method, fold=None):
    master = random.Random(WALK_SEED)
    if fold is not None:
        for k in range(len(nodes)):
            ws = master.getrandbits(32)
            arr = random_walk(obj, int(nodes[k]), "fitness", WALK_LEN, seed=ws)
            col = np.asarray(arr[:, 1], dtype=np.int64)
            fold.update(struct.pack("<q", col.size))
            fold.update(col.tobytes())
    else:
        for k in range(len(nodes)):
            ws = master.getrandbits(32)
            random_walk(obj, int(nodes[k]), "fitness", WALK_LEN, seed=ws)


WORKLOADS = [
    ("local_search_best", wl_local_search, "best-improvement", "climb"),
    ("local_search_first", wl_local_search, "first-improvement", "climb"),
    ("hill_climb_best", wl_hill_climb, "best-improvement", "climb"),
    ("hill_climb_first", wl_hill_climb, "first-improvement", "climb"),
    ("random_walk", wl_random_walk, None, "walk"),
]


def _reseed():
    random.seed(GLOBAL_SEED)


def run_workload(prep, nodes, fn, method, reps):
    # identity pass (full trajectories -> one streaming digest)
    obj = prep()
    _reseed()
    h = hashlib.sha1()
    fn(obj, nodes, method, fold=h)
    traj_hash = h.hexdigest()[:16]
    del obj
    # timing passes: prep (build) + hot loop, split; reseed each rep
    rss0 = _rss_mb()
    sampler = _Sampler()
    sampler.start()
    build_t, loop_t = [], []
    for _ in range(reps):
        _reseed()
        gc.collect()
        t0 = time.perf_counter()
        obj = prep()
        t1 = time.perf_counter()
        fn(obj, nodes, method, fold=None)
        t2 = time.perf_counter()
        build_t.append(t1 - t0)
        loop_t.append(t2 - t1)
        del obj
    peak = sampler.stop()
    return dict(traj_hash=traj_hash,
                build_s=_agg(build_t), loop_s=_agg(loop_t),
                total_s=_agg([b + l for b, l in zip(build_t, loop_t)]),
                peak_rss_mb=round(peak, 1), delta_mb=round(peak - rss0, 1))


def bench_dataset(ds, n_climb, n_walk, reps, kill_s=1800.0):
    print(f"\n{'=' * 70}\n  {ds['name']}  [{ds['type']}]\n{'=' * 70}", flush=True)
    X, f = load_xy(ds)
    gc.collect()
    t0 = time.perf_counter()
    ls = CLS[ds["type"]](maximize=True).build_from_data(X, f, verbose=False)
    n = ls.graph.vcount()
    print(f"  built: {n:,} nodes / {ls.graph.ecount():,} edges "
          f"in {time.perf_counter() - t0:.1f}s   (cache_api={USE_CACHE})", flush=True)
    del X, f
    gc.collect()

    out = {"n_configs": n, "n_edges": ls.graph.ecount(), "workloads": {}}
    signal.setitimer(signal.ITIMER_REAL, kill_s)
    try:
        for wname, fn, method, tier in WORKLOADS:
            n_list = n_walk if tier == "walk" else n_climb
            wout = {}
            for N in n_list:
                nodes = pick_nodes(n, N)
                prep = make_prep(ls, tier)
                r = run_workload(prep, nodes, fn, method, reps)
                wout[str(N)] = r
                tot, bld, lp = r["total_s"], r["build_s"], r["loop_s"]
                print(f"  {wname:<20} N={N:>7}  total {tot['mean']:.4f}s "
                      f"(build {bld['mean']:.4f} + loop {lp['mean']:.4f}) "
                      f"±{tot['std']:.4f}  Δ{r['delta_mb']:.0f}MB  #{r['traj_hash']}",
                      flush=True)
            out["workloads"][wname] = wout
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    del ls
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    ap.add_argument("--n-climb", default="10000,100000", dest="n_climb")
    ap.add_argument("--n-walk", default="1000,10000", dest="n_walk")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="")
    ap.add_argument("--check", default="", help="reference JSON; assert traj hashes match")
    args = ap.parse_args()

    signal.signal(signal.SIGALRM, _on_alarm)
    n_climb = [int(x) for x in args.n_climb.split(",") if x]
    n_walk = [int(x) for x in args.n_walk.split(",") if x]
    datasets = DATASETS
    if args.only:
        wanted = set(args.only.split(","))
        datasets = [d for d in DATASETS if d["name"] in wanted]

    print(f"[algo_bench:{args.label}] {len(datasets)} dataset(s) "
          f"climb-N={n_climb} walk-N={n_walk} x {args.reps} reps "
          f"(threads=1, cache_api={USE_CACHE})", flush=True)
    results = {ds["name"]: bench_dataset(ds, n_climb, n_walk, args.reps) for ds in datasets}
    out = dict(label=args.label, n_climb=n_climb, n_walk=n_walk,
               reps=args.reps, cache_api=USE_CACHE, datasets=results)

    path = args.out or f"/tmp/graphfla_algo_{args.label}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nWROTE " + path, flush=True)

    if args.check:
        with open(args.check) as fh:
            ref = json.load(fh)["datasets"]
        ok = True
        mism = []
        for dname, d in results.items():
            for wname, wls in d["workloads"].items():
                for N, r in wls.items():
                    try:
                        rh = ref[dname]["workloads"][wname][N]["traj_hash"]
                    except KeyError:
                        continue
                    if rh != r["traj_hash"]:
                        ok = False
                        mism.append(f"{dname} {wname} N={N}: {rh} != {r['traj_hash']}")
        print("\n=== TRAJECTORY IDENTITY CHECK ===")
        if ok:
            print("PASS: all trajectory hashes match baseline.")
        else:
            print("FAIL: trajectory divergence:")
            for m in mism:
                print("  " + m)
            sys.exit(1)


if __name__ == "__main__":
    main()
