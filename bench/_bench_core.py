#!/usr/bin/env python
"""Single-build benchmark worker for GraphFLA landscape construction.

Runs ONE landscape build in an isolated process so peak memory
(``getrusage``) is attributable to that build alone. Captures:

  * total wall-clock for ``build_from_data`` (perf_counter)
  * per-module timing parsed from the always-on ``@timeit`` stdout
  * RSS at import / after data load / after build  (psutil)
  * process peak RSS via ``resource.getrusage`` (ru_maxrss)
  * resulting n_configs / n_edges / resolved strategy

Prints exactly one JSON line prefixed with ``REPJSON ``.

This script is intentionally self-contained (no bench package import) so it
can be launched as ``python _bench_core.py ...`` inside the Modal container.
NOT committed behaviour-affecting code: pure measurement.
"""
import argparse
import contextlib
import gc
import io
import json
import os
import re
import resource
import sys
import time

# Pin threads BEFORE numpy import so all builds are single-threaded and
# comparable (the campaign forbids parallel/threaded optimisation).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import psutil  # noqa: E402
    _PROC = psutil.Process()

    def _rss_mb():
        return _PROC.memory_info().rss / 1024 ** 2
except Exception:  # pragma: no cover - psutil should be present
    def _rss_mb():
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2


_TIMEIT_RE = re.compile(r"Method (\w+) executed in ([\d.eE+-]+) seconds\.")


def _peak_rss_mb():
    # ru_maxrss is KiB on Linux, bytes on macOS. Assume Linux (Modal).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _load_landscape_class(cls_name):
    from graphfla.landscape.protein import ProteinLandscape
    from graphfla.landscape.dna import DNALandscape
    from graphfla.landscape.boolean import BooleanLandscape
    from graphfla.landscape.ordinal import OrdinalLandscape
    return {
        "protein": ProteinLandscape,
        "dna": DNALandscape,
        "boolean": BooleanLandscape,
        "ordinal": OrdinalLandscape,
    }[cls_name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--type", required=True,
                    choices=["protein", "dna", "boolean", "ordinal"])
    ap.add_argument("--path", required=True)
    ap.add_argument("--xcols", required=True, help="comma-separated column(s)")
    ap.add_argument("--fcol", required=True)
    ap.add_argument("--data-types", default="", help="json dict or empty")
    ap.add_argument("--strategy", default="", help="force strategy or empty=auto")
    ap.add_argument("--maximize", default="1")
    ap.add_argument("--rep", type=int, default=0)
    args = ap.parse_args()

    rss_after_import = _rss_mb()

    xcols = args.xcols.split(",")
    if args.type == "ordinal":
        df = pd.read_csv(args.path)
    else:
        # Read sequence/bitstring column as str: pandas would otherwise parse
        # all-numeric bitstrings (e.g. "0000000001") as ints, corrupting them.
        df = pd.read_csv(args.path, dtype={xcols[0]: str})
    # Drop rows with missing fitness / features (unmeasured genotypes). Standard
    # preprocessing, applied identically to baseline and every variant.
    df = df.dropna(subset=[args.fcol] + xcols).reset_index(drop=True)
    X = df[xcols] if args.type == "ordinal" else df[xcols[0]]
    f = df[args.fcol]
    data_types = json.loads(args.data_types) if args.data_types else None
    maximize = args.maximize == "1"
    strategy = args.strategy or None

    cls = _load_landscape_class(args.type)
    ls = cls(maximize=maximize)

    gc.collect()
    rss_after_load = _rss_mb()

    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        ls.build_from_data(
            X, f,
            data_types=data_types,
            neighborhood_strategy=strategy,
            verbose=False,
        )
    total_s = time.perf_counter() - t0

    rss_after_build = _rss_mb()
    peak_mb = _peak_rss_mb()

    modules = {}
    for m in _TIMEIT_RE.finditer(buf.getvalue()):
        modules[m.group(1)] = float(m.group(2))

    try:
        n_configs = ls.graph.vcount()
        n_edges = ls.graph.ecount()
    except Exception:
        n_configs = getattr(ls, "_n_configs", None)
        n_edges = getattr(ls, "_n_edges", None)

    rec = {
        "name": args.name,
        "type": args.type,
        "rep": args.rep,
        "total_s": total_s,
        "modules": modules,
        "n_configs": n_configs,
        "n_edges": n_edges,
        "rss_import_mb": round(rss_after_import, 1),
        "rss_load_mb": round(rss_after_load, 1),
        "rss_postbuild_mb": round(rss_after_build, 1),
        "peak_rss_mb": round(peak_mb, 1),
        "construct_delta_mb": round(peak_mb - rss_after_load, 1),
    }
    print("REPJSON " + json.dumps(rec))


if __name__ == "__main__":
    main()
