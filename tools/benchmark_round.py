"""Record an isolated construction round for comparison across implementations."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}


def source_digest(root):
    digest = hashlib.sha256()
    for path in sorted((root / "graphfla").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def worker(args):
    import gc
    import resource

    sys.path.insert(0, str(args.source_root))
    sys.path.append(str(ROOT))
    import pandas as pd
    from benchmarks._datasets import load_dataset

    cls, X, fitness, options = load_dataset(args.dataset)
    digest = hashlib.sha256()
    for value in (X, fitness):
        digest.update(
            pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes()
        )
    digest.update(json.dumps(options, sort_keys=True).encode())

    def build():
        return cls().build_from_data(X, fitness, verbose=False, **options)

    if args.mode == "time":
        for _ in range(args.warmups):
            build()
        samples, cpu = [], []
        for _ in range(args.repeat):
            gc.collect()
            start_cpu, start = time.process_time(), time.perf_counter()
            landscape = build()
            samples.append(time.perf_counter() - start)
            cpu.append(time.process_time() - start_cpu)
            shape, n_vars = landscape.shape, landscape.n_vars
            del landscape
        result = {"seconds": samples, "cpu_seconds": cpu}
    else:
        gc.collect()
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        landscape = build()
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1 if sys.platform == "darwin" else 1024
        shape, n_vars = landscape.shape, landscape.n_vars
        result = {
            "peak_rss_bytes": int(after * scale),
            "setup_peak_rss_bytes": int(before * scale),
        }
    result.update(
        input_sha256=digest.hexdigest(),
        shape=shape,
        variable_sites=n_vars,
        source_sha256=source_digest(args.source_root),
    )
    print(json.dumps(result, allow_nan=False))


def run(args):
    sys.path.insert(0, str(ROOT))
    from benchmarks._datasets import DATASETS

    datasets = args.datasets or DATASETS
    unknown = set(datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite a recorded round: {args.output}")
    env = {**os.environ, **THREAD_ENV, "PYTHONHASHSEED": "0"}
    source_hash = source_digest(args.source_root)
    packages = ["numpy", "pandas", "igraph", "scipy", "scikit-learn", "asv"]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    output = {
        "schema_version": 1,
        "label": args.label,
        "revision": revision,
        "source_root": str(args.source_root),
        "source_sha256": source_hash,
        "source_ref": args.source_ref,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "threads": THREAD_ENV,
            "packages": {p: importlib.metadata.version(p) for p in packages},
        },
        "protocol": {
            "processes": args.processes,
            "repeat": args.repeat,
            "warmups_per_process": args.warmups,
            "timeout_seconds": args.timeout,
            "memory": "Fresh-process peak RSS including imports, input preparation and one build",
        },
        "datasets": {},
    }
    for dataset in datasets:
        results = []
        for mode, count in (("time", args.processes), ("memory", args.processes)):
            for _ in range(count):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--source-root",
                    str(args.source_root),
                    "--dataset",
                    dataset,
                    "--mode",
                    mode,
                    "--repeat",
                    str(args.repeat),
                    "--warmups",
                    str(args.warmups),
                ]
                proc = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=args.timeout,
                )
                if proc.returncode:
                    raise RuntimeError(f"{dataset} {mode}: {proc.stderr}")
                value = json.loads(proc.stdout)
                if proc.stderr:
                    value["diagnostics"] = proc.stderr.strip()
                if value["source_sha256"] != source_hash:
                    raise RuntimeError("Source changed during the benchmark round")
                results.append(value)
        identities = {
            (r["input_sha256"], tuple(r["shape"]), r["variable_sites"]) for r in results
        }
        if len(identities) != 1:
            raise RuntimeError(f"Non-reproducible benchmark input or shape: {dataset}")
        seconds = [s for r in results for s in r.get("seconds", [])]
        median = statistics.median(seconds)
        output["datasets"][dataset] = {
            "input_sha256": results[0]["input_sha256"],
            "shape": results[0]["shape"],
            "variable_sites": results[0]["variable_sites"],
            "workers": results,
            "median_seconds": median,
            "mad_seconds": statistics.median(abs(s - median) for s in seconds),
            "median_peak_rss_bytes": statistics.median(
                r["peak_rss_bytes"] for r in results if "peak_rss_bytes" in r
            ),
        }
        print(f"{dataset}: {median:.4f} s; {results[0]['shape']}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(".partial.json")
        partial.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    args.output.with_suffix(".partial.json").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--source-ref", default="working-tree")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=["time", "memory"], default="time", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.processes < 1 or args.repeat < 1 or args.warmups < 0:
        parser.error(
            "processes and repeat must be positive; warmups cannot be negative"
        )
    if not args.worker and args.output is None:
        parser.error("--output is required")
    worker(args) if args.worker else run(args)


if __name__ == "__main__":
    main()
