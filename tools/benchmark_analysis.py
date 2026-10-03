"""Run exactly one analysis metric in bounded fresh processes.

Example: python tools/benchmark_analysis.py --metric ee --output run.json
No construction, unrelated metric, or literature reproduction is timed.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools._resource_guard import ResourceLimitError, run_guarded  # noqa: E402

THREADS = dict.fromkeys(
    [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ],
    "1",
)
SNAPSHOT_METRICS = {"ee", "gamma", "gamma_star", "r_s_ratio", "walsh_hadamard"}


def worker(args):
    import numpy as np
    import pandas as pd
    import scipy
    import igraph
    import sklearn
    import resource
    from benchmarks.analysis._workloads import build_case
    from benchmarks.analysis import prepare_call
    from graphfla.analysis import robustness as ee

    module = None
    if args.baseline_kernel:
        spec = importlib.util.spec_from_file_location(
            "ee_baseline"
            if args.metric == "ee"
            else "graphfla.analysis._r_s_baseline"
            if args.metric == "r_s_ratio"
            else "graphfla.analysis.epistasis._gamma_baseline",
            args.baseline_kernel,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if args.metric == "ee":
            ee._landscape_ee_statistics = module._landscape_ee_statistics
    landscape = build_case(args.case)
    calls = (
        {
            "fraction": lambda: ee.evolvability_enhancing_fraction(landscape),
            "effects": lambda: ee.evolvability_effects(landscape),
        }
        if args.metric == "ee"
        else {args.metric: prepare_call(landscape, args.metric)}
    )
    if module is not None and args.metric in {"gamma", "gamma_star"}:
        calls = {args.metric: lambda: getattr(module, args.metric)(landscape, n_jobs=1)}
    if module is not None and args.metric == "r_s_ratio":
        calls = {args.metric: lambda: module.r_s_ratio(landscape)}
    if args.metric == "walsh_hadamard":
        from graphfla.analysis import walsh_hadamard
        from benchmarks.analysis.walsh_hadamard import fit_options

        options = fit_options(args.walsh_method)
        if module is not None:
            calls = {
                args.metric: lambda: module.walsh_hadamard(
                    landscape, max_order=2, max_cells=1e6
                )
            }
        else:
            calls = {args.metric: lambda: walsh_hadamard(landscape, **options)}
    results = {}
    for name, call in calls.items():
        call()  # Untimed warmup; construction is also outside the timer.
        samples = []
        for _ in range(args.repeats):
            gc.collect()
            start = time.perf_counter()
            value = call()
            samples.append(time.perf_counter() - start)
            del value
        results[name] = samples
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    if args.snapshot and args.metric == "ee":
        table = ee.evolvability_effects(landscape)
        arrays = {}
        for col in table:
            if col == "is_ee":
                arrays[col] = (
                    table[col].astype("Int8").fillna(-1).to_numpy(dtype=np.int8)
                )
            elif table[col].dtype == object:
                arrays[col] = table[col].to_numpy(dtype=str)
            else:
                arrays[col] = table[col].to_numpy()
        arrays["fraction"] = np.asarray(ee.evolvability_enhancing_fraction(landscape))
        np.savez_compressed(args.snapshot, **arrays)
    elif args.snapshot and args.metric in {"gamma", "gamma_star", "r_s_ratio"}:
        np.savez_compressed(
            args.snapshot, **{args.metric: np.asarray(calls[args.metric]())}
        )
    elif args.snapshot and args.metric == "walsh_hadamard":
        table = calls[args.metric]()
        if module is not None and landscape.kind != "boolean":
            # Legacy labels are artificial factor codes. Decode with the exact
            # input factorization before comparing all corrected labels/values.
            data = landscape.get_data()[list(landscape.data_types)]
            labels = [pd.unique(data[c]) for c in data]

            def decode(term):
                if term == "WT":
                    return term
                parts = []
                for mutation in term.split("-"):
                    source, pos, target = mutation.split("_")
                    states = labels[int(pos) - 1]
                    parts.append(
                        f"{states[ord(source) - 49]}_{pos}_{states[ord(target) - 49]}"
                    )
                return "-".join(parts)

            table["term"] = table.term.map(decode)
        table = table.sort_values(["order", "term"])
        np.savez_compressed(
            args.snapshot,
            order=table.order.to_numpy(),
            positions=np.array([repr(x) for x in table.positions]),
            term=table.term.to_numpy(dtype=str),
            coefficient=table.coefficient.to_numpy(),
        )
    input_digest = hashlib.sha256(landscape.get_data().to_json(orient="split").encode())
    input_digest.update(np.asarray(landscape.graph.vs["fitness"]).tobytes())
    input_digest.update(repr(landscape.graph.get_edgelist()).encode())
    input_digest.update(repr(getattr(landscape, "_neutral_neighbors", None)).encode())
    return {
        "samples_seconds": results,
        "peak_rss_bytes": rss,
        "input_sha256": input_digest.hexdigest(),
        "vertices": landscape.n_configs,
        "edges": landscape.n_edges,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "igraph": igraph.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }


def compare_snapshots(baseline, candidate):
    import numpy as np

    checks = {}
    with (
        np.load(baseline, allow_pickle=False) as old,
        np.load(candidate, allow_pickle=False) as new,
    ):
        assert set(old.files) == set(new.files)
        for key in old.files:
            assert old[key].shape == new[key].shape, f"Output shape changed: {key}"
            assert old[key].dtype == new[key].dtype, f"Output dtype changed: {key}"
            if old[key].dtype.kind == "f":
                np.testing.assert_allclose(
                    new[key],
                    old[key],
                    rtol=1e-10,
                    atol=2e-12,
                    equal_nan=True,
                    err_msg=key,
                )
            else:
                np.testing.assert_array_equal(new[key], old[key], err_msg=key)
            checks[key] = "equivalent"
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--memory-limit-mib", type=int, default=1024)
    parser.add_argument("--compare", type=Path)
    parser.add_argument(
        "--baseline-kernel",
        type=Path,
        help="Trusted local metric module for a supported snapshot metric; never downloads code",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--walsh-method", choices=["ols", "lasso", "lasso_cv"], default="ols"
    )
    args = parser.parse_args()
    from benchmarks.analysis import METRIC_MODULES
    from benchmarks.analysis._workloads import cases_for_metric

    if args.metric != "ee" and args.metric not in METRIC_MODULES:
        parser.error("Choose 'ee' or one public metric: " + ", ".join(METRIC_MODULES))
    if (
        not 1 <= args.processes <= 5
        or not 1 <= args.repeats <= 10
        or not 0 < args.timeout <= 60
        or not 64 <= args.memory_limit_mib <= 4096
    ):
        parser.error(
            "Limits: 1..5 processes, 1..10 repeats, 0 < timeout <= 60 seconds, "
            "64..4096 MiB process-tree RSS"
        )
    if args.compare and args.metric not in SNAPSHOT_METRICS:
        parser.error(
            "Output-equivalence comparison requires a supported snapshot metric"
        )
    if args.baseline_kernel and args.metric not in SNAPSHOT_METRICS:
        parser.error("--baseline-kernel requires a supported snapshot metric")
    if (
        args.baseline_kernel
        and args.metric == "walsh_hadamard"
        and args.walsh_method != "ols"
    ):
        parser.error("The legacy Walsh baseline supports only OLS")
    if args.worker:
        if args.case not in cases_for_metric(args.metric):
            parser.error("Worker case is not registered for the selected metric")
        print(json.dumps(worker(args), allow_nan=False))
        return
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    partial = output.with_suffix(".partial.json")
    snapshots = output.with_suffix(".snapshots")
    if output.exists() or partial.exists() or snapshots.exists():
        parser.error("Refusing to overwrite a benchmark run")
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots.mkdir()
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    source = args.baseline_kernel or REPO / (
        "graphfla/analysis/epistasis/gamma.py"
        if args.metric in {"gamma", "gamma_star"}
        else "graphfla/analysis/epistasis/walsh_hadamard.py"
        if args.metric == "walsh_hadamard"
        else "graphfla/analysis/_roughness.py"
        if args.metric == "r_s_ratio"
        else "graphfla/analysis/_evolvability.py"
    )
    protocol = [
        Path(__file__).resolve(),
        REPO / "benchmarks/analysis/_workloads.py",
        REPO / "benchmarks/analysis/_shared.py",
        REPO / "tools/_resource_guard.py",
    ]
    if args.metric == "walsh_hadamard":
        protocol.append(REPO / "benchmarks/analysis/walsh_hadamard.py")
    protocol_hashes = {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in protocol
    }
    implementation = {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((REPO / "graphfla").rglob("*.py"))
    }
    report = {
        "schema_version": 1,
        "metric": args.metric,
        "walsh_method": args.walsh_method if args.metric == "walsh_hadamard" else None,
        "python": sys.version,
        "platform": platform.platform(),
        "host_id": hashlib.sha256(platform.node().encode()).hexdigest(),
        "thread_environment": THREADS,
        "processes": args.processes,
        "repeats": args.repeats,
        "timeout_seconds_per_process": args.timeout,
        "memory_limit_bytes": args.memory_limit_mib * 1024**2,
        "kernel_sha256": (
            hashlib.sha256(source.read_bytes()).hexdigest()
            if args.metric in SNAPSHOT_METRICS
            else None
        ),
        "protocol_sha256": protocol_hashes,
        "source_sha256": implementation,
        "workloads": {},
    }
    cases = cases_for_metric(args.metric)
    if baseline:
        for key in [
            "python",
            "platform",
            "host_id",
            "thread_environment",
            "metric",
            "walsh_method",
            "protocol_sha256",
            "processes",
            "repeats",
            "timeout_seconds_per_process",
            "memory_limit_bytes",
        ]:
            if baseline[key] != report[key]:
                raise ValueError(
                    f"Cannot compare different protocols/environments: {key}"
                )
    partial.write_text(json.dumps(report, indent=2) + "\n")
    for case in cases:
        runs = []
        snapshot = snapshots / f"{case}.npz"
        for index in range(args.processes):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--metric",
                args.metric,
                "--case",
                case,
                "--repeats",
                str(args.repeats),
            ]
            if index == 0 and args.metric in SNAPSHOT_METRICS:
                cmd.extend(["--snapshot", str(snapshot)])
            if args.metric == "walsh_hadamard":
                cmd.extend(["--walsh-method", args.walsh_method])
            if args.baseline_kernel:
                cmd.extend(["--baseline-kernel", str(args.baseline_kernel.resolve())])
            try:
                done = run_guarded(
                    cmd,
                    cwd=REPO,
                    timeout=args.timeout,
                    memory_bytes=args.memory_limit_mib * 1024**2,
                    env={**os.environ, **THREADS, "PYTHONHASHSEED": "0"},
                )
            except (
                subprocess.CalledProcessError,
                ResourceLimitError,
                KeyboardInterrupt,
            ) as exc:
                report["failure"] = {
                    "case": case,
                    "process": index,
                    "error": f"{type(exc).__name__}: {exc}",
                    "stderr": str(getattr(exc, "stderr", "")),
                }
                partial.write_text(json.dumps(report, indent=2) + "\n")
                raise RuntimeError(f"Benchmark failed: {case}\n{exc}") from exc
            run = json.loads(done.stdout)
            run["monitored_tree_peak_rss_bytes"] = done.monitored_peak_rss_bytes
            if run["peak_rss_bytes"] > args.memory_limit_mib * 1024**2:
                report["failure"] = {
                    "case": case,
                    "process": index,
                    "error": "Worker high-water RSS exceeded the limit",
                }
                partial.write_text(json.dumps(report, indent=2) + "\n")
                raise ResourceLimitError(report["failure"]["error"])
            runs.append(run)
        row = {"runs": runs, "measurements": {}}
        for name in runs[0]["samples_seconds"]:
            medians = [statistics.median(r["samples_seconds"][name]) for r in runs]
            median = statistics.median(medians)
            row["measurements"][name] = {
                "process_medians_seconds": medians,
                "median_seconds": median,
                "mad_seconds": statistics.median(abs(x - median) for x in medians),
            }
        if args.metric in SNAPSHOT_METRICS:
            row["snapshot"] = str(snapshot.relative_to(output.parent))
            row["snapshot_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        if baseline:
            previous = baseline["workloads"][case]
            for run in runs:
                for key in ["vertices", "edges", "versions", "input_sha256"]:
                    assert run[key] == previous["runs"][0][key], key
            old_snapshot = args.compare.resolve().parent / previous["snapshot"]
            assert (
                hashlib.sha256(old_snapshot.read_bytes()).hexdigest()
                == previous["snapshot_sha256"]
            )
            row["output_checks"] = compare_snapshots(old_snapshot, snapshot)
            for name, measurement in row["measurements"].items():
                old = previous["measurements"][name]
                measurement["speedup"] = (
                    old["median_seconds"] / measurement["median_seconds"]
                )
                delta = old["median_seconds"] - measurement["median_seconds"]
                gate = max(
                    0.05 * old["median_seconds"],
                    3 * (old["mad_seconds"] + measurement["mad_seconds"]),
                )
                measurement["noise_gate_seconds"] = gate
                measurement["decision"] = (
                    "within_noise"
                    if abs(delta) <= gate
                    else "improved"
                    if delta > 0
                    else "regressed"
                )
        report["workloads"][case] = row
        partial.write_text(json.dumps(report, indent=2) + "\n")
        print(
            case,
            {k: round(v["median_seconds"], 6) for k, v in row["measurements"].items()},
            flush=True,
        )
    partial.rename(output)


if __name__ == "__main__":
    main()
