"""Re-execute dataset notebooks with this checkout, one bounded fresh kernel each."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "tutorials/datasets"
REPORTS = ROOT / "docs/.build/notebook-execution"
sys.path.insert(0, str(ROOT / "docs/_support"))
from notebook_sources import source_hash


def execute(path):
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="graphfla-kernel-") as directory:
        spec = Path(directory) / "tutorial"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "GraphFLA tutorial",
                    "language": "python",
                }
            )
        )
        manager = KernelManager(
            kernel_name="tutorial",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]),
        )
        try:
            NotebookClient(
                nb,
                km=manager,
                timeout=240,
                allow_errors=False,
                resources={"metadata": {"path": str(path.parent)}},
            ).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    nb.metadata.graphfla = {
        "source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "executed_source_sha256": source_hash(nb),
        "executed_code_cells": sum(c.cell_type == "code" for c in nb.cells),
    }
    nbformat.write(nb, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks", nargs="*", help="Filenames in tutorials/datasets; default all"
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        execute(args.worker)
        return
    import psutil

    REPORTS.mkdir(parents=True, exist_ok=True)
    paths = (
        [NOTEBOOKS / name for name in args.notebooks]
        if args.notebooks
        else sorted(NOTEBOOKS.glob("*.ipynb"))
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT), env.get("PYTHONPATH")])
    )
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env[key] = "1"
    results = []
    for path in paths:
        print("Running", path.name, flush=True)
        start, peak, failure = time.monotonic(), 0, None
        with (REPORTS / (path.stem + ".log")).open("w") as log:
            process = subprocess.Popen(
                [sys.executable, __file__, "--worker", str(path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            while process.poll() is None:
                tree = []
                try:
                    parent = psutil.Process(process.pid)
                    tree = [parent] + parent.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in tree if p.is_running())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    rss = 0
                peak = max(peak, rss)
                if rss > 5 * 1024**3 or time.monotonic() - start > 360:
                    failure = "memory or time limit exceeded"
                    for child in reversed(tree):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    process.kill()
                    break
                time.sleep(0.5)
            process.wait()
        result = {
            "notebook": path.name,
            "exit_code": process.returncode,
            "wall_seconds": time.monotonic() - start,
            "peak_rss_bytes": peak,
            "guard_failure": failure,
        }
        results.append(result)
        (REPORTS / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if process.returncode or failure:
            raise SystemExit((REPORTS / (path.stem + ".log")).read_text()[-6000:])


if __name__ == "__main__":
    main()
