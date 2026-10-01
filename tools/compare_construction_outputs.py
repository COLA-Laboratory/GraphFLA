"""Compare complete construction outputs from two source trees, in fresh processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark_round import ROOT, THREAD_ENV, source_digest


def worker(source_root, dataset):
    sys.path.insert(0, str(source_root))
    sys.path.append(str(ROOT))
    import numpy as np
    import pandas as pd
    from benchmarks._datasets import build_dataset

    landscape = build_dataset(dataset)
    graph = landscape.graph

    def array_hash(values):
        return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()

    def attributes_hash(attributes):
        digest = hashlib.sha256()
        for name in attributes.attributes():
            digest.update(name.encode())
            values = np.asarray(attributes[name])
            digest.update(str(values.dtype).encode())
            digest.update(pd.util.hash_array(values).tobytes())
        return digest.hexdigest()

    state = {
        name: getattr(landscape, name)
        for name in (
            "data_types",
            "config_dict",
            "n_vars",
            "lo_index",
            "go_index",
            "plateaus",
            "_neutral_neighbors",
        )
    }
    state_hash = hashlib.sha256(
        json.dumps(state, sort_keys=True, default=int).encode()
    ).hexdigest()
    return {
        "shape": landscape.shape,
        "variable_sites": landscape.n_vars,
        "edges_sha256": array_hash(np.asarray(graph.get_edgelist(), dtype=np.int64)),
        "vertex_attributes_sha256": attributes_hash(graph.vs),
        "edge_attributes_sha256": attributes_hash(graph.es),
        "configs_sha256": array_hash(landscape._configs_array),
        "metadata_sha256": state_hash,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.dataset)))
        return
    if args.before is None or args.output is None:
        parser.error("--before and --output are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(ROOT))
    from benchmarks._datasets import DATASETS

    output = {
        "before_source_sha256": source_digest(args.before),
        "after_source_sha256": source_digest(args.after),
        "datasets": {},
    }
    for dataset in DATASETS:
        signatures = []
        for source in (args.before, args.after):
            result = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    str(source),
                    "--dataset",
                    dataset,
                ],
                env={**os.environ, **THREAD_ENV, "PYTHONHASHSEED": "0"},
                capture_output=True,
                text=True,
                check=True,
                timeout=180,
            )
            signatures.append(json.loads(result.stdout))
        if signatures[0] != signatures[1]:
            raise ValueError(f"Construction outputs differ for {dataset}: {signatures}")
        output["datasets"][dataset] = signatures[0]
        print(f"{dataset}: identical", flush=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
