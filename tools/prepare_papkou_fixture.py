"""Extract the two construction fixtures from Papkou's published archive."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path


SOURCES = {
    "in_data/fitness_data_wt.rds": "fe7289c2bee25a1fb659cdda53264c0953124d9ddaeb1ac1eb893b1b2209ff08",
    "computation_find_reciprocal_epistasis/graph_largest_component.ncol": "7d4448eaec8a9357c5af2c4a3025a263a2fa4ece5db7f766fe3cd8b0e13ee174",
}


def write_gzip(path, content):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    import pyreadr

    source = args.source
    output = Path(__file__).resolve().parents[1] / "tests/fixtures/papkou2023"
    raw_path = source / "in_data/fitness_data_wt.rds"
    edge_path = source / (
        "computation_find_reciprocal_epistasis/graph_largest_component.ncol"
    )
    source_hashes = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (raw_path, edge_path)
    }
    for name, expected in SOURCES.items():
        if source_hashes[name] != expected:
            raise ValueError(f"Unexpected source checksum: {name}")
    data = pyreadr.read_r(str(raw_path))[None][["SV", "m"]]
    data.columns = ["sequence", "fitness"]
    buffer = io.StringIO()
    data.to_csv(buffer, index=False)
    write_gzip(output / "fitness.csv.gz", buffer.getvalue().encode())
    write_gzip(output / "edges.ncol.gz", edge_path.read_bytes())
    manifest = {
        "doi": "10.5281/zenodo.8228920",
        "license": "CC-BY-4.0",
        "sources": source_hashes,
        "fixtures": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("fitness.csv.gz", "edges.ncol.gz")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
