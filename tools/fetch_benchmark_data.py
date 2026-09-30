"""Restore pinned DMS CSV fixtures from their public ProteinGym/RNAGym archives."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/data"


def check(content, expected, label):
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", choices=["proteingym", "rnagym"])
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for collection, archive in manifest["archives"].items():
        if args.collection and collection != args.collection:
            continue
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.zip"
            with (
                urlopen(archive["url"], timeout=60) as response,
                path.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            check(path.read_bytes(), archive["sha256"], collection)
            with zipfile.ZipFile(path) as source:
                for name, spec in manifest["datasets"].items():
                    if spec["collection"] != collection:
                        continue
                    content = source.read(spec["member"])
                    check(content, spec["member_sha256"], name)
                    target = ROOT / spec["file"]
                    temporary = target.with_suffix(".tmp")
                    with temporary.open("wb") as raw:
                        with gzip.GzipFile(
                            filename="", fileobj=raw, mode="wb", mtime=0
                        ) as out:
                            out.write(content)
                    check(temporary.read_bytes(), spec["sha256"], target.name)
                    temporary.replace(target)
                    print(f"Restored {target.name}")


if __name__ == "__main__":
    main()
