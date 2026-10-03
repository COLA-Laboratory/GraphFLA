"""Extract Johnson 2019 measurements from eLife 76491 Supplementary file 1.

Usage: python tools/prepare_fitness_trends_fixture.py SOURCE.xlsx OUTPUT_DIR
Source is downloaded separately; this script never uses GraphFLA or the network.
"""

import gzip
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    x = pd.read_excel(source, "BYxRM_x")
    effects = pd.read_excel(source, "BYxRM_s")
    metadata = pd.read_excel(source, "data_by_mutation")
    experimental = set(metadata.loc[metadata.Type == "Experiment", "Edge"])
    data = (
        effects[effects.Edge.isin(experimental)]
        .merge(x, on="Sample", validate="many_to_one")
        .dropna(subset=["s", "Fitness"])
    )
    data = data[["Edge", "Sample", "Fitness", "s"]].sort_values(["Edge", "Sample"])
    if data.duplicated(["Edge", "Sample"]).any():
        raise ValueError("Duplicate mutation/background measurements")
    output = destination / "johnson2019.csv.gz"
    output.write_bytes(gzip.compress(data.to_csv(index=False).encode(), mtime=0))
    manifest = {
        "source_url": "https://cdn.elifesciences.org/articles/76491/elife-76491-supp1-v2.xlsx",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "license": "CC-BY-4.0",
        "license_source": "https://api.elifesciences.org/articles/76491 (copyright)",
        "attribution": "Milo S. Johnson and Michael M. Desai, eLife 11:e76491 (2022), Supplementary file 1, republishing Johnson et al., Science 366:490-493 (2019) data.",
        "derived_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": len(data),
        "mutations": data.Edge.nunique(),
        "conversion": "BYxRM_s experimental insertions selected using data_by_mutation.Type; inner join BYxRM_x on Sample; retain finite paired s/Fitness; select Edge,Sample,Fitness,s; sort; CSV float roundtrip; gzip mtime=0. No >=50 filter applied until test.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    prepare(*sys.argv[1:])
