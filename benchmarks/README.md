# GraphFLA benchmarks

Performance benchmarks for GraphFLA, written for
[airspeed velocity (`asv`)](https://asv.readthedocs.io/) — the benchmarking tool
used by NumPy, SciPy, pandas and scikit-learn. These are **not** part of the
pytest suite; they measure runtime and memory, and `asv` can track them across
commits to flag regressions.

## Layout

| file | group | what it measures |
|------|-------|------------------|
| `construction.py` | `Construction` | `build_from_data` time + peak memory + edge count, over the curated real landscapes and synthetic cubes |
| `analysis.py`     | `Analysis`     | each landscape-analysis metric on two moderate real landscapes (boolean + protein) |
| `algorithms.py`   | `Trajectories` | the `HillClimb` / `RandomWalk` walkers on two large real landscapes |
| `_datasets.py`    | —              | the real-landscape loader + synthetic generators |

## Datasets

The benchmarks run primarily on the **real empirical landscapes committed in
`data/`** — a curated set spanning encodings and sizes:

| name | kind | size | source |
|------|------|------|--------|
| `WReOs`   | ordinal | ~500   | W–Re–Os alloy design |
| `CR6261`  | boolean | ~1.9k  | Phillips 2021 antibody (CR6261) |
| `TrpB3I`  | protein | ~7.8k  | Johnston 2024 |
| `Westmann`| dna     | ~17.8k | Westmann 2024 |
| `CR9114`  | boolean | ~65k   | Phillips 2021 antibody (CR9114) |
| `GB1`     | protein | ~150k  | Wu 2016 |
| `Papkou`  | dna     | ~260k  | Papkou 2023 DHFR |

Because this data ships in the repo, `asv` can use it reproducibly at any commit.
A benchmark is **skipped** automatically (via `NotImplementedError`) if its data
file is not present on a given checkout. `construction.py` additionally includes
a few deterministic **synthetic** NK / ordinal cubes (`synthetic-*`) for quick,
fully self-contained runs.

## Running

Install asv (a dev dependency):

```bash
pip install asv
```

Quick local run against the **current** environment (no isolated build — fastest
for development):

```bash
asv run --python=same                       # full run in the active env
asv dev                                      # one iteration each (smoke check)
asv run --python=same -b Construction        # only one group
asv run --python=same -b 'Analysis.*CR6261'  # one group, one dataset (subset)
```

The `Analysis` group on the larger real landscapes is the slowest (a couple of
motif/regression metrics take ~1–2 s on the protein landscape); subset with `-b`
when iterating.

Full, reproducible run (asv builds an isolated env and installs graphfla per
commit — slower the first time):

```bash
asv run
```

Compare two revisions and report regressions (the typical regression check):

```bash
asv continuous main HEAD
asv run main..HEAD               # benchmark every commit in a range
```

Browse results as an HTML report:

```bash
asv publish && asv preview
```

Results land in `.asv/` (git-ignored).
