# GraphFLA benchmarks

Performance benchmarks for GraphFLA, written for
[airspeed velocity (`asv`)](https://asv.readthedocs.io/) — the benchmarking tool
used by NumPy, SciPy, pandas and scikit-learn. These are **not** part of the
pytest suite; they measure runtime and memory, and `asv` can track them across
commits to flag regressions.

## Layout

| file | group | what it measures |
|------|-------|------------------|
| `construction.py` | `Construction` | `build_from_data` time + peak memory, per landscape kind/size |
| `analysis.py`     | `Analysis`     | each landscape-analysis metric on a fixed NK landscape |
| `algorithms.py`   | `Trajectories` | the `HillClimb` / `RandomWalk` walkers |
| `_datasets.py`    | —              | synthetic, deterministic landscape generators (no external data) |

All landscapes are synthetic (Kauffman NK cubes / ordinal grids), so the suite
is portable and reproducible — no data files or machine-specific paths.

## Running

Install asv (a dev dependency):

```bash
pip install asv
```

Quick local run against the **current** environment (no isolated build — fastest
for development):

```bash
asv run --python=same            # full run in the active env
asv dev                          # one iteration each (smoke check)
asv run --python=same -b Analysis    # only the Analysis group
```

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
