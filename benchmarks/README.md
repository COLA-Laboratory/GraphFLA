# Performance benchmarks

[ASV](https://asv.readthedocs.io/en/stable/writing_benchmarks.html) measures
runtime and process peak memory. Data loading is excluded from runtime.
All inputs are local, deterministic and pinned; missing data fails explicitly.
Install with `python -m pip install -r requirements-dev.txt -e .`.

## Coverage

| Module | Workloads |
|---|---|
| `construction.py` | All seven landscape classes; 21 datasets; neighborhood strategies and edit radii; full build time, peak RSS, output size and variable sites |
| `analysis.py` | All 32 public analysis functions; Boolean, protein, RNA and mixed HPO inputs; fixed seeds, one analysis worker |
| `landscape.py` | Cold and warm lazy properties, configuration materialization, data export, local optima network, GraphML read/write |
| `algorithms.py` | Search cache, both hill-climb strategies, random walks; time and peak memory |
| `utilities.py` | Distances, filters, all sampling functions, all concrete problem generators |

Result dataclasses, abstract protocols, logging, registration/accessor methods
and visualization rendering are not computational benchmarks. Tests cover API
contracts separately. A benchmark executes a method; it does not validate the
scientific interpretation of its result. Analysis implementations are unchanged.

`Analysis` prepares landscape caches outside measurement. The separate cold
property cases invalidate the relevant cache on every invocation; their timing
includes that small invalidation cost. Memory measurements include the process,
imports and prepared input, not just allocations attributable to the operation.
`LandscapeOperations` prepares basins and configuration tuples before measuring
LON and serialization operations. Separate cases measure their construction.
ASV fixes native thread counts and the hash seed in its environment matrix.

## Inputs

The original empirical set contains WReOs, CR6261, TrpB3I, Westmann, CR9114 and
GB1. Papkou now uses the complete **author** fitness table pinned in
`tests/fixtures/papkou2023/`; both full and threshold-filtered builds are measured.
The previous untracked `Papkou2023_DHFR_RAW.csv` dependency has been removed.

Synthetic cases cover Boolean and ordinal grids, DNA/RNA/general sequences,
and an HPO grid combining categorical activation, ordinal depth/width and a
Boolean switch. Synthetic HPO fitness is seeded; it is not an ML training run.

[ProteinGym v1.3](https://zenodo.org/records/15293562) and
[RNAGym](https://github.com/MarksLab-DasLab/RNAGym) add full-background DMS inputs:

| Dataset | Variants | Input sites | Variable sites | Role |
|---|---:|---:|---:|---|
| PG-GB1 | 149,360 | 448 | 4 | Dense four-site combinations with background |
| PG-BRCA2 | 265 | 3,418 | 53 | Very long, sparse input |
| PG-POLG | 15,711 | 2,185 | 851 | Many variable sites, sparse sampling |
| RG-BRCA1 | 886 | 5,592 | 400 | Very long nucleotide background |
| RG-tRNA | 4,175 | 72 | 10 | Relatively dense observed allele space |
| RG-ribozyme | 255 | 230 | 4 | Almost complete four-site RNA space |
| RG-LINE1 | 69,583 | 146 | 146 | Large sparse RNA space |

These are complete substitution assays, without subsampling. RNAGym's T-coded
sequences are normalized to U. Fitness is `DMS_score`, with its supplied sign.
GraphFLA itself removes invariant sites from computation and preserves them in
exported data. Isolated variants may be removed by the existing build contract;
input counts and constructed graph sizes are recorded separately.

`data/manifest.json` pins archive, member and compressed fixture checksums,
assay identifiers and transformations. `log10_observed_space_occupancy` compares
variant count with the product of **observed** per-site alphabet sizes, not a
claim of coverage of the full biological sequence space. The gzip files contain
the original CSV bytes. To restore them: `python tools/fetch_benchmark_data.py`.
ProteinGym and RNAGym distribute their benchmark resources under MIT; attribution
and source links remain in the manifest and here.

## Running ASV

```bash
python -m asv check --python=same
python -m asv run --python=same --quick --dry-run --show-stderr
python -m asv run --python=same -b Construction
python -m asv continuous main HEAD
python -m asv publish
python -m asv preview
```

`--quick` verifies execution and is unsuitable for performance claims. ASV
isolated environments are preferred for commit comparisons. The current
environment mode requires installing the intended checkout first. Pin dependency
versions when comparing different runs; result metadata records installed versions.

## Iterative construction optimization

```bash
python tools/benchmark_round.py --label baseline --output benchmarks/results/baseline.json
# Change the implementation; pass pytest, including the exact Papkou graph.
python tools/benchmark_round.py --label candidate --output benchmarks/results/candidate.json
python tools/compare_benchmark_rounds.py benchmarks/results/baseline.json benchmarks/results/candidate.json
```

Each round uses three fresh timing processes, one unmeasured warmup per process,
and three measured builds per process. Three additional processes each build once
for peak RSS. Garbage collection runs before each timing sample; object disposal
is outside the timer. BLAS/OpenMP threads and `PYTHONHASHSEED` are fixed. Record
wall and CPU time, raw samples, median, MAD, input/source digests and environment.
Results are immutable: the runner refuses to overwrite a round. Interrupted runs
leave a `.partial.json` progress file. The iterative RSS runner targets macOS/Linux;
ASV remains the portable benchmark entry point.

Compare identical datasets, parameters, source inputs and environments. Keep the
machine otherwise idle; do not run tests, profilers or other benchmarks alongside
a measured round. Repeat the baseline if background load or thermals changed.
The comparison uses process medians and an effect/noise gate; it is not a formal
significance test. Timing differences below 5% or within measured dispersion do
not justify a performance claim. Reject unexplained regressions above that gate.

Keep successful changes only after the full correctness suite passes. Record
rejected experiments too. Stop after three consecutive distinct hypotheses fail
to improve meaningful workloads, or when the remaining cost requires an API or
scientific change outside scope. This is a practical stopping rule, not proof of
a theoretical performance ceiling.
