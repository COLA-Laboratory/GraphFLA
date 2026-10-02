# Performance benchmarks

Results: [initial optimization](RESULTS.md) ·
[compact construction experiment](ARCHITECTURE_EXPERIMENT.md).

[ASV](https://asv.readthedocs.io/en/stable/writing_benchmarks.html) measures
runtime and process peak memory. Data loading is excluded from runtime.
All inputs are local, deterministic and pinned; missing data fails explicitly.
Install with `python -m pip install -r requirements-dev.txt -e .`.

## Select a scope

Construction and analysis are independent benchmark groups. Each analysis metric
has its own module; selecting EE does not execute other metrics or construction
benchmarks. Input construction happens in setup and is excluded from metric time.

| Group / module | Workloads |
| --- | --- |
| `construction.build` | Existing 21 datasets, seven classes, neighborhood strategies and radii |
| `construction.utilities` | Distances, filters, samplers and problem generators |
| `analysis.ee` | EE fraction and effects table, six bounded landscapes, 64–1,024 input configurations |
| `analysis.idiosyncratic_index`, `analysis.global_idiosyncratic_index` | Single mutation / landscape mean separately, six inputs with at most 1,024 configurations and 72 positions |
| `analysis.gamma`, `analysis.gamma_star` | Each scalar separately, six inputs of 64–1,024 configurations, up to 72 positions and 16 alleles |
| `analysis.<function_name>` | One other public metric on Boolean-6 and categorical-3×3 inputs |
| `analysis.landscape` | Cold/warm lazy properties, export, LON and GraphML operations |
| `analysis.trajectories` | Search caches, hill-climbing and random walks |

The registry in `analysis/__init__.py` maps public functions to modules and gives
reasons for exclusions. The deprecated EE wrapper is covered by basic compatibility
tests and shares the canonical calculation. Dataclasses, abstract protocols,
registration/accessor methods and visualization rendering are not timed.
A basic test checks that every public analysis function is benchmarked or explicitly
excluded and that its bounded call remains valid.

To add a metric, copy a small `analysis/<function>.py` subclass, register it, and
bind any required parameters in `_shared.prepare_call`. Start with `SMALL_CASES`;
add a dedicated bounded workload only when it answers a specific performance
question. Especially for epistasis enumeration, do not reuse the large construction
corpus by default. Benchmark modules must not load data or calculate metrics at
import time. Set a timeout and keep setup out of the timer. Scientific correctness
belongs in basic/oracle and [literature tests](../validation/TESTING.md).

Calls with seed parameters use fixed seeds. The existing single-mutation
`idiosyncratic_index` has no seed parameter; its numerical draws vary and it is
only a timing workload. `profile` measures an explicitly named small subset.
Metrics may populate their own lazy dependencies; setup does not compute unrelated
metrics. Separate cold-property cases invalidate the relevant cache on each call.
Process memory includes imports and the prepared input, not only metric allocations.

The reorganization changes ASV IDs (for example,
`construction.Construction.time_build` → `construction.build.Construction.time_build`).
Historical results remain intact. Compare old/new source using the same benchmark
harness or the construction round runner, rather than joining different IDs.

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

## Targeted ASV execution

```bash
# EE only: smoke check (24 workload/output/measurement combinations).
python -m asv run --python=same --quick --dry-run -b '^analysis.ee\.' --show-stderr
# Repeated timing for one selected metric, or construction only.
python -m asv run --python=same -b '^analysis.ee\.'
python -m asv run --python=same -b '^analysis.gamma\.'
python -m asv run --python=same -b '^construction\.build\.'
```

Omitting `-b` explicitly selects the entire suite and may be expensive. Normal
iteration and CI use a metric selector. `--quick` verifies execution and is not
performance evidence. With `--python=same`, first install the intended checkout
with `python -m pip install --no-deps -e .` in the active environment; another
worktree's editable install would benchmark the wrong implementation. ASV isolated
environments are preferred for commit comparisons. Pin dependencies between runs.

## Bounded analysis comparisons

```bash
python tools/benchmark_analysis.py --metric ee --output baseline.json
# Change only the implementation; keep the workload and runner fixed.
python tools/benchmark_analysis.py --metric ee --output candidate.json --compare baseline.json
# Gamma metrics each have six bounded fixtures and scalar output snapshots.
python tools/benchmark_analysis.py --metric gamma --output gamma.json
python tools/benchmark_analysis.py --metric gamma_star --output gamma-star.json
# Idiosyncrasy only: bounded inputs, time and process-tree RSS.
python tools/benchmark_analysis.py --metric global_idiosyncratic_index --output iid.json --timeout 30 --memory-limit-mib 1024
python tools/benchmark_analysis.py --metric idiosyncratic_index --output single-iid.json --timeout 30 --memory-limit-mib 1024
```

`--metric` and `--output` are required; there is no all-metric default. EE records
both public outputs. Other functions are independently selectable by name.
Output-equivalence comparison supports EE tables/fractions and gamma/gamma-star
scalar snapshots, including matching NaNs.

Each workload uses three fresh processes, one warmup per function, and three timed
samples per process. Construction, GC, serialization and output checks are untimed.
Defaults cap each process at 30 seconds (maximum configurable 60), repetitions at
10 and processes at five. BLAS/OpenMP threads and the hash seed are fixed. Missing
inputs, timeout, failed comparisons and subprocess errors fail the run; interrupted
runs retain a partial JSON. Existing results and snapshot directories are never
overwritten. This runner uses `resource` on macOS/Linux; ASV is the portable entry.

The runner also polls worker-plus-descendant RSS every 50 ms. The default limit
is 1,024 MiB, adjustable to 64–4,096 MiB with `--memory-limit-mib`. Limit violations
terminate the isolated process group and retain a partial failure report. The
worker's high-water RSS is checked after completion too. This watchdog can miss
short allocation spikes, and summing process RSS can double-count shared pages;
it is not an OS allocation reservation. Fixed registered inputs are the primary
size guard; hidden worker calls reject unregistered cases before construction.
ASV retains its separate timeout and does not inherit this RSS watchdog.

Reports include raw samples, process medians/MAD, total process peak RSS, input and
source hashes, runner/workload hashes and environment versions. Comparison rejects
changed protocols, inputs or environments. EE snapshots compare labels, counts and
nullable decisions exactly, floats at `rtol=1e-10, atol=2e-12` with matching NaNs.
Snapshot hashes prevent silently edited references. The numerical tolerance covers
summation-order roundoff on these bounded workloads; it is not a paper tolerance.

A change is called improved/regressed only beyond both 5% and three times the sum
of process-median MADs. This is a practical noise gate, not a significance test.
Keep the machine otherwise idle and repeat the baseline as a drift control.
RSS is a process high-water mark, including imports; it does not isolate allocation
cost or justify claims about arbitrarily large landscapes. A trusted local old
EE kernel or gamma module can be supplied with `--baseline-kernel`. EE keeps the
public wrapper fixed; gamma calls the selected module's original public function
with the same signature and workload. No code is downloaded. See
[EE results](EE_RESULTS.md) and [gamma results](GAMMA_RESULTS.md).

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
