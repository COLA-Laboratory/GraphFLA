# Idiosyncrasy bounded performance checks

2026-10-02. Both public functions were measured separately; no implementation
optimization or speedup claim is made. The production source is unchanged from
the EE integration baseline `9014395`. Construction is outside the timer.

## Measurements

Each workload uses three fresh processes, one untimed warmup and three timed
calls per process. The reported time is the median of the three process medians;
raw samples and MADs are retained in the JSON reports. BLAS/OpenMP threads and
hash seed are fixed. The global function uses `seed=0, n_jobs=1`. The single
function has no seed parameter and is a timing-only stochastic workload.
Correctness is established separately by synthetic and full-population tests.

| Workload | Vertices / features | Single mutation (ms) | Global mean (ms) | Largest worker peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Boolean 6 | 64 / 6 | 1.359 | 2.399 | 209.45 |
| Boolean 10 | 1,024 / 10 | 3.470 | 4.839 | 209.14 |
| Categorical 4×5 | 1,024 / 5 | 2.529 | 3.901 | 209.06 |
| Categorical 16×2 | 256 / 2 | 1.922 | 7.384 | 209.41 |
| Sparse Boolean 12 | 1,002 / 12 | 3.840 | 5.186 | 209.28 |
| Long Boolean 72 | 560 / 72 | 10.485 | 27.504 | 214.61 |

Peak memory includes the interpreter, imports and prepared landscape. The table
uses the largest worker high-water RSS across both functions and all six fresh
processes for that workload. The report also records sampled process-tree RSS.
The sparse input starts with 1,024 rows and loses 22 isolates under the existing
builder contract. The long input is 70 adjacent background states crossed with
eight focal states: it exercises byte-key matching without enumerating 2**72
genotypes. The multiallelic case increases the number of directed mutation types.

Raw reports: [global](results/idiosyncrasy-2026-10-02/global.json) and
[single mutation](results/idiosyncrasy-2026-10-02/single.json). They include all
samples, input hashes, production/protocol hashes, platform and dependency
versions. All 36 processes used matching inputs across the two runs. The two
runs were sequential, with no correctness suite or other benchmark launched
alongside these timing rounds. These are local observations on macOS arm64,
Python 3.13.11, NumPy 2.4.2, pandas 2.3.3 and igraph 0.11.9.

## Bounds and reproduction

```sh
python tools/benchmark_analysis.py --metric global_idiosyncratic_index --output global.json --timeout 30 --memory-limit-mib 1024
python tools/benchmark_analysis.py --metric idiosyncratic_index --output single.json --timeout 30 --memory-limit-mib 1024
python -m asv run --python=same --quick --dry-run -b '^analysis\.(global_idiosyncratic_index|idiosyncratic_index)\.'
```

All measured processes completed within **30 s and 1,024 MiB**. The runner
permits only registered cases, at most five processes and ten repetitions, a
maximum 60 s timeout and configurable RSS limits of 64–4,096 MiB. Its new
watchdog polls RSS every 50 ms, including descendants, and kills only its own
isolated process group on timeout, excess memory or interruption. Worker peak
RSS is checked at completion as well. Failure reports remain as partial JSON;
existing runs are never overwritten. `psutil` is an explicit development
dependency. Tests exercise completion, nonzero exits, wall timeout, child
memory accounting/termination and rejection of an unregistered huge case.

Polling cannot reserve memory or prevent every brief allocation spike, and
summed RSS may count shared pages more than once. The fixed input sizes are
the primary protection against combinatorial expansion. No claim is made
about arbitrarily large inputs or all-core execution. The two-worker path is
checked for numerical equality in the literature/basic tests, not timed here.

Both ASV selectors also passed all 24 input/function/measurement combinations.
That smoke run had a separate 90 s / 1,536 MiB watchdog; ASV's quick timings are
not pooled with the measurements above. The common guard changes the runner
protocol hash, so old EE reports are preserved and cannot be compared as if
they used this new protocol. No EE measurements or production code were changed.
