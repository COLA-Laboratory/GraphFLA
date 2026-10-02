# Gamma performance results

2026-10-02. Both public signatures and scientific definitions are unchanged.
This follow-up optimizes the numerically corrected `ed829ed` implementation;
it does not compare against an older version lacking the extreme-scale fix.

## Changes

Each unordered position pair is grouped once, then evaluated in both focal
directions. For one focal allele pair with m observed background alleles,
numeric products are summed with prefix sums, and each squared effect receives
weight `(m-1)/2`. This replaces a quadratic enumeration of background allele
pairs with linear array operations without averaging square ratios. Prefix
products avoid cancellation from subtracting two large squared sums. Gamma-star
uses integer-valued sign sums; each public entry computes only its needed moments.
The power-of-two fallback still handles extreme units and overflowing differences.

Groups with fewer than four configurations cannot contain a complete square
and are discarded before allocation. A dense worker grid is capped at 1,048,576
float64 cells (8 MiB). Above this cap, or below 1/8 occupancy, observed-group
dictionaries enumerate shared backgrounds without allocating the full allele
Cartesian product. This cap is for that grid, not a guarantee on total process
memory: input arrays, grouping, temporaries and parallel workers also use memory.

For dense groups with allele counts A and B, moment arithmetic changes from
O(A²B²) to O(A²B + B²A) per background and position pair. Grouping and the number
of variable pairs still matter; high-dimensional sparse inputs are not free.
No sampling, configuration truncation or scientific tolerance was introduced.

## Controlled measurements

Each metric runs independently on six registered inputs of 64–1,024 input
configurations, at most 72 variables and 16 alleles per variable. Each round
uses three fresh processes, one untimed warmup and three timed calls per process.
Construction, snapshots and output checks are outside the timer. Native threads
and the hash seed are fixed. Every process has a **30 s / 1,024 MiB** process-tree
watchdog. Runs were sequential, with no tests or competing benchmark run alongside.

Baseline, repeated-baseline control and final reports, all raw samples and scalar
NPZ snapshots are in [results/gamma-2026-10-02](results/gamma-2026-10-02/).
The same runner/workload protocol hashes apply to all six rounds. Baseline source
is saved as `baseline-kernel.py.txt`; `measured-kernel.py.txt` matches the final
source hash. Snapshot paths remain valid relative to the reports. The comparator
checks shapes/dtypes and values with matching NaNs at `rtol=1e-10, atol=2e-12`.
Independent equation/literature tests additionally protect against shared errors.

The old-code control was somewhat faster than the first baseline. The table uses
the **faster old median per input**, and speedup claims must exceed both 5% and
three times the combined process-median MAD against both baselines.

| Workload | Gamma old → new (ms) | Ratio | Gamma-star old → new (ms) | Ratio |
| --- | ---: | ---: | ---: | ---: |
| boolean-6 | 1.685 → 1.427 | 1.18×, within noise | 1.718 → 1.497 | 1.15×, within noise |
| boolean-10 | 8.783 → 7.106 | 1.24× | 8.566 → 7.819 | 1.10× |
| categorical-4x5 | 8.225 → 4.190 | 1.96× | 7.999 → 4.163 | 1.92× |
| categorical-16x2 | 183.957 → 3.450 | 53.32× | 179.967 → 3.598 | 50.01× |
| sparse-boolean-12 | 12.816 → 8.549 | 1.50× | 12.709 → 9.428 | 1.35× |
| long-boolean-72 | 1177.729 → 469.291 | 2.51× | 1170.252 → 477.360 | 2.45× |

All twelve public outputs match their baseline snapshots. The smallest workload
has no defensible speedup claim under the noise gate; the other ten comparisons
pass that gate. Peak worker RSS was at most **214.27 MiB**. This includes imports
and the built graph; it does not isolate metric allocations. RSS polling every
50 ms may miss brief peaks and may count shared child pages more than once.
The sparse Boolean input retains 1,002 vertices after the existing construction
policy removes 22 isolates; the long input has 560 vertices and 72 variables.

```sh
git show ed829ed:graphfla/analysis/epistasis/gamma.py > /tmp/gamma-baseline.py
python tools/benchmark_analysis.py --metric gamma --baseline-kernel /tmp/gamma-baseline.py --output baseline.json
python tools/benchmark_analysis.py --metric gamma --baseline-kernel /tmp/gamma-baseline.py --compare baseline.json --output control.json
python tools/benchmark_analysis.py --metric gamma --compare control.json --output final.json
# Repeat independently with --metric gamma_star and fresh output paths.
```

The runner refuses existing result paths and incomparable protocols, environments
or inputs. Local environment: macOS arm64, Python 3.13.11, numpy 2.4.2,
pandas 2.3.3, scipy 1.17.1, igraph 0.11.9, scikit-learn 1.8.0.
ASV smoke execution also passed all 24 metric/workload/measurement combinations,
under a separate 120 s / 1 GiB outer guard; it took 29.31 s and is not timing
evidence. Analysis benchmarks remain separate from construction benchmarks.

## Experiment history and limits

The first candidate still re-compacted background/allele IDs even when every
group was retained. Skipping that redundant work improved the Boolean cases;
the final rounds above measure this refinement. Preliminary candidate timings
remain local and are not mixed into the final comparison.

The square-sum shortcut `((sum s)**2 - sum(s**2))/2` was avoided for numeric
effects because it can erase a small but valid numerator; a regression checks
gamma near 1e-200. Its use for bounded integer sign sums is exact. Further
optimization was unnecessary after the verified gains. These measurements do
not promise similar speedups on unbounded or arbitrarily sparse landscapes.
