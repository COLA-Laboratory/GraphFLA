# EE performance results — 2026-10-02

The EE calculation now batches non-focal neighborhood moments by degree and
node/site identity. Edge-site comparisons and neighbor matrices are chunked;
centered two-pass population variances preserve the mathematical convention.
No public signature, denominator, t-test or BH family changed. Construction is
excluded from these measurements and its implementation was not modified.

## Method and evidence

Six deterministic workloads contain 64–1,024 input configurations. Each round
uses three fresh processes per workload, one untimed warmup per public function,
and three measured calls per process. Native threads and hash seed are fixed.
Each process is limited to 30 seconds. Runs were sequential, without correctness
tests or another benchmark running alongside them. The method is documented in
[README](README.md#bounded-analysis-comparisons).

The baseline kernel is from `962a5e04c6b1b312af92c456ba8f2a2931793415`, loaded
through the unchanged public wrapper. The checked-in runner and workload were
identical for the baseline, optimized and repeated-baseline control rounds.
Raw samples, source/input/environment hashes, output checks and NPZ snapshots
are retained byte-for-byte in [results/ee-2026-10-02](results/ee-2026-10-02/).
Snapshot filenames in those JSON reports remain valid relative to that directory.
`measured-kernel.py.txt` preserves the optimized source matching the recorded hash.
The final source differs only in a docstring clarification of index storage
(O(N+E)) and the one-wide-row block exception; executable ASTs are identical.

The control baseline was 6–15% faster than the first baseline: there was measurable
run-to-run drift. The table therefore uses the **faster** of the two old-code
medians for each comparison. Every improvement still exceeds both the 5% gate
and three times the sum of process-median MADs against either baseline. These
are local timing measurements, not claims of a universal speedup.

| Workload | Retained vertices / edges | Fraction old → new (ms) | Speedup | Effects old → new (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| boolean-6 | 64 / 192 | 4.29 → 1.88 | 2.29× | 5.27 → 2.76 | 1.91× |
| boolean-10 | 1,024 / 5,120 | 83.96 → 12.46 | 6.74× | 85.58 → 15.07 | 5.68× |
| categorical-4x5 | 1,024 / 7,680 | 59.55 → 14.98 | 3.98× | 63.67 → 18.64 | 3.42× |
| categorical-8x3 | 512 / 5,376 | 28.82 → 9.65 | 2.99× | 31.53 → 12.24 | 2.58× |
| sparse-boolean-12 | 1,002 / 1,509 | 25.33 → 5.85 | 4.33× | 27.47 → 7.31 | 3.76× |
| ordinal-8x3 | 512 / 1,344 | 14.89 → 3.44 | 4.33× | 15.91 → 4.69 | 3.39× |

All six full public tables plus scalar outputs match: pair/allele labels,
neighbor counts, status and nullable decisions exactly; floats within
`rtol=1e-10, atol=2e-12`, including matching NaNs, shapes and dtypes. The same
checks pass for the repeated old-code control. Independent synthetic and
complete author-data tests additionally guard against a shared baseline error.

Median process peak RSS is approximately 209–258 MiB for optimized runs.
The largest increase over the faster control baseline is about 7 MiB (<3%).
RSS includes Python, imports, input graphs and return tables; it is not an
isolated allocation measurement. Temporary moment blocks target 131,072 cells,
with a single exceptionally wide row allowed to exceed that target. Arrays for
edges, node/site keys and output still scale with represented graph size. The
sparse workload loses 22 isolated inputs under the existing construction policy;
its retained dimensions are reported above, not silently called 1,024 vertices.

## Reproduction

```sh
git show 962a5e04c6b1b312af92c456ba8f2a2931793415:graphfla/analysis/_evolvability.py > /tmp/ee-baseline.py
python tools/benchmark_analysis.py --metric ee --baseline-kernel /tmp/ee-baseline.py --output baseline.json
python tools/benchmark_analysis.py --metric ee --compare baseline.json --output optimized.json
python tools/benchmark_analysis.py --metric ee --baseline-kernel /tmp/ee-baseline.py --compare baseline.json --output control.json
```

Use a quiet machine and the same environment/harness for all rounds. The runner
refuses existing output paths. Local environment for these reports:
`macOS-26.3-arm64-arm-64bit-Mach-O`, Python `3.13.11`; numpy 2.4.2, pandas 2.3.3, scipy 1.17.1, igraph 0.11.9, scikit-learn 1.8.0.

## Audit trail and limits

Profiling the original Boolean-10 calculation found 10,240 repeated `np.var`
calls and the corresponding mean calls dominating the moment calculation.
Batching that computation gave the measured improvement. An alternative that
subtracts focal second moments from totals was rejected on numerical grounds;
a dedicated stress test now protects small residual neighborhood variance.

A preliminary benchmark run failed on a synthetic categorical column-label
mismatch; the adapter was corrected before accepted rounds. Early ASV checks
also exposed a stale editable install and duplicate discovery of the common
base class. The install was pointed at this worktree and the base made private;
discovery now has a regression test. Preliminary rounds used an evolving harness
and are not pooled with the three frozen rounds above. Their logs remain local.

Only bounded EE workloads were optimized. The complete Wagner datasets were
used for correctness, not for timing claims; other metrics and large-data
performance have not been claimed to improve. Further optimization is unnecessary
for this scope after the measured gains and equivalent outputs.
