---
title: Construction benchmarks
---

# Construction benchmarks

This comparison measures the cost of building a complete binary fitness landscape with GraphFLA and with an explicitly naive neighborhood search. Both methods receive the same configurations and fitness values and produce the same graph, vertex attributes, directed edges, fitness differences and local-optima count.

## Results

Each number is the median of three fresh processes. Runtime covers the full landscape build; peak memory is the process high-water RSS, including Python, imported dependencies, input data and construction. Input generation and correctness checks are outside the timed interval.

| Configurations | Variables | Directed edges | Naive time (s) | GraphFLA time (s) | Naive peak (MiB) | GraphFLA peak (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 8 | 1,024 | 0.0216 | 0.0036 | 103.7 | 103.5 |
| 1,024 | 10 | 5,120 | 0.317 | 0.0069 | 113.1 | 104.6 |
| 4,096 | 12 | 24,576 | 5.41 | 0.0199 | 238.7 | 109.3 |
| 8,192 | 13 | 53,248 | 22.4 | 0.0388 | 629.1 | 117.0 |
| 16,384 | 14 | 114,688 | 93.4 | 0.0893 | 2,133.6 | 136.8 |
| 65,536 | 16 | 524,288 | — | 0.441 | — | 279.5 |
| 262,144 | 18 | 2,359,296 | — | 2.04 | — | 898.1 |
| 1,048,576 | 20 | 10,485,760 | — | 9.23 | — | 3,520.3 |

Measured on an Apple M4 Pro, macOS 26.3, Python 3.9.6, using a single BLAS/OpenMP thread. The saved record includes library versions, exact source revision, per-process samples, input hashes and, where both methods run, graph hashes. The smallest input has effectively unchanged peak RSS; Python and dependency memory dominate at that scale.

## Larger sizes for the naive baseline

The naive baseline is measured up to 16,384 configurations, where its distance matrix alone occupies 2 GiB. At 65,536 and 1,048,576 configurations that matrix would occupy 32 GiB and 8 TiB, so the homepage shows modelled values for the baseline at these two sizes. GraphFLA is measured at every size.

The model has two parts:

-   **Time** is proportional to the pairwise work, $N(N-1)/2$ pairs of configurations, each compared at every variable. The constant is fitted to the median time at 16,384 configurations. On the smaller measured sizes the model underestimates the measured time by 3.3% at 8,192 configurations and by 7.5% at 4,096; the gap grows at smaller sizes, where fixed overheads dominate. Modelled times are therefore lower bounds rather than overestimates.
-   **Peak memory** is GraphFLA's measured peak at the same size plus the $8N^2$ bytes of the dense matrix. On every measured size this agrees with the naive peak to within 0.6%.

With this model, the baseline needs about 28 minutes and 32 GiB at 65,536 configurations, and about 6.3 days and 8.0 TiB at 1,048,576 configurations. The model parameters and their validation errors are stored with the measurements.

## What is compared

The naive baseline uses two nested Python loops to compute Hamming distances between every unordered pair of configurations. It writes both symmetric entries of a dense, 64-bit floating-point distance matrix and then extracts the pairs at distance one. Edges point toward higher fitness. At 4,096 configurations, that matrix alone occupies 128 MiB.

GraphFLA uses its default neighbor strategy, which resolves to active neighbor lookup on these complete Boolean spaces. The benchmark replaces only the edge builder in the naive worker process. Both methods keep the same preprocessing, graph allocation, topology filtering, attribute attachment, plateau handling and local-optima detection. No production package code is modified.

These results compare this specific Python-loop, dense-matrix implementation. They do not compare every possible pairwise method: vectorized, compiled, condensed-matrix and memory-efficient implementations have different costs. The measurements concern construction, rather than analysis metrics, and vary with hardware and input structure.

## Reproduce

[Download the recorded measurements](assets/benchmarks/construction.json) and [the benchmark script](assets/benchmarks/benchmark_home.py). The script belongs in the repository's docs/scripts directory and runs in an environment with GraphFLA's dependencies. It reuses the repository's process resource guard.

```sh
python docs/scripts/benchmark_home.py
```

GraphFLA runs at 8, 10, 12, 13, 14, 16, 18 and 20 variables; the naive baseline runs at 8 to 14 variables. Each GraphFLA worker has a 600-second wall-time limit and an 8 GiB process-tree memory limit; each naive worker has 300 seconds and 3 GiB. Methods alternate execution order across repetitions. A small, identical 16-row warmup resolves library initialization before measurement. The fixed random seed is 20261004; every observed fitness is distinct.

Wherever both methods run, each run is checked against identical input and canonical graph hashes. The expected number of directed edges is independently checked against the complete binary hypercube formula, configurations × variables / 2, at every size. The full record retains individual measurements rather than only the homepage speed and memory summaries.
