---
title: Construction benchmarks
---

# Construction benchmarks

This comparison measures the cost of building a complete binary fitness landscape with GraphFLA and with an explicitly naive neighborhood search. Both methods receive the same configurations and fitness values and produce the same graph, vertex attributes, directed edges, fitness differences and local-optima count.

## Results

Each number is the median of three fresh processes. Runtime covers the full landscape build; peak memory is the process high-water RSS, including Python, imported dependencies, input data and construction. Input generation and correctness checks are outside the timed interval.

| Configurations | Variables | Directed edges | Naive time (ms) | GraphFLA time (ms) | Naive peak (MiB) | GraphFLA peak (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 8 | 1,024 | 20.07 | 3.29 | 101.8 | 101.7 |
| 1,024 | 10 | 5,120 | 302.93 | 6.27 | 111.0 | 102.5 |
| 4,096 | 12 | 24,576 | 5179.13 | 17.98 | 236.3 | 107.0 |

Measured on an Apple M4 Pro, macOS 26.3, Python 3.9.6, using a single BLAS/OpenMP thread. The saved record includes library versions, exact source revision, per-process samples, input hashes and graph hashes. The smallest input has effectively unchanged peak RSS; Python and dependency memory dominate at that scale.

## What is compared

The naive baseline uses two nested Python loops to compute Hamming distances between every unordered pair of configurations. It writes both symmetric entries of a dense, 64-bit floating-point distance matrix and then extracts the pairs at distance one. Edges point toward higher fitness. At 4,096 configurations, that matrix alone occupies 128 MiB.

GraphFLA uses its default neighbor strategy, which resolves to active neighbor lookup on these complete Boolean spaces. The benchmark replaces only the edge builder in the naive worker process. Both methods keep the same preprocessing, graph allocation, topology filtering, attribute attachment, plateau handling and local-optima detection. No production package code is modified.

These results compare this specific Python-loop, dense-matrix implementation. They do not compare every possible pairwise method: vectorized, compiled, condensed-matrix and memory-efficient implementations have different costs. The measurements concern construction, rather than analysis metrics, and vary with hardware and input structure.

## Reproduce

[Download the recorded measurements](assets/benchmarks/construction.json) and [the benchmark script](assets/benchmarks/benchmark_home.py). The script belongs in the repository's docs/scripts directory and runs in an environment with GraphFLA's dependencies. It reuses the repository's process resource guard.

```sh
python docs/scripts/benchmark_home.py
```

Workloads are limited to 256, 1,024 and 4,096 configurations. Each worker has a 45-second wall-time limit and a 768 MiB process-tree memory limit. Methods alternate execution order across repetitions. A small, identical 16-row warmup resolves library initialization before measurement. The fixed random seed is 20261004; every observed fitness is distinct.

Each pair of runs is checked against identical input and canonical graph hashes. The expected number of directed edges is independently checked against the complete binary hypercube formula, configurations × variables / 2. The full record retains individual measurements rather than only the homepage speed and memory summaries.
