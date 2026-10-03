# Returns/costs bounded performance

Both public metrics were measured separately for Pearson, Spearman and OLS on
six fixed workloads: Boolean-6, Boolean-14, categorical 4×5 and 64×2, sparse
Boolean-12 and ordinal 8×3. Inputs contain 64–16,384 configurations and at most
258,048 improving edges. Construction is outside the timer. The baseline is a
straightforward NumPy/SciPy **pooled-edge** implementation, not the former
node-mean statistic. Each of 36 workload/statistic/metric combinations passed
output-equivalence checks against that independent implementation.

Two fresh processes per combination, one warmup and three timed repeats per
process; BLAS/OpenMP threads fixed to one. Each worker has a 30 s watchdog and
1 GiB process-tree RSS limit. No workload exceeded either limit. Reported time
is the median of process medians; memory includes Python imports and the graph.

| Metric | Statistic | Time range, ms | Maximum worker RSS, MiB | Maximum sampled tree RSS, MiB |
| --- | --- | ---: | ---: | ---: |
| DRI | Pearson | 0.193–72.344 | 278.64 | 282.64 |
| DRI | OLS slope | 0.189–74.139 | 275.91 | 296.61 |
| DRI | Spearman | 0.271–67.721 | 343.16 | 357.25 |
| ICI | Pearson | 0.201–73.855 | 274.62 | 305.30 |
| ICI | OLS slope | 0.178–72.194 | 268.86 | 256.31 |
| ICI | Spearman | 0.308–61.560 | 341.17 | 352.50 |

The reference materializes all edges. Pearson/OLS instead use blocks of 32,768
edges and centered moment merging: O(V+E) time, O(V+B) auxiliary storage. This
avoids Python edge-list memory growth with E, but has a runtime cost: on the
largest workloads the reference is about 1.7–1.8× faster. These are not claims
of a universal speedup. Their final worst measured time remains below 75 ms.

Exact Spearman inherently retains/sorts O(E) observations. The first version
filled those arrays through the streaming path; after measurement it was
changed to direct igraph edge extraction. On the largest categorical case,
DRI Spearman fell from about 100 ms to 68 ms and ICI from 94 ms to 62 ms. Final
comparison against the vector reference is mixed (0.82–1.27× across all
Spearman workloads); the change is a bounded allocation/runtime tradeoff.

No complete genotype cube, dense adjacency matrix, unbounded neighborhood
expansion or subprocess parallelism is added. Results do not promise a fixed
memory ceiling on arbitrarily large input graphs, particularly for exact ranks.
The separate six-test empirical validation on the full 324,044-edge Papkou graph
and Johnson data also passes a 60 s / 1 GiB watchdog (about 580 MiB tree RSS).

Run one statistic explicitly:

```sh
python tools/benchmark_analysis.py --metric diminishing_returns_index --trend-method pearson --output dri.json --timeout 30 --memory-limit-mib 1024
python tools/benchmark_analysis.py --metric increasing_costs_index --trend-method spearman --output ici-spearman.json --timeout 30 --memory-limit-mib 1024
python -m asv run --python=same --quick --dry-run -b '^analysis[.](diminishing_returns_index|increasing_costs_index)[.]'
```

The runner records source/protocol/input hashes, versions, raw samples, worker
and process-tree memory, and exact selected-statistic snapshots. Comparisons
reject changed statistic choices. The ASV modules expose all three statistics
on the same six datasets; their setup is untimed. For this run, Python 3.13.11,
NumPy 2.4.2, pandas 2.3.3, SciPy 1.17.1, igraph 0.11.9 and scikit-learn 1.8.0
were fixed. Detailed initial/final reports and the reference implementation are
retained locally in `.codex-local/proposals/dr-ic/`.

ASV smoke validation also passed all 72 parameter/measurement combinations.
Each metric was run separately under the same 60 s / 1 GiB cap: DRI finished
in 42.22 s (368.30 MiB sampled tree RSS), ICI in 43.06 s (338.75 MiB).
The initial combined ASV invocation exceeded 60 s and was terminated; splitting
the independent metric runs retained the same inputs and caps. ASV first exposed
an unrelated editable-install import from another worktree; a private local
venv pointing to this checkout resolved it without modifying that shared install.
Only the successful isolated runs count as ASV evidence.
