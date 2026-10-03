# Returns/costs basic-test audit

## Previous coverage and replacements

The old suite had 60 frozen node-mean values (two metrics across 30 catalog
landscapes) plus a fully neutral smoke test. Those frozen values cannot validate
the newly specified edge population. Their historical bytes remain unchanged in
`tests/_golden_references.py`; their entries were removed from the active golden
comparison. They were not regenerated with production output.

`tests/test_fitness_trends.py` supplies 113 focused cases:

- An unequal-degree graph distinguishes equal-edge pooling from node averaging
  for Pearson, Spearman and OLS, and distinguishes each metric's background.
- Independent SciPy statistics verify retained Boolean, categorical and ordinal
  graphs with missing configurations and nonzero construction epsilon.
- Equivalent maximize/minimize encodings, additive translations, 1e-250/1e250
  units and opposite-sign values near float64 limits test numerical behavior.
- Stale/missing edge attributes, neutral edges, empty and one-edge populations,
  constant backgrounds/effects, unbuilt/malformed graphs and nonfinite fitness
  exercise explicit contracts. Isolated extreme fitness has no statistical weight.
- Forced small blocks and permuted edges check population preservation and
  stable reductions. The profile interface agrees with direct calls.
- A fully additive landscape with fixed unequal effects has nonzero pooled
  coefficients: this prevents interpreting every nonzero DRI/ICI as interaction.

Tests assert actual coefficients or explicit degeneracies, not only output type,
range or sign. The three intentional scales (per-edge, per-node, per-mutation)
are never used interchangeably as oracles. The related idiosyncrasy functions'
ASTs are identical to baseline; their algorithms were not changed.

## Coverage and execution

After the final scientific/numerical implementation, targeted basic tests cover
82/82 statements and 30/30 branches of `_fitness_trends.py`. Coverage alone is
not proof of scientific validity; the independent estimand tests and separate
literature evidence provide that context. Eight additional harness tests verify
metric/statistic selection and baseline snapshot forwarding.

Full basic suite: **2,244 passed, 2 existing strict xfailed** (24.43 s on the
recorded local environment). These two precision xfails predate this task and
are not counted as passing evidence. Full dedicated literature suite:
**52 passed** (52.96 s), including six new tests. After the Spearman/isolated-node
refinement the six affected literature tests were rerun successfully; their
resource-guarded run used about 580 MiB process-tree RSS under a 60 s / 1 GiB cap.
No network calls occur inside these tests. API docstrings: eight examples pass.

Artifact check: 216 inputs/history artifacts verified; all 36 pre-existing case
files keep their original SHA-256 hashes. Three new cases distinguish Johnson's
paper counts from two independent Papkou raw/clipped coefficient targets.
The current contract/template and CI separation are reused, not duplicated.

Environment: the existing local EE Python 3.13.11 test environment with NumPy
2.4.2, SciPy 1.17.1, pandas 2.3.3 and igraph 0.11.9. Commands use the current
checkout on `sys.path`; no editable installation in another session was changed.
Initial direct system-Python coverage failed because pytest-cov was absent;
coverage was rerun in the existing test environment. A module-level coverage
source selector triggered an igraph reimport error; the package-level selector
ran successfully. Neither failed attempt is passing evidence.

Detailed raw logs, JUnit, coverage JSON, command/resource records and local
narrative length checks are retained in `.codex-local/proposals/dr-ic/`.

After the final removal of the unused historical extractor calls, all 1,351
related golden/metric/harness tests passed again. ASV's 72 combinations passed
in a private environment after an initial unrelated-worktree import failure.
A combined ASV smoke run exceeded its 60 s cap; two independent per-metric runs
then passed under that same cap. The failures and recovery are retained locally.
