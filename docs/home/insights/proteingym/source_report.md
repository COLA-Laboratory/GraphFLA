# ProteinGym data preparation report

Dataset: ProteinGym DMS substitution benchmark, v1.3.
Official record: https://zenodo.org/records/15293562 (v1.3, published 2025-04-27).

## Source files

- `DMS_ProteinGym_substitutions.zip` — 43,021,128 bytes; SHA-256 `3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921`
- `DMS_substitutions.csv` — 208,734 bytes; SHA-256 `a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308`
- `DMS_substitutions_Spearman_DMS_level.csv` — 141,235 bytes; SHA-256 `f432423b87f79ac9778dfac86e3d95be041d246bc618dc0a406b35b0b7466437`
- `proteingym_dms_citations.json` — 12,932 bytes; SHA-256 `24820f710360764ba402a698ddc0162a613cbb31baf075526a9e6341b8be307f`

The model outcomes come from the official zero-shot per-DMS Spearman table at
ProteinGym repository commit `144fe22b07dfaeec2b366f2346203a9838a55b4c`. The current 103-column CSV has
97 model columns; the remaining columns are the assay key and benchmark metadata.
This process uses those published correlations directly. It performs no model
training or inference and does not download the 1.9 GB model-score archive.

## Eligibility and input validation

All 217 substitution assay CSVs were scanned using every measured row. Eligibility
is based on the arithmetic mean number of colon-separated substitution tokens in
`mutant`, strictly greater than 1.5. No variant rows were filtered to calculate
that mean. The v1.3 archive contains 2,465,767 rows; the
selected 36 assays contain
1,788,585 complete measured
rows. Every selected row was checked against the official target sequence: the
WT letter and position match, the labels reproduce `mutated_sequence`, and the
fitness value is finite. Every selected genotype is unique. The 36-assay cohort
is recomputed from the current v1.3 release and is not copied from an earlier
paper table with rounded mutation-depth summaries.

## GraphFLA method

Each assay is passed to `ProteinLandscape` with all measured rows after removing
only positions that are invariant throughout that assay. The original target
length and retained variable length are saved per assay. GraphFLA's standard
construction then removes configurations with no observed one-edit neighbor;
both the raw `n_variants` and resulting `n_graph_configs` are saved. Eligibility
is always computed before construction from every assay row. The landscape uses
processed `DMS_score` with maximization, one-edit neighbors, and `epsilon=0`.
No assay-level row is sampled or removed by this preparation script. The feature
set is:

- `epistasis.magnitude`: Magnitude epistasis
- `epistasis.sign`: Sign epistasis
- `epistasis.reciprocal_sign`: Reciprocal sign epistasis
- `diminishing_returns_index`: Diminishing returns index
- `increasing_costs_index`: Increasing costs index
- `global_idiosyncratic_index`: Global idiosyncratic index
- `r_s_ratio`: Roughness-to-slope ratio
- `gamma`: Gamma
- `gamma_star`: Gamma star
- `local_optima_ratio`: Local optima ratio
- `autocorrelation`: Fitness autocorrelation
- `fdc`: Fitness-distance correlation
- `evolvability_enhancing_fraction`: Evolvability-enhancing mutation fraction
- `global_optima_accessibility`: Global optimum accessibility

`classify_epistasis` uses reproducible GraphFLA motif sampling with a fixed seed;
the resolved cutoff is recorded per assay. Gamma and gamma-star use the bundled
exact sparse traversal with GraphFLA's existing pooled moment kernels. Its
equivalence checks on two full ProteinGym assays are in
`gamma_validation.json`; no genotype rows are subsampled. For large landscapes,
FDC uses an exact chunked Hamming calculation to the nearest global optimum;
the single- and tied-optimum checks are in `fdc_validation.json`.
`global_idiosyncratic_index` uses one worker, `min_pairs=3`, and the fixed seed.
`autocorrelation` uses 1,000 random walks of at most 20 visited states, lag 1,
and the same fixed seed. Other scalar settings are recorded in the JSON
provenance. For landscapes with at least 50,000 retained graph configurations,
the EE fraction uses a bundled degree-bucketed implementation of GraphFLA's
nonfocal-moment and p-value/BH kernels; it was checked against the public API
and the prior exact helper on two full landscapes with zero numerical
difference. The neutrality metric is excluded.

Per-assay feature status: {'complete': 36}. Missing/non-finite values are JSON
`null`; explanations are in `provenance.graphfla.metric_failures`.

## Reproduction

From the repository root, run:

```bash
./.venv-ci39/bin/python docs/home/insights/proteingym/prepare_proteingym.py
```

The script verifies source SHA-256 values, scans and validates the official data,
and processes at most two assays concurrently in separate single-threaded
subprocesses. A worker is stopped above the configured resident-memory ceiling
or assay timeout;
individual metric limits and incomplete metric reasons are recorded in the output.
Use `--limit 3` for an explicitly partial preview.
