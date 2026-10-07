# ProteinGym × GraphFLA dataset

This folder contains the landing-page scatter-plot data joined from ProteinGym
v1.3 substitution assays, GraphFLA landscape features, and ProteinGym’s published
zero-shot per-assay Spearman scores.

`data.json` is the frontend input. It has one record for each eligible assay,
14 GraphFLA features, and the 97 current zero-shot model score columns. The
assay publication is listed on each record; exact sources, file hashes,
calculation parameters, and explanations for null feature values are in
`provenance.json`. `source_report.md` summarizes the filtering, validation, and
construction conventions. `prepare_proteingym.py`, `sparse_gamma.py`, and
`bounded_ee_fast.py` are the reproducible preparation code. The validation
JSON files document exact equivalence checks for gamma, EE, and FDC, including
tied global optima.

The strict eligibility rule, as in the NeurIPS 2025 paper, is a mean of more than one substitution token per
measured assay row. This mean is computed from every row before landscape
construction. The landscape receives the complete measured population after
invariant sequence positions are removed. GraphFLA then removes configurations
with no observed one-edit neighbor; records keep both the measured variant count
and the graph configuration count.

Model values come directly from ProteinGym’s published zero-shot per-DMS
Spearman table. No model was trained or run for this dataset. Feature metrics
are computed from each GraphFLA landscape with the parameters in
`provenance.json`; exact source hashes and the pinned performance-table commit
are recorded there.

To regenerate the data from the repository root, use the scientific environment:

```bash
./.venv-ci39/bin/python docs/home/insights/proteingym/prepare_proteingym.py
```

The script verifies SHA-256 hashes, processes at most two assays concurrently,
limits standard workers to 2 GiB resident memory, and applies per-metric and
per-assay timeouts. `--limit 3` creates a clearly marked partial preview. The
cap can be raised for a single resource-heavy assay with
`--worker-only DMS_ID --max-rss-mb 8192`; `--assemble-only` then rebuilds the
joined files from assay caches. Any metric that cannot be computed is `null`
in `data.json`, with its reason in `provenance.json`.

## Sources

- [ProteinGym v1.3 release record](https://zenodo.org/records/15293562)
- [ProteinGym official repository](https://github.com/OATML-Markslab/ProteinGym)
- [NeurIPS ProteinGym benchmark paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee-Paper-Datasets_and_Benchmarks.pdf)
