# Gamma empirical inputs

Three full 32-genotype numerical tables. `provenance.json` pins source URLs,
versions, raw/derived hashes, attribution and conversion rules.

- `csi.csv`: Franke, Klözer, de Visser & Krug (2011),
  [Table S1](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1002134.s009&type=supplementary),
  CC BY. Retain arg/pyr/leu/oli/crn, fixing fwn/phe/lys to zero. Printed W values
  are preserved; read genotype as a string, then use ln(W). The source table
  was independently extracted again from the PDF and all 32 entries checked.
- `csi2008.csv`: de Visser, Park & Krug,
  [arXiv:0807.3002v1](https://arxiv.org/abs/0807.3002), Table 1 pp.32–33.
  The earlier printed CS I values have different rounding. No rank-based
  tie-breaking or transformation is used beyond ln(W).
- `tem_mic.csv`: the numerical table released with
  [Ogbunugafor (2026)](https://doi.org/10.1093/genetics/iyag140),
  [commit ae47fda1](https://github.com/OgPlexus/DEFPreflect/tree/ae47fda1f973c25551d4f45129287fff25ac4d6b).
  Byte-identical to `1. FINAL_DataBinary_MIC.csv`; use five binary coordinates
  and ln(MIC), not its base-10 `log_MIC` column. These are numerical measurements;
  no license for the entire author repository or 2008 manuscript is asserted.

All tests use exact ties, full squares, maximization and no randomness. Each
input supplies 80 squares / 640 directed comparisons; the production graph
retains all 32 configurations. Independent targets are direct equation sums,
not GraphFLA snapshots. Only the 2011 csI gamma case is promoted as a printed
Figure 4 result; gamma-star and TEM Figure 4 discrepancies remain explicit.

```sh
python -m validation.tests --literature-study Ferretti2016 -q
python -m validation.gamma
```

The diagnostic command prints failed published comparisons as well as equation
agreement. Its completion is not certification that every target matches.
See [the review](../../../../validation/GAMMA_REVIEW.md) for full distinctions.
