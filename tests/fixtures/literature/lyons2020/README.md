# Lyons 2020 tRNA idiosyncrasy fixture

`trna.csv.gz` contains 28,530 viable genotype/fitness rows in original source
order. `provenance.json` records the original file and fixture hashes, author
commit, source URLs and preprocessing. Source data are numerical measurements
from Li et al. (2016), as redistributed and analyzed by Lyons et al. (2020).
This extraction does not assert a license for the entire author repository.

To rebuild from the hash-verified author `ExpData.txt`:

```python
import gzip
import numpy as np
import pandas as pd

df = pd.read_csv(source_path, sep="\t")
keep = df.Fit > 0.5
out = pd.DataFrame({
    "sequence": df.loc[keep, "Seq"],
    "fitness": np.log(df.loc[keep, "Fit"].to_numpy() / 0.5),
})
fixture_path.write_bytes(gzip.compress(
    out.to_csv(index=False, float_format="%.17g").encode(), mtime=0,
))
```

The 37,007 floor-valued rows are excluded. No viable genotype is dropped for
lacking single-mutant neighbors: 3,903 such isolates still belong in the random
control pool. Read with `float_precision="round_trip"`. The 72-character input
has invariant positions 0, 70 and 71; the authors enumerate offsets 1 through 69.

The published global targets and tolerances are frozen in
`validation/cases/lyons.trna.iid.v1.json`. Run
`python -m validation.idiosyncrasy` for independent enumeration, an exact
author-seed replay of the production kernel, and a public-function comparison
with the same random stream. The latter uses the existing GraphML importer to
retain the full population; normal `build_from_data` prunes isolates.

The paper's Figure 1a illustrative index is 0.49, while the released Figure 1a
code with seed 4033 gives 0.5080668756. The separate, pre-existing global-notebook
policy (`n**2+3`) gives 0.4905242284 for that mutation. The two policies are not
silently interchanged. See `validation/IDIOSYNCRASY_REVIEW.md`.


Run the dedicated suite with
`python -m validation.tests --literature-study Lyons2020 -q`. Default
`python -m pytest` excludes these empirical tests. The original global case is
unchanged; separate cases record the input population, public seed-0 independent
check and Figure 1a procedural discrepancy. Each supplies an explicit source
locator and evidence role under `validation/TESTING.md`.

Resource review (2026-10-02, macOS arm64 / Python 3.13.11): this 450,155-byte
compressed input is used in full. The five tests share one reproduction;
pytest took 10.68 s, and sampled worker-plus-child peak RSS was 1,186,660,352
bytes. The supervised invocation imposed 60 s and 1,536 MiB limits. RSS can
count shared pages more than once and the 50 ms sampler can miss short peaks.
These observations are correctness-run resource checks, not metric timing.
