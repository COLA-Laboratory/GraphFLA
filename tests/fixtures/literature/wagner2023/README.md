# Wagner 2023 EE fixtures

Small numerical extracts from the protein (Lite et al. 2020, GEO GSE153897) and
RNA (Domingo et al. 2018, Supplementary Table S1) datasets analyzed by Wagner
(2023), doi:10.1038/s41467-023-39321-8. These are empirical facts and author
numerical results; no author source code is redistributed. This extraction
does not assert a license for the entire repository or source datasets.

`provenance.json` records the input URLs, local acquisition paths, original
hashes, author commit, preprocessing and hashes of every fixture. The CSV.gz
files keep input order and full float64 values (17 significant decimal digits).
Read with `float_precision="round_trip"`. RNA keeps the ten varied positions
and the original measurement SE; variance=6*SE**2 is applied explicitly only
when reproducing the author's measurement-error procedure.

The two compressed NPZ files preserve the author tables' row order, with
source/target indices mapped to the CSV rows, printed four-decimal `delfit`,
and the two signed significance flags. Load with `allow_pickle=False`.
They occupy approximately 1.1 MB together rather than copying the full output
tables, which contain many unrelated columns.

`python -m validation.ee_mutations` independently generates single substitutions
(codon-restricted for protein), checks every signed decision, reproduces the
four printed beneficial/deleterious counts, and verifies the corrected estimator
against a separate oracle. The original target-variance duplication and the
rounded classification are replayed only in validation. See
`validation/EE_MUTATIONS_REVIEW.md` for the two RNA printed-effect discrepancies,
the corrected-formula results and the public API's uncertainty limitations.

The executable regression entry is now
`python -m validation.tests --literature-study Wagner2023 -q`. It is separate
from the default basic tests, cites versioned cases, and also checks each public
table decision against the independent oracle. See `validation/TESTING.md` for
the reusable contract and `validation/EE_TEST_AUDIT.md` for resource observations.
