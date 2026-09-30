# Published landscape fixtures

These numerical inputs are compact extracts of reviewed author artifacts.
They are not generated from GraphFLA outputs. Original sources, licenses,
checksums and extraction details accompany each fixture in `provenance.json`
or a directory-specific README. Tests verify the fixture checksum.

- **Wu et al. (2016):** [eLife article](https://doi.org/10.7554/eLife.16965),
  [publisher supplements](https://cdn.elifesciences.org/articles/16965/elife-16965-supp-v2.zip),
  CC BY. Select `Variants`/`Fitness` from Dataset 1 and
  `Variants`/`Imputed fitness` from Dataset 2, mark measured/imputed membership,
  concatenate and sort. All 160,000 variants are required, including 10,639
  imputed entries. Expected counts and rounded accessibility come from the
  article's Results, p. 9 and Figure 4A (peaks), and p. 11 / Figure 4—figure
  supplement 2C (accessibility), with methods on pp. 20–22.
- **Johnston et al. (2024):** [PNAS article](https://doi.org/10.1073/pnas.2400439121),
  [CaltechDATA archive](https://doi.org/10.22002/h5rah-5z170), data licensed CC0.
  Select no-stop measured rows and the 871 imputed rows using the source paths
  in `provenance.json`. Retain `AAs`, `fitness`, `active` and measured/imputed
  membership. The paper's 520-peak target counts active measured candidates
  against the complete 160,000-vertex neighborhood. `active` is the author's
  provided classification; its upstream replicate thresholding is not retested
  by this peak check. Main Results and SI Figure S29 specify the target.
- **Bank / Reia and Campos:** see `bank2016_reia2020/README.md` for the unmodified
  small downstream archive and the separate read-count provenance issue.

The compressed CSVs retain source float64 values with 17 significant digits.
Read them using `float_precision="round_trip"`. Gzip timestamps are fixed to
zero for deterministic output. No values are thresholded, rounded, re-estimated
or tuned to match a target during curation. The extracted tables occupy about
3.3 MB compressed, so the empirical tests run offline without spreadsheet
libraries or multi-gigabyte author archives.
