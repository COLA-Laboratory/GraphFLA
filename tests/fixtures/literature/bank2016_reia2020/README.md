# Hsp90 fitness input

Unmodified bytes of `Empirical_Landscapes/HSP90_fitness_landscape.txt`, saved
with a `.tsv` extension. This file contains 640 amino-acid variants and one
stop-codon control. The test excludes that control.

- Source: [Reia and Campos (2020), Dryad archive](https://doi.org/10.5061/dryad.41ns1rn9r), version 49951.
- Article: [Analysis of statistical correlations between properties of adaptive walks in fitness landscapes](https://doi.org/10.1098/rsos.192118), Figure 1.
- Original experiment: [Bank et al. (2016)](https://doi.org/10.1073/pnas.1612676113).
- Archive license: CC0. Retrieved 2026-09-30 through Dryad's visible download link.
- Archive SHA-256: `6cc5fa44a824f24aee65958b813c3243c754103eae42081e164d715bb1a4bb6b`.
- File SHA-256: `06a4e14f7486540149ca7f82fe206d1d17650595515201bdd8282cd11bba6988`.

Reia and Campos acknowledge Bank for supplying the landscape. Their Figure 1
independently specifies all six peak sequences used as the expected result.
This input does not include Bank's posterior samples. Its epistasis fractions
do not reproduce Bank's reported fractions and are not certified test targets.

The project's `data/BioSequence/Bank2016a.csv` is a different input: its values
match final-timepoint replicate-1 read counts averaged across synonymous
sequences, to six decimal places. It has not been replaced by this fixture.
