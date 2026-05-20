# CSNN Chemfaces × 7-hGPCR yeast screen — data card

## Source
Thomsen, Sommer et al., "Labels as a feature: Network homophily for
systematically annotating human G protein-coupled receptor drug
interactions," *Nat. Commun.* (2025).
DOI: [10.1038/s41467-025-59418-6](https://doi.org/10.1038/s41467-025-59418-6).

Built from the Source Data file (MOESM6) accompanying the Nature
article. The main ML/analysis code + the ChEMBL/IUPHAR training data
lives at Zenodo 10.5281/zenodo.12532113 (not needed for this benchmark).

## Background in one paragraph
The authors constructed yeast platform strains expressing each of 7
human GPCRs — 5HT4B, ADRA2A, ADRA2B, OPRM1, MTNR1A, CHRM3, 5HT1A — with
NanoLUC as the reporter, then screened 550 compounds drawn primarily
from the ChemFaces natural-product library + Prestwick Chemical Library
references. For each compound × GPCR pair the assay output is the raw
luminescence (RLU). The paper refers to "all-vs-all 3,773 DTIs" (539
compounds × 7 GPCRs after QC); the published Source Data (Fig 4c) has
the cleaned 550 × 7 = 3,850 matrix with no missing values.

To give the optimiser a scalar that makes GPCR columns comparable we
compute a **robust Z-score per GPCR**: Z = (RLU − median(RLU_gpcr)) /
(1.4826 × MAD_gpcr). Higher Z = stronger agonist-like signal. The paper
uses |Z| > 3 as a positivity cut-off (178 such hits across 3,850).

## Search space X
Every row in `gpcr_dti.csv` is one (compound, hGPCR) pair:

| column | type | meaning |
|---|---|---|
| `catnr` | string | compound catalog number, e.g. `CFN96101` (ChemFaces) |
| `compound_name` | string | common name, e.g. `Melatonin`, `Loperamide_hydrochloride` |
| `hgpcr` | string | one of {5HT4B, ADRA2A, ADRA2B, OPRM1, MTNR1A, CHRM3, 5HT1A} |

Search space size: **3,850 pairs = 550 compounds × 7 GPCRs** (full
factorial, no missing cells). The optimiser's two categorical features
are `catnr` and `hgpcr`.

### The 7 GPCRs

ADRA2A, ADRA2B, 5HT1A, 5HT4B, CHRM3, MTNR1A, OPRM1. Hit rates vary
substantially across receptors — broader pharmacophore receptors have
higher hit rates than narrow-pocket receptors like OPRM1.

## Objectives f

| column | type | direction | meaning |
|---|---|---|---|
| `z_score` | float | **maximize** | robust Z-score vs per-GPCR background (median / MAD). > 3 = positive hit. Highly right-skewed — potent reference agonists can produce extreme Z values. |
| `rlu`     | float | — | raw NanoLUC luminescence (arbitrary units, per-GPCR scale differs by ~10×). Kept as reference. |
| `smiles`  | string | — | compound SMILES. Metadata. |

## Caveats

- **Z-score is extremely right-skewed.** Potent known agonists from the
  Prestwick reference library can produce Z values orders of magnitude
  above background because the per-GPCR MAD is tiny (most compounds give
  near-background RLU). Use `log10(1 + z_score)` or tanh if you want a
  more friendly distribution for surrogate models.
- **Yeast ≠ mammalian pharmacology.** The platform is sensitive enough
  to recover known agonists, but (a) downstream signalling biases the
  RLU away from strict Gi/Gq/Gs behaviour, and (b) several GPCRs give
  much lower dynamic range in yeast (paper Supplementary Fig S21). The
  hit rate and magnitudes shouldn't be taken as in-vivo affinities.
- **550 > 539 count mismatch.** Paper reports 539 compounds / 3,773 DTIs
  after QC; Source Data Fig 4c has 550 / 3,850. We use the 550 number
  because it's the published table.
- **RLU medians differ ~20× across GPCRs** (ADRA2A ~18,000; OPRM1 ~500).
  That's why Z-score uses per-GPCR centering.
- **CSNN paper trains a ML model** using ChEMBL/IUPHAR Ki values as
  weak labels and the experimental Z-score as the anchor. We don't need
  the ML training data for this benchmark — we only use the experimental
  screen. But the Z-score distribution here is biased toward compounds
  the authors expected to be active (literature-guided subsetting from
  ChemFaces).
- **Duplicate compounds are not exhaustively dedup'd.** Different salt
  forms (e.g. Loperamide_hydrochloride vs Loperamide) may appear as
  separate `catnr`s. Check by SMILES if that matters.

## Files in this directory
```
gpcr_dti.csv            # 3,850 rows — the main benchmark
compound_metadata.csv   # 550 compounds — catnr, name, SMILES
DATA_CARD.md            # this file
```
