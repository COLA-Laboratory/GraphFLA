# Qu MedPoly curated dataset — data card

## Source
Qu, Liu, Yang, Zheng, Huang, Wang, Xie, Zuo, Xia, Sun, Zhou, Xie, Lu, Zhu,
Yu, Liu, Zhou, Dai & Leung, "Selective utilization of medicinal
polysaccharides by human gut Bacteroides and Parabacteroides species,"
*Nat. Commun.* **16** (2025). DOI:
[10.1038/s41467-025-55845-7](https://doi.org/10.1038/s41467-025-55845-7).

Built from the paper's consolidated **Source Data** xlsx
(`Source_Data.xlsx`, article supplement MOESM7_ESM.xlsx) and
**Supplementary Data** xlsx (`Supplementary_Data.xlsx`, MOESM2_ESM.xlsx).

Sheets used:
- `Source_Data.xlsx` / `Fig. 1E, 2C` — the full raw growth table (long-form
  OD600 time course, 16,845 rows; 28 target strains + 3 Alistipes outgroups
  x 25 polysaccharide labels x up to 2 batches x 3 technical replicates x
  5 time points 0/12/24/36/48 h).
- `Supplementary_Data.xlsx` / `SupTable1` — 28-strain BacPUS reference
  (strain_id → species).
- `Supplementary_Data.xlsx` / `SupTable3A/B` — 17-herb source taxonomy and
  20-polysaccharide structural properties.

## Background in one paragraph
The authors assemble **BacPUS** (Bacteroides PolySaccharide Utilization
Set), a collection of 28 human gut Bacteroides and Parabacteroides
isolates, and screen each strain against 20 medicinal polysaccharides
extracted from 17 herb/mushroom sources (Ginseng, Dendrobium, Ganoderma,
Lycium, Astragalus, etc.; Dendrobium / Ganoderma / Lycium each yield two
preparations). Each strain is grown in minimal medium with the
polysaccharide as sole carbon source; bacterial density (**OD600**) is
measured at 0, 12, 24, 36 and 48 h, in 3 technical replicates per batch
(1–2 batches per polysaccharide). The 48 h OD600 is the paper's scalar
phenotype (Fig. 1E heatmap). Downstream experiments in the paper explain mechanistically why certain
strain × polysaccharide pairings show high selectivity.

## Search space X
Every row is one **(strain, polysaccharide)** combination:

| column | type | meaning |
|---|---|---|
| `strain_id`             | string | BacPUS ID, e.g. `DA183` |
| `strain`                | string | species name with underscore, e.g. `Bacteroides_uniformis` |
| `genus`                 | string | `Bacteroides` or `Parabacteroides` |
| `polysaccharide`        | string | paper's short label, e.g. `GPs`, `DPs_1`, `DPs_2` |
| `polysaccharide_name`   | string | full name, e.g. `Ginseng polysaccharides` |
| `polysaccharide_source` | string | source organism, e.g. `Panax ginseng` |
| `source_kingdom`        | string | `Plant` (17 of 20) or `Fungus` (3 of 20) |

Search space size: **560 pairs** = 28 strains x 20 polysaccharides.

### The 28 strains (BacPUS)

23 Bacteroides + 5 Parabacteroides. Three strain-species pairs appear
twice (different isolates of the same species): `B. fragilis` (DA486,
DA557), `B. clarus` (DA1439, DA647). Four strain_ids (DA347, DA486,
DA1439, DA1479) carry GTDB-reclassified alternate names in the raw growth
table (`Bacteroides_sp900066265`, `fragilis_A`, `sp003545565`,
`intestinalis_A`); we use the paper's SupTable1 canonical names.
`DA14`, `DA647`, `DA672` replaced contaminated `DA916`, `DA261`, `DA1466`
(paper's SupTable1 footnote).

See `strains.csv` for the full list.

### The 20 polysaccharides

17 source herbs/mushrooms; Dendrobium (DPs_1, DPs_2), Ganoderma lucidum
(GLPs_1, GLPs_2), and Lycium barbarum (LBPs_1, LBPs_2) each contribute two
polysaccharide preparations (different extraction methods or purification
protocols — identical structural descriptors in SupTable3B). Three of
the 20 are fungal (GLPs_1, GLPs_2, PPs from *Poria cocos*); the other 17
are plant-derived.

See `polysaccharides.csv` for the full list and source taxonomy.

## Objectives f

Single objective — **maximize** bacterial growth at 48 h.

| column | type | description |
|---|---|---|
| `OD600_48h`        | float | mean OD600 across all available replicates (both batches if present, 3 reps each). Paper's primary phenotype (Fig. 1E). |
| `OD600_48h_sd`     | float | standard deviation across all replicates |
| `n_rep`            | int   | 3 (single batch) or 6 (both batches × 3 reps). 224 / 560 pairs have n_rep=3; 336 / 560 have n_rep=6. |
| `OD600_48h_batch1` | float | batch 1 mean (always present) |
| `OD600_48h_batch2` | float | batch 2 mean (14 of 20 polysaccharides) or NaN |

The paper's negative control (NC, no carbon source) has mean OD600 ~0.005
at 48 h; the positive control (PC, glucose) reaches ~0.70 on average and
up to ~1.13 — anything substantially above the NC level indicates
utilization.

The paper highlights specific strain × polysaccharide utilization patterns,
including broad-spectrum substrates and narrow-specialist pairings. The
optimizer should discover these patterns from the data.

## Caveats

- **Paper-highlighted patterns have been removed from this card.**
  The paper's narrative flags specific strain × polysaccharide patterns
  that constitute the optimization target. These have been stripped to
  avoid leaking the answer to the optimizer.

- **Batch imbalance.** 12 of 20 polysaccharides were repeated in a second
  batch (AdPs, AnPs, CPPs, HAPs, PLPs, PPs, PQPs, PSPs, RAPs, RNPs, RPPs,
  SPs — `n_rep=6`). The other 8 (APs, DPs_1, DPs_2, GLPs_1, GLPs_2,
  LBPs_1, LBPs_2, GPs) have batch 1 only (`n_rep=3`). Both GPs and
  DPs_1/DPs_2 — the paper-highlighted polysaccharides — are n=3 only.
  This does not change the mean-OD600 ranking used in the paper's
  heatmap, but the uncertainty around Dendrobium selectivity is somewhat
  higher than for the 6-rep polysaccharides.

- **GTDB species reassignments.** DA347 (`B. finegoldii` in SupTable1 /
  `B. sp900066265` in SupTable4), DA486 (`B. fragilis` / `B. fragilis_A`),
  DA1439 (`B. clarus` / `B. sp003545565`), DA1479 (`B. intestinalis` /
  `B. intestinalis_A`) have two names in the raw data. We kept the paper's
  SupTable1 names (the ones used in the main text and figures).

- **Two pairs of same-species duplicates.** DA486 and DA557 are both
  labelled *B. fragilis*; DA1439 and DA647 both *B. clarus*. These are
  genuinely distinct isolates (different DA_num, contamination-replacement
  origin in the case of DA647) and were screened independently — their
  growth profiles are NOT identical (see `growth.csv`).

- **Alistipes outgroups dropped.** The source table also contains DA25
  (*Alistipes shahii*), DA253 (*A. onderdonkii*) and DA338 (*A. finegoldii*)
  as a distant-relative outgroup; per the paper's 28 × 20 scope we drop
  these. If you want a 31-strain extended benchmark, re-read the source
  sheet.

- **GPs sub-fractions excluded.** The source sheet also has `GPs_S`,
  `GPs_L`, `GPs_mix` — these are Ginseng polysaccharide size-fraction
  follow-up experiments used only in Fig. 2 of the paper, not part of the
  main 28 × 20 screen.

- **Negative OD600 values.** A handful of pairs show slightly negative
  mean OD600 (min = -0.012) due to plate-reader blank drift; these
  indicate no growth, not negative biomass. Treat as noise around zero.

- **Objective is OD600 at 48 h only** — the growth-curve shape (12 h / 24
  h / 36 h columns) is discarded here. The underlying raw sheet has these
  if you want AUC or lag-time objectives; re-read `Source_Data.xlsx` /
  `Fig. 1E, 2C`.

- **No single objective is a "yield" or "bioactivity" measure** — OD600 is
  a proxy for total biomass, which in this assay reflects the bacterium's
  ability to utilize the polysaccharide as sole carbon source. It does
  not measure immunomodulatory or anti-tumor potency of the
  polysaccharide-bacterium pair, which is what the authors ultimately
  care about; those downstream assays are outside the screen.

## Files in this directory
```
growth.csv              # 560 rows — main benchmark (28 strains x 20 polysaccharides)
strains.csv             #  28 rows — strain_id, species, genus
polysaccharides.csv     #  20 rows — short label, full name, source, kingdom
DATA_CARD.md            # this file
```
