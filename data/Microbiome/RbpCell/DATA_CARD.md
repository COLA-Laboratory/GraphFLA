# Dufloo RBP × NCI-60 pseudotype infectivity — data card

## Source
Dufloo, Andreu-Moreno, Moreno-García, Valero-Rello & Sanjuán,
"Receptor-binding proteins from animal viruses are broadly compatible with
human cell entry factors," *Nat. Microbiol.* (2025).
DOI: [10.1038/s41564-024-01879-4](https://doi.org/10.1038/s41564-024-01879-4).

Built from the article's deposited supplementary workbooks:

| file (local) | Springer filename | contents used |
|---|---|---|
| `Supp_Tables_1-6.xlsx`  | `MOESM3_ESM.xlsx` | Supp Tables S1 (virus/RBP metadata), S4 (per-virus HEK293 screen outcome), S5 (NCI-60 cell line metadata), S6 (the 82×51 infectivity matrix). |
| `Source_Data_Fig1.xlsx` | `MOESM4_ESM.xlsx` | Duplicate of a subset of S6 — not re-used here. Kept for reference. |
| `Supp_Info.pdf`         | `MOESM1_ESM.pdf`  | 24 supplementary figures. Kept for reference only. |

## Background in one paragraph
The authors cloned 102 animal-virus receptor-binding proteins (RBPs) covering
14 enveloped-virus families and packaged each onto a VSV pseudotype (or a
lentiviral backup when VSV failed). Of the 102 constructs, 82 produced
infectious pseudotypes on HEK293 or HUVEC screening cells. These 82
pseudotypes were then inoculated onto a 51-member sub-panel of the NCI-60
human cancer-cell-line collection. Per well, GFP-positive fraction was
measured, a positive-call `Infection yes/no` was made, and the authors
reported a per-RBP-rescaled scalar **log2(R + 1)** where R is the GFP-positive
fraction divided by the per-RBP maximum across the 51 cells. Higher
log2(R+1) ⇒ greater relative pseudotype entry into that cell line (for that
RBP).

The central finding is that interspecies-barrier compatibility is
surprisingly weak at the receptor-binding step — most RBPs can enter
multiple human cell lines. Hit rates vary substantially across viral
families.

## Search space X
Every row in `compatibility.csv` is one (rbp, cell_line) pair on the tested
grid:

| column | type | meaning |
|---|---|---|
| `rbp` | string | virus common name used as the RBP identifier, e.g. `Lassa virus`, `Vesicular stomatitis virus`. 82 unique. |
| `cell_line` | string | NCI-60 short name, e.g. `A549`, `HEK293T-like`, `SK-OV-3`. 51 unique. |
| `viral_family` | string | one of 14 families (see below). Metadata; not part of the candidate choice. |
| `genus` | string | ICTV genus. Metadata. |

Search space size: **4,182 pairs = 82 RBPs × 51 cells** (full factorial over the
tested sub-pool). The 20 non-producer RBPs are captured in
`nonproducer_rbps.csv` — they are not part of the candidate pool because no
pseudotype could be made.

### The 14 viral families

| family | n_rbps |
|---|---|
| Rhabdoviridae    | 17 |
| Peribunyaviridae | 11 |
| Arenaviridae     |  9 |
| Hantaviridae     |  7 |
| Nairoviridae     |  6 |
| Phenuiviridae    |  5 |
| Togaviridae      |  5 |
| Orthomyxoviridae |  4 |
| Bornaviridae     |  3 |
| Coronaviridae    |  3 |
| Filoviridae      |  3 |
| Flaviviridae     |  3 |
| Paramyxoviridae  |  4 |
| Matonaviridae    |  2 |

## Objectives f

| column | type | direction | meaning |
|---|---|---|---|
| `log2_R_plus_1` | float | **maximize** | paper's per-RBP-rescaled entry score: `log2(R+1)` where `R = %GFP_pos / max_cells(%GFP_pos)` for that RBP. Range 0 .. 6.66. The scalar is **rescaled within each RBP**, so cross-RBP magnitudes are not comparable — higher just means "this cell line is near the best for this RBP". |
| `gfp_pct` | float | — | raw % GFP-positive cells. Kept as reference. Not used as the primary scalar because it is not normalised for pseudotype infectious titre. |
| `gfp_pct_empty_ctrl` | float | — | % GFP-positive in an empty-plasmid pseudotype control on the same cell line. The paper calls "infection" when `gfp_pct - control > threshold`. |
| `infection_call` | int (0/1) | — | paper's binary positivity call per pair. 64.5 % positive overall. |

## Caveats

- **READ THIS: the scalar is per-RBP-normalised, not a raw titre.** Two RBPs
  both showing `log2_R_plus_1 = 6.66` is not evidence they are equally
  infectious — it means each is at its own maximum on the cell panel. If you
  need absolute titre, use `gfp_pct` instead.
- **Only 82 of 102 cloned RBPs are in the grid.** The other 20 are in
  `nonproducer_rbps.csv` because no infectious pseudotype could be made from
  the construct. Don't silently pad them with zeros — that would bias any
  optimizer.
- **log2(R+1) = 0 is common** (34.5 % of cells are "no infection"). The
  distribution is thus zero-heavy; optimisers that assume Gaussian residuals
  should handle the floor explicitly.
- **Per-RBP replication is unclear** from the supplementary tables — the
  paper says "biological replicates" but Table S6 is a single scalar per
  pair. Treat every cell as n=1 for the purpose of benchmarking.
- **NCI-60 sub-panel, not the full 60.** The tested set is 51 lines; 9 NCI-60
  lines are absent. Tissue-of-origin coverage is still broad (Breast, CNS,
  Colon, Leukaemia, Lung, Melanoma, Ovarian, Prostate, Renal).
- **Cell-line metadata in Tables S8/S9/S10** contains per-surface-receptor
  transcript abundances — useful for featurising the cell axis, but not
  loaded by `curate_dataset.py` (they are not used as candidates, just
  descriptors). Read them directly from `Supp_Tables_1-6.xlsx` if needed.

## Files in this directory
```
compatibility.csv      # 4,182 rows — the main benchmark
rbp_metadata.csv       # 82 rows   — per-RBP virus family/accession/size
cell_metadata.csv      # 51 rows   — NCI-60 tissue/age/sex
nonproducer_rbps.csv   # 20 rows   — RBPs that didn't yield pseudotypes
DATA_CARD.md           # this file
```
