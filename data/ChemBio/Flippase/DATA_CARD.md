# Chua capsule-flippase × CPS-precursor Bar-seq — data card

## Source
Chua, Wong, Chun, Ng Chyi Shien, Su, Maiwald, Chew, Lin, Hockenberry, Luo,
Sham, "Massively parallel barcode sequencing revealed the interchangeability
of capsule transporters in *Streptococcus pneumoniae*,"
*Science Advances* 11(4), 24 Jan 2025.
DOI: [10.1126/sciadv.adr0162](https://doi.org/10.1126/sciadv.adr0162).
Open-access PDF via PMC: [PMC11759038](https://pmc.ncbi.nlm.nih.gov/articles/PMC11759038/).

Built from:

| local file | Science Advances filename | contents used |
|---|---|---|
| `adr0162_data_s1.xlsx` | `sciadv.adr0162_data_s1.zip` → `adr0162_data_s1.xlsx` | 5 sheets: (A) 81×81 substrate Tanimoto, (B) 91×91 flippase RMSD, (C) per-serotype library sizes, (D) 80×79 output-library Bar-seq RPM, (E) 80×79 fold-change (×100). Downloaded via EuropePMC. |
| `Supp_Info.pdf` | `sciadv.adr0162_sm.pdf` | 15 supplementary figures + 13 supp tables. Kept as reference; not parsed. |

## Background in one paragraph
Pneumococcal capsular polysaccharides (CPS) — the main virulence factor
with 107 known serotypes — are assembled on undecaprenyl-phosphate in the
cytoplasm, then flipped across the plasma membrane by the MOP-family
flippase CpsJ (Wzx). The authors built a barcoded library of 83 cpsJ
alleles (one per non-redundant pneumococcal serotype) plus 4 gain-of-
function cps23BJ(P254S X) double-mutants isolated by selection for lipid-II
flipping (i.e. that can substitute the essential peptidoglycan flippase
MurJ/YtgP). The pool was separately introduced into 79 isogenic capsule-
switch strains, the native cpsJ was deleted, and Bar-seq counted which
flippase barcodes survived selection. After excluding 3 non-functional
ectopic cassettes (cps1J, cps7FJ, cps33DJ) and 3 amplification failures,
the final matrix is **80 flippases × 79 serotype substrates = 6320 cells**.
Hierarchical clustering on the fold-change heatmap groups flippases into
"strictly specific", "type-specific", and "relaxed" (Cps7AJ, Cps33BJ,
Cps33CJ, Cps34J + all 4 cps23BJ** mutants). Pathologically relaxed
flippases are toxic in some backgrounds because they flip incomplete
precursors and sequester Und-P.

## Search space X
Every row in `flippase_complementation.csv` is one (flippase, substrate)
combination:

| column | type | meaning |
|---|---|---|
| `flippase_id` | string | 80 unique. 76 wild-type CpsJ alleles (e.g. `2J`, `19B`), 4 gain-of-function mutants `23BJ(T38M)`, `23BJ(I246T)`, `23BJ(Q276L)`, `23BJ(L414P)`. The prefix `Cps` and the P254S background of the mutants are implied and spelled out in `flippase_long`. |
| `serotype_id` | string | 79 unique. Pneumococcal CPS serotypes tested as substrates: `1, 2, 4, 5, 6A, 6B, 6C, 6D, 7A, 7B, 7C, 7F, 8, 9A, 9L, 9N, 9V, 10A, 10C, 10F, 12A, 12B, 12F, 13, 14, 15A, 15B, 15C, 15F, 16F, 17A, 17F, 18A, 18B, 18C, 18F, 19A, 19B, 19C, 19F, 20, 21, 22A, 22F, 23A, 23B, 23F, 24A, 24B, 24F, 27, 28A, 28F, 29, 31, 32A, 32F, 33A, 33B, 33C, 33D, 33F, 34, 35A, 35B, 35C, 36, 39, 40, 41A, 41F, 42, 43, 44, 45, 46, 47A, 47F, 48`. |
| `flippase_long`, `substrate_long` | string | `Cps<flippase>` and `Serotype <id> CPS precursor`. Metadata only. |

Search space size: **6320 combinations** (80 × 79). Paper-stated total.

### Flippase metadata (80 rows, `flippase_metadata.csv`)

| column | type | meaning |
|---|---|---|
| `native_serotype` | string | Serotype whose capsule the flippase natively transports (e.g. `23B` for `23BJ(T38M)`). |
| `is_gof` | bool | `True` for the 4 cps23BJ(P254S X) double-mutants isolated for lipid-II complementation. |
| `flippase_group` | string | `relaxed` (4 paper-named), `gof_mutant` (4 cps23BJ**), `strict_named` (6 explicitly named in Fig S13 / S7: 2J, 4J, 5J, 14J, 31J, 45J), `unclassified` (66). Paper's full 3-way hierarchical clustering on Fig 2E is not shipped as a table in Data S1, so most flippases are `unclassified` and this column is metadata, **not** the optimization target. |

### Substrate metadata (79 rows, `substrate_metadata.csv`)

| column | type | meaning |
|---|---|---|
| `library_input_N` | int | Number of transformants in the input library before native-cpsJ deletion. Range 7k (serotype 27) .. 2.4M (serotype 15C). |
| `library_output_N` | float | Transformants in the output library after Erm selection. NaN for 11 serotypes reported as "TNTC" (too numerous to count). |
| `library_output_TNTC` | bool | Flag for the 11 TNTC serotypes. |
| `n_sugars` | int | Repeat-unit sugar count. Only serotype 17A (=8) is filled from paper text; others NaN. |

## Objectives f

| column | type | direction | meaning |
|---|---|---|---|
| `output_rpm` | float | **maximize** | Reads per million of the flippase's barcode in the post-deletion output library. Range 0..49.4. **This is the recommended primary scalar**: within each serotype it directly measures which flippase barcodes survived Erm selection. Cleanly recovers paper claims (see `validate_curated.py`). |
| `fold_change` | float | maximize | Paper's "LogFold-change (%)" heatmap value = (output RPM / input RPM) * 100. Range 1..406. Binarized as `fold_change >= 100` (equivalently ratio >= 1) in the paper's ML analysis. |
| `ratio` | float | maximize | `fold_change / 100`. Plain ratio-scale. |
| `log2_fold_change` | float | maximize | `log2(ratio)`. Signed, zero-centered. NaN where `fold_change == 0`. |
| `is_cognate` | bool | — | `True` iff the flippase's `native_serotype` matches `serotype_id`. 80 of 6320 rows. |
| `is_gof` | bool | — | `True` for the 4 cps23BJ** GoF mutant rows × 79 serotypes = 316 cells. |
| `flippase_group` | string | — | Copied from `flippase_metadata.csv`. |

## Caveats

**READ THIS: `fold_change` and `output_rpm` tell different stories.** On
serotype 17A the paper says only Cps17AJ and Cps17FJ should transport the
(largest, 8-sugar) precursor. By `output_rpm` those two flippases rank
#1-2 (cleanly). But by `fold_change` they rank #5 and outside top-5 because
low-abundance barcodes in the input library inflate ratios. Similarly, the
extreme `fold_change` values (fc > 100) in low-library-size serotypes often reflect noise spikes from low-abundance barcodes, not real biological wins. **Treat `fold_change` as the paper's reported metric
but use `output_rpm` when you need a clean per-serotype ranking.**

- **Input-library size varies 300×.** 7k (serotype 27) to 2.4M
  (serotype 15C / serotype 22A). Low-input serotypes give noisy
  fold_change values. The 4 serotypes with `library_output_N < 5k`
  (27, 12F, 12B, 31) are particularly unreliable.
- **11 serotypes report `library_output_N = "TNTC"`** (too numerous to
  count, i.e. very high survival). Mapped to NaN with a flag.
- **3 flippase alleles were pre-excluded** (cps1J, cps7FJ, cps33DJ —
  non-functional when expressed ectopically at the CEP locus) and 3 more
  (cps10BJ, cps25AJ, cps25FJ, cps38J) were excluded for amplification /
  capsule-switch-strain issues. So 87 alleles were initially cloned, 80
  passed QC. See Results → "high-throughput approach".
- **79 serotypes tested (not 107).** The paper's capsule-switch collection
  covers 79 of the 107 pneumococcal serotypes.
- **4 rows are gain-of-function double mutants** `cps23BJ(P254S X)` with
  X ∈ {T38M, I246T, Q276L, L414P}. Paper says all 4 can replace the
  essential peptidoglycan flippase YtgP/MurJ. Their cognate column is
  `23B` (native), not a separate column.
- **`is_cognate` is a strain-serotype notion.** For the 4 GoF mutants,
  `is_cognate=True` means `serotype_id == 23B`. They have `23B` as their
  native serotype.
- **Cognate cells can have fold_change < 100.** Median cognate ratio is
  0.90 (fold_change 90). The paper's binarization threshold at ratio ≥ 1
  is stringent — many genuinely-complementing native pairs fall just
  short. The `flippase_complementation.csv` preserves the raw signal and
  does not binarize.
- **`flippase_group` is hand-coded from the paper text and figures**,
  covering only the paper's explicitly-named exemplars: 4 relaxed + 4 GoF
  + 6 strict. The full 3-way hierarchical-clustering assignment in Fig 2E
  / Fig 3 is not a deliverable in Data S1; 66 flippases are labelled
  `unclassified`. **This column is metadata only — it is NOT the
  optimization target.**
- **Auxiliary similarity matrices** (`substrate_tanimoto.csv`,
  `flippase_rmsd.csv`) cover slightly broader sets than the benchmark
  (81×81 substrates, 91×91 flippases) because they include alleles that
  were dropped from the Bar-seq experiment. Join via serotype / flippase
  name if you want structural/chemical priors.
- **Bar-seq toxicity confounds the scalar.** Relaxed flippases
  (Cps7AJ/33BJ/33CJ/34J, and Cps23BJ**) are *toxic* in some backgrounds
  (Fig S6/S7) because they flip incomplete precursors. The fold-change
  metric conflates "doesn't complement" with "complements but is toxic".
  If you need the toxicity signal separately it's not in Data S1 — would
  need to re-analyse input-library reads before the ∆cpsJ selection.

## Files in this directory
```
adr0162_data_s1.xlsx                    # 221 kB, 5 sheets (raw)
Supp_Info.pdf                           # 8.7 MB, Figs S1-S15 + Tables S1-S13
paper.txt                               # 55 kB, main text
curate_dataset.py                       # raw → curated/*.csv
validate_curated.py                     # paper-claim assertions
curated/
  flippase_complementation.csv          # 6320 rows — main benchmark
  flippase_metadata.csv                 #   80 rows — per-flippase columns
  substrate_metadata.csv                #   79 rows — per-serotype columns
  substrate_tanimoto.csv                # 81 × 81 chemical similarity of CPS units
  flippase_rmsd.csv                     # 91 × 91 AlphaFold-model RMSD matrix
  DATA_CARD.md                          # this file
```
