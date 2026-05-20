# Kooij cyanimide × 12-DUB HTS screen — data card

## Source
Kooij, R. et al., "High-Throughput Synthesis and Screening of a
Cyanimide Library Identifies Selective Inhibitors of ISG15-Specific
Protease mUSP18," *Angew. Chem. Int. Ed.* (2025).
DOI: [10.1002/anie.202510941](https://doi.org/10.1002/anie.202510941).
PMCID: [PMC12668310](https://europepmc.org/article/PMC/PMC12668310).
Preprint: [bioRxiv 2025.04.18.649523](https://www.biorxiv.org/content/10.1101/2025.04.18.649523v1).

Built from the Wiley SI zip (EuropePMC supplementaryFiles endpoint):
`ANIE-64-e202510941-s001.xlsx` (HTS + controls) and `-s002.xlsx`
(CA structures). Both files are copied into the parent folder.

## Background in one paragraph
Kooij and co-workers use an Echo acoustic liquid handler to execute
amide-coupling reactions in-plate between **16 cyanimide-bearing amine
building blocks (BB01..BB16)** and **~469 structurally diverse
carboxylic acids (CA001..CA469)**, producing a 7,536-member crude
library in which every product carries the DUB-privileged *cyanimide*
warhead. Each crude reaction mixture is screened **without purification**
against a 12-member panel of deubiquitinases / ubiquitin-like proteases
(UCHL1/3/5, OTUD1, OTUB2, USP8, USP16, mUSP18, USP24, USP30, USP32,
SENP1) in a fluorogenic substrate assay, giving **90,168 single-point
%inhibition values** (12 × 7,536 minus ~264 missing wells for OTUB2/USP8).
The discovery goal is a **selective mUSP18 hit**: from the primary
screen the authors identify selective hits and, after re-synthesis +
analog campaign, advance a lead to nanomolar potency against mUSP18
with high selectivity over other DUBs. The final optimised analog is
NOT in the primary 7,536-member grid — this benchmark represents the
HTS stage only.

## Search space X

Every row in `dub_hts.csv` is one (compound, DUB) measurement; every
row in `dub_hts_wide.csv` is one compound with 12 DUB columns.

| column | type | meaning |
|---|---|---|
| `compound_id` | str | `BB##CA##` unique reaction ID. 7,504 unique compound IDs over 7,536 HTS wells (32 replicate wells). |
| `smiles` | str | Crude-coupling product SMILES (pyrrolidine-3-amine cyanimide + CA carboxylic-acid coupling partner). |
| `mw` | float | Molecular weight (Da) of the intended product. |
| `bb` | str | 16 amine fragments `BB01..BB16` (full structures shown in Supp. Fig. SBB). |
| `ca` | str | 469 carboxylic-acid fragments `CA001..CA469` (SMILES in `ca_structures.csv`). |
| `plate` | str | Echo plate ID (e.g. `NCN-1`, `NCN-2`, …). |
| `well` | str | 384-well position (`A01`..`P24`). |
| `dub` | str | Target protease, one of 12 (`UCHL1`, `UCHL3`, `UCHL5`, `OTUD1`, `OTUB2`, `USP8`, `USP16`, `mUSP18`, `USP24`, `USP30`, `USP32`, `SENP1`). |

Search space size for primary optimization: **7,536 compounds × 12 DUBs
= 90,432 design points**, of which 90,168 are measured (264 OTUB2/USP8
wells are missing).

A reduced single-objective framing (mUSP18 only) gives **7,536 design
points**. The natural combinatorial axes are `bb × ca` (16 × 469 = 7,504
distinct products).

## Objectives f

| column | type | direction | meaning |
|---|---|---|---|
| `inhibition` | float | **maximize** | % enzyme inhibition at the single-point HTS dose. Paper normalises to DMSO (0 %) and NEM (100 %) controls; values > 100 or < 0 are noise. |

### Recommended composite objective (selectivity)
For the paper's actual discovery goal, define
```
selectivity(c) = mUSP18(c) − max_{d ≠ mUSP18} d(c)
```
on the wide table. A positive selectivity score means the compound
preferentially inhibits mUSP18 over all other DUBs in the panel.
Higher selectivity indicates better mUSP18-specific inhibitors.

## Caveats

- **Crude reaction screening.** Every %inhibition value is from an
  *unpurified* Echo-dispensed coupling reaction. Apparent hits may be
  starting-material artefacts; the CA-only (`ca_controls.csv`) and
  BB-only (`bb_controls.csv`) counter-screens on mUSP18 are included for
  de-artefacting. 475 of the 469 CAs are screened as bare acids and also
  with DIC/HOBT activation; the `pct_inh_ca_activated` column is the
  safer baseline.
- **The paper's final lead is NOT in this grid.** The paper's headline
  optimised compound is a *post-HTS analog* resynthesised after the screen.
  The primary grid contains the seed hits from which the optimised analog
  was derived.
- **Missing wells.** OTUB2 has 7,296 measured wells (240 missing) and
  USP8 has 7,512 (24 missing) because two plates of those assays failed
  QC. The other 10 DUBs have the full 7,536 wells.
- **Values can exceed 100 % and go strongly negative.** USP30 tops
  out at 102.40, USP8 at 104.92, OTUD1 goes to −154.46 %. These are
  well-level read-noise and fluorogenic-substrate interference; treat
  `inhibition` as continuous and unbounded, not as a [0, 100]
  probability.
- **32 replicate compound IDs.** 7,536 wells but 7,504 unique
  `compound_id` strings — some BB × CA pairs are dispensed twice across
  plates as intra-assay reproducibility controls. Wide-format CSV keeps
  the 7,536 rows; compound_id is therefore not a primary key. Use
  `(plate, well)` or the row order for a unique key.
- **Single-point screen dose.** The HTS assay is a fixed-concentration
  (typically 10 µM crude) single-replicate read. The IC50s in the paper
  (35 nM for BB07CA902 etc.) come from purified resynthesised compounds
  in a separate secondary assay; that data is NOT in the xlsx.
- **mUSP18 is the mouse ortholog.** The paper uses mouse USP18 because
  human USP18 is hard to express; 35 nM IC50 on mUSP18 translates into
  "USP18 ISGylation activity in cells" (Fig. 5 of the paper). The other
  11 panel DUBs are human.

## Files in this directory
```
dub_hts.csv           # 90,168 rows — long-format benchmark (compound × DUB)
dub_hts_wide.csv      # 7,536 rows × 19 cols — wide, one row per HTS well
ca_controls.csv       # 475 rows — CA-only mUSP18 counter-screen
bb_controls.csv       # 16 rows — BB-only 12-DUB counter-screen
ca_structures.csv     # 471 rows — CA## → SMILES lookup
DATA_CARD.md          # this file
```
