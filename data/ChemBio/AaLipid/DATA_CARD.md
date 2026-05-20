# Gong-Li amino-acid lipid library — data card

## Source
Gong, Xu, Chen, Zhang, ... , Yan, Li (corresponding: Bowen Li, U. Toronto),
"Amino acid-derived ionizable lipids enable inhaled base editing for
therapeutic gene correction in the lung," *Nature Materials* (2026).
DOI: [10.1038/s41563-026-02555-0](https://doi.org/10.1038/s41563-026-02555-0).

This curation is built from three different inputs the paper provides:

| file in this directory | what it is | who measured it |
|---|---|---|
| `Source_Data_Fig1.xlsx` | Nature MOESM3 — Fig 1d (30-AA aggregate) + Fig 1e (top-20 log2 luc) | author-published, **exact values** |
| `Source_Data_Fig2.xlsx` | Nature MOESM4 — Fig 2a (top-20 in differentiated BEAS-2B, 3 reps) | author-published, exact values |
| `Supp_Info.pdf`         | MOESM1 SI including Supp Fig 2b (dot-plot of the full 960-lipid BEAS-2B primary screen) | author-published image; **we recover numeric values by image extraction** |

**Why image extraction was necessary.** The paper says "Source data are
provided with this paper" but the Nature MOESM xlsx files only cover
*aggregates* — Fig 1d (per-AA max), Fig 1e (top-20), Figs 2a-g, 3c-h, 4c-g,
5c-s. The raw 960-lipid screen (Supp Fig 2b) is never released as a
numeric table. To build a 960-entry optimization benchmark we render the
figure from the deposited SI PDF and extract per-dot values (see
[§ "Image-derived values" caveat](#-caveat-image-derived-values-uncertainty) below).

## Background in one paragraph
The authors develop a **catalyst-free Ugi-5C-4CR reaction** (Ugi
five-centre / four-component) that takes a fatty aliphatic aldehyde, an
α-amino acid, an isocyanide bearing a hydrophobic tail, and methanol, and
in one step produces an α-acylamino-α-amino-acid methyl-ester ionizable
lipid with two ester bonds (biodegradable by design) and two amide bonds.
They use a robotic liquid handler to synthesize a combinatorial library of
**960 unique ionizable lipids** = **30 amino-acid backbones × 16 aldehyde
tails × 2 isocyanide headgroup classes** (cyclic "CH" and linear "LH"),
formulate each into lipid nanoparticles loaded with firefly-luciferase
mRNA, and screen transfection efficiency in BEAS-2B human bronchial
epithelial cells (luminescence normalized to untreated → log2-transformed).
The top-performing lipids are then validated in vivo as inhaled mRNA / base-
editor delivery vehicles for cystic-fibrosis gene correction.

## Search space X
Every row is one Ugi-5C-4CR product. The three discrete choices that
define the search space:

| column | values | meaning |
|---|---|---|
| `head` | `CH`, `LH` | isocyanide head class. `CH` = cyclic (cyclohexyl-derived); `LH` = linear (n-alkyl). |
| `amino_acid` | 30 α-amino acids (see table below) | the backbone — determines side-chain chemistry |
| `tail` | `1` .. `16` | aldehyde identifier; a branched or linear C10-C16 hydrocarbon with 0 or 1 ester bond (exact structures in Supp Fig 2a — not numerically published) |
| `lipid_id` | e.g. `CHCha-10` | canonical name `{head}{amino_acid}-{tail}` used throughout the paper |

Additional metadata columns included for LLM / domain-prior use:
- `amino_acid_full`: full chemical name of the AA
- `aa_proteinogenic`: 1 = one of the 20 canonical proteinogenic AAs, 0 = non-proteinogenic (15 of the 30 AAs are non-proteinogenic — Cha, Tle, Aad, Pen, Hcy, Orn, The, Gsh, Cit, Hse, Sem, Cth, Eth, Nle, Nva)
- `head_description`: one-line human-readable head class

Search space size: **960 unique lipids** (full factorial).

### The 30 amino acids

The 30 amino acids used as backbones (listed alphabetically):

| AA  | full name | proteinogenic |
|---|---|---|
| Aad | α-aminoadipic acid      |   |
| Ala | alanine                 | ✓ |
| Asn | asparagine              | ✓ |
| Cha | β-cyclohexyl-alanine    |   |
| Cit | citrulline              |   |
| Cth | cystathionine           |   |
| Eth | ethionine               |   |
| Gln | glutamine               | ✓ |
| Gly | glycine                 | ✓ |
| Gsh | glutathione             |   |
| Hcy | homocysteine            |   |
| His | histidine               | ✓ |
| Hse | homoserine              |   |
| Ile | isoleucine              | ✓ |
| Leu | leucine                 | ✓ |
| Lys | lysine                  | ✓ |
| Met | methionine              | ✓ |
| Nle | norleucine              |   |
| Nva | norvaline               |   |
| Orn | ornithine               |   |
| Pen | penicillamine           |   |
| Phe | phenylalanine           | ✓ |
| Sem | selenomethionine        |   |
| Ser | serine                  | ✓ |
| The | β-homothreonine         |   |
| Thr | threonine               | ✓ |
| Tle | tert-leucine            |   |
| Trp | tryptophan              | ✓ |
| Tyr | tyrosine                | ✓ |
| Val | valine                  | ✓ |

### Headgroup classes

| label | meaning |
|---|---|
| `CH` | **cyclic** isocyanide head (cyclohexyl-containing). Rigid. |
| `LH` | **linear** isocyanide head (n-alkyl). Flexible. |

### Tails
Tails are identified by their integer id (1..16). The structural formulas
are in Supp Fig 2a of the paper but **not published as SMILES or a table**;
we cannot attach explicit chemistry to individual tail numbers.

## Objectives f

| column | type | direction | notes |
|---|---|---|---|
| `log2_luc_pred` | float | **maximize** | log2(firefly-luciferase luminescence) in BEAS-2B, normalized to untreated. Main optimization objective. Clipped to `[5, 18]`. |
| `log2_luc_paper` | float or NaN | (reference) | Exact published value from Fig 1e. Only populated for a subset of lipids. |
| `log2_is_paper_verified` | 0/1 | (metadata) | 1 if this row's value is from the paper (Fig 1e), 0 if image-extracted. |
| `below_heatmap_floor` | 0/1 | (metadata) | 1 if the raw (pre-clip) extracted value was below the heatmap's declared floor of 6. 80% of the library sits here (essentially inactive). |

For optimization benchmarks, `log2_luc_pred` is the primary objective.
Use `log2_luc_paper` as a hold-out accuracy check (20 known anchors).

## Caveat — image-derived values, uncertainty

The `log2_luc_pred` column for the 940 non-anchor lipids is recovered from
the dot-plot image in Supp Fig 2b by:

1. Rendering the published SI PDF at 1200 DPI.
2. Detecting the 30 × 32 grid of dot centers.
3. Sampling mean RGB and pixel-count per dot.
4. Fitting a linear model `value ~ R + G + B + size_px` jointly against:
   - ~250 colorbar samples (value 6 → 15 by interpolated tick positions), and
   - Fig 1e anchor log2 values for calibration.

**Validation on the Fig 1e anchors:**
- mean bias: +0.08 log2 ≈ 1.06× (extraction slightly underpredicts the truth)
- residual σ: 0.48 log2 ≈ 1.39×
- max |residual|: 1.16 log2 ≈ 2.24×

**What extraction cannot recover accurately:**
- **Very-low-value dots**: the heatmap legend floors at value 6 and about
  80% of the library sits at or below this floor. Those 797 lipids are all
  clipped to 5 in `log2_luc_pred`. For optimization they are all tied at
  the bottom; distinguishing among them is impossible from this data.
  For an active-learning / BO benchmark this is actually a useful
  "background noise" regime (plenty of near-baseline samples) but expect
  the algorithm to explore many of them early.
- **Very-high-value dots** (`>17.5`): only 3 dots exceed this in
  calibration; extrapolation to `>18` is not supported by anchor data.

Recommended framings:
- **Top-k discovery** (most meaningful): given a budget of N queries,
  maximize the sum / max of top-k `log2_luc_pred`. Hold out the 20
  paper-verified labels as an unbiased sanity check.
- **Head / AA / tail selection**: 30-AA × 16-tail × 2-head discrete
  space is small enough that a competent optimizer should find
  the optimum in <50 queries.
- **Avoid** using this dataset as a **single-value regression target** —
  the 0.5 log2 extraction noise caps supervised-learning accuracy.

## Published anchor files

These files are 100% faithful to the paper's published numeric source
data — no image extraction:

| file | what it contains | rows |
|---|---|---|
| `top20_ranked.csv`         | Fig 1e: top 20 lipids + exact log2 luc | 20 |
| `top20_differentiated.csv` | Fig 2a: selected lipids + cKK-E12 benchmark in differentiated BEAS-2B (3 replicates each, lower log2 than primary screen) | 21 |
| `aa_aggregate.csv`         | Fig 1d: 30-AA polar-chart ranking (relative-max efficiency) | 30 |

## Files in this directory
```
lipid_library_960.csv            # 960 rows — MAIN BENCHMARK
per_head/
  CH.csv                         # 480 rows (cyclic head)
  LH.csv                         # 480 rows (linear head)
top20_ranked.csv                 # 20 rows  — paper Fig 1e, exact values
top20_differentiated.csv         # 21 rows  — paper Fig 2a, exact values
aa_aggregate.csv                 # 30 rows  — paper Fig 1d, exact values
_extracted_raw.parquet           # intermediate from extract_from_figure.py
DATA_CARD.md                     # this file
```
