# NCATS PANC-1 drug-combination synergy — data card

## Source
Zakharov, Dietrich, Shen, Chaudhry, Guha, Wang, Michael, Sittampalam,
McKnight, Southall et al., "AI-driven discovery of synergistic drug
combinations against pancreatic cancer," *Nat. Commun.* (2025).
DOI: [10.1038/s41467-025-56818-6](https://doi.org/10.1038/s41467-025-56818-6).

Built from the NCATS GitHub release at
https://github.com/ncats/PANC1/tree/main/datasets :

| local file | source | contents used |
|---|---|---|
| `Training_set.csv` | `ncats/PANC1/datasets/Training_set.csv` | the 496-row (pair, gamma, label) training table — our primary input. |
| `307_synergisitc_combos.csv` | same repo | reference only; a 307-row superset that includes ML-validation pairs not in the training set. |
| `Source_Data.xlsx` | MOESM4 from the Nature article | duplicates a subset of Training_set.csv rearranged per figure. Not parsed. |

## Background in one paragraph
The NCATS team screened 32 compounds (from their 1,785-member MIPE
library) as all 32 × 31 / 2 = 496 unordered pairs against PANC-1
pancreatic-cancer cells, each pair tested in a 10 × 10 dose matrix in
biological duplicate. For each pair they compute the **Gamma** synergy
score (Dietrich et al. 2024; Holford-Finney variant): **gamma < 1.0 =
synergistic, > 1.0 = antagonistic**. The paper's operational definitions:
gamma < 0.95 = "synergistic" (hit), gamma < 0.5 = "strongly synergistic".
The single agents are tested on PANC-1 with log AC50 values from −8.7 to
−5.5 (≈ 2 nM – 3 µM), so the pairs span a wide potency range.

## Search space X
Every row in `combinations.csv` is one unordered (compound_a, compound_b)
pair with `compound_a < compound_b` alphabetically:

| column | type | meaning |
|---|---|---|
| `compound_a` | string | first compound name (alphabetically earlier). 32 unique. |
| `compound_b` | string | second compound name. 32 unique. |

Each of the 32 compounds appears in exactly 31 pairs. The candidate pool
size is 496.

## Objectives f

| column | type | direction | meaning |
|---|---|---|---|
| `gamma` | float | **minimize** (framework flips → maximize 1 − gamma) | NCATS gamma synergy score. 1.0 = no interaction; < 0.95 = synergistic; < 0.5 = strong synergy; > 1.0 = antagonistic. Range 0.041..1.350. |
| `label_synergistic` | 0/1 | — | paper's hit label: 1 iff gamma < 0.95. 256 / 496 pairs hit. |
| `moa_a`, `moa_b` | string | — | mechanism of action per compound. Metadata; not exposed to optimiser as a feature. |

## Caveats

- **Direction flip.** The BenchmarkDataset loader converts gamma to
  `f = 1 - gamma` so the benchmark is consistently "higher = better
  synergy". Raw gamma is stored in the CSV for transparency.
- **PANC-1 only.** All measurements are on a single pancreatic-cancer
  cell line. Do not extrapolate to other cancers or cell types.
- **Duplicate structure.** Each pair was screened twice biologically; the
  gamma in Training_set.csv is the **average** of the two replicates.
  Pearson correlation between replicates is 0.83 (paper Fig 2b).
- **Synergistic counts differ by analysis scope.** Our 496-pair training
  set has 256 pairs with gamma < 0.95 and 10 with gamma < 0.5. The paper
  also cites 307 synergistic and 26 strongly-synergistic — those numbers
  include a further 88-combination ML-validation round that is
  intentionally NOT in the benchmark pool. Use `combinations.csv` as the
  enumerable candidate set.
- **MoA annotations are coarse.** Some compounds map to multiple targets;
  the `moa_a` / `moa_b` strings are the NCATS primary-target annotation
  (e.g. "HDAC Inhibitor", "Proteasome Inhibitor"). For compounds with no
  literature MoA, the field may be blank.
- **MoA-level synergy patterns.** Proteasome inhibitor + HDAC inhibitor
  is a well-known synergistic MoA class in oncology from pre-publication
  pharmacology literature. The optimizer should discover which specific
  pairs within this and other MoA classes yield the highest gamma scores.

## Files in this directory
```
combinations.csv        # 496 rows — the main benchmark
compound_metadata.csv   # 32 rows  — compound name, NCGC ID, SMILES, log AC50, MoA
DATA_CARD.md            # this file
```
