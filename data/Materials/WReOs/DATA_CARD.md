# Wang W-Re-Os curated dataset — data card

## Source
Wang, Liu, Li, Lai, Chen, Lu & Chen, "High-throughput discovery of
ultrahigh-temperature multi-principal element alloys by combinatorial
additive manufacturing," *Nat. Commun.* **17**, 668 (2026).
DOI: [10.1038/s41467-025-66119-z](https://doi.org/10.1038/s41467-025-66119-z).

Built from the paper's consolidated **Source Data** xlsx
(`Source_Data.xlsx`), which holds one sheet per figure panel (24 sheets
total). We use Figures 2a + 3a/3c/3d/3e/3f.

## Background in one paragraph
The authors use **combinatorial additive manufacturing** (laser directed
energy deposition with three independently fed metal hoppers) to rapidly
deposit a bulk compositional library of the W-Re-Os ternary refractory
alloy system. Three coaxial nozzles feed W, Re and Os elemental powders
into the laser melt pool; by dynamically adjusting the three mass flow
rates they print ~500 cylindrical patch samples (Ø5 mm × 3 mm) on a single
substrate, covering the ternary simplex in compositional steps of 3 at.%.
The library is then screened by (a) room-temperature Vickers indentation
(hardness + pile-up → plasticity proxy), (b) high-temperature Vickers
indentation up to 1000 °C, and (c) room-temperature XRD / EBSD phase
mapping. A TOPSIS (Technique for Order Preference by Similarity to Ideal
Solution) composite score combines normalized H1000, retention rate, and
pile-up into a single scalar ranking.

## Search space X
Every row is one **W-Re-Os at.% composition** on a 3 at.% ternary simplex:

| column | type | meaning |
|---|---|---|
| `composition_id` | string | formatted key, e.g. `W42Re30Os28` |
| `W`, `Re`, `Os` | int, at.% | always sum to 100 |
| `phase_label` | string | room-temp phase region (see below) |

Search space size: **496 compositions** (the paper's stated count). The
three pure endpoints (pure W, pure Re, pure Os) are in the source data for
the measurement sheets but NOT in the phase map sheet, so we drop them
from the curated set so every row has a phase label.

### The 5 phase regions

From the paper's XRD + EBSD analysis:

| label | count | meaning |
|---|---|---|
| `HCP`           | 311 | hexagonal close-packed solid solution (dominant region; Re-rich side) |
| `HCP+BCT-ReW`   |  93 | dual-phase hypoeutectic — hard BCT lamellae in ductile HCP matrix |
| `BCT-ReW`       |  58 | single-phase ReW intermetallic; typically cracks during AM due to brittleness |
| `BCT-ReW+BCC`   |  33 | BCT / BCC dual-phase (W7Os3-type BCT) |
| `BCC`           |   1 | pure BCC solid solution — only a single composition fits this region cleanly (W-rich corner); paper notes BCC is a narrow strip |

The HCP+BCT-ReW eutectic pocket features hard BCT lamellae strengthening
a ductile HCP matrix — a promising region for combined hardness and plasticity.

## Objectives f

All five scalars are **maximized**.

| column | type | description |
|---|---|---|
| `H25_HV`    | float | Vickers hardness at 25 °C (HV). Proxy for room-temp yield strength (σy ≈ HV/3). |
| `H1000_HV`  | float | Vickers hardness at 1000 °C (HV). High-temp yield strength. |
| `R1000`     | float | retention rate H1000 / H25. 1.0 = no softening; typical refractory alloys are 0.5–0.8. |
| `pileup_um` | float | average height of pile-up around the indent (µm). Positive values = plastic flow; **0 = the indentation cracked** (brittle). Proxy for room-temp plasticity. |
| `TOPSIS_Ci` | float | paper's composite score ∈ [0, 1], combining normalized H1000, R1000, pile-up. Ci > 0.5 is the paper's "exceptional" band. |

The TOPSIS_Ci composite score ranges from ~0 (cracked / single-objective-poor)
to ~0.6 (exceptional all-rounder). Ci > 0.5 is the paper's "exceptional" band.

## Caveats
- **Only H25 and H1000 are in the source data**, not the intermediate
  temperatures (200 / 400 / 600 / 800 °C). Those maps exist in Supplementary
  Figs. 4–5 of the paper but the Source_Data.xlsx sheets only expose H25
  (Fig. 3a) and H1000 (Fig. 3c).
- **pileup_um = 0 is a cracked indent, not a missing measurement.** Treat it
  as a hard penalty for brittleness when optimizing plasticity. If you want
  a soft label use `pileup_um > 0` as a "crack-free" indicator.
- **Pure-element endpoints (100,0,0), (0,100,0), (0,0,100) were dropped**
  because the paper's Fig. 2a phase map does not cover them. All 5 measured
  hardness/Ci maps do include the endpoints, so if you need them, re-read
  Source_Data.xlsx Figs 3a/3c/3d/3e/3f directly.
- **TOPSIS Ci is not a physical quantity** — it's a linear combination of
  rank-normalized H1000/R1000/pile-up. The paper uses it for down-selection.
  It correlates strongly but not monotonically with the three underlying
  metrics. Single-objective optimization of Ci is a valid proxy for the
  paper's task, but the more faithful framing is **multi-objective on
  (H1000, R1000, pile-up)** and rank by Pareto front.
- **All measurements are single-pass on one patch sample each** — there are
  no replicates. Noise is unmodelled. The paper's error bars (Fig. 3b) come
  from multiple indents per patch, not re-fabricated patches.
- **BCC single-phase region has only 1 composition.** If you stratify by
  phase, BCC is effectively a singleton and optimizers should be warned.
- **Phase label is a static property** (same compositional coordinates → same
  phase) and does not change between 25 °C and 1000 °C — the paper
  explicitly checks thermal stability and finds the phases are preserved up
  to 1400 °C.

## Files in this directory
```
simplex.csv                      # 496 rows — the main benchmark
phase_groups/
  HCP.csv                        # 311 rows
  HCP_plus_BCT_ReW.csv           #  93 rows
  BCT_ReW.csv                    #  58 rows
  BCT_ReW_plus_BCC.csv           #  33 rows
  BCC.csv                        #   1 row
DATA_CARD.md                     # this file
```
