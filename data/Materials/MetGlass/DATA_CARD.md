# Huang SputterGlass curated dataset — data card

## Source
Huang, Kube, Johnson, Sohn, Mehta & Schroers,
"Glass formation during combinatorial sputtering in binary alloys,"
*Acta Materialia* **296**, 121240 (2025).
DOI: [10.1016/j.actamat.2025.121240](https://doi.org/10.1016/j.actamat.2025.121240).

Built from the paper's Supplementary Information (`Supp_Info.docx`,
Elsevier `mmc1.docx`), specifically **Table S1**, which reports the
amorphous composition range per binary system. The per-alloy XRD spectra
themselves are not publicly deposited ("available from corresponding
author on reasonable request").

## Background in one paragraph
The authors use **combinatorial magnetron sputtering** with a three-gun,
tapeless-masking setup to deposit each binary alloy system as a
continuous composition gradient on a 100 mm wafer, spanning ~A5B95 to
~A95B5 in two rows (row 1 centered at A23B77, row 2 at A77B23). Each
system is interrogated at **66 pre-determined positions** with automated
EDX (composition) and high-energy synchrotron XRD (structure) at SSRL
beamline 1-5, 12.7 keV. **57 binary systems × 66 alloys = 3762 alloys**
are classified as crystalline, amorphous (= metallic glass), or mixed.
Approximately **17%** of the 3762 alloys form a glass under the estimated
sputtering cooling rate of ~1e8 K/s. The paper's headline finding is
that **crystal-structure mismatch between the two elements is the
strongest single indicator of glass formation**, while commonly cited
descriptors (atomic-size ratio, heat of mixing, deep eutectics) are
inadequate.

## Reconstruction note — READ THIS
The raw per-alloy XRD matrix is not public. Table S1 instead reports,
per system, the interval `[AM_beg, AM_end]` in at.% of element A that is
amorphous. Because the paper explicitly frames each system's amorphous
region as a **contiguous interval of at.% A** (Fig. 3b and Fig. 4 of the
paper use this exact convention), we reconstruct the per-alloy glass
label as:

    is_glass(x_A) = 1  if  AM_beg <= x_A <= AM_end  (and AM_Range > 0)
                    0  otherwise

This reproduces the paper's Fig. 4 summary **exactly at the binary-range
level**, and approximately at the per-composition level (19.8% glass vs
paper's 17% — the 2-3% gap comes from integer at.% rounding and from
the paper sometimes classifying the edge bins as Crystalline+Amorphous
mixed rather than pure amorphous). See Caveats below.

## Search space X
Every row of `alloys.csv` is one **binary alloy composition**:

| column | type | meaning |
|---|---|---|
| `composition_id` | string | formatted key, e.g. `Mg37Ni63` |
| `system_id` | string | `{A}-{B}` (e.g. `Mg-Ni`) |
| `element_A`, `element_B` | string | elemental symbols |
| `at_A`, `at_B` | int, at.% | always sum to 100, `at_A` ∈ [5, 95] |
| `xtal_A`, `xtal_B` | string | FCC / BCC / HCP / ... (pure-element structure) |
| `xtal_mismatch` | int | 1 if `xtal_A != xtal_B` else 0 |

Search space size: **3762 alloys = 57 systems × 66 compositions**.
The 57 systems are drawn from 16 elements (Mg, Al, Ti, V, Cr, Mn, Fe,
Co, Ni, Cu, Y, Zr, Nb, Mo, Ag, Hf, Ta, W, Pt, Au — check `systems.csv`
for the exact list). Each system is sampled on a 66-point gradient.

### Per-system descriptors (from Table S1)
| column | units | meaning |
|---|---|---|
| `AM_beg`, `AM_end` | at.% A | amorphous interval of system (may be NaN if system has no glass) |
| `AM_Range` | at.% | `AM_end - AM_beg` (0 if no glass anywhere) |
| `dHmix` | kJ/mol | Miedema mixing enthalpy at the mid-composition |
| `H_A`, `H_B` | kJ/mol | cohesive energies of pure A, B |
| `dHmix_normal` | dimensionless | `dHmix / ((H_A + H_B) / 2)` |
| `r_A_over_r_B` | dimensionless | atomic radius ratio (Goldschmidt) |
| `slope_K_per_at` | K / at.% | ∣dT_L/dx_A∣ at the most gradient-rich end |
| `Tm_min` | K | lowest melting / eutectic temperature in system |
| `TL_min` | K | lowest liquidus temperature |
| `T_min_all` | dimensionless | normalized `Tm_min / ((T_m,A + T_m,B)/2)` |

## Objectives f

| column | type | direction | notes |
|---|---|---|---|
| `is_glass` | int ∈ {0, 1} | **maximize** | primary scalar — 1 if the alloy forms a metallic glass under sputtering, 0 otherwise |

Secondary scalars (useful for multi-objective framing):
| column | type | direction | notes |
|---|---|---|---|
| `AM_Range` | int, at.% | maximize | per-system glass-forming ability. Same for all 66 alloys of a given system; a system-level objective. |
| `xtal_mismatch` | int ∈ {0, 1} | maximize | the paper's best predictor — trivial to maximize directly, included as a sanity oracle |

**Argmax** (under `is_glass`): many alloys are tied at `is_glass = 1`.
For a per-system objective, use `AM_Range` to rank glass-forming systems
by the width of their amorphous composition window.

## Caveats
- **Per-alloy labels are reconstructed from Table S1**, not from the
  individual XRD spectra (which are not public). The reconstruction
  assumes the amorphous region is a single contiguous interval in at.%
  A — this is the paper's own convention (Fig. 3b, Fig. 4). The
  reconstruction may differ slightly from the paper's stated glass
  fraction; the gap reflects (a) integer rounding of
  `AM_beg`/`AM_end`, (b) the fact that edge bins sometimes are
  Crystalline+Amorphous mixed in the paper's three-way classification.
  **Do not interpret per-alloy glass labels as ground truth at the
  ~1-at.% boundary**; they are faithful only at ~2-3 at.% resolution.
- **No missing data** in `is_glass` — every alloy is labelled. But
  per-system descriptors (`Tm_min`, `TL_min`, etc.) are missing for
  some systems where no melting-point data was tabulated in the paper.
  These are left as NaN in `systems.csv` and broadcast into `alloys.csv`.
- **26 of 57 systems have no glass formation at all** (AM_Range = 0).
  Their 66 alloys are all `is_glass = 0`. Examples: Al-Cu, Ni-Cu, Mg-Al,
  Ti-V — all same-crystal-structure systems with shallow dHmix.
- **The 5%..95% grid endpoints are approximate.** The paper states
  "~A5B95 to ~A95B5"; we use 66 evenly-spaced values, each rounded to
  the nearest integer at.%. Two-row centerings at A23B77 / A77B23
  produce the actual physical composition; our idealized evenly-spaced
  grid is a mild simplification.
- **`is_glass` is binary, not continuous.** For BO/AL this is a
  classification problem, not regression. If you need a continuous
  optimization target, use `AM_Range` (per-system, 57 distinct values)
  or promote `is_glass` to a probability via kernel smoothing.
- **Systems are not i.i.d. across X.** 57 binary systems × 66
  compositions means the search space has two natural axes — choosing
  WHICH system to interrogate is the expensive step (requires setting
  up a new sputter deposition), while moving along composition within a
  system is cheap (XRD-scan a different wafer position). Optimizers
  treating the 3762 alloys as i.i.d. will overestimate query cost.
- **Pure-element endpoints are excluded** (at_A=0 and at_A=100 are not
  sampled in the 66-point grid). The paper notes that pure elements are
  mostly crystalline under the same cooling rate.

## Files in this directory
```
alloys.csv                     # 3762 rows — the main benchmark
systems.csv                    # 57 rows — one row per binary system
per_system/
  Mn-Ta.csv                    # 66 rows per system (57 files total)
  Mg-Ni.csv
  Zr-Cu.csv
  ...
DATA_CARD.md                   # this file
```
