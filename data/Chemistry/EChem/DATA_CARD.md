# Eisenberg Aldehyde-Electro-Reduction curated dataset — data card

## Source
Eisenberg J. B., Lee K., Schmidt J. R. & Choi K.-S. (2025).
"Understanding the Competition between Alcohol Formation and
Dimerization during Electrochemical Reduction of Aromatic Carbonyl
Compounds." *J. Am. Chem. Soc.* (2025).
DOI: [10.1021/jacs.5c10757](https://doi.org/10.1021/jacs.5c10757).
PMCID: PMC12616691.

Built from **Table S1** of the paper's Supporting Information PDF
(`raw/ja5c10757_si_001.pdf`, pp. S8-S9), which reports all numerical
data used to plot the paper's Figures 3, 4, and S3.

## Background in one paragraph
Benzaldehyde (BAL), a lignocellulose-derived aromatic carbonyl, can be
electrochemically reduced along two competing pathways: **hydrogenation**
to benzyl alcohol (BA) by two-electron proton-coupled electron transfer
(PCET), and **reductive hydrodimerization** to hydrobenzoin via a ketyl
radical that must first desorb from the electrode and couple with a
second ketyl in solution. The competition is set by a single kinetic
race on the electrode surface — PCET of the adsorbed ketyl (→ alcohol)
vs. desorption of the ketyl radical (→ dimerization). The authors scan
this competition on a **4 × 3 × 3 = 36-run factorial** of electrode
material, solution pH, and applied potential, and interpret the results
with constrained-DFT / configuration-interaction (CDFT-CI) calculations
of the PCET activation barrier vs. ketyl desorption energy. The take-home
message: weak-binding electrodes (**Bi**) and basic pH (**OH* coverage
blocks PCET**) strongly favour dimerization; strong-binding electrodes
(**Cu**) and acidic pH favour alcohol (via PCET + hydrogenolysis). The
most dimerization-selective condition measured is **Bi / pH 13 / -0.8 V
vs. RHE (100% selectivity, 76% BAL conversion)**.

## Search space X (36 design points, full factorial)

| column | type | meaning |
|---|---|---|
| `electrode` | categorical(4) | `Cu`, `Pb`, `Bi`, `graphite` — bulk rod electrode |
| `pH` | categorical(3) | `2` (phosphate buffer), `7` (phosphate buffer), `13` (0.1 M KOH) |
| `potential_V` | categorical(3) | `-0.6`, `-0.8`, `-1.0` V vs. RHE, constant-potential electrolysis |
| `pH_regime` | derived string | `acidic` / `neutral` / `basic` — alias of `pH` |

All three variables are treated categorically because the paper picks
exactly three representative levels of pH and potential; the three
voltages span the BAL-reduction onset region (paper Fig. S6 LSVs).
Each design point was a separate galvanostatic electrolysis in an
H-cell to a total charge of 1 e− per BAL molecule (so the current
density and time per run are coupled — both are reported as
`current_density_mA_cm2` and `electrolysis_time_min`).

Search space size: **36 rows**, exactly one replicate per (electrode, pH,
V). No repeats.

## Objectives f

The paper frames selectivity along **five** scalar directions (all in %
unless marked):

| column | dir | what it is |
|---|---|---|
| `conversion_pct`            | max | % of BAL consumed after passing 1 e−/BAL |
| `sel_dimerization_pct`      | max | % selectivity → **hydrobenzoin dimer** (C-C coupled) |
| `sel_hydrogenation_pct`     | max | % selectivity → **benzyl alcohol** via PCET |
| `sel_hydrogenolysis_pct`    | max | % selectivity → toluene (C-O bond cleaved) |
| `sel_other_pct`             | —   | remainder (condensation / lost / analytical error) |
| `dimer_yield`               | max | = conv × sel_dim / 100, in % points |
| `alcohol_yield`             | max | = conv × (sel_hydrog + sel_hydrogenol) / 100 — **paper's "monomer reduction" pathway** |
| **`alcohol_to_dimer_ratio`**| tunable | (sel_hydrog + sel_hydrogenol) / sel_dim — the paper's explicit competition scalar. <1 ⇒ dimer wins, >1 ⇒ alcohol wins |
| `selectivity_scalar`        | tunable | sel_dim − (sel_hydrog + sel_hydrogenol) — signed, bounded to [−100, 100] |
| `electrolysis_time_min`     | —   | minutes to pass 1 e−/BAL (coupled to current) |
| `current_density_mA_cm2`    | —   | average j during that electrolysis |

**Recommended primary scalar for optimisation benchmarks:**
`alcohol_to_dimer_ratio` — this is the exact quantity the paper's
mechanistic model predicts (the PCET-vs-desorption rate ratio). It is
unbounded above but well-conditioned on a log scale.

For BO/AL tasks, a robust drop-in alternative is
`selectivity_scalar ∈ [−100, 100]` (the signed, bounded version):
maximise it for dimer-biased optimisation, minimise it for alcohol.

## Objective ranges

The 36-cell factorial spans a wide selectivity range from fully dimer-selective
to fully alcohol-selective. The `selectivity_scalar` spans approximately
[−100, +100]. Use `selectivity_scalar` (maximize for dimer, minimize for
alcohol) as the primary single-scalar objective for BO benchmarks.

## Caveats
- **No replicates.** Every (electrode, pH, V) combination was run exactly
  once. Noise in selectivity measurements is not characterised in the
  paper beyond the analytical error bar on product quantification.
- **"Other" is not zero.** Rows where sel_other is large (Cu at pH 2;
  graphite at pH 2) reflect genuine missing carbon in the mass balance —
  some fraction of converted BAL went to species not identifiable by
  HPLC/GC (condensation products, electrode film, aldol side-products).
  The dimer/hydrogenation/hydrogenolysis columns remain reliable as
  *relative* selectivities among identified products.
- **One small negative sel_other** (Bi / pH 13 / -0.8 V, `sel_other_pct =
  -0.24 %`) is an analytical artefact the paper preserves as-is; we keep
  it. It is within the ±2-3% integration error expected for this assay.
- **Conversion is capped at ~1 e-/BAL** by design — electrolyses were
  stopped after passing exactly 9.649 C × (time/min), so conversion
  cannot exceed ~50% via a single 2-e− hydrogenation. The maximum
  observed conversion is 76%, achieved when dimerization dominates (only
  1 e−/BAL is needed for a hydrodimer per radical, so 100% dimer
  selectivity can in principle give 100% conversion).
- **Current density is an outcome, not a control variable** — the cell
  was controlled at constant potential, so j varied with conditions
  (0.17 → 28 mA cm⁻² across the library). If treating this as a
  reaction-engineering benchmark, consider including `current_density` as
  an *auxiliary* descriptor but not a design variable.
- **Trifluoroethanol experiments in Fig. S5 are not included** — those
  form a separate 6-run "co-solvent" study that changes the chemistry
  (proton donor engineered to suppress dimer desorption), so they are
  mechanistically distinct and outside the 36-run factorial.
- **Only pH 2 / 7 / 13 are measured**, not intermediate pH. If your BO/AL
  benchmark needs a continuous pH axis, either treat pH as categorical
  (default here) or fit a surrogate on the three anchor points.

## Files in this directory
```
aldehyde_electro_36.csv          # 36 rows — the main benchmark
DATA_CARD.md                     # this file
```
Raw SI (upstream):
```
../raw/ja5c10757_si_001.pdf      # paper's SI, Table S1 on pp. S8-S9
../raw/si_layout.txt             # pdftotext -layout dump (for audit)
```
