# Mancuso S. epidermidis intraspecies antagonism — data card

## Source
Mancuso, Baker, Qu, Tripp, Balogun & Lieberman, "Intraspecies warfare
restricts strain coexistence in human skin microbiomes," *Nat. Microbiol.*
**10**, 1581–1592 (2025).
DOI: [10.1038/s41564-025-02041-4](https://doi.org/10.1038/s41564-025-02041-4).

Primary source: the authors' GitHub repository
[github.com/cpmancuso/Sepidermidis-antagonism](https://github.com/cpmancuso/Sepidermidis-antagonism).
We build the curated tables directly from the `.mat` interaction structures
in `Interaction Analysis/`, which the paper's MATLAB pipeline uses to compute
every figure involving the antagonism matrix.

Backup source (included as `Source_Tables.xlsx` in the parent folder): the
paper's Supplementary Tables MOESM4 from Nature. Tables S6 and S7 store the
same isolate-level matrix for TSA and M9+S media but **only the binary ZOI
call (0/1), not the numeric AUC**; we therefore use the `.mat` files for the
primary continuous objective.

## Background in one paragraph
The authors profile **intraspecies warfare** among commensal
*Staphylococcus epidermidis* strains on human skin. Starting from 2,025
sequenced skin isolates from 18 people across 6 families (from a prior study
by Baker et al.), they down-select 122 representative *S. epidermidis*
isolates that span 59 "lineages" (genetic clusters separated by <90 core
genome SNVs) and add 23 other skin-species isolates for interspecies context.
They then run every ordered (producer, receiver) pair in a high-throughput
**spot-on-lawn** assay: the producer isolate is spotted on top of a lawn of
the receiver, and after 72 h of growth a zone of inhibition (ZOI) around the
spot signals that the producer secretes or presents a molecule that kills
the receiver. Lawn pixel intensity around each spot is radially integrated
to give a **ZOI area under the curve (ZOI_AUC)** in arbitrary intensity ×
pixel units — the dataset's scalar objective. The same screen is repeated
on a skin-mimicking medium (M9 minimal salts + artificial sweat, "M9+S") to
probe medium-dependent production.

Their headline biological finding is that antagonism is (a) prevalent —
~8% of random (producer, receiver) pairs show a detectable ZOI, with 44%
of isolates antagonizing at least one other lineage, (b) mechanistically
heterogeneous and poorly predicted by phylogeny, and (c) **depleted among
strains co-residing on the same person** relative to random assemblages,
implicating warfare as a significant barrier to strain coexistence in the
skin microbiome.

## Search space X
Every row is one **directed (producer, receiver) pair** of *S. epidermidis*
isolates:

| column | type | meaning |
|---|---|---|
| `producer_isolate_id` | string | e.g. `SE-34.1` (62 isolates of lineage 34, replicate index 1). Prefixed with `SE-` so CSV readers keep it string-typed. |
| `receiver_isolate_id` | string | same format |
| `producer_lineage_id` | int | genetic lineage of the producer, 1..92 |
| `receiver_lineage_id` | int | same for receiver |
| `producer_subject` | string | person the producer was cultured from, e.g. `5PA` (family 5 parent A); 17 distinct subjects in 6 families |
| `receiver_subject` | string | same for receiver |
| `producer_family` | int | 1 / 2 / 4 / 5 / 7 / 8 (paper's 6 families) |
| `receiver_family` | int | same for receiver |

The pair is **directed**: `(A, B)` (A producing, B receiving) is a different
row from `(B, A)`. The main CSV includes all 122² = 14,884 directed pairs,
matching the paper's Abstract ("14,884 pairwise interactions between *S.
epidermidis* isolates"). The self-pair diagonal (`producer == receiver`) is
kept in the CSV but has near-zero antagonism by construction.

## Objectives f

Two columns per row, **both with higher = stronger antagonism**:

| column | type | description |
|---|---|---|
| `ZOI_AUC`  | float | Negative integral of locally-corrected pixel intensity across the inhibition zone (arbitrary intensity × pixel units). 0 = no detectable inhibition. Paper's detection threshold is AUC ≥ 50 (see `ZOI_call`). **This is the primary scalar objective** — it preserves the continuous strength signal. |
| `ZOI_call` | 0/1 int | Paper's final binary call after their full filter chain: depth ≥ 8 pixel intensity units AND (AUC ≥ 50 OR AUC ≥ 1000) AND signal ≥ 4× local noise AND spot not blank (see `make_ZOI_calls.m` in the repo). Use this for thresholded analyses and for comparing to the paper's "8% positive" headline. |

Maximization in both directions: a large-AUC pair indicates strong producer
antimicrobial activity against that receiver.

## Files in this directory
```
antagonism_TSA.csv              # 14,884 rows, TSA media     (primary)
antagonism_M9S.csv              # 14,884 rows, M9+sweat      (secondary medium)
antagonism_TSA_lineage.csv      #  3,481 rows = 59 x 59, TSA lineage-level
antagonism_M9S_lineage.csv      #  3,481 rows = 59 x 59, M9+S lineage-level
DATA_CARD.md                    # this file
```

**Lineage-level tables**: the paper dereplicates the 122 isolates to 59
lineages (clusters separated by ≥90 core genome SNVs), each represented by
≥1 isolate. The lineage-level ZOI_AUC and ZOI_call in our tables are the
**max across isolate replicates** per (prod_lineage, recv_lineage) cell —
this matches the paper's stated convention ("since the ZOI calling step is
so conservative, the maximum ZOI observed between a pair of lineages was
chosen as representative"). `n_isolate_pairs` records how many isolate
pairs were aggregated per cell.

## Known headline numbers (verified by `validate_curated.py`)
- 14,884 directed pairs at isolate level; 3,481 at lineage level.
- TSA ZOI_call positivity: 6.6% overall (paper: "~8% of spots show ZOIs").
- M9+S ZOI_call positivity: 2.8% — lower growth under M9+S suppresses many
  interactions (Extended Data Fig. 6).
- Self-pair (producer == receiver) ZOI_call positivity: 0.8% (1/122 on
  each medium — the paper notes one case of residual self-signal on
  lineage 51 that they rescored; we keep it in the CSV).
- Intralineage ZOI_call positivity: 0.9% vs interlineage 6.7% on TSA
  (paper Fig. 2d: intralineage antagonism "negligible").
- A small number of "superantagonist" lineages (carrying lantibiotic BGCs)
  show broad-spectrum killing across the panel. These lineages come from
  different families.
- Same-subject off-lineage pairs have lower ZOI_call rates vs cross-subject
  pairs, consistent with co-residency depletion dynamics.

## Caveats

- **Directed pairs.** (A inhibits B) and (B inhibits A) are separate rows;
  the 14,884 count is 122² and reflects this. A symmetric-only view would
  halve that count to ~7,320 unordered pairs (excluding the diagonal). An
  optimizer that treats producer and receiver symmetrically will miss
  direction-specific signals.

- **Self-pair diagonal.** 122 rows have `producer_isolate_id ==
  receiver_isolate_id`. By construction these should be zero (an isolate
  does not kill itself), and in the post-threshold `ZOI_call` they
  effectively are (1/122 positive). But raw `ZOI_AUC` on the diagonal has a
  noise floor (median 37 on TSA; max 597): small AUC values arise from
  baseline fluctuations around self-spots. **Do not treat AUC ≈ 0 on the
  diagonal as the benchmark's "true minimum" — treat the diagonal as a
  control/structural zero.**

- **Two growth media.** TSA (rich) gives the denser antagonism map and is
  the paper's primary screen — **we recommend `antagonism_TSA.csv` as the
  default benchmark**. M9+S (minimal + artificial sweat) is harder to grow
  on so fewer interactions surface; only ~23% of TSA-positive interactions
  are positive in M9+S and only ~43% of M9+S-positive are positive in TSA.
  Both media agree on the headline co-residency depletion signal.

- **M9+S detection threshold differs.** In the paper the M9+S image-analysis
  threshold was changed to depth ≥ 4 / AUC ≥ 0 to compensate for lower
  growth, so the M9+S `ZOI_call` column uses those (looser) thresholds.
  Raw M9 AUC values are therefore NOT directly comparable in magnitude to
  TSA AUC values — the two media have different effective noise scales.

- **Replicate structure at isolate level.** The paper includes 6 isolates
  at duplicated plate positions as in-experiment replicates (isolates 26.1,
  32.1, 34.1, 36.1, 48.1, 51.1 each appear at 2–4 positions). We average
  ZOI_AUC across replicates and take max of ZOI_call; the curated CSV
  therefore has 122 unique isolates, not 137 array positions. If you need
  per-position variance, load the `.mat` directly.

- **Idiosyncratic isolates are NOT masked** in our curated CSV. The paper
  masks isolates 20.3, 37.3, 49.5, 70.3 (carrying vraFG-pathway loss-of-
  function mutations that acquired derived sensitivity on specific hosts)
  in figures beyond Fig. 4 because their behavior differs from their
  lineage. We keep them in — optimizers that aggregate at the lineage
  level will see their signal, optimizers that work isolate-by-isolate
  can filter them out if desired (the stock numbers are 211, 313, 292, 332
  in the repo metadata).

- **11 replacement isolates are NOT in the curated CSV.** After the main
  TSA screen the authors added 11 *S. epidermidis* isolates as replacements
  for non-*S. epidermidis* slots that had grown poorly (1.1, 1.2, 16.3,
  20.4, 24.2, 24.3, 37.4, 58.3, 77.1, 77.2, 83.1). These isolates have
  partial data: they were measured as lawns but have no spot data for
  most producer positions. Including them would introduce NaNs in ~80% of
  their spot-column cells, so we restrict the benchmark to the 122
  fully-measured isolates (the paper's headline count). Future extensions
  could add an `antagonism_TSA_extended_148x148.csv` with NaN-handling.

- **Non-*S. epidermidis* species are NOT in the curated CSV.** The full
  192-position screen also covers 23 isolates from *S. aureus*, *S.
  hominis*, *S. capitis*, *S. lugdunensis*, *S. pasteuri* and *M. luteus*
  (yielding ~21,025 cross-species pairs). They are outside the paper's
  primary focus — "14,884 pairwise intraspecies interactions" is what's
  benchmarked here. An interspecies follow-up CSV can be built from the
  same `.mat` matrices if needed.

- **Co-residency signal is REAL BIOLOGY, not noise.** The paper's central
  claim is that same-subject pairs show depleted antagonism relative to
  random cross-subject pairs — this is recoverable from our CSV
  (`validate_curated.py` Check 7: 3.5% vs 6.9% on TSA). A benchmark user
  looking for "difficult" optimization pockets should note that low-AUC
  pairs cluster on same-subject / same-family rows, which means the
  metadata features (`producer_subject`, `producer_family`, ...) are
  informative covariates even before genomic features.

- **Arbitrary-unit scaling.** ZOI_AUC values have no absolute calibration;
  they are intensity-integrated pixel counts. Across-medium comparison
  ratios are meaningful (an AUC of 2000 is substantially stronger than
  200 on the same medium), but the raw numbers cannot be converted to
  e.g. MIC values. The paper's `MIC_isolate_list.xlsx` has MICs for a
  small follow-up panel, not for the main 122×122 screen.

- **Missing paper-claimed 18 people.** The paper mentions 18 people but
  only 17 distinct `Subject` codes appear in the curated CSV (subjects
  1AA, 1PA, 1PB, 2AA, 2PA, 2PB, 4AA, 4AB, 5PA, 5PB, 7AA, 7AB, 7PA, 8AA,
  8AB, 8AC, 8PB). The 18th subject presumably contributed only
  non-*S. epidermidis* isolates or was excluded post-screen; we did not
  chase this down further as it does not affect the 14,884-pair count.

## Reproducibility
- `../curate_dataset.py` builds all four CSVs from the repo `.mat` files
  deterministically; rerun to regenerate.
- `../validate_curated.py` asserts the paper-level claims above. It exits
  non-zero if any assertion fails.
- The paper's MATLAB `interaction_analysis.m` pipeline ingests the same
  `.mat` files; our outputs are an alternative export of its input matrix
  rather than a reprocessing.
