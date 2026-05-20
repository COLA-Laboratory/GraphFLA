# Roux PollutantsGut curated dataset — data card

## Source
Roux, Lindell, Grießhammer, Smith, Krishna, Guan, Rad, Faria, Blasche,
K. R. Patil, Kleinstreuer, Maier, Kamrad & K. R. Patil, "Industrial and
agricultural chemicals exhibit antimicrobial activity against human gut
bacteria in vitro," *Nat. Microbiol.* **10**, 3107-3121 (2025).
DOI: [10.1038/s41564-025-02182-6](https://doi.org/10.1038/s41564-025-02182-6).

Built from two complementary sources:

1. **Nature Supplementary Data 3 workbook** (`Source_SuppData3.xlsx`,
   MOESM3 from the article page):
     * `SuppData1`  — 1,076 compounds × 315 metadata columns
                      (Compound Class, PubChem CID, InChIKey, SMILES,
                      IUPAC, 290 PESTICIDE CLASSIFICATION binary flags,
                      industrial chemical group, metabolite-of,
                      pesticide-related comment).
     * `SuppData2`  — 22 bacteria × 9 cols (phylum, prevalence, per-
                      strain inhibitory hit count).
     * `SuppData3`  — **23,672 rows** (1,076 × 22) with the paper's
                      replicate-medianed `Median normalised AUC` and
                      pre-computed `Significant growth inhibition`
                      Boolean.
     * `SuppData4`  — 600 rows (8 strains × 11 compounds × 5-8
                      concentrations) — the validation dose-response.

2. **Mendeley Data** DOI [10.17632/g7hy84t2r6.1](https://doi.org/10.17632/g7hy84t2r6.1),
   file `20220607_pesticide_library_metadata.txt`. Used to pull
   **CAS numbers (`BCPC_CAS Reg. No.`)** for the 829 pesticides, joined
   to SuppData1 by PubChem CID (826/1,076 matched).

## Background in one paragraph
Industrial and agricultural chemicals such as pesticides, fungicides,
flame retardants, plasticizers, PAHs, bisphenols, PFAs, mycotoxins and
processing contaminants are ubiquitously encountered by humans yet
their effect on the gut microbiome is poorly catalogued. The authors
assemble a library of **1,076 pollutants** covering 829 pesticides
(BCPC-classified into herbicides, fungicides, insecticides, acaricides,
nematicides, rodenticides, plant / insect growth regulators, safeners
and more), 119 pesticide metabolites, 75 pesticide-related compounds,
48 industrial chemicals and 5 mycotoxins. The library is assayed
against **22 phylogenetically diverse, prevalent human gut bacteria**
in mGAM medium under anaerobic conditions at **20 µM single dose with
three biological replicates**, monitoring OD₆₀₀ for 24 h and computing
**relative area under the growth curve (rAUC) = AUC(well) / median
AUC(DMSO controls on plate)**. Of the 1,076 × 22 = 23,672 pair-wise
interactions, the paper calls **588 as inhibitory hits** (168 distinct
chemicals hit at least one strain). Fungicides (~30% hit rate) and
industrial chemicals (~27%) are the most impactful classes; TBBPA
(tetrabromobisphenol-A, a brominated flame retardant) is highlighted
as a particularly broad-spectrum inhibitor. A dose-response validation
sub-screen (`dose_response.csv`) re-tests 11 strong hits on 8 bacteria
over 5–8 concentrations from 2.5 to 320 µM.

## Search space X
Every row of `single_dose.csv` is one **(chemical, bacterium)** pair.
The candidate space is the Cartesian product:

| column | type | meaning |
|---|---|---|
| `chemical_id`           | string | paper's compound label (e.g. `TBBPA (3,3,5,5-TETRABROMOBISPHENOL A)`) |
| `chemical_iupac`        | string | PubChem IUPAC name |
| `chemical_CAS`          | string | CAS registry number (829 pesticides populated; empty for most industrial / metabolite / mycotoxin rows) |
| `chemical_InChIKey`     | string | canonical 27-char key |
| `chemical_SMILES`       | string | PubChem isomeric SMILES |
| `chemical_PubChemCID`   | int    | PubChem CID (3 rows empty) |
| `chemical_class`        | string | `pesticide`, `pesticide metabolite`, `pesticide-related`, `industrial chemical`, or `mycotoxin` |
| `indicated_application` | string | For pesticides: semicolon-joined union of top-level BCPC activities (`herbicide`, `fungicide`, `insecticide`, `acaricide`, `nematicide`, `rodenticide`, `regulator`, `safener`, ...). For industrial chemicals: the paper's `INDUSTRIAL CHEMICAL group` (e.g. `PAH (Polycyclic aromatic hydrocarbon)`, `Plasticizer`, `Bisphenol`, `Brominated flame retardant`). Empty for metabolites / pesticide-related compounds that the paper did not classify. |
| `bacterium_species`     | string | underscore-separated Linnaean name (`Bacteroides_thetaiotaomicron`, `Parabacteroides_distasonis`, ...) |
| `bacterium_strain_id`   | string | `NT5xxx / DSM xxxx` combined code |
| `bacterium_phylum`      | string | `Bacteroidota`, `Bacillota`, `Actinomycetota`, `Pseudomonadota`, `Fusobacteriota`, `Verrumicrobiota` |

Search space size: **23,672** pairs (1,076 chemicals × 22 bacteria).

### Bacterium panel (22 strains, 6 phyla)
| phylum | count | strains |
|---|---|---|
| Bacteroidota     | 9 | `B. caccae, B. clarus, B. dorei, B. stercoris, B. thetaiotaomicron, B. uniformis, Odoribacter splanchnicus, Parabacteroides distasonis, P. merdae` |
| Bacillota        | 7 | `C. difficile, Coprococcus comes, Eubacterium rectale, L. gasseri, L. paracasei, Roseburia intestinalis, Streptococcus salivarius` |
| Actinomycetota   | 2 | `Bifidobacterium longum subsp. longum, Collinsella aerofaciens` |
| Pseudomonadota   | 2 | `Escherichia coli ED1a, E. coli IAI1` |
| Fusobacteriota   | 1 | `Fusobacterium nucleatum subsp. animalis` |
| Verrumicrobiota  | 1 | `Akkermansia muciniphila` |

### Chemical classes (1,076 compounds)
| `chemical_class`        | count | hit rate (chem with ≥1 hit) |
|---|---|---|
| pesticide               | 829 | 138 / 829 = 16.6% |
| pesticide metabolite    | 119 |   6 / 119 =  5.0% |
| pesticide-related       |  75 |  10 /  75 = 13.3% |
| industrial chemical     |  48 |  13 /  48 = 27.1% |
| mycotoxin               |   5 |   1 /   5 = 20.0% |

## Objectives f

The primary scalar is **`rAUC` = relative area under the 24 h growth
curve, normalised to DMSO controls on the same plate**. **Lower rAUC =
stronger inhibition.** The paper does not define a "maximize" direction —
the biological question is "which compound inhibits most" — so in our
framework the `BenchmarkDataset` loader will set `maximize=True` on
`f = 1 − rAUC` (inhibition strength, bounded roughly in [−0.3, 1.0]
with higher = more inhibitory). The curated CSV stores **raw rAUC
directly**; the sign flip is done inside the loader.

| column | type | description |
|---|---|---|
| `rAUC`              | float | Median-of-3-replicate normalised AUC. 1.0 = DMSO-equivalent growth; 0.0 = complete killing; >1.0 = growth promotion. Min in dataset = 0.004 (hexachlorophene vs Clostridium difficile-like), max = 3.59 (outlier growth promotion). |
| `is_inhibitory_hit` | bool  | Paper's official call: `rAUC < 0.8 AND p_adj < 0.05 in ≥2 of 3 replicates`. Exactly **588 True rows**, matching the paper's headline. |

### Dose-response sub-file (`dose_response.csv`)
Same columns as `single_dose.csv` plus:

| column | type | description |
|---|---|---|
| `concentration_uM`     | float | One of 2.5, 5, 10, 20, 40, 80, 160, 320 µM |
| `is_hit_this_conc`     | bool  | Hit flag at this specific concentration |
| `is_hit_any_conc`      | bool  | Hit at ≥1 tested concentration |
| `was_main_screen_hit`  | bool  | Was this (chemical, strain) pair also flagged in the primary 20 µM screen? |

600 rows = 8 bacteria × 11 compounds × variable (5 or 8) concentration
points per pair. The 11 compounds are strong hits from the primary
screen (abamectin, chlorothalonil, closantel, cyhalofop-butyl,
emamectin benzoate, fenazaflor, fluazinam, imazalil sulfate, prochloraz,
propiconazole, pyraclostrobin). Dose-response monotonicity is strong:
median Pearson correlation between concentration and rAUC across
evaluated (compound, strain) pairs is **−0.71**, and 74 / 87 pairs
show negative dose dependence.

## Data validation

| claim (paper) | this dataset |
|---|---|
| 1,076 compounds × 22 bacteria = 23,672 pairs | 23,672 rows  OK |
| 588 inhibitory hits | 588  OK (exact) |
| 168 distinct inhibitory chemicals | 168  OK |

## Caveats
- **Direction of optimization.** `rAUC` in the CSV is **lower = more
  inhibitory**. The `BenchmarkDataset` loader must set
  `maximize=True` with `f = 1 − rAUC` (or `f = −rAUC`). Single-objective
  framings can also use `is_inhibitory_hit` as a binary oracle.
- **Single dose (20 µM).** The primary matrix is all at 20 µM. A
  chemical that looks inactive here may be potent at higher concentrations
  (`dose_response.csv` shows several compounds that only become hits at
  40–320 µM) and vice versa (rare cases of hormesis).
- **Replicates are medianed.** `rAUC` is the median of three biological
  replicates (paper's spec). Per-replicate data and the corresponding
  per-strain, per-rep normalised_auc / z-scores / p-values **are**
  available for the main pesticide screen and the xenobiotic screen via
  Mendeley (`5eff6745/results.txt` 11 MB, and
  `4e697abf/results.txt` 4.9 MB). We did not propagate per-replicate
  uncertainty into the curated CSV — downstream noise models should
  assume light Gaussian jitter.
- **Hit criterion.** `is_inhibitory_hit` is the paper's consensus
  (`rAUC < 0.8 AND p_adj < 0.05 in ≥2 of 3 replicates`). Different cutoffs
  would change the hit count, but 0.8 is the paper's reference value —
  we honour it.
- **Solubility / precipitation.** The authors note that TBBPA and some
  other lipophilic compounds are at or near their solubility ceiling
  at 20 µM; growth effects cannot be fully disentangled from
  precipitate-dependent OD artefacts. We do not expose a separate
  solubility flag because the Mendeley metadata does not either —
  treat compounds with xlogp > 5 (available via SuppData1's
  `PUBCHEM_xlogp`) with some caution.
- **CAS availability is partial.** Only 826 / 1,076 rows have a CAS
  number (the 829 BCPC-classified pesticides minus 3 without a clean
  BCPC match). Metabolites and most industrial chemicals have
  `chemical_CAS = ""`. Use `chemical_InChIKey` (1,073 / 1,076 populated)
  as the canonical structural identifier for LLM / search applications.
- **Taxonomy.** Phyla use the modern Bacteroidota / Bacillota /
  Actinomycetota / Pseudomonadota names as the paper does; older
  literature (Bacteroidetes / Firmicutes / Actinobacteria /
  Proteobacteria) corresponds to the `PhylumFig` column of SuppData2
  which we did not carry into the curated CSV.
- **2 extra bacteria** (`Eggerthella lenta` and some extras) appear in
  the earlier xenobiotic screen on Mendeley but are NOT in the primary
  1,076-compound screen — we follow the paper's canonical 22-bacterium
  panel from SuppData3.
- **3 pesticide-related rows without a class.** Three SuppData1 rows
  have empty `INDUSTRIAL CHEMICAL group` and no PESTICIDE CLASSIFICATION
  flags set (pesticide metabolites without a parent indicated). Their
  `indicated_application` is empty; they are still included in the
  22,572 row matrix.

## Files in this directory
```
single_dose.csv                  # 23,672 rows — the main benchmark
dose_response.csv                #    600 rows — 8×11×5-8 dose-response
DATA_CARD.md                     # this file
```
