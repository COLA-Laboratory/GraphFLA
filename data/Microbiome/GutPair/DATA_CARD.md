# Kamrad gut-bacterial pairwise co-culture — data card

## Source
Kamrad, Aulakh, Mozzachiodi, Blasche, Scheidweiler, Basile, Guan, Bradley,
van den Berg, Mülleder, Ralser & Patil,
"Interspecies interactions drive bacterial proteome reorganization and
emergent metabolism," *Nat. Ecol. Evol.* (2026).
DOI: [10.1038/s41559-026-03030-4](https://doi.org/10.1038/s41559-026-03030-4).

Built from:

| local file | Springer filename | contents used |
|---|---|---|
| `Supp_Table1.xlsx` | `MOESM3_ESM.xlsx` | Supp. Table 1 (strain metadata), Supp. Table 4 (pairwise fold-change + ecological interaction classes). |
| `Source_Data.xlsx` | `MOESM4_ESM.xlsx` | Source Data for Figs 1, 2, 3, 6 and Extended Data Figs 2, 5 — kept for reference; not parsed by `curate_dataset.py`. |

## Background in one paragraph
The authors co-cultured all pairs of 15 human-gut bacterial strains (13
commensals + 2 pathogens: *C. difficile*, *K. aerogenes*; strain list
spans Bacteroidetes, Firmicutes, Proteobacteria and Actinobacteria) in
modified Gifu anaerobic broth (mGAM) for 24 h, then quantified
(a) total culture density OD₆₀₀, (b) the absolute protein biomass of
each species via DIA-PASEF proteomics, and (c) targeted supernatant
metabolomics. From the proteomic biomass ratios they derive a directed
per-(focal, partner) **fold-change of relative abundance** —
`relAbu(focal in coculture) / relAbu(focal in monoculture)`. This scalar
measures how much a given partner boosts (>1) or suppresses (<1) the
focal species' biomass. 104 of the 105 possible unordered pairs passed QC
(one pair with *B. longum* was dropped for poor proteomic coverage).

## Search space X
Every row in `fold_change.csv` is one directed (focal, partner) pair:

| column | type | meaning |
|---|---|---|
| `focal_species`   | string | 2-letter code (Bt, Bu, Pv, Ca, Cd, Cs, Ec, Ar, Ka, La, Lg, Ls, Sc, Pm, Rg). 15 unique. |
| `partner_species` | string | same 15-species alphabet. Directed pair; (A, B) ≠ (B, A) in general. |
| `focal_species_long` / `partner_species_long` | string | full species names. Metadata, not part of the candidate choice. |

Search space size: **223 directed pairs** (15 × 15 = 225; two cells are
missing because of proteomic coverage drops — one for Pv(focal)
and one for Rg(focal)).

### Species codes

| code | species | phylum |
|---|---|---|
| Bt | *Bacteroides thetaiotaomicron* | Bacteroidetes |
| Bu | *Bacteroides uniformis* | Bacteroidetes |
| Pv | *Phocaeicola vulgatus* (= *B. vulgatus*) | Bacteroidetes |
| Pm | *Parabacteroides merdae* | Bacteroidetes |
| Sc | *Segatella copri* (= *Prevotella copri*) | Bacteroidetes |
| Ka | *Klebsiella aerogenes* | Proteobacteria |
| Ec | *Escherichia coli* | Proteobacteria |
| Rg | *Ruminococcus gnavus* | Firmicutes |
| Ls | *Lacrimispora saccharolytica* | Firmicutes |
| Ar | *Agathobacter rectalis* (= *Eubacterium rectale*) | Firmicutes |
| Cs | *Clostridium sporogenes* | Firmicutes |
| Cd | *Clostridioides difficile* | Firmicutes |
| La | *Lactobacillus acidophilus* | Firmicutes |
| Lg | *Lactobacillus gasseri* | Firmicutes |
| Ca | *Collinsella aerofaciens* | Actinobacteria |

Supp. Table 4 uses the pre-2016 taxonomy (Bv, Pc, Er); we rename them to
Pv / Sc / Ar in the curated CSV.

## Objectives f

| column | type | direction | meaning |
|---|---|---|---|
| `fold_change` | float | **maximize** | `relAbu(focal in co-culture) / relAbu(focal in mono-culture)`. Range ≈ 0.19..2.59. 1.0 means no effect; >1 means the partner boosts the focal. |
| `p`    | float | — | uncorrected p-value for "fold-change ≠ 1" from limma moderated t-test. |
| `padj` | float | — | Benjamini-Hochberg FDR-adjusted p-value. |
| `interaction_type` | string | — | ecological-interaction class of the **unordered** {focal, partner} pair: 76 × exploitation, 74 × competition, 50 × amensalism, 4 × commensalism, 2 × mutualism, 2 × neutralism. NaN for self-pairs. |

## Caveats

- **Directed pairs.** (focal = A, partner = B) and (focal = B, partner = A)
  are separate rows. In the wild you'd need to measure both to get the full
  (A↔B) interaction. That's why the effective candidate pool is 223
  directed cells, not 104 unordered.
- **Self-pair diagonal is 1.0 tautologically** (15 rows with
  `focal_species == partner_species`). Within numerical rounding — two
  cells are actually 0.999999… Don't treat these as learnt information.
- **2 missing cells** (one for Pv focal, one for Rg focal) — the original
  Supp. Table 4 has NaN there. Curated CSV drops them, so the pool is 223,
  not 225. `BenchmarkDataset.as_candidate_pool()` reflects this.
- **Scalar is relative-abundance fold change**, not absolute biomass. Two
  species growing equally well in co-culture (both at 50% abundance) with
  a total OD₆₀₀ of 8.0 show the same `fold_change ≈ 1` as two species
  growing equally badly (both at 50%) with total OD ≈ 0.5. If you want an
  "absolute productivity" objective, join with the total OD₆₀₀ per
  co-culture from Source_Data.xlsx Fig1F (not parsed here).
- **Interaction type is per unordered pair** — A→B and B→A share the
  same class label because it classifies the pair, not the direction.
- **Bt-starch-metabolism is a context-dependent finding.** Paper shows
  Bt promotes Cd in mGAM medium but suppresses it in BHI. Our scalar is
  mGAM-only; don't generalise to other media.
- **`B. longum` was excluded at QC**, so 14 species × 14 species + 15
  self-pairs would give 211; the paper reports 15 species × 15 = 225 and
  we get 223 because two cells (not entire species) failed QC.

## Files in this directory
```
fold_change.csv        # 223 rows — main benchmark
species_metadata.csv   # 15 rows — per-species code, Latin name, genome ID
DATA_CARD.md           # this file
```
