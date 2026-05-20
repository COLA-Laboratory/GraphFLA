# McGuire_Metal-Depoly

## Source

McGuire, T. M.; Buchard, A.; Williams, C. K.
**"Chemical Recycling of Polyesters and Polycarbonates: Why Is Zinc(II) Such
an Effective Depolymerization Catalyst?"**
*J. Am. Chem. Soc.* **2025**, *147*, 43077-43085.
DOI: [10.1021/jacs.5c16346](https://doi.org/10.1021/jacs.5c16346)

Raw TGA traces (kinetic mass-loss curves per run) deposited at
Oxford University Research Archive, DOI:
[10.5287/ora-aqa4e6xvr](https://doi.org/10.5287/ora-aqa4e6xvr),
149.6 MB zip. Our curation keeps only the kinetic CSV folders and downsamples
each trace to 300 points (preserves the sigmoid shape; brings the local cache
from 628 MB to 0.9 MB).

## Background in one paragraph

Eight commercial metal 2-ethylhexanoate salts, M(Oct)_n (M = Zn, Co, Mg, Sn,
Ca, Ba, Bi, Y; n = 2 for divalent, 3 for trivalent) are screened for the
melt-phase ring-closing depolymerization of six dihydroxy-telechelic
polyesters / polycarbonates (DP ~ 100) to their cyclic monomers. Reactions
are run neat (solventless) in a TGA under 25 mL/min N2 at the isothermal
setpoint and catalyst:polymer loading noted per polymer. The paper establishes
a structure-performance linear free energy relationship: log(k_obs) correlates
exponentially with the metal's Lewis acidity (aqueous pKh), with Zn(II) being
the fastest catalyst for every polymer tested. The study benchmarks catalyst
performance against four polymer backbone properties: chain linker length
(6 vs 7 atoms), carbonyl type (ester vs carbonate), end-group (primary vs
secondary hydroxyl), and methyl-substitution at the ring-closure carbon.

## Search space X

Categorical 8 x 4 grid = 32 (metal, polymer) screening points on the primary
benchmark (depoly_rates.csv). An auxiliary 6-row LFER file covers the two
methyl-substituted polymers where only three catalysts (Co, Mg, Sn) have
raw kinetic traces in the public ORA archive; Zn/Ca/Ba/Y/Bi kinetic data for
P3MeVL / P4MeCL are shown qualitatively in the paper's Fig 4a but not
archived, so those 10 cells are absent.

| column                 | type        | values / meaning                                                                     |
|------------------------|-------------|--------------------------------------------------------------------------------------|
| `metal`                | categorical | one of {Zn, Co, Mg, Sn, Ca, Ba, Y, Bi} - the catalyst metal center                   |
| `polymer`              | categorical | one of {PVL, PDTC, PHL, PCL} for the primary grid                                     |
| `T_C`                  | float       | screening temperature / °C; polymer-specific (130, 150, or 160)                       |
| `loading`              | string      | [cat]:[polymer repeat unit] ratio string ("1:100" or "1:1000")                       |
| `cat_polymer_ratio`    | float       | same as `loading` but numeric                                                         |
| `linker_atoms`         | int         | polymer backbone repeat unit linker length (6 or 7)                                   |
| `carbonyl`             | string      | "ester" or "carbonate"                                                                |
| `polymer_substitution` | string      | "none" / "3,3-diMe" / "3-Me" / "4-Me"                                                |
| `metal_oxidation`      | int         | +2 or +3                                                                              |
| `pKh`                  | float       | metal-aqua hydrolysis constant (Brown & Ekberg 2016); used as Lewis-acidity descriptor |
| `ionic_radius_A`       | float       | Shannon 6-coordinate ionic radius / Å                                                 |

Search-space size: 32 primary cells (Cartesian 8 x 4). No missing primary
cells. Temperature and loading are not free variables - they are tied to the
polymer to match the paper's published screening condition.

## Objectives f

| column              | type    | direction | notes                                                                                       |
|---------------------|---------|-----------|---------------------------------------------------------------------------------------------|
| `k_rate_s_inv`      | float   | maximize  | depolymerization rate constant k_obs / s^-1 (sigmoidal fit; see Caveats)                   |
| `log10_k_rate`      | float   | maximize  | log10 of `k_rate_s_inv`; preferred for LFER work and for BO kernels on rate constants       |
| `k_rate_std_s_inv`  | float   | info      | inter-replicate std dev; n_runs in {2, 3, 6}                                                |
| `n_runs`            | int     | info      | TGA replicate count per cell                                                                |
| `fit_methods`       | string  | info      | "sigmoid" (normal) or "pseudo_1st" (fallback, rare)                                         |

Primary objective for optimization experiments: **maximize `log10_k_rate`**
(equivalently `k_rate_s_inv`).

Primary objective for optimization experiments: **maximize `log10_k_rate`**.
Note that each polymer row uses polymer-specific (T, loading) conditions,
so absolute k values are not directly comparable across polymers — within-polymer
metal rankings are the more informative comparison.

## Caveats

- **Not truly 8 x 6.** The paper claims coverage of six oxygenated polymers,
  but the public ORA archive contains full 8-metal replicate traces only for
  four polymers (PVL, PDTC, PHL, PCL) = 32 cells. For the two methyl-
  substituted polymers (P3MeVL, P4MeCL), only Co, Mg, Sn raw traces are
  deposited (Zn/Ca/Ba/Y/Bi kinetic data appear in Fig 4a of the paper but
  not in the archive). These 6 extra cells ship in `depoly_rates_LFER_extra.csv`
  and are **not** included in the primary optimization grid.

- **Conditions vary per polymer.** The (T, loading) combination is polymer-
  specific so that the fastest catalyst finishes within a TGA window. As a
  result, absolute k values **cannot be compared across polymers** directly;
  for example, PHL's k values look 4x higher than PDTC not because PHL is
  intrinsically more reactive but because PHL was run at 150 °C / 1:100 vs
  PDTC at 150 °C / 1:1000 (10x more catalyst). Within a polymer column, the
  k values are directly comparable and reproduce the paper's catalyst ranking.
  **If using this as a single-objective BO benchmark across the full 32-cell
  grid, this cross-polymer incomparability is a real feature of the landscape
  and not a bug.**

- **Sigmoid fit choice.** `k_obs` is the inflection slope of a logistic fit
  to fractional-mass-loss with asymptote fixed at 1.0 (full depolymerization),
  matching the functional form used in the paper's Methods (McGuire, Ning,
  Buchard, Williams JACS 2025, ref. 29 of the DOI under curation). For slow
  catalysts whose TGA trace shows only a partial ramp, the sigmoid
  extrapolates past the observation window; this is consistent with the
  paper's convention but gives a softer dynamic range (our PVL k_max/k_min
  = 90x vs the paper abstract's 1000x claim from their SI Table S4 numbers).
  A pseudo-first-order fallback is used only when the sigmoid optimizer
  fails to converge (0 runs in the current curation, but kept for robustness).

- **TGA abort artefact.** The instrument procedure aborts when wgt% drops
  below 5, producing a spurious terminal jump in the CSVs for very-slow
  catalysts (e.g. Sn/PDTC at 150 °C). We filter `wgt% > 5` before fitting;
  without that filter, the sigmoid optimiser places a narrow step near the
  trace end and inflates k by ~1000x.

- **Replicates per cell.** Most cells have n=2; the paper's key 4 fast
  catalysts (Zn, Co, Mg, Sn) on PVL have n=3-6 (extra Eyring-analysis runs
  at 130 C pool into the LFER condition). Report the inter-run std as
  `k_rate_std_s_inv`; it is typically 5-20% of the mean.

- **No P3MeVL / P4MeCL Zn data.** For any cross-polymer generalisation study,
  the 10 missing cells are informative: they correspond to the paper's
  methyl-substituted substrates for which the archive only deposited Co, Mg,
  Sn runs. Any method that models these as "missing-at-random" will be
  biased because the gap is structured (Me-substitution + specific metals).

- **pKh reference values.** Metal hydrolysis constants come from Brown &
  Ekberg (2016) as cited in the paper. Some values differ slightly in the
  literature (e.g. Zn pKh is reported as 8.96-9.0 across sources); if your
  pipeline uses a different Lewis-acidity descriptor (e.g. aqua pKa from
  Kumar & Blakemore 2021), regenerate `pKh` in `metal_descriptors.csv`.

- **Sn in PVL is a paper anomaly.** Paper Figure 2b reports the PVL ranking
  ... > Sn > Ba > Bi. Our fits put Sn below Ba (but above Bi), giving
  Spearman rho = 0.93 overall and identical top-3 {Zn, Co, Mg}. The Sn/PVL
  trace only reaches 23-34% mass loss in 300 min, so Sn's k is sigmoid-
  extrapolated and sensitive to the specific end-of-window weighting.

## Files in this directory

```
McGuire_Metal-Depoly/
├── curate_dataset.py                   fit TGA traces -> k_obs, assemble grid
├── validate_curated.py                 8 checks against paper claims
├── raw/
│   └── tga_runs/                       114 downsampled TGA CSVs (300 pts each)
│       ├── PVL, 130 deg, Table S4/                 20 files (8 metals x 2-3 replicates)
│       ├── PCL, 160 deg, Table S5/                 17 files
│       ├── PHL, 150 deg, Table S6/                 17 files
│       ├── PDTC, 150 deg, Table S5/                17 files
│       ├── LFER, Co, Table S8 and Fig S22/         14 files (Co on 5 polymers, 130 C)
│       ├── LFER, Mg, Table S9 and Fig S23/         15 files
│       └── LFER, Sn, Table S23 and Fig S24/        14 files
└── curated/
    ├── DATA_CARD.md                    this file
    ├── depoly_rates.csv                32 rows = 8 metals x 4 polymers (primary grid)
    ├── depoly_rates_LFER_extra.csv      6 rows = {Co,Mg,Sn} x {P3MeVL,P4MeCL}
    ├── metal_descriptors.csv            8 rows; pKh, ionic_radius_A, oxidation
    └── run_level_fits.csv             114 rows; per-replicate k_obs + fit_method
```
