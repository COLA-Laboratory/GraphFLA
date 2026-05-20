# Puttreddy PyNO-AgTFA metallogel — data card

## Source
Puttreddy, Thongrom, Rautiainen, Lahtinen, Kukkonen, Haag, Moilanen, Lundell & Rissanen,
"Free-Standing Supramolecular Pyridine N-Oxide-Silver(I) Metallogels," *Adv. Mater.* (2025).
DOI: [10.1002/adma.202502818](https://doi.org/10.1002/adma.202502818)  |  PMID: 40590158  |  PMCID: [PMC12447065](https://pmc.ncbi.nlm.nih.gov/articles/PMC12447065/).

Built from:

| local file | journal filename | contents used |
|---|---|---|
| `Supp_Info.docx` | `ADMA-37-2502818-s002.docx` | Supplementary Information. We parse Tables **S1**, **S2** (the 27×8 gelation matrix at 0.25 w/v%), **S3**, **S4** (same matrix at 0.5 w/v%), and **S11** (CGC in toluene for the 11 gel-forming PyNOs). |

The inner CIF bundle `ADMA-37-2502818-s001.zip` (66 crystal structures) is **not** used here — it's structural characterization of the metallogel product, not the gelation-screen scalar we want to optimize.

## Background in one paragraph
The authors mix a **pyridine N-oxide (PyNO)** ligand with **silver(I) trifluoroacetate (AgTFA)** at 1:1 stoichiometry in a solvent and observe whether the mixture gels. They screen 27 PyNOs (varying substituents: alkyl, aryl, methoxy, tert-butyl, cyano, nitro, isoquinoline, etc.) across 8 solvents — 4 nonpolar/polar-aprotic (benzene, toluene, EtOAc, acetone) and 4 polar protic/aprotic (DMSO, DMF, EtOH, H₂O). The key design rule that emerges: **gelation requires (a) the N-oxide functional group** (control pyridines without N-O never gel — Tables S5–S6) and **(b) an electron-donating ring substituent** (cyano and nitro derivatives 20, 21 never gel). The strongest gelators achieve "supergelator" status (CGC < 0.1 w/v%) in toluene and can be moulded into free-standing rods able to hold their own weight in solvent.

## Search space X

Primary benchmark (`gelation_main.csv`): each row is one (PyNO, solvent) cell.

| column | type | values | meaning |
|---|---|---|---|
| `pyno_id` | int | 1 .. 27 | compound index as in the paper's Figure S1. Discrete choice; identities are partially known (see `pyno_identity.csv`). |
| `solvent` | string | benzene, toluene, etoac, acet, dmso, dmf, etoh, h2o | 8 solvents. |
| `pyno_wv_pct` | float | 0.25 | PyNO concentration in w/v%. Fixed in the primary benchmark; the `gelation_0_5wv.csv` file re-screens at 0.5 w/v%. |
| `pyno_agtfa_ratio` | string | "1:1" | stoichiometry. Constant. |

**Search space size:** 27 × 8 = **216 cells** at each of two concentrations.

### PyNO identity (`pyno_identity.csv`)
Only the 27 numeric IDs appear in machine-parseable SI text. Names are given in **Figure S1 (a schematic image)**; per curation policy we do not OCR figures, so `pyno_name` / `pyno_smiles` are populated **only** for compounds referenced unambiguously elsewhere in the SI:

| pyno_id | confirmed name | class |
|---|---|---|
| 1 | pyridine N-oxide | electron-donor |
| 2 | 2-methylpyridine N-oxide | electron-donor |
| 4 | 4-methylpyridine N-oxide | electron-donor |
| 8 | 3,4-dimethylpyridine N-oxide | electron-donor |
| 17 | 4-*tert*-butylpyridine N-oxide | electron-donor |
| 20 | 4-cyanopyridine N-oxide | electron-withdrawer |
| 21 | 4-nitropyridine N-oxide | electron-withdrawer |
| 23 | 2-methyl-4-nitropyridine N-oxide | mixed |
| all other | *unknown* (encoded in Figure S1 only) | unknown |

### Solvent properties (`solvent_properties.csv`)
Reference values (CRC Handbook / NIST) for optional featurization:

| solvent | polarity bucket | ε (25 °C) | b.p. °C |
|---|---|---|---|
| benzene | nonpolar-apolar | 2.28 | 80 |
| toluene | nonpolar-apolar | 2.38 | 111 |
| etoac | polar-aprotic | 6.0 | 77 |
| acet | polar-aprotic | 20.7 | 56 |
| dmso | polar-aprotic | 46.7 | 189 |
| dmf | polar-aprotic | 36.7 | 153 |
| etoh | polar-protic | 24.6 | 78 |
| h2o | polar-protic | 78.5 | 100 |

## Objectives f

Two co-existing scalars — primary is binary, secondary is continuous but defined only on the 11 gel-forming PyNOs.

| file | column | type | direction | notes |
|---|---|---|---|---|
| `gelation_main.csv` | `gel` | {0, 1} | **maximize** | 1 = gel forms; 0 = "No gel" as recorded in Table S1/S2 at 0.25 w/v%. |
| `gelation_0_5wv.csv` | `gel` | {0, 1} | **maximize** | Higher-concentration rescreen (Tables S3/S4) at 0.5 w/v%. |
| `cgc_toluene.csv` | `cgc_wv_pct` | float | **minimize** | critical gel concentration in toluene for the gel-forming PyNOs. Lower = stronger gelator. |

**Typical optimization framing:** binary classification over the 216-cell matrix (most BO/AL baselines). If you want a continuous regression task, join `cgc_toluene.csv` onto `gelation_main.csv` on `(pyno_id, solvent='toluene')` and treat the 11 finite CGCs as the objective; non-gelling cells become censored (+∞).

## Caveats
- **Most PyNO identities are Figure-S1-only.** 19 of 27 `pyno_name` values are NaN because Figure S1 is a schematic and we do not do figure extraction. This does not block BO/AL (they can treat `pyno_id` as a categorical choice) but it does limit featurization. If an LLM-featurized run needs structures, cross-reference Figure S1 manually or request the authors' deposited ligand list.
- **Class labels (`pyno_class`) are only set when the identity is confirmed.** 19 rows carry `pyno_class = "unknown"`. The paper's generalization "electron-donors gel, electron-withdrawers do not" is the mechanistic summary, but we did not back-fill classes by gel behaviour (that would be label leakage).
- **All 8 polar protic/aprotic cells are always "No gel."** At **both** 0.25 and 0.5 w/v%, every cell in {DMSO, DMF, EtOH, H₂O} is non-gelling (108 / 108 at each concentration). If you down-sample the benchmark randomly you will see a strong solvent-class marginal — any sensible BO/AL will exploit it. This is a real, reproducible experimental finding (the authors explicitly attribute non-gelation in polar solvents to disruption of Ag–O coordination by competing donors), but it is also a large "dead zone" that makes the matrix sparse.
- **Binary outcome is already thresholded by the authors.** The SI cells are literally "gel" or "No gel" — no borderline / partial-gel / precipitate sub-labels. SEM evidence (Figures S2–S55) supports the classification, but the scalar we optimize is the experimenter's visual inversion-tube call.
- **CGC is toluene-only.** Table S11 reports CGC only in toluene; no CGC values for benzene / EtOAc / acetone. If you try to regress CGC as a function of (pyno, solvent), you're limited to 11 labelled cells.
- **0.25 vs 0.5 w/v%: monotone but not identical.** Moving from 0.25 → 0.5 w/v% flips two cells from "No gel" to "gel": (PyNO 3, acetone) and (PyNO 11, toluene) — a tight confirmation of concentration-dependent gelation. Don't mix rows from both tables in the same train/test split.
- **No data deposit.** The paper does not deposit the gelation matrix as CSV/XLSX/Mendeley/Zenodo — we reconstruct it by parsing the .docx SI. Tables S1–S4 are hand-entered; any typos in the source reproduce here. CIF structures are the only separately deposited artefact (CCDC numbers 2420930–2421021; see `ADMA-37-2502818-s001.zip` on PMC if needed).
- **Stoichiometry is fixed at 1:1 PyNO:AgTFA.** Table S9 explores mixed-ligand systems (PyNO1 + PyNO2) and Table S7 probes alternative silver salts (AgPF6 / AgBF4 / AgSbF6 / AgClO4 for PyNO 4 only). These 41 + 8 cells are NOT in the primary benchmark — curating them would require introducing a second ligand column and a salt column. If needed for richer benchmarks, add them as follow-up CSVs.

## Files in this directory
```
gelation_main.csv       # 216 rows — 27 PyNOs x 8 solvents at 0.25 w/v%   (primary)
gelation_0_5wv.csv      # 216 rows — same grid at 0.5 w/v%                (secondary)
cgc_toluene.csv         # 11 rows — CGC in toluene for gel-forming PyNOs  (continuous)
pyno_identity.csv       # 27 rows — pyno_id, pyno_name (8 known / 19 NaN), pyno_smiles, pyno_class
solvent_properties.csv  # 8 rows — reference dielectric / boiling point / polarity bucket
DATA_CARD.md            # this file
```
