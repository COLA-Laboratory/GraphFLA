# Scope ledger — definition of done

**Scope rule (set by the user, 2026-09-30):** within the **supplied corpus**, only
landscapes with **more than 1,024 variants**. No supplied-corpus study at or below
1,024 variants is worked. The cutoff selects which supplied studies to spend effort
on; it does not reject a core-literature paper whose landscape happens to be small.
The Ferretti csI reproduction below uses 32 genotypes and is in scope for exactly
that reason.
Plus: for each GraphFLA metric family, any highly relevant paper the supplied corpus
missed — in particular the paper that PROPOSED the metric.

When every row below is `done`, `closed` or `blocked`, the tranche is finished.

## Scope-selection caveat

`corpus_registry.json` records `priority_size: 0` for entries whose size was never
parsed, so a mechanical `priority_size > 1024` filter silently drops them. Two such
entries exist:

- `DomingoDL18` — recorded 0, actually 2^6 x 3^4 = **5,184** → **in scope**
- `Ogbunugafor22` — recorded 0, actually (2^4) x 12 conditions → out of scope

Anyone re-deriving this list must resolve size from `landscapes[].theoretical_size_expression`,
not from `priority_size`.

Exactly at 1,024 and therefore OUT of scope: `SchulzTWW25`, `BakerleeNSRD22`.
(Noted only because Bakerlee 2022 is the closest thing in the corpus to a core reference
for the diminishing-returns / increasing-costs family. It stays out under the scope rule.)

## Supplied corpus, >1,024 variants — 19 studies

| study_key | variants | state | result |
| --- | --- | --- | --- |
| PapkouRM23 | 262,144 | done | 514 peaks; 740,211 motifs incl. 85,203 reciprocal-sign, exact |
| KuoJCCLHWC20 | 262,144 | closed | definition mismatch: paper's conditional-mean equations are not GraphFLA's OLS |
| PodgornaiaL15 | 160,000 | blocked | Science SI behind a robot check; needs author functional classes |
| WuDOLS16 | 160,000 | done | 30 peaks, 15 above WT; 92.638% vs published 93% |
| JalalTSCLTNLL20 | 160,000 | closed | path definition differs; workbook provenance unresolved |
| TuSE22 | 160,000 | closed | preprint states ">1,200 peaks", not an exact target |
| JohnstonAWLPYA24 | 160,000 | done | 520 peaks among author-designated active candidates |
| WestmannGW24 | 65,536 | done | 2,092 peaks: 58 above WT, 2,034 below |
| SooSFW21 | 65,536 | closed | paper never defines which Ferretti gamma it plots; plot-only |
| BendixsenCOH19 | 16,384 | done | HDV 982 and ligase 68 peaks, exact; lead-verified |
| PhillipsLMDCJCMWD21 | 65,536 | in flight | CR6261 done (4 OLS R2); CR9114 under review |
| PhillipsMBDSD23 | 65,536 | in flight | |
| WongKK18 | 32,768 | in flight | |
| MoulanaDP22 | 32,768 | done | 0 monotone-increasing Wuhan->BA.1 paths (lead-verified); order-5 R2 mismatch 0.99406 vs printed 0.97754 |
| MoulanaDPCRGSBD23 | 32,768 | closed | no GraphFLA metric counterpart; escapers are dropped not censored, so n_lo describes a punctured sub-landscape |
| PoelwijkSR19 | 8,192 | done | order-2 R2 0.87335 vs printed 0.87 (lead-verified); per-order fractions are plot-only |
| LiteGNLGL20 | 8,000 | done | no direct metric overlap; ParE3 W>0.5 = 1847/7882 reproduced (lead-verified). Codon-accessibility is WAGNER's rule, not Lite's |
| DomingoDL18 | 5,184 | done | paper prints NO local-optima count; froze S1 integrity + Wagner per-edge oracle instead |
| BaezaCenturionMSVL19 | 3,072 | closed | CSV identity confirmed against Table S3; 3,072 x 11 DNA exact. Scaling-law paper, no like-for-like metric target |

## Core literature the supplied corpus missed

| family | reference | state | result |
| --- | --- | --- | --- |
| evolvability_enhancing_mutations | Wagner, Nat Commun 14:3624 (2023) | done | cited DOI does not exist; implemented inequality is the one the paper rejects |
| idiosyncratic_index | Lyons et al., Nat Ecol Evol (2020) | done | landscape aggregate IS defined by the paper; GraphFLA's baseline is the analytic limit, not the paper's control |
| gamma, gamma_star, r_s_ratio | Ferretti et al., JTB 396:132 (2016) | done | **first independent empirical reproduction of gamma and r/s.** A. niger csI: peaks 4, sinks 2, gamma 0.3273->0.33, r/s 0.8928->0.89 all match Fig 4c (lead-verified). gamma* 0.2692 vs 0.25 differs |
| fdc, autocorrelation, neutrality, accessibility, path length, basin, extradimensional bypass | — | in flight | core references to be found and verified |
| higher_order_epistasis, walsh_hadamard, classify_epistasis | — | in flight | |
| gradient_intensity, fitness_flattening_index, neighbor_fitness_correlation, diminishing_returns/increasing_costs scalars | — | in flight | may have NO published origin; that is the expected and acceptable answer |

## Not in this tranche

Production fixes. Two confirmed metric-definition divergences (EE mutations,
idiosyncrasy baseline) are recorded in `metric_provenance_findings.md` and deliberately
left unfixed for a later session.

## Carried forward for the packaging / development session

- **API gap.** Moulana 2022's headline result is the number of monotone fitness-increasing
  paths between two named genotypes. GraphFLA exposes no public metric for it:
  `local_optima_accessibility` only targets local optima, and BA.1 is not one. The
  reproduction had to reach into `ls.graph.subcomponent`. A public
  "accessible paths between two configurations" helper would make this a first-class test.
- **Record convention.** Agents place paper statistics that have no GraphFLA counterpart in
  `overlaps` with `metric: "none"`. Those rows are evidence about the paper, not metric
  comparisons, so a record can legitimately carry `reproduced_*` rows and still have
  `verdict: "no_overlap"`. Read `metric` before reading `outcome`.
- **NaN is not censoring.** In the Moulana inputs a NaN fitness mixes QC failures with true
  non-binders, and the two have different meanings. Any promoted case must state which
  population it describes.
- **Bibliography errors found so far.** `KuoJCCLHWC20` has an entirely wrong author list;
  `MoulanaDP22` is truncated to 3 of 10 authors. The supplied table is not a citation source.

## Ferretti Figure 4c — detail

The csI sub-landscape of de Visser et al. 1997 is the first empirical input on which
GraphFLA reproduces gamma and r/s against a published number. Recipe, lead-verified:

- loci: argH12, pyrA5, leuA1, oliC2, crnB12 (the csI membership stated in de Visser 1997,
  Results p.1502 / Table 4 p.1503 -- NOT inferred)
- values: Franke et al. 2011 Supplementary Table S1, relative mean mycelium growth rate W
- transform: f = ln(W)
- source file: papers/ferretti2016/sources/Franke2011_deVisser_CSI_codex.csv

| statistic | GraphFLA | Fig 4c | |
| --- | --- | --- | --- |
| peaks | 4 | 4 | match |
| sinks | 2 | 2 | match |
| gamma | 0.327295 | 0.33 | match at printed precision |
| r/s | 0.892831 | 0.89 | match at printed precision |
| gamma* | 0.269231 | 0.25 | **differs** |

gamma* is the one miss. Ferretti's gamma* (Eqs. 10-12) carries an epsilon-tolerance
trichotomy for calling an effect positive / neutral / negative; GraphFLA uses `np.sign`
with no tolerance. That is a plausible definitional cause and should be checked before
gamma* is treated as reproducible.

The TEM / beta-lactamase half of Figure 4c did **not** reproduce. Averaging the source MIC
replicates and applying ln, log10 or log2 all give gamma 0.8353, gamma* 0.6136,
r/s 0.4441 against printed 0.85 / 0.59 / 0.43. Unresolved; the Weinreich Supporting Online
Material and the MAGELLAN `.fl` inputs remain access blockers.

### Data defect found in the GraphFLA repository

`data/BioSequence/Weinreich2006Tan2011_Weinreich2006.csv` stores fitness 15.32333629 for
the all-ones genotype, which back-transforms to 2^15.3233 = **41,000**, while the Weinreich
2006 source MIC table prints **4,100** -- a factor of ten, on the global optimum. The other
31 rows were not contradicted. This is a defect in bundled data, not in any metric. It is recorded rather than
fixed: this tranche changes code only where a confirmed defect already had a
failing test, and does not touch bundled datasets.
