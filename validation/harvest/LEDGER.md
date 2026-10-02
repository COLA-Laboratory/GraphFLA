# Scope ledger — definition of done

**2026-10-02 gamma addendum:** [the calculation review](../GAMMA_REVIEW.md)
corrects the later claim that ties reveal a gamma-star/classification bug.
Ferretti's published Eq. (12) assumes no neutral mutations; its failure outside
that domain is expected. New source-pinned equation checks and a numerical
scale fix are complete. The empirical gamma-star/TEM mismatches below remain
unresolved; they are not hidden by the new passing equation tests.

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
31 rows were not contradicted. This is a defect in bundled data, not in any metric, and is
recorded here rather than fixed because this tranche makes no repository changes.


## Second wave — metric-definition audits (2026-10-01)

Scope set by the user: audit DRI/ICI's approach, anchor extradimensional_bypass, check the
Walsh-Hadamard implementation, and reproduce the Papkou supplementary figures. The user
also ruled that **no literature validation is required** for `autocorrelation`,
`neutrality`, `basin_fitness_correlation`, `mean_path_length_to_global_optimum`,
`mean_distance_to_global_optimum`, `global_optima_accessibility`, `local_optima_ratio`,
`neighbor_fitness_correlation`, `gradient_intensity` and `fitness_flattening_index`.
Do not spend effort anchoring those; it is a decision, not an omission.

### New literature anchors

| metric | source | result |
| --- | --- | --- |
| `walsh_hadamard` | PLOS Comput Biol 10.1371/journal.pcbi.1012132, Table 1 | all 9 published coefficients match at the table's printed 2-decimal precision (lead-verified). Deviation from the **printed** values is up to 3.3e-3, which is rounding; the 2.3e-15 figure is against a direct V2H2y recomputation from the printed phenotypes, not against the table |
| `extradimensional_bypass` | Papkou 2023, Supplementary text S3 p.20, printed 23% | 19,377 / 85,203 = 0.22742 (lead-verified) |
| `classify_epistasis` | Papkou 2023 Fig. S20B | 408,065 / 246,943 / 85,203 exact |
| `basin_fitness_correlation` | Papkou 2023 Fig. S12 | four Spearman rho reproduce to 2 dp (bonus; this family needs no anchor) |

Lead-verified structural check for Fig. S12: decoding the nucleotide recoding and reading
codon 2 (DHFR position 27) partitions the 514 peaks into Asp 40, Glu 34, Cys 43, other 397,
exactly the figure's N values. This confirms our 514 peaks are the paper's 514 peaks
residue by residue, which is stronger evidence than any correlation in that figure.

### Confirmed non-reproducible, with reasons

- **Papkou Fig. S21** (`higher_order_epistasis`, printed R2 0.28/0.78/0.93/0.97):
  `definition_incompatible`. The paper fits penalized LASSO in a Walsh-Hadamard basis;
  GraphFLA computes unregularized one-hot polynomial OLS R2. The evaluation split is NOT
  STATED IN PAPER. Do not tune toward these numbers.
- **Papkou Fig. S13**: the paper is internally inconsistent. Panel D states 134,662
  non-peak variants; 135,178 - 514 = 134,664, and the paper's own Methods elsewhere also
  says 134,664. The discrepancy is 2, not an error in our peak set.
- **Papkou Fig. S22**: the printed 324,044 "mutations" population **is exactly** the
  filtered directed edge set GraphFLA reproduces. Identity confirmed. The regression slope
  itself is plot-only.

### Open defects found in this wave

1. **`walsh_hadamard` reports wrong positions.** Invariant sites are dropped in
   preprocessing and coefficient labels are not mapped back. `GAC -> GCC` mutates position
   2; GraphFLA returns `A_1_C`, `positions=(1,)`. The coefficient is right, the biological
   location is wrong, which would corrupt any residue-contact analysis. Lead-verified.
2. **`walsh_hadamard` hides rank deficiency.** On an incomplete landscape the fit is
   non-identifiable and the chosen coefficients are returned with no warning.
3. **`higher_order_epistasis` docstring is wrong.** It says values near 1 mean stronger
   epistasis *of the given order*; it is the cumulative R2 of all terms *up to* that order,
   so a perfectly additive landscape returns 1.0 at every order.
4. **`walsh_hadamard` categorical encoding breaks at 47 states** (`chr(48+c)` collides with
   the `_` delimiter) and reports remapped labels with no label map.
5. **`diminishing_returns_index` / `increasing_costs_index` do not match the field, and
   appear not to match GraphFLA's own paper.** See below.

### DRI / ICI verdict

The literature does **not** converge on a landscape-level scalar. It reports per-mutation
Pearson coefficients or slopes, or a described trend, from regressing a mutation's fitness
gain on its background's fitness. The maintainer's expectation was correct.

GraphFLA instead correlates node fitness against a **node-level mean** of improving-edge
gains. The behaviour audit calls this "wrong as a general representation-robust measure;
on a fixed raw scale with maximize=True it is an accurate but differently weighted
node-level correlation", and reports sensitivity to House-of-Cards structure, to fitness
scale, and to the maximize encoding.

**Needs the maintainer's own check:** the literature survey states that Huang, Zhou & Li
2025 (NeurIPS 38, 39477-39532, 10.52202/085713-1180) defines *pooled edge-level* Pearson
coefficients, whereas the code computes Pearson across node means. GraphFLA's current
output nevertheless matches that paper's Table A2 values. The likely reading is that the
paper's written definition does not describe the code that produced its own table. This is
the maintainer's own publication, so the lead did not act on it.


## Third wave — prominent-paper scouting and anchoring (2026-10-01)

Goal: find further prominent papers usable as reproducibility anchors, weighted toward the
thinnest metrics. Nine studies worked. Every claimed reproduction below was re-run by the
lead; two agent claims did not survive that check and are corrected in place.

### New anchors, lead-verified

| metric | source | result |
| --- | --- | --- |
| `r_s_ratio` | Szendro et al. 2013, J Stat Mech P01005, Table 2, on Chou2011 | r/s 0.121797 vs printed 0.122; `n_lo` 1 vs 1; order-1 Fourier 0.989489 vs 0.989; order-2 increment 0.009206 vs 0.009 |
| `r_s_ratio` | Song & Zhang 2021, Evolution 75:2658, Table 1, on Kuo2020 | 4.0628980 vs printed 4.063 — **only under the paper's reference-allele coding**, see defect 1 |
| `fitness_distribution` | Li et al. 2025, Cell Systems 16:101387, Table S1 | kurtosis AND max-normalised Cauchy location both match at printed precision on **14 of 15** bundled landscapes; the sole miss is `Tu2022_T7`, independently flagged as being on an inconsistent scale. Record the max-normalisation as case preprocessing: `cauchy_loc` is not scale-invariant |
| `global_idiosyncratic_index` | Lyons et al. 2020 | target 0.612 reproduced by an independent implementation; GraphFLA differs, gap decomposed below |
| `n_lo`, `classify_epistasis` | Crona et al. 2013, J Theor Biol 317:1 | two-locus Example 1 reproduced exactly |

Third-party confirmation: Ferretti, Weinreich, Tajima & Achaz 2018, Heredity 121:466
(10.1038/s41437-018-0110-1) independently reprints the A. niger csI values already anchored
here — 4 peaks, 2 sinks, gamma 0.33, r/s 0.89 — corroborating this ledger's reading of
Ferretti 2016 Fig. 4c.

### Defects found (recorded, NOT fixed — fixing is a later session's job)

1. **`r_s_ratio` depends on which allele the one-hot encoding drops.** On `Kuo2020.csv`
   (4-letter RNA alphabet) the same input gives r/s = 3.7324 dropping A (GraphFLA's current
   behaviour, `get_dummies(drop_first=True)`), 2.6628 dropping C, 2.9838 dropping G, and
   4.0629 dropping U. A **53% spread** from an arbitrary encoding choice. Roughness `r` is
   invariant; `s = mean(|beta|)` is not. The paper's own convention is U, which is why its
   printed 4.063 matches exactly there. Lead-verified. Undocumented.
2. **`gamma_star` and `classify_epistasis` are mutually inconsistent on data with ties.**
   Ferretti's own identity `gamma* = 1 - phi_s - 2*phi_rs` holds to ~1e-17 on tie-free
   landscapes but fails by **0.0449** on `Weinreich2006Tan2011_Weinreich2006.csv`: gamma_star
   gives 0.613636 against 0.658537 from the identity. Independent enumeration by the lead
   confirms 80 undirected squares of which **39 contain a tie**, leaving 41 classifiable.
   The two functions resolve ties over different populations. Affects every MIC /
   doubling-dilution dataset, including the 15 `Mira2015_TEM_*` landscapes.
3. **`global_idiosyncratic_index` gap decomposed.** Independent implementation of Lyons'
   procedure gives 0.6121 +/- 0.0046 against the printed 0.612. GraphFLA gives 0.5874
   (lead-verified digit for digit). Two causes: the analytic `sqrt(2)*sigma` baseline instead
   of the paper's matched-n resampled control accounts for about -0.018, and construction
   drops **3,903 of 28,530** viable genotypes (13.7%) as isolated, because Lyons uses all
   measured pairs rather than a connected mutational graph. `min_pairs=3` has no effect here.
4. **`n_lo` disagrees with `sum(graph.vs['is_lo'])`** on Kuo2020: 2390 against 2392. Plateau
   members versus plateaus, silently.
5. **`fitness_distribution` docstring claims the whole set is scale-invariant.** `cauchy_loc`
   is not, which is why Li et al. max-normalise, and is the likely cause of its 10/15 rather
   than 14/15 agreement.
6. **`ProteinLandscape` cannot build the His3 landscape**: Pokusaeva 2019's amino-acid
   sequences are variable length and the build fails. A capability gap, not a wrong number.

### Two lead overturns that were themselves wrong, and were reverted

Both are recorded because the failure mode is instructive: **re-running a claim under the
wrong preprocessing looks exactly like a refuted claim.** An external review caught both.

- **`fitness_distribution`**: the lead reported `cauchy_loc` matching only 10/15 and
  downgraded the claim. That run used **raw** fitness. Li et al. max-normalise, and
  `cauchy_loc` is not scale-invariant; under max-normalisation it matches **14/15**, the sole
  miss being `Tu2022_T7`, which is independently flagged as being on an inconsistent scale.
  The agent was right. The normalisation must be recorded as case preprocessing on promotion.
- **Szendro Weinreich rows**: the lead got order-1 Fourier 0.918275 against the agent's
  0.893836 and downgraded it. That run used the **whole five-locus landscape**; the recorded
  procedure averages the **ten four-locus faces**, which reproduces 0.893835535 / 0.063756852
  / 0.106164465 against the printed 0.894 / 0.064 / 0.106. The lead also wrongly accused the
  record of a column mis-mapping: the third column is `Fsum`, the total epistatic fraction
  1 - F1, which the record already identified correctly. The agent was right.

### Data issues recorded

- The documented tenfold error in `Weinreich2006Tan2011_Weinreich2006.csv` does **not**
  explain the standing TEM mismatch: correcting 41,000 to 4,100 moves gamma 0.8254 to 0.8351
  against a target of 0.85, leaves gamma_star at 0.6136 against 0.59, and r/s 0.4455 to
  0.4440 against 0.43. Hypothesis tested and rejected.
- `Tu2022_T7.csv` is on a fitness scale inconsistent with its sibling `Tu2022_TEV.csv`.
- `Kuo2020.csv` is rounded to 3 decimals: 2,473 distinct values over 197,890 rows.

### Best unworked target

Weinreich, Lan, Jaffe & Heckendorn 2018, J Stat Phys 172:208 (10.1007/s10955-018-1975-3).
Table 3 prints Fourier-Walsh order-wise variance reductions for three empirical landscapes
whose inputs are **all local** (`Palmer_DHFR_ic75`, `Bridgham2009`, `Mira2015_TEM_AM`), and
Table 1 prints (L, number of maxima) for 17 landscapes, around 15 of them local. Settle the
increment-versus-cumulative R-squared question first — it is exactly what tripped the
Szendro mapping above.

### Ruled out, recorded so nobody repeats the search

Schenk 2013 MBE, Fragata 2018 Heredity and Schulz 2025 PNAS are plot-only. de Visser & Krug
2014 and Bank 2022 contain no tables. arXiv 1303.3842 was never peer reviewed and records
mutations rather than fitnesses. The MAGELLAN preprint prints no gamma. Blanquart &
Bataillon 2016 reports different statistics. Hinz 2024, Ghenu 2023 and Pressman 2019 are
gamma-plot-only or Levenshtein-based. Song & Zhang is ruled out as a *gamma* anchor — its
code uses a different denominator population from Ferretti eq. (3) — while remaining valid
for r/s. The MAGELLAN landscape library is unreachable (timeout): a blocker, not a dead end.

`fdc`'s origin is confirmed as Jones & Forrest 1995, ICGA pp. 184-192 (**no DOI exists**;
Santa Fe Institute WP 1995-02-022). Note GraphFLA defaults to Spearman where the origin uses
Pearson. The four remaining `fitness_distribution` descriptors — cv, quartile coefficient,
median/mean ratio and relative range — have **no published origin**: a settled negative.


### Follow-up closed after the third wave (lead-verified)

Li et al. 2025 also prints a non-magnitude-epistasis fraction and a local-optima fraction per
landscape. These were completed on the two largest landscapes and remain
`definition_incompatible` on all eight tested:

| landscape | n_configs | n_lo | lo fraction | printed | non-magnitude epistasis vs printed |
| --- | --- | --- | --- | --- | --- |
| GB1 | 149,361 | 184 | 0.00123 | 0.005 | 0.5651 vs 0.40 |
| TEV | 159,132 | 1,109 | 0.00697 | 0.060 | 0.6652 vs 0.56 |

All four structural figures were re-run by the lead and match exactly. The divergence is a
property of the definitions, not of landscape size or sparse activity: Li et al. restrict
starting variants to active ones and fold additive squares into magnitude. Reopen only with
the active cutoffs from the authors' code. The Cauchy-location and kurtosis columns are
unaffected and remain the anchor.
