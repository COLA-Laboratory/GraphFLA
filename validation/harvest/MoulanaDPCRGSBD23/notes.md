# MoulanaDPCRGSBD23 — Moulana et al. 2023, antibody affinity landscapes

Agent: claude-opus-5 · recorded 2026-09-30T19:28:12Z · dossier `papers/Moulana2023`

## 1. Bibliography

Verified against Crossref (`https://api.crossref.org/works/10.7554/eLife.83442`, HTTP 200).
Title, journal (eLife), volume 12, article e83442, year 2023 and the full nine-author list all
match the supplied bibliography exactly. **No errors found.** Crossref lower-cases the DOI to
`10.7554/elife.83442`. PMC id `PMC9949795`, PMID 36803543.

## 2. Acquisition

The prior session already had the Europe PMC `fullTextXML` (sha256 `e9b2fb8e…b0cde`) — reused,
not re-downloaded. The author-site PDF that 404'd was not retried; it is redundant given the XML.
New this session:

| what | route | result |
| --- | --- | --- |
| eLife supplementary bundle | Europe PMC `supplementaryFiles` for `PMC9949795` | 200, 968 431 B |
| Supplementary file 1 | member `elife-83442-supp1.docx` | isogenic vs Tite-seq −log K_D,app table |
| authors' K_D tables ×4 | `github.com/desai-lab/omicron_ab_landscape` raw `data/cleaned_Kds_RBD_<ab>_proper.csv` | 200 each, 34–39 MB |
| CV R² per epistatic order | same repo, `data/CV_rsquared.csv` | 200, 844 B |

Full log: `papers/Moulana2023/logs/acquisition_claude.jsonl`. The 34–39 MB tables were not copied
whole into the dossier; gzipped column subsets are stored there and the raw URLs + sha256 are
logged.

## 3. What the `fitness` column is — established exactly

Name map: `CB6` = LY-CoV016 (etesevimab), `CoV555` = LY-CoV555 (bamlanivimab),
`REGN10987` = imdevimab, `S309` = sotrovimab precursor.

`fitness` = **−log₁₀ K_D,app for binding to that monoclonal antibody** (larger = tighter antibody
binding = *less* escape). It is **bit-for-bit identical** to the authors' published `log10Kd`
column, with NaN wherever the authors have NaN:

| file | rows | non-NaN | exact matches vs author `log10Kd` | `n_configs` in GraphFLA | `n_lo` |
| --- | --- | --- | --- | --- | --- |
| `Moulana2023_CB6.csv` | 32 768 | 16 511 | 16 511 / 16 511 | 16 511 | 100 |
| `Moulana2023_CoV555.csv` | 32 768 | 19 867 | 19 867 / 19 867 | 19 865 (2 isolated dropped) | 391 |
| `Moulana2023_REGN10987.csv` | 32 768 | 23 686 | 23 686 / 23 686 | 23 686 | 174 |
| `Moulana2023_S309.csv` | 32 768 | 32 768 | 32 768 / 32 768 | 32 768 | 210 |

(`log10Kd_pinned` is identical to `log10Kd` for every row of all four antibodies, so it offers no
censored-value alternative.)

## 4. Non-binder handling — the choice that dominates everything

This is the single most consequential fact about these four inputs.

Paper, Methods → "Sequence data processing":

> "We then averaged the inferred KD,s values across the two replicates for each antibody after
> removing values with poor fit (r2<0.8 or SE>1). Variants were defined as non-binders if the
> difference between the maximum and the minimum of their estimated log-fluorescence over all
> concentrations was lower than 1 (in log-fluorescence units). This value was set by measuring the
> distribution for known non-binders (see Figure 1—figure supplement 1)."

Paper, Results:

> "These KD,app range from 0.1 nM to 1 μM (which is our limit of detection and likely corresponds
> to non-specific binding)"

Paper, Methods → "Epistasis analysis", on what the authors do with them:

> "Some phenotypic variables log⁡KD,app are unavailable in our dataset due to the upper limit of
> the assay concentration: we are unable to precisely infer KD,app for the low-affinity (or
> non-binding) variants, particularly when the true −log⁡KD,app<6 (the highest concentration used).
> To address this issue, we augmented our linear model with a lower boundary, following a Tobit
> left-censored model (Tobin, 1958)."

**The GraphFLA inputs do the opposite.** Complete escapers are `NaN` and are therefore *dropped*,
not censored and not imputed at the detection limit. Consequences:

* the landscapes cover **50.4 % / 60.6 % / 72.3 % / 100.0 %** of the 32 768 genotypes for
  CB6 / CoV555 / REGN10987 / S309;
* three of the four are **punctured hypercubes**, not combinatorially complete;
* **Omicron BA.1 itself is absent** from CB6, CoV555 and REGN10987 (it is a non-binder);
* any peak count or accessibility statistic computed on them is a statistic about the *binder
  sub-landscape*. The `n_lo` values in the table above (100 / 391 / 174 / 210) have **no paper
  counterpart** and must not be promoted as literature-anchored benchmarks.

If a future session wants complete 2¹⁵ landscapes here, the honest construction is to impute
non-binders at the detection limit (−log₁₀ K_D,app = 6) — which is a *modelling decision this
paper does not make*, so it would need its own justification rather than being presented as the
paper's method.

## 5. Metric families searched, and what is actually printed

Exhaustive case-insensitive sweep of the full text + Methods:

| term | hits |
| --- | --- |
| peak | 0 |
| local maxim… | 0 |
| local optim… | 0 |
| rugged… | 0 |
| sign epistasis | 0 |
| reciprocal sign | 0 |
| accessible path | 0 |
| mutational path | 0 |
| fraction of path | 0 |
| optima | 2 — both "optimum epistasis model" |

The analytical vocabulary of this paper is *enrichment of escape mutations*, *rpart decision
trees* and a *Tobit-censored truncated epistasis model*. None of these is a GraphFLA metric.
Hence the verdict `no_overlap`.

## 6. Printed numbers examined, one by one

### 6a. Measured-variant counts → `n_configs` (mismatch)

> "Of the 32,768 variants in our library, we obtain KD, app for at least ~30,000 variants to each
> of the mAbs (32,603 for LY-CoV016, 31479 for REGN10987, 27485 for LY-CoV555, and 32650 for S309)
> after removing variants with poor titration curves (r2<0.8 or σ>1; see Methods)."
> — Results, para. 2

| antibody | printed | `ls.n_configs` |
| --- | --- | --- |
| LY-CoV016 (CB6) | 32 603 | 16 511 |
| REGN10987 | 31 479 | 23 686 |
| LY-CoV555 (CoV555) | 27 485 | 19 865 |
| S309 | 32 650 | 32 768 |

The gap is structural: the printed count *includes* complete escapers (whose K_D,app is above the
limit of detection), which cannot enter a landscape keyed on −log₁₀ K_D,app.

Beyond that, the printed counts could **not** be reconstructed from the deposited tables at all.
Candidate rules tried, all reported (CB6 / CoV555 / REGN10987 / S309):

| rule | value | printed |
| --- | --- | --- |
| `is_clean OR is_non_binder` | 32 112 / 29 892 / 30 479 / 32 602 | 32 603 / 27 485 / 31 479 / 32 650 |
| ≥1 replicate with r² ≥ 0.8 and σ ≤ 1, OR flagged non-binder | 32 717 / 32 038 / 32 333 / 32 768 | — |
| ≥1 replicate with r² ≥ 0.8, OR flagged non-binder | 32 721 / 32 341 / 32 350 / 32 768 | — |
| `is_clean` alone | 15 760 / 13 088 / 19 888 / 32 602 | — |

None matches. The deposited `*_proper.csv` tables appear to come from a different processing pass
than the printed counts; the paper states no rule more precise than the sentence quoted.

### 6b. Escape fractions (2 reproduced to precision, 1 mismatch, 1 exact)

> "with 51% of the variants fully escaping LY-CoV016 (defined as having KD,app above the limit of
> detection), 65% fully escaping LY-CoV555, 36% fully escaping REGN10897, and no variants fully
> escaping S309 (Figure 1A)"
> — Results, para. 2 (note the paper's typo "REGN10897" for REGN10987)

Recomputed as `is_non_binder / (is_clean OR is_non_binder)` from the authors' own flags:

| antibody | printed | recomputed | NaN fraction of the GraphFLA input |
| --- | --- | --- | --- |
| LY-CoV016 | 51 % | 16 470 / 32 112 = **51.29 %** | 16 257 / 32 768 = 49.61 % |
| LY-CoV555 | 65 % | 17 898 / 29 892 = **59.88 %** | 12 901 / 32 768 = 39.37 % |
| REGN10987 | 36 % | 10 837 / 30 479 = **35.56 %** | 9 082 / 32 768 = 27.72 % |
| S309 | 0 | 0 / 32 602 = **0 %** | 0 |

LY-CoV016, REGN10987 and S309 agree with the printed values; **LY-CoV555 does not** (59.9 % vs
65 %) — and LY-CoV555 is also the antibody whose printed measured-variant count is furthest from
anything reconstructible, so the two discrepancies are plausibly the same processing-version
issue. **Warning for the packaging session:** the NaN fraction of the CSV is *not* the escape
fraction (right-hand column above). NaN also covers QC failures, and 261 / 5 426 / 2 173 / 0
flagged non-binders still carry a fitted `log10Kd`.

### 6c. BA.1 / Wuhan Hu-1 reference affinities (Supplementary file 1)

Printed (`TiteSeq −log KD,app` column, `elife-83442-supp1.docx`):

| strain | antibody | printed | GraphFLA input |
| --- | --- | --- | --- |
| Omicron BA.1 | LY-CoV016 | NB | NaN ✓ |
| Omicron BA.1 | LY-CoV555 | NB | NaN ✓ |
| Omicron BA.1 | REGN10987 | NB | NaN ✓ |
| Omicron BA.1 | S309 | 8.62 ± 0.18 | 8.5866 |
| Wuhan Hu-1 | LY-CoV016 | 10.27 ± 0.07 | 9.8054 |
| Wuhan Hu-1 | LY-CoV555 | 10.35 ± 0.02 | 10.1940 |
| Wuhan Hu-1 | REGN10987 | 10.52 ± 0.18 | 9.9593 |
| Wuhan Hu-1 | S309 | 9.00 ± 0.08 | 9.3432 |

The categorical half (NB / NB / NB / binding) reproduces exactly. The numeric half does not: BA.1
vs S309 is within the printed standard error, but all four Wuhan Hu-1 values are off by 0.16–0.56
log units. The paper does not say whether this table's Tite-seq column comes from the library
genotype `000000000000000` or from the separately spiked-in clonal Wuhan Hu-1 strain sorted in the
same experiment — **NOT STATED IN PAPER**. Do not use Supplementary file 1 as a per-genotype
oracle.

### 6d. S309 "~17 % below BA.1" (mismatch)

> "The picture is more complex for S309, where BA.1 has reduced affinity relative to Wuhan Hu-1,
> but ~17% of variants have lower affinity than BA.1."
> — Results, para. 3

S309 is the one complete input (0 NaN), so this is a clean check. **Computed 6 333 / 32 768 =
19.33 %.** All variants tried, none tuned into agreement:

| variant | value |
| --- | --- |
| threshold = library BA.1 value 8.58662, over all 32 768 | **19.33 %** ← the paper's stated method |
| same, `is_clean` subset only (32 602) | 19.38 % |
| `<=` instead of `<` | 19.33 % |
| threshold = printed Tite-seq BA.1 value 8.62 | 21.68 % |
| threshold = printed isogenic BA.1 value 8.81 | 39.50 % |
| threshold that *would* give exactly 17 % | 8.5556 |

Recorded as `mismatch`. The method the paper's own wording justifies is the first row.

### 6e. Optimal epistatic order → `higher_order_epistasis` (definition-incompatible)

> "This indicates that epistasis does play a significant role in all cases (up to second order for
> REGN10987, to third order for LY-CoV555, and to fourth or higher order for LY-CoV016, S309, and
> ACE2)."
> — Results, "Inference of epistatic affinity landscapes"

vs GraphFLA:

> "Calculates the fraction of variance in fitness that can be explained by interactions between
> variables up to the specified order using polynomial regression."
> — `graphfla/analysis/epistasis/higher_order.py`

Three independent divergences, any one of which is disqualifying:

1. the paper reports an **argmax over held-out 10-fold CV R²**; GraphFLA returns an **in-sample
   OLS R²** at a fixed order, which is monotone in order and cannot select an interior optimum;
2. the paper fits an **L2-regularised Tobit left-censored likelihood**; GraphFLA fits
   unregularised OLS;
3. the Tobit model's whole purpose is to use the non-binders — exactly the rows the GraphFLA input
   drops.

No run was made. The per-order R² curve is plotted (Figure 3A), so it is not a target under
rule 4; its numeric backing was located at `desai-lab/omicron_ab_landscape` →
`data/CV_rsquared.csv` (saved in the dossier) and does confirm the printed ordering:

```
LY-CoV016  1 .9798  2 .9886  3 .9911  4 .9919
LY-CoV555  1 .9286  2 .9379  3 .9408  4 .9406   ← peak at 3
REGN10987  1 .7425  2 .8635  3 .8485  4 .8368   ← peak at 2
S309       1 .7349  2 .8649  3 .9237  4 .9347  5 .9311  6 .9168   ← peak at 4
ACE2       1 .8664  2 .9581  3 .9831  4 .9903  5 .9916  6 .9903   ← peak at 5
```

## 7. Honest caveats

* The verdict is `no_overlap`, argued from an exhaustive keyword sweep rather than from a quick
  skim. The value delivered by this study is input provenance, not a reproduction target.
* Nothing was tuned. The LY-CoV555 escape fraction and the S309 17 % claim are recorded as
  mismatches with every variant tried listed.
* The two metrics flagged as under review upstream (`evolvability_enhancing_mutations`,
  `idiosyncratic_index` / `global_idiosyncratic_index`) have no overlap with this paper.
* Landscape statistics computed on these four files are statistics about a **binder
  sub-landscape on a punctured hypercube**. This should be stated wherever they are used.
* The run script is stored at `papers/Moulana2023/reproduction_claude.py` (it builds all five
  landscapes — the four antibodies here plus the 2022 ACE2 one — and prints every number quoted
  above). Run it as
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/arwen/Documents/GitHub/GraphFLA python3 …`.
