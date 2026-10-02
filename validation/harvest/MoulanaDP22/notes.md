# MoulanaDP22 — Moulana et al. 2022, ACE2 affinity landscape

Agent: claude-opus-5 · recorded 2026-09-30T19:28:12Z · dossier `papers/Moulana2022`

## 1. Bibliography

Verified against Crossref (`https://api.crossref.org/works/10.1038/s41467-022-34506-z`, HTTP 200).

| field | Crossref | supplied bibliography |
| --- | --- | --- |
| title | Compensatory epistasis maintains ACE2 affinity in SARS-CoV-2 Omicron BA.1 | same |
| journal / vol / art | Nature Communications 13, 7011 | same |
| year | 2022 | same |
| authors | Alief Moulana; Thomas Dupic; Angela M. Phillips; Jeffrey Chang; Serafina Nieves; Anne A. Roffler; Allison J. Greaney; Tyler N. Starr; Jesse D. Bloom; Michael M. Desai | **truncated to the first three** |

PMC id `PMC9668218`, PMID 36384919.

## 2. Acquisition

The prior session failed on two routes (author-site PDF, brotli decode error; Europe PMC
supplementary bundle, read timeout). Both were replaced:

| what | route | result |
| --- | --- | --- |
| main text + Methods | Europe PMC `fullTextXML` for `PMC9668218` | 200, 201 960 B, sha256 `50809a8a…b205c` |
| Supplementary Information | Europe PMC `supplementaryFiles` (retry, 240 s budget) | 200, 9 426 658 B — the route that timed out before now succeeds |
| authors' K_D table | `github.com/desai-lab/compensatory_epistasis_omicron` raw | 200, 59 130 095 B |

Full log: `papers/Moulana2022/logs/acquisition_claude.jsonl`.
The Supplementary Information PDF (`MOESM1`) turned out to contain **only figure captions** — no
printed statistics. **`MOESM3_ESM.zip` was the payoff**: it holds Supplementary Data 1, whose
header prints two numbers that nothing in the main text does — `Params: 4944` and
`Performance: 0.9775420737902647` (see §7). That file is now in the dossier.

## 3. What the `fitness` column is — established exactly

`data/BioSequence/Moulana2022_ACE2.csv` (sha256 `71b466b5…6460e`), 32 768 rows, 203 NaN.

`fitness` = **−log₁₀ K_D,app for binding to human ACE2**, the same quantity and sign convention
the paper plots. Derivation, reproduced on **32 768 / 32 768 rows**:

```
fitness(g) = mean over replicates r ∈ {a, b, x} of  log10Kd_r(g)
             keeping only r with  r2_r >= 0.8  AND  sigma_r <= 1
             NaN when no replicate passes          (203 genotypes)
```

Source table: `cleaned_Kds_RBD_ACE2_withx.tsv` (3 replicates). Controls:

* the 2-replicate `cleaned_Kds_RBD_ACE2.tsv` matches **0** rows exactly;
* the `log10Kd` column of the 3-replicate file matches 29 531 / 32 565;
* dropping the `sigma <= 1` half of the filter matches 29 531 / 32 768 and leaves 0 NaN.

So both halves of the filter are load-bearing. Note that the paper's Methods state only the
r² half:

> "We then averaged the inferred KD,s values across the three replicates after removing values
> with poor fit ( r2<0.8)."
> — Methods, "Tite-Seq" (PMC9668218)

The σ ≤ 1 criterion is stated in the companion 2023 paper ("r2<0.8 or SE>1") and is evidently
what the authors' code applies here too, but for this paper it is **NOT STATED IN PAPER**.

**Censoring: none.** Unlike the 2023 antibody landscapes there are no non-binders:

> "We find that all 32,768 RBD intermediates between Wuhan Hu-1 and Omicron BA.1 have detectable
> affinity to ACE2, with KD,app ranging between 0.1 μM and 0.1 nM"
> — Results and discussion, para. 2

The only missing values are fit-quality failures (203 / 32 768 = 0.62 %).

## 4. Metric families searched, and what is actually printed

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

So `n_lo`, `local_optima_ratio`, `classify_epistasis`, `gamma`, `r_s_ratio` and the robustness
family have **no printed counterpart**. (For context only, GraphFLA gives `n_lo = 22`,
`n_configs = 32 565`, `n_edges = 243 098`, `global_optima_accessibility = 0.99392` on this input;
none of these is anchored to the paper and none should be promoted as a literature benchmark.)

## 5. The one real target — accessible paths Wuhan Hu-1 → BA.1

> "In fact, there are no paths from Wuhan Hu-1 to Omicron BA.1 that do not contain at least one
> step that decreases ACE2 affinity."
> — Results and discussion, para. 2 (printed, categorical)

GraphFLA side, for the definition comparison:

> "This metric represents the fraction of configurations in the landscape that can reach the
> specified local optimum (or optima) via any monotonic, fitness-improving path."
> — `graphfla/analysis/navigability.py`, `local_optima_accessibility`

**Divergence.** GraphFLA has no metric that counts accessible paths between two named genotypes.
Its navigability family only targets *local optima*, and BA.1 is not one here (out-degree > 0),
so `local_optima_accessibility(ls, i_BA1)` raises. Two things were therefore run:

1. **Direct-path DP** (the strict reading of the paper's wording: 15 forward steps, no
   back-mutation, no step decreasing affinity), over the full 2¹⁵ hypercube.
   * strict, unmeasured genotypes block a step → **0 paths**
   * optimistic, any step touching an unmeasured genotype is allowed → **0 paths**
     (this is an upper bound, so the 203 missing measurements cannot change the answer)
   * DP validated on controls: fitness = popcount → 15! = 1 307 674 368 000 paths;
     fitness = −popcount → 0.
2. **GraphFLA's own improving-edge digraph**: `i_wt in ls.graph.subcomponent(i_ba1, mode="in")`
   → `False`. BA.1 has 10 876 monotone ancestors and Wuhan Hu-1 is not among them, so BA.1 is
   unreachable from Wuhan Hu-1 even allowing indirect, back-mutating monotone walks.

**Outcome: `reproduced_exact`.** This is a cheap (sub-second on 32 k nodes), deterministic,
categorical target. Worth noting for the packaging session: exposing a public
"count accessible paths between two configurations" helper would convert this into a
first-class GraphFLA test rather than a test on a private graph primitive.

## 6. Secondary printed numbers (input-integrity checks, not GraphFLA metrics)

> "However, most (~60%) of the intermediate RBD sequences actually show a weaker binding affinity
> to ACE2 than the ancestral Wuhan Hu-1 RBD."

Computed **62.11 %** (20 225 / 32 565). Also 61.72 % if the 203 unmeasured genotypes stay in the
denominator. Consistent with "~60 %" but not equal to it at two significant figures — recorded as
`reproduced_with_precision` with the exact figure so the packaging session can set its own
tolerance.

> "the BA.1 RBD exhibits a slight (threefold both by Tite-seq and by isogenic measurements)
> improvement in binding affinity compared to Wuhan Hu-1"

f(Wuhan Hu-1) = 9.033694, f(BA.1) = 9.478874, Δ = 0.445181 → **2.787-fold**, rounds to the
printed "threefold".

## 7. The fifth-order epistasis model — model class matches, printed R² does not

Supplementary Data 1 (`Supplementary_Data_1-ACE2_5order_biochem_coefficients.txt`,
sha256 `586b41a7…6cc10`) prints, verbatim:

```
Params: 	4944
Performance: 	0.9775420737902647
Term	Coefficient	Standard Error	p-value	95% CI lower	95% CI upper
Intercept	8.950403005300428
```

and `Description of Additional Supplementary Files` (MOESM2) defines the header:

> "All coefficient values from the biochemical epistasis model (truncated at the fifth order).
> Performance corresponds to the R2 of the fit, and params denotes the number of parameters
> inferred."

The paper's model, Methods → "Epistasis analysis":

> "The full K-order model can be written: −log10 K_D,s = β0 + Σ_{i=1..K} Σ_{c∈C_i} β_c x_{c,s},
> where β_c denotes the coefficient for the combination of mutation c … and is equal to 1 if the
> sequence contains all the mutations in [c] and to 0 otherwise. This choice is called
> 'biochemical' or 'local' epistasis and is the one used in the main text."
> … "Finally, we trained a K=5 model over the complete dataset to get the final coefficients."

### The model classes are identical — verified, not assumed

The biochemical basis (subset indicators, |c| ≤ 5) *is* the degree-≤5 monomial basis on 15 binary
variables that GraphFLA's `PolynomialFeatures(degree=5)` spans. Both have
1+15+105+455+1365+3003 = **4944** parameters, matching the printed `Params: 4944` exactly. An
independent re-implementation (explicit 4944-column design matrix, `np.linalg.lstsq`, no sklearn)
reproduces GraphFLA's R² **to 10 decimal places** on every fitness version tried:

| fitness version | n | GraphFLA `higher_order_epistasis(order=5)` | independent biochemical OLS | fitted intercept |
| --- | --- | --- | --- | --- |
| repository input (3-rep, r²/σ filtered) | 32 565 | 0.9940605643 | 0.9940605643 | 8.960962 |
| authors' 3-replicate `log10Kd` (`_withx`) | 32 768 | 0.9935601335 | 0.9935601335 | 8.954319 |
| authors' 2-replicate `log10Kd` | 32 768 | 0.9902770034 | 0.9902770034 | 8.890372 |

**This is a useful result for GraphFLA independent of the paper**: `higher_order_epistasis` really
does implement the biochemical/local epistasis model of Poelwijk-style analyses, and the redundant
`x_i^k` columns that `PolynomialFeatures` emits for binary inputs do not corrupt the R².

### But the printed R² does not reproduce — `mismatch`

Printed **0.9775421** vs GraphFLA **0.9940606** (repo input). The three values above are exact
in-sample least-squares optima for that model class, so **no in-sample OLS fit of this model to
any deposited fitness version can produce 0.9775421** — it is below all of them. The printed
number must therefore be held-out, or regularised, or fitted to data not deposited. The paper's
own stated method ("trained a K=5 model over the complete dataset") implies the in-sample reading,
which is exactly what GraphFLA computes and exactly what mismatches. Recorded as `mismatch` with
`definition_match: "unknown"`, not tuned.

Supporting evidence that the model class is right and only the data/fit procedure differ: fitted
coefficients correlate with the 4944 published ones at Pearson **0.9606 / 0.9767 / 0.9810** across
the three versions (mean |Δ| 0.026 / 0.021 / 0.020), and the fitted intercepts 8.9610 / 8.9543 /
8.8904 bracket the published 8.9504 without matching it. A correlation near 1 across all 4944
terms also confirms the term-index convention (`1…15` ↔ `pos1…pos15`).

## 8. Epistasis order K = 5 — definition-incompatible

> "We chose the value of K that maximizes the prediction performance (R²) averaged over all ten
> testing datasets. For this dataset we found an optimal value of K = 5 (Supplementary Fig. 5)."
> — Methods, "Epistasis analysis"

vs GraphFLA:

> "Calculates the fraction of variance in fitness that can be explained by interactions between
> variables up to the specified order using polynomial regression."
> — `graphfla/analysis/epistasis/higher_order.py`

The paper's K = 5 is an **argmax over held-out 10-fold cross-validated R²**; GraphFLA's
`higher_order_epistasis(order=k)` is an **in-sample, unregularised OLS R²**, which is
non-decreasing in k and can never select an interior optimum. No run was made.

The per-order R² curve itself is plot-only (Supplementary Fig. 5), so it is not a reproduction
target under rule 4. Its numeric backing was nevertheless located in the authors' companion
repository, `desai-lab/omicron_ab_landscape` → `data/CV_rsquared.csv`, whose ACE2 rows are

```
order 1 0.866356 · 2 0.958085 · 3 0.983056 · 4 0.990313 · 5 0.991563 · 6 0.990331
```

peaking at order 5 and so confirming the printed K = 5. Saved at
`papers/Moulana2023/sources/CV_rsquared_claude.csv` (it covers both papers).

## 9. Honest caveats

* `n_configs` is **32 565**, not the 32 768 the paper's "all 32,768 … have detectable affinity"
  might suggest. The paper prints no post-QC genotype count: **NOT STATED IN PAPER**.
* Nothing here was tuned to make a number match. The two `reproduced_with_precision` outcomes
  record the exact computed values (62.11 %, 2.787×) next to the paper's approximations, and the
  fifth-order R² is recorded as a `mismatch` with all three data versions reported rather than
  being presented as "close enough".
* Also stored: `papers/Moulana2022/reproduction_claude.py` (the landscape/path/DP run). The
  epistasis-model fits used two further throwaway scripts whose logic is fully described in §7;
  they are cheap to re-derive (a 4944-column design matrix and `np.linalg.lstsq`).
* The two metrics flagged as under review upstream (`evolvability_enhancing_mutations`,
  `idiosyncratic_index` / `global_idiosyncratic_index`) have **no overlap** with this paper, so
  no `definition_match: "unknown"` entry was needed.
