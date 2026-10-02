# Idiosyncratic epistasis: definition, correction and four-paper review

Reviewed 2026-10-02 on `codex/package-fixes`, after integration `e46936e`.
Only the idiosyncratic-index calculations are changed. Construction, DRI/ICI,
other metrics, the public signatures and scalar return types are unchanged.

## Which results can validate this metric?

| Supplied paper | What it actually measures | Role in this correction |
| --- | --- | --- |
| [Lyons et al. 2020](https://doi.org/10.1038/s41559-020-01286-y), *Idiosyncratic epistasis creates universals in mutational effects and evolutionary trajectories* | SD of one mutation's effects across backgrounds, divided by SD of the same number of random genotype-pair differences; mean across mutations. Main Results and Methods, PDF pp. 2 and 8. | Direct definition and numerical anchor: tRNA mean 0.612, SEM 0.005, 828 directed mutations. |
| [Bakerlee et al. 2022](https://doi.org/10.1126/science.abm4774), *Idiosyncratic epistasis leads to global fitness-correlated trends* | Background-averaged additive/interaction coefficients, LASSO fits, partitioned variance and fitness-correlated regressions. Fig. 2 and Figs. 3–4, PDF pp. 3–5. | Conceptual comparator and potential future regression/interaction benchmark; no reported Lyons SD-ratio target was found in the supplied paper or reviewed code. |
| [Johnson and Desai 2022](https://doi.org/10.7554/eLife.76491), *Mutational robustness changes during long-term adaptation in laboratory budding yeast populations* | Fitness model (XM), population/timepoint indicator model (IM), and combined model (FM), fit by OLS; forward selection requires a BIC improvement greater than 2. Fig. 3 and Methods, PDF pp. 7 and 19. | Model comparison and variance explained, not a random-pair SD ratio. Its coefficients, R² and BIC cannot serve as numerical I_id targets. |
| [Collesano et al. 2024](https://doi.org/10.1371/journal.pcbi.1012380), *Energy landscapes of peptide-MHC binding* | Linear and pairwise energy models versus a nonlinear global map; regression/MSE comparisons. For direct background effects, resample backgrounds at fixed ancestral energy and center their effects; summarize RMS (S4 Fig). PDF pp. 5 and 9. | A conditional, dimensionful effect statistic and model comparison; no matching dimensionless Lyons index is reported. The paper explicitly notes undersampling of matching backgrounds in measured data. |

The four papers do not provide four interchangeable index targets. Recreating
their entire model-fitting analyses would test different computations and is
outside this metric correction. Agreement with one published number alone also
does not establish correctness for all possible inputs: empirical anchors here
are combined with independent enumeration and explicit boundary tests.

This distinction matters scientifically. For example, `f(x) = (sum(x))**2` is
a nonlinear transformation of an additive trait, yet its mutation effects vary
over backgrounds, so its Lyons I_id is positive. I_id cannot by itself classify
global versus specific/residual epistasis in the other papers' sense.

## Reviewed author artifacts

- Lyons: [repository at 0a6c2ce](https://github.com/lyonsdm/idiosyncrasy/tree/0a6c2ce3a277d56679dae52b222106d05af93983).
  `01_trna/31_fitness/get_fitness.py` supplies the fitness scale;
  `01_trna/32_idiosyncrasy/distributions.ipynb`, code cells 2 and 3, supply the
  illustrative and global analyses. Cell 3 resets the RNG to `n**2+3` for each
  directed mutation. The n-order model notebook instead uses one stream seeded
  4032 and separate draws per directed mutation. The particular seed policy is
  therefore an artifact detail, not part of the index definition.
- Bakerlee: [Zenodo publication archive 6352707](https://zenodo.org/records/6352707),
  repository snapshot `710a5ff`; `2_Coefficient_modeling/lasso_2.py` and the
  `5_FCT_analysis` notebooks expose coefficient modeling and York regressions.
  The README notes paths/dependencies require adaptation. These are useful
  reproducibility resources for their own analyses, not a second I_id oracle.
- Johnson: [VTn_pipeline at 02d2b41](https://github.com/mjohnson11/VTn_pipeline/tree/02d2b41d54dd22487df1c75f9e381411c5ef0376),
  `scripts/VTn_modeling.py` explicitly selects by BIC and emits R², likelihood,
  BIC, parameters and coefficients. Supplementary File 1 supplies figure data.
- Collesano: [MHC-gem at eb74790](https://github.com/lcollesano/MHC-gem/tree/eb74790645f7c19e18e3c742466635b5dc8fb69d)
  supplies curated affinity data and prediction matrices/tools. The supplied
  paper's fixed-energy RMS procedure is different from unrestricted random-pair
  normalization; the repository inventory does not provide a Lyons index target.

## Correct estimator and previous differences

For mutation m observed in n_m matching backgrounds, let
`s_m(b) = f(b with B) - f(b with A)`. Independently draw n_m pairs `(U_i,V_i)`
with replacement from the specified genotype population. Then

```text
I_m = SD({s_m(b)}) / SD({f(V_i) - f(U_i)})
I_landscape = mean_m(I_m)
```

Both SDs use `ddof=0` as in the author Python code. Sampling with replacement
allows repeated genotypes and self-pairs. Mutations receive equal weight,
not weight proportional to background count. Reverse mutations count too.

The old implementation used `sqrt(2)*SD(f)` for every denominator. This is the
population SD of independent differences, not a finite matched-size sample SD.
Taking an expectation does not commute with taking a reciprocal, so it is not
an exact replacement even when many mutation ratios are averaged. On all viable
tRNA genotypes it gives 0.5942481263 instead of the author's 0.6121000731.

The old default build also removed 3,903 isolated viable genotypes, leaving
24,627 rather than 28,530 control-pool members. That shifted its analytic result
again to 0.5873601678 (historical reproduction). Isolates supply no matched
mutation effects but still affect the random-pair distribution.

## Numerical reproduction

Run `python -m validation.idiosyncrasy`. It verifies the input hash and uses an
independent sequence-substitution oracle. The production matcher uses encoded
background groups and a byte-key fallback, so it does not reuse that oracle.

| Check | Result | Evidence interpretation |
| --- | ---: | --- |
| Paper global I_id | 0.612 ± 0.005 SEM | Printed target |
| Independent reconstruction with author seeds | 0.6121000730811921 ± 0.0045646619468353 | Reproduces both printed values at their precision |
| Production matcher/kernel with those same seeds | 0.6121000730811921 | All 828 ratios agree; maximum absolute difference 2.6e-15 |
| Public global function, `seed=0`, complete population | 0.6141679589301013 | Independent same-stream oracle: 0.6141679589301012 |
| Public function, `n_jobs=1` versus `n_jobs=2` | Exactly equal | Parallel scheduling does not alter the RNG stream |
| G→A at offset 10 | 88 backgrounds, observed SD 0.1265239532 | Matches the printed count and rounded observed SD 0.13 |

The public API has one seed, whereas the tRNA notebook reinitializes its seed
for each background count. It is incorrect to demand identical output from
these different random streams. We therefore check exact author replay through
the shared production numeric kernel, and the public function against an
independent calculation with its own fixed stream. The published SEM is across
the 828 directed ratios; it is not a Monte Carlo error estimate or an independent
biological confidence interval, since reverse mutations are correlated.

### Figure 1a discrepancy retained

The text reports a single-mutation ratio 0.49 with control SD 0.26. The released
Figure 1a cell's seed 4033 instead yields SD 0.2490301165 and ratio 0.5080668756;
its control range is also different from the printed range. Repeating the
calculation with the raw fitness ratios, rather than differences of their logs,
gives the same disagreement, ruling out the log-transform algebra as its cause.

The separate global cell's documented `n**2+3` seed gives 0.4905242284 for this
mutation and rounds to the text's 0.49. This is a useful partial agreement but
does not reproduce Figure 1a's stated control range. No seed search, target-based
preprocessing choice or enlarged tolerance was used. The figure-level source
discrepancy remains unresolved; it does not prevent the global mean/SEM replay.

## Behavior and input contract

- Signatures and return types are unchanged. `global_idiosyncratic_index(seed=...)`
  now uses the seed. `seed=None` and the single-mutation function draw fresh local
  random streams without changing NumPy's global RNG state.
- Each directed mutation receives one matched-size control. The global function
  draws in column/source-allele/target-allele order, after parallel matching.
- `min_pairs=3` remains the GraphFLA default; valid values are integers >= 2.
  The paper specifies no minimum for this index. All 828 tRNA mutations have at
  least 3 backgrounds, so this guard does not affect the reproduction.
- A flat landscape or no eligible mutations returns NaN. A zero-SD sampled
  control warns and returns NaN; the global result remains NaN if any eligible
  control fails. Such mutations are not silently discarded or resampled. Low
  sample sizes or discrete fitness pools can make this event appreciable.
- No clipping to [0,1], implicit log transformation, or error-noise correction is
  applied. Duplicate genotypes, missing configuration values, nonfinite fitness
  and invalid mutations are rejected rather than arbitrarily deduplicated.
- The input population remains `landscape.get_data()` (retained nodes). The
  regression reproduces the full study using the existing public GraphML import
  with genuine single-step improving edges and all viable vertices. This adapter
  is for verification; it does not make normal `build_from_data` retain isolates.

## API discussion after this correction

1. A direct `X, f` or explicit reference-population input would avoid building a
   graph for a statistic that needs no graph edges, and preserve isolated inputs.
2. Single-mutation `seed`/RNG support is needed for convenient reproducibility.
   A documented seed policy or supplied controls would allow exact artifact
   replay without calling internal helpers.
3. A per-mutation table could expose labels, background counts, observed/control
   SD, ratio and validity alongside the aggregate. Repeated controls and an
   explicit analytic alternative need distinct names/semantics: either changes
   the finite-sample estimator and must not silently replace it.

These interface changes are proposals only. The current patch deliberately
preserves both public function signatures for the requested follow-up discussion.

## Verification coverage

`tests/test_idiosyncrasy.py` checks independent directed enumeration, equal
mutation weighting, matched control size and replacement, reverse labels,
missing backgrounds, seed effectiveness, parallel equality, global RNG
isolation, long-sequence fallback, invalid input and undefined controls.
`validation/tests/test_idiosyncrasy.py` checks the frozen paper case, per-mutation
author replay, the complete population and the retained Figure 1a discrepancy.
Only this metric's synthetic golden values were replaced, using an independent
literal reference with seed 0. Other golden metrics were left unchanged.

The follow-up [test audit](IDIOSYNCRASY_TEST_AUDIT.md) records the dedicated literature cases, strengthened basic tests, resource limits and bounded performance checks. Scientific assertions now live in `validation/tests/test_idiosyncrasy.py`; the replay driver returns detailed observations.
