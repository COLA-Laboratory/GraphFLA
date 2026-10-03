# Changelog

All notable changes to GraphFLA are recorded here. Entries that change a
returned statistic are marked, because downstream analyses depend on them.

## Unreleased

### Analysis API consistency

- All public analysis functions now have explicit return annotations and
  consistent NumPy-style parameter and result documentation. Existing metric
  names, positional arguments, defaults and literature definitions remain.
  Examples use small, self-contained landscapes and explicit random seeds.
- `idiosyncratic_index` accepts keyword-only `seed=None`, matching the global
  index's local RandomState convention. Existing calls retain their behavior.
- `single_mutation_effects` and `all_mutation_effects` append a `position`
  column identifying each row's configuration variable. The original six
  columns retain their order and values. Both functions preserve their column
  schema and numeric dtypes when the result is empty.
- `mean_distance_to_global_optimum` now honors an explicit `distance_func`
  even when default distances are cached. Calls without a custom function
  continue to reuse the cache; custom calculations do not overwrite it.
- Corrected descriptions of fitness-distribution units, gradient normalization,
  signed fitness-flattening trends, plateau counts and path sampling. These
  documentation corrections do not change metric calculations.

### Changed — pooled fitness trends

- The completed DRI/ICI fixes are integrated: `diminishing_returns_index` and
  `increasing_costs_index` follow the pooled-edge definition in Huang et al.
  (2025), Appendix C.3.2, instead of correlating node-level mean effects.
  Returned values can therefore change. Minimization uses negated fitness;
  numerical scaling and undefined-result handling are also corrected.
  Method names, defaults and scalar returns are retained. See
  `validation/FITNESS_TRENDS_REVIEW.md` for the evidence and scope.

### Changed — integrated epistasis order analysis

- `walsh_hadamard` now returns a sklearn `Bunch` with `.coefficients`,
  `.order_summary` and `.fit_info`. Existing DataFrame consumers should use
  `.coefficients`. One design is shared by all nested orders; the highest-order
  solution is reused. Tall OLS problems share an augmented QR compression.
- The order summary reports cumulative training `r2`, additional `delta_r2`,
  RMSE, model dimensions/rank and per-fit alpha. It separately reports the
  highest-order model's exact uniform-product variance fractions, accounting
  for multistate covariance, and Lasso nonzero counts. These model fractions
  are not automatically observed-data R-squared contributions. Constant fitness
  gives NaN R-squared/increments; negative Lasso increments are preserved.
- `higher_order_epistasis` and its module have been removed, including their
  public exports, benchmark entry and `profile()` / `list_metrics()` entry.
  Use `walsh_hadamard(landscape, max_order=k).order_summary`; read an existing
  result's `.order_summary` directly. No compatibility alias remains. The
  score-only fitting branch has also been removed; nonidentifiable OLS fits
  raise, and regularization requires explicit `method="lasso"`.
  See `validation/EPISTASIS_ORDER_REVIEW.md`.

### Changed — Walsh-Hadamard coefficients

- `walsh_hadamard` preserves original one-based positions after invariant-site
  removal and uses real allele labels for all discrete variable types. Reserved
  label delimiters are escaped; categorical inputs with 47 or more states no
  longer collide with the internal encoding. Corrected labels may change joins.
- Default OLS now raises `ValueError` when coefficients cannot be uniquely
  fitted. Incomplete data remains supported when the chosen design has full
  column rank. `max_order=0` fits only a constant; the legacy `WT` row denotes
  the model's uniform product-space mean, not reference fitness.
- Explicit `method="lasso"` supports a positive `alpha` or `alpha="cv"`, with
  `cv`, `random_state`, `max_iter` and `tol` controls. The constant is unpenalized
  and features are not standardized. This differs from the cited author's
  penalized-constant analysis pipeline; no automatic estimator switching occurs.
  The coefficient table retains four columns, with fitting/reference metadata
  in DataFrame attributes. Save attributes separately when exporting to CSV.
- The dense design is built directly in blocks, with model-size checks before
  term enumeration. `max_cells` now includes all columns and defaults to `1e7`
  rather than `1e9`; solver workspace is additional. Dedicated literature tests
  reproduce Faure et al. (2024) Table 1 and independently check multistate
  transforms, pinned author matrices and a bounded Lasso procedure. See
  `validation/WALSH_HADAMARD_REVIEW.md` for evidence and interpretation limits.

### Validation and performance

- Literature tests now run separately with `python -m validation.tests`, with
  study/case filters, paper references, evidence roles, mandatory input hashes,
  JUnit provenance, a permanent contract/template and an independent CI job.
  Default pytest runs only the basic suite; historical case identities remain.
- Performance benchmarks are split into `construction` and individually selectable
  `analysis` modules. The bounded comparison runner can benchmark EE alone.
- EE neighborhood moments use bounded vectorized blocks with centered variances.
  Public API and statistical definitions are unchanged; basic oracles, complete
  author-data comparisons and output snapshots guard the optimization.

### Changed — EE analysis API

- `evolvability_enhancing_fraction(landscape, *, fdr=0.01, effect_type="all")`
  is the canonical scalar entry point. It uses neighborhood fitness variation
  for all data domains; experimental measurement-error options are not exposed.
  Selecting an effect type filters only the numerator, keeping all ordered
  neighbor pairs in the denominator and the two complete BH testing families.
- `evolvability_effects(landscape, *, fdr=0.01)` returns one row per directed
  mutation, including original position/allele labels, raw and BH adjusted
  p-values, and nullable EE decisions for insufficient neighborhoods.
- `evolvability_enhancing_mutations` remains available with a FutureWarning
  and its legacy `epsilon`/`auto_calculate` behavior. New entry points neither
  require nor populate the unrestricted neighbor-fitness cache.
- `profile()` and `list_metrics()` use the canonical scalar name and column.
  The old name in include/exclude or parameter keys warns and resolves to the
  new name; including both names computes EE once. Legacy profile overrides
  for `epsilon`/`auto_calculate` raise an explicit migration error instead of
  being silently ignored. Use the old function directly for those options.

### Changed — affects returned values

- `evolvability_enhancing_mutations` now excludes the focal site, tests
  `delta_mean > max(0, delta_fitness)` with two-sided t tests and BH FDR 0.01,
  and counts both mutation directions over represented ordered neighbor pairs.
  Additive landscapes now return zero. The t-test variance uses both
  endpoints, correcting a duplicated-target term in the cited author's code.
  The scalar combines beneficial, deleterious and neutral EE mutations; it is
  not the paper's beneficial-only fraction. Insufficient neighborhoods return
  NaN when no pair is testable. Existing results need recalculation. See
  `validation/EE_MUTATIONS_REVIEW.md` for exact author replay, RNA measurement
  error limitations, and the finalized general API. Construction is unchanged.
- `single_mutation_effects` and `all_mutation_effects` now report `mean_effect`
  as `mutation_to` minus `mutation_from`, matching the row's own label and the
  convention already used by `fitness_effect_distribution`. Up to and including
  0.3.0 the sign was reversed, and `test_type="positive"` correspondingly tested
  for a fitness *decrease*. Results computed with an earlier release should be
  re-derived, or the sign of `mean_effect` flipped and `test_type` swapped.
  `median_abs_effect` is unaffected.
- `r_s_ratio`'s degeneracy test is now invariant under an affine rescaling of
  fitness, `f -> a*f + b`, as its definition implies: roughness and slope both
  scale by `|a|` and the intercept absorbs `b`. It previously used an absolute
  tolerance and reported `inf` for fitness in very small units. Note that the
  residual root-mean-square is still computed by squaring, so the *returned
  value* remains subject to float64 overflow beyond roughly 1e150 and underflow
  below roughly 1e-150. That arithmetic limit is unchanged by this release.
- `r_s_ratio` returns NaN rather than `inf` for constant fitness, where
  roughness and slope are both zero and the ratio is undefined. A purely
  epistatic landscape, with positive roughness and zero slope, still gives
  `inf`. Constant fitness is detected by exact equality, so values whose
  standard deviation is a non-zero round-off (such as a repeated 0.1) are
  recognised. The degeneracy test compares the slope against the fitness
  *range*, so it stays correct at scales where squaring the deviations would
  overflow or underflow.

### Fixed

- `fitness_effect_distribution` no longer raises on a single-position landscape,
  where every genotype shares one trivial genetic background.
- Landscape construction no longer rejects a fully neutral landscape as
  edgeless. Such a landscape has neighbours but no directed (improving) edges;
  construction now fails only when neither directed edges nor neutral pairs are
  found.
- `diminishing_returns_index` and `increasing_costs_index` return NaN with a
  warning on an edgeless landscape instead of raising from the sparse-adjacency
  fallback.
- Largest-component filtering (`tau` with `filter_mode`) now counts neutral
  pairs toward connectivity. Because a neutral pair carries no directed edge, a
  plateau previously looked like a set of singleton components, and filtering a
  neutral landscape could silently reduce it to a single genotype. The
  functional filter severs neutral pairs on the same rule it applies to directed
  edges, so a below-threshold plateau can no longer win component selection over
  the functional region.

### Documentation

- The `gamma` and `gamma_star` return-value descriptions contradicted their own
  Notes and Ferretti et al. (2016): an additive landscape gives `gamma = 1` and
  a House-of-Cards landscape gives `gamma` near 0, not the reverse.

## 0.3.0

- See the repository history for releases up to 0.3.0.
