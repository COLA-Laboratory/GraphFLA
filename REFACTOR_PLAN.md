# GraphFLA → scikit-learn-grade refactor — working plan

Branch: `refactor/sklearn-quality`. Living document. **Phase: DISCUSS & RECORD** (not delivering yet —
deliver only after the user says they are leaving).

## Progress log (delivery)
Baseline: 1267 tests pass. Each phase below is committed green on `refactor/sklearn-quality`.
- ✅ **Phase 0** — exceptions.py taxonomy; fixed broken `graphfla.Landscape` export; dead `__getattr__`/`setup_module` removed. (c4df238)
- ✅ **Phase 1.1** — semantic `kind` split from registry `_strategy_key`; WHT/distance discriminators fixed (DNA WHT keys now `A_1_C`, values identical). (0d92710)
- ✅ **Phase 1.2** — upfront build validation + `NotBuiltError` + `verbose` coercion. (e782e32)
- ✅ **Phase 1.3** — freeze-after-build: `determine_*` → private `_compute_*`. (6ffcce8)
- ✅ **Phase 1.4** — estimator protocol: get_params/set_params, params-forward repr, structured describe(). (96a5ff4)
- ✅ **Phase 4** — unified Walk API (Walk/WalkResult/HillClimb/RandomWalk); perf-preserved via `HillClimb.descend`. (7400dfa)
- ✅ **Phase 7** — full public-analysis naming sweep (rename table applied; `ffi`/`calculate_evol_enhance` deleted). (4176879)
- ✅ **Phase 5a** — Walsh-Hadamard → tidy DataFrame (the flagship "mess"). (8316a66)
- ✅ **Phase 5b** — classify_epistasis → `EpistasisClassification` dataclass (kills prose-string keys). (0a5be01)
- ✅ **Phase 9a** — dead-code removal (euclidean_distance, EdgeResult.strategy, include_configs, ~100 lines commented
  networkx, unused filter_data re-export) + `_pythonize` deduped 6→1 (analysis/_utils.py). (f15afff)
- Perf verified: walk/basin neutral (descend allocation-free); build 60ms / basins 30ms on 16k nodes. All green at 1265 tests.

**Gripe coverage:** #1 returns (WHT→DataFrame, classify→dataclass, kind fix) ✅ | #2 naming ✅ | #3 walk API ✅ |
#4 file size ⏭ deferred (structural) | #5 errors/warnings ⚠ partial (validation + exception taxonomy + NotBuiltError done;
logging migration + narrow-excepts pending) | #6 dedup/dead-code ✅. Plus: estimator protocol, freeze-after-build,
future-annotations (partial rollout).

### Remaining (not yet delivered — for next session)
- **Phase 5c (more return contract)**: `extradimensional_bypass` prose keys → dataclass (same pattern as classify);
  navigability `mean_path_length_to_local_optima` / `local_optima_accessibility` / `mean_distance_to_local_optima`
  dict-or-list[dict] polymorphism → always the collection form (DataFrame). Golden extractor translates to keep refs frozen.
- **Phase 6 (value-changing, E1/E2 — approved, regenerate affected goldens)**: non-finite-fitness build guard;
  NaN-normalize degenerate metric returns + divide guards; thread `rng` into first-improvement (HillClimb already has
  `seed=`; wire it through basin's plateau-exit path + any first-improvement caller).
- **Phase 8 (logging)**: route ~115 `print()`s through the logger; `warnings.warn` for pseudo-warnings; fix the
  `logger = random_walk(...)` shadow (already moot — random_walk gone — but audit ruggedness).
- **Phase 9 (mechanical/polish)**: `_pythonize`×6 + `_pack_rows`×2 → `analysis/_utils.py`; alphabet triplication →
  import from `_data`; dead-code (commented networkx in utils, `euclidean_distance`, `EdgeResult.strategy`,
  `include_configs`); narrow the ~32 broad `except` with `raise ... from e`; finish `from __future__ import annotations`
  on untouched modules; type-annotation backfill; docstring-drift fixes.
- **Structural decomposition (deferred)**: landscape `_io.py`/`_build.py`; `neighbors/` subpackage (+ generator-honesty
  raise, D6); `_data/` subpackage (handlers own parsing, D5, + `len(X)==len(f)` / reserved-`fitness` guards, column-prefix
  unify); `analysis/epistasis/` split; move `basin`/`optima`/`plateaus` into `landscape/`.
- **Final**: full `bench/` run for the formal perf gate; adapt out-of-scope modules (plotting/lon/sampling/filters/
  problems/examples) when you signal; packaging deferred (version `0.1.dev0` vs `0.2.5`, `py.typed`).

## Process / rules of engagement
- First clarify the hard, consultation-worthy decisions; the easy nits (the 5 the user listed) get batched later.
- Maintain this file as the single source of truth for what to do.
- Engage the user with questions; record every resolved decision here under "Decisions".

## Redlines (must hold)
1. **High-level API**: the sklearn-style *pipeline shape* is frozen
   (`DNALandscape(...).build_from_data(X, f)`; `from graphfla.analysis import <fn>; fn(landscape)`).
   Low-level/intermediate signatures and analysis **return shapes** MAY change. Public *names*: see Decision 3.
2. **No behavior change**: constructed landscape + every calculated value stay numerically identical.
   Gate = existing pytest suite (`tests/`). Do NOT write new verification scripts; do NOT change result-asserting tests.
3. **No perf regression** (runtime & memory), tolerance ~10–20%. Do NOT optimize further. Gate = existing `bench/` harness.
- No backward-compat shims/aliases — overwrite directly and update all internal callers.

## Out of scope (until the very end, when user signals)
`plotting/`, `lon.py`, `sampling.py`, `filters.py`, `problems/`, `examples/`, `docs/`. Do not factor their needs
into any single change. Tests: may adapt to API changes, but never change a result/calculation assertion.

## Verified facts (de-risked — do not re-litigate)
- **`type`-overload is NOT a behavior bug.** `DNALandscape().type == 'sequence_<id>'` (a registry key), never `'dna'`.
  Consequences, both verified value-preserving:
  - `_get_default_distance_metric` returns `mixed_distance` for sequences, but `mixed_distance ≡ hamming_distance`
    for all-categorical data (verified: max diff `0.0` on `dist_to_go`). No numeric impact.
  - `walsh_hadamard_coefficient` falls into the generic `else` branch instead of the `dna/rna/protein` branch.
    Verified: coefficient **values are identical** across branches (max abs diff `0.0`); only the **key labels**
    differ (`1_1_2` vs `A_1_C`). So fixing the overload + redesigning the WHT return is redline-#2-safe.
- Semantic `.type` reads: `epistasis.py` (boolean / dna-rna-protein / generic WHT branches, motif `_assign_roles`),
  `landscape.py:693` (`ordinal` strategy warning), `:1538` (distance), `:1271/:1277` (legit registry lookup).
  Only the sequence subclasses break the semantic checks.
- Walk callers (all internal, low-level): `random_walk` ← `ruggedness.autocorrelation`; `hill_climb` ←
  `correlation.basin_fit_corr` (uses `return_trace`) & `algorithms/basin.py`. `local_search` has **no internal caller**.
  `plotting.py` also calls `hill_climb` (out of scope for now).

## The 7 workstreams (menu)
1. **Module decomposition** — `landscape.py`, `_neighbors.py`, `epistasis.py`, `_data.py`, `utils.py` grab-bag. (large / low risk)
2. **Analysis return contract** — one shape policy; kill nested-string-dict WHT + dict-or-list polymorphism. (large / med)
3. **Naming/terminology** — abbreviation policy, verb-vs-noun, gamma pair, `type` rename, strategy-param triple-name. (med / med)
4. **Unified walk/trajectory API** — common contract + WalkResult; seed everywhere; bounds checks. (med / med)
5. **Validation & errors** — input checks, unify is-built guard, replace print-as-warning with `warnings.warn`. (med / med)
6. **Duplication & dead code** — `_pythonize`×6, alphabets×3, commented networkx, unused fns/fields/params. (med / low)
7. **Encapsulation** — leaky `landscape.graph`/`_check_built` access; the `self`-named mutating analysis helpers. (med / low)

## Major decisions (status: PENDING user)
- **D1 — `type`/WHT cleanup**: introduce a stable semantic `kind` attr separate from the registry key; redesign the
  WHT return. Verified value-preserving. → fold into Decisions 2 & 3. *(low risk; confirm key-label format.)*
- **D2 — Analysis return contract form**: dataclass+DataFrame vs plain-dict vs DataFrame-everywhere. (redline #1 low-level)
- **D3 — Public naming aggressiveness**: moderate vs aggressive vs internal-only. (brushes redline #1 — confirm interpretation)
- **D4 — Walk API form**: Walk classes+WalkResult vs unify-functions vs generator protocol. (redline #3 perf — per-walk not per-step)
- **D5 — `_data.py`↔`landscape.py` ingestion boundary**: single `prepare_landscape_data` entry; handler classes own parsing.
- **D6 — `_neighbors.py` decomposition + generator-honesty invariant**: subpackage; raise when Manhattan generator meets
  Hamming strategy (latent hazard, currently masked by `OrdinalLandscape` hard-coding `strategy='active'`).
- **D7 — Placement of the 4 mutating `determine_*` helpers** (first param literally `self`): move to a landscape-internal
  `_compute.py` so `analysis/` is pure readers. Also: keep `Landscape.determine_local_optima/global_optimum` public?

## Mechanical fixes (no consultation — batch during delivery)
- `len(X)==len(f)` check with clear ValueError (`_data.py`).
- Raise if an X column is named `'fitness'` (reserved) — currently silently overwrites it (`_data.py:680`).
- Upfront `build_from_data` validation: `filter_mode∈{any,both}`, `neighborhood_strategy∈{None,auto,active,pairwise,broadcast}`, `epsilon>=0`, `n_edit>=1`.
- Extract `_pythonize`(×6) + `_pack_rows`(×2) into `analysis/_utils.py`.
- Import DNA/RNA/PROTEIN alphabets from `_data.py` in the subclasses (kill triplication).
- `boolean.py` import `Landscape` from `.landscape` not `.sequence`; hoist lazy imports unless circular-import reason.
- Delete dead: `include_configs` param, `EdgeResult.strategy`, `euclidean_distance`, ~100 lines commented networkx (`utils.py:323-423`),
  regression block in `diminishing_returns_index` (`epistasis.py:765-777`), always-true `if sum(int_col)>=0` guard.
- Fix wrong annotations: `fitness_distance_corr -> float`, `hill_climb` return type; wrap nan/np-float leaks in `_pythonize`.
- Replace `print`-as-warning with `warnings.warn` (`epistasis.py:282,542`); gate/remove WHT progress prints; narrow bare `except`.
- `analysis/__init__.py` `__all__` hygiene: resolve `ffi` alias, drop `calculate_evol_enhance`, privatize `get_motif_node_indices`.
- Collapse duplicate `__getattr__` branches in `graphfla/__init__.py`; fix the broken `graphfla.Landscape` `__all__` entry.
- Rename `self`→`landscape` param in the 4 mutating helpers.
- Drop `id(alphabet)` registry-key scheme → constant `'sequence'` key (coordinate with D1).
- Centralize `random.Random(seed) if seed else random` idiom; lift `batch_size=10000` constant in `basin.py`.
- Single `_coerce_fitness`/`_align_index` helper for the 4 copy-pasted handler blocks (`_data.py`).
- Single validated `configs_array` shape/dtype check at top of `build_edges`.

## Additional issues found (beyond the 5 you raised) — round-2 mining of the first sweep
These came out of the detailed per-area findings (not in your original 5). Folded into the workstreams above
unless noted; a few are genuinely new and may warrant a decision (flagged ★).
- **Misleading dead params** ★: `n_jobs` is accepted but **ignored** in several analysis functions; `seed` is
  accepted-but-inert in another. A param that does nothing is worse than no param. → either honor it or remove it.
- **`__getitem__`/`__contains__` ambiguity** ★: conflate integer node-index vs string node-name lookups; ambiguous
  for string-encoded landscapes. → define one lookup semantics (likely: index by position, `.config_index(name)` for names).
- **User-visible column-name inconsistency** ★: `get_data()` columns are `bit_i` (boolean) / `pos_i` (sequence) /
  `var_i` (ordinal/default) — different vocab per landscape type, in user-facing output. → one scheme (e.g. always `pos_i`).
- **Degenerate-case contract is undefined/inconsistent** ★: conceptually-similar "no data" situations return `0.0`
  in one fn, `NaN` in another, `[]` in a third. → define a single documented degenerate-input contract per metric class.
- **Error-swallowing hides root causes**: bare `except Exception` re-raised as generic `RuntimeError` (analysis);
  `determine_basin_of_attraction` swallows all per-node climb exceptions; `build_from_graph` silently defaults on parse
  failure. → narrow excepts, preserve cause (`raise ... from e`), fail loudly on corrupt input. (→ workstream 5)
- **`build_edges` has a 12-param keyword-only signature** with redundant/derivable args → tighten once the neighbors
  package is split (→ workstream 1).
- **Producer containers leak internals**: edge producers return ndarray vs Python list inconsistently (→ workstream 1/6).
- **Lazy-cache vs docstring drift**: the class docstring documents eager `Attributes` that are actually lazy-computed;
  state split across mirror fields (`n_lo`/`lo_index`, configs-triplet/counts) (→ workstream 7 + docstrings).
- **`distribution_fit_effects` (→`dfe`) and `idiosyncratic_index` overlap** heavily but return different shapes from
  different modules → reconcile shapes under the return contract (→ workstream 2).
- **Public symbols leak outside `__all__`** across analysis/algorithms → curate `__all__` per module (→ workstream 5/6).

## Round-2 sweep: cross-cutting workstreams (beyond the 5 + the first 7)
8. **Estimator object protocol** — `get_params`/`set_params`/clone; `describe()` → structured report (drop ANSI,
   stop returning None); params-forward `__repr__` (show maximize/epsilon/kind/alphabet, informative pre-build);
   `__eq__`/`__hash__` decision; fix `verbose=None` poisoning the bool `self.verbose` (landscape.py:534,671). (med/med)
9. **Custom exception taxonomy** — new `graphfla/exceptions.py` (`GraphFLAError` base, `NotBuiltError(RuntimeError)`);
   replace the 3 inconsistent not-built idioms; chain ~32 broad `except Exception` with `raise ... from e`; narrow
   swallow-all sites (navigability.py:493, ruggedness.py:288 swallows→NaN, basin.py:223, landscape.py:789). (med/med)
10. **Single RNG convention** — `_check_random_state` helper; thread an `rng` through walks/sampling; fix `seed=None`
   aliasing global `random` (ruggedness.py:92). [rng-into-first-improvement is value-changing → decision E2]. (med/med)
11. **Logging over print()** — route ~115 `print()`s (40 landscape, 26 _data, 18 _neighbors) through the module
   logger; `warnings.warn` for pseudo-warnings; rename the `logger = random_walk(...)` shadow (ruggedness.py:98). (large/med)
12. **Packaging & typing distribution** — add `graphfla/py.typed`; single-source `__version__` (**0.1.dev0 vs setup.py
   0.2.5 mismatch**); add `pyproject [project]` table; dedup deps; fix malformed `tqdm>=4.40.0a`; reconcile
   python_requires>=3.8 vs 3.9-3.11 classifiers. (med/low)
13. **Degenerate-input numeric contracts** — non-finite-fitness build guard; NaN-normalize flat/empty/constant metric
   returns; guard divides-by-mean/std and mean-of-empty. [VALUE-CHANGING → decision E1]. (large/high)
14. **Cache-invalidation / freeze lifecycle** — freeze-after-build (aligns with privatized `_compute_*`); fix
   `configs.setter` leaving `_configs_array` stale (landscape.py:367), the `go` stale-snapshot (navigability.py:48),
   raw igraph `KeyError` on early-return properties, dead `_n_configs`/`_n_edges` mirrors. [policy → decision E4]. (med/med)
15. **Type-annotation backfill** — `_neighbors.py` (30% return cov), `epistasis.py` (7%), algorithms (0 returns),
   utils, the 6 subclass `__init__` (`-> None`); fix `default=None`-without-`Optional` (ruggedness.py:54,
   navigability.py:340). (large/low)
16. **Docstring-vs-signature drift** — undocumented `seed` params; phantom `BaseLandscape` type; `mean_path_lengths_go`
   wrong xref + false "and variance" claim vs `-> float`; missing `Raises` on hill_climb/local_search; NameError
   `Examples` block in `build_from_data`; stray whitespace; honest `include_configs` doc. (med/low)

## Round-2 decisions (RESOLVED)
- **E1 — Degenerate-input contract** = **IN (value-changing, approved)**: reject non-finite fitness at build (ValueError)
  + NaN-normalize flat/empty/constant metric returns + guard divides. Deliberate behavior change on degenerate inputs;
  regenerate the affected golden-test values as an isolated slice. Non-degenerate values untouched.
- **E2 — RNG for first-improvement walks** = **IN (value-changing, approved)**: thread an `rng` so first-improvement is
  seedable (changes which successor is chosen). Coordinate with the Walk/WalkResult unification; regenerate any
  first-improvement test fixtures.
- **E3 — Additive workstreams** = **IN: estimator protocol (8), exceptions (9), logging (11)**.
  **Packaging/typing-dist (12) = DEFERRED** (skip for now; flag the `0.1.dev0` vs `0.2.5` version mismatch + missing
  `py.typed` for a later one-liner pass).
- **E4 — Lifecycle** = **freeze-after-build**: privatize the `_compute_*` steps to build-time only; document the built
  landscape as immutable (fit-once). Fixes the configs-setter / go-snapshot / re-run-determine_* stale-cache class wholesale.
- **E5 — `from __future__ import annotations`** = **adopt package-wide** (in-scope modules); prerequisite for the typing backfill.
- **E6 — Broken `graphfla.Landscape` export** = **fix by importing it** at top level (+ delete dead
  `_exported_config_functions` and the duplicated `__getattr__` branches).
- **Fold into kind+IO work**: unify `get_data()` column prefix to `pos_i` + expose `feature_columns`; persist `alphabet`
  through the GraphML round-trip so `build_from_graph` reconstructs a real SequenceLandscape. Output/serialization-surface
  changes (allowed) — not calculation changes.

### Round-2 mechanical fixes (append to batch)
- Delete the dead side-effecting `setup_module` hook + its `os`/`random` imports (`__init__.py:76-87`); move any test
  seed into `tests/conftest.py`.
- Remove unused `from ._data import filter_data` (`utils.py:11`); give `utils.py` an explicit `__all__`.
- Collapse dead `_exported_config_functions` + byte-identical `__getattr__` elif/else (`__init__.py:64-73`).
- Fix `tqdm>=4.40.0a` → `tqdm>=4.40.0`; reconcile python_requires vs classifiers.
- `default=None`-without-`Optional` fixes (ruggedness.py:54, navigability.py:340, `configs_series_from_array`).
- `print('Warning: ...')` → `warnings.warn` (epistasis.py:542 et al.).
- Parameterize bare Protocol generics in `_neighbors.py` (`Tuple`→`Tuple[int,...]`, etc.) + `build_edges` `Callable`.
- Phantom `BaseLandscape` → `Landscape` (correlation.py:108, robustness.py:111).
- Add missing `landscape` param to numpydoc of fdc/fitness_flattening_index; add undocumented `seed` to
  classify_epistasis/extradimensional_bypass/mean_path_length* docstrings.
- Add `Raises` to hill_climb/local_search; fix mean_path_lengths_go xref + drop false variance claim.
- `-> None` on the 6 subclass `__init__` + `SearchCache.__init__` (+ type its `graph` param).

## Decisions (RESOLVED)
- **D2 — Analysis return contract** = **dataclasses + tidy DataFrames**. Fixed-schema multi-stat → typed
  `@dataclass` (snake_case fields, attribute access); per-term / per-node / tabular → tidy long-form DataFrame.
  Scalars stay plain `float`. Values numerically identical; only containers change.
- **D4 — Walk API** = **Walk classes + `WalkResult` dataclass**. `Walk` base with `.run(start) -> WalkResult`;
  `HillClimb` / `RandomWalk` subclasses hold their own knobs. Per-walk overhead only (perf-safe). Kills the
  `return_trace` tuple-arity switch; uniform `seed`; uniform start-node bounds check.
- **Decomposition** = **full subpackages** for all four god-files; re-export public symbols (paths stable).
- **D3 — Naming policy** = aggressive but principled (no blanket full-spelling; no aliases; overwrite + update callers):
  - One canonical term per concept used everywhere.
  - **Keep genuine FLA-literature acronyms** as canonical names: `fdc`, `dfe`, `r_s_ratio`, `gamma`/`gamma_star`,
    `walsh_hadamard`, `lon`.
  - **Spell out `lo`/`go`** → `local_optima` / `global_optima` (resolves the existing internal inconsistency).
  - **Spell out homegrown/ambiguous tokens**: `fit`→`fitness`, `corr`→`correlation` (outside acronyms),
    `dist`→`distance`, `evol`→`evolvability`. Drop `ffi` alias and deprecated `calculate_evol_enhance`.
  - Verb prefix only for side-effecting ops; pure metrics are noun phrases.

### Public rename table (analysis) — REVIEW & flag any you dislike
| current | new | note |
|---|---|---|
| `fitness_distance_corr` | `fdc` | acronym kept; matches advertised `graphfla.analysis.fdc` |
| `fitness_flattening_index` | `fitness_flattening_index` | unchanged |
| `ffi` | _(dropped)_ | redundant alias |
| `basin_fit_corr` | `basin_fitness_correlation` | fit→fitness, corr→correlation |
| `neighbor_fit_corr` | `neighbor_fitness_correlation` | |
| `fitness_distribution` | `fitness_distribution` | unchanged |
| `distribution_fit_effects` | `fitness_effect_distribution` | user pick (not `dfe`) |
| `lo_ratio` | `local_optima_ratio` | lo→local_optima |
| `autocorrelation` | `autocorrelation` | unchanged |
| `r_s_ratio` | `r_s_ratio` | notation kept |
| `gradient_intensity` | `gradient_intensity` | unchanged |
| `global_optima_accessibility` | `global_optima_accessibility` | unchanged |
| `local_optima_accessibility` | `local_optima_accessibility` | unchanged |
| `mean_path_lengths` | `mean_path_length_to_local_optima` | confirm semantics |
| `mean_path_lengths_go` | `mean_path_length_to_global_optima` | |
| `mean_dist_lo` | `mean_distance_to_local_optima` | |
| `mean_dist_go` | `mean_distance_to_global_optima` | |
| `neutrality` | `neutrality` | unchanged |
| `single_mutation_effects` | `single_mutation_effects` | unchanged |
| `all_mutation_effects` | `all_mutation_effects` | unchanged |
| `evol_enhance_mutations` | `evolvability_enhancing_mutations` | |
| `calculate_evol_enhance` | _(dropped)_ | deprecated twin |
| `higher_order_epistasis` | `higher_order_epistasis` | unchanged |
| `classify_epistasis` | `classify_epistasis` | KEEP (user pick) |
| `idiosyncratic_index` | `idiosyncratic_index` | unchanged |
| `global_idiosyncratic_index` | `global_idiosyncratic_index` | unchanged |
| `diminishing_returns_index` | `diminishing_returns_index` | unchanged |
| `increasing_costs_index` | `increasing_costs_index` | unchanged |
| `gamma_statistic` | `gamma` | pairs with `gamma_star` |
| `gamma_star` | `gamma_star` | unchanged |
| `walsh_hadamard_coefficient` | `walsh_hadamard` | returns tidy DataFrame now |
| `extradimensional_bypass_analysis` | `extradimensional_bypass` | drop vague `_analysis` |
| `get_motif_node_indices` | `_get_motif_node_indices` | privatize (not in `__all__`) |

Landscape: rename `type` attr (shadows builtin) → `kind` (semantic), registry keyed on it; serialized
`landscape_type` string updated. Algorithms names handled inside the Walk redesign + D7 (below).
Navigability family: `mean_path_lengths`→`mean_path_length_to_local_optima`,
`mean_path_lengths_go`→`mean_path_length_to_global_optimum`, `mean_dist_lo`→`mean_distance_to_local_optima`,
`mean_dist_go`→`mean_distance_to_global_optimum` (LO=optima/plural, GO=optimum/singular).

- **D5 — Data layer** = **handler classes own their parsing**, one `prepare_landscape_data` entry point.
- **D6 — Neighbor generators** = **raise** when an ordinal/Manhattan generator meets a Hamming-only strategy
  (`pairwise`/`broadcast`); `_select_strategy` never auto-picks Hamming for an ordinal generator. No current
  landscape hits it → behavior/tests unchanged. Make fast-path dispatch polymorphic (capability attr on the
  generator) so custom generators aren't silently slow.
- **D7 — `determine_*`** = **privatize** `Landscape.determine_local_optima/determine_global_optimum` →
  `_compute_local_optima`/`_compute_global_optimum`; move the 4 `self`-named mutating analysis helpers
  (`determine_global_optimum`/`determine_accessible_paths`/`determine_dist_to_go` in navigability,
  `determine_neighbor_fitness` in correlation) into a landscape-internal `landscape/_compute.py`. `analysis/`
  becomes pure (landscape → metric) readers. Update the `is self` regression test.
- **Algorithms boundary** = **move** `basin`/`optima`/`plateaus` into `landscape/` as internal modules; `algorithms/`
  holds only the pure Walk classes.

## Detailed design specs (recorded)

### Target package layouts (re-export public symbols; import paths stay stable)
```
landscape/
  __init__.py        # re-export Landscape + subclasses (unchanged public surface)
  landscape.py       # core Landscape: accessors/properties, dunders, describe, get_data, get_lon, kind
  _build.py          # build pipeline as functions taking the landscape (resolve/preprocess/construct/postprocess/remap/finalize/build_edges/build_graph)
  _io.py             # GraphML read/write; build_from_graph/to_graph become thin delegators
  _compute.py        # _compute_local_optima/_global_optimum + the 4 moved mutating helpers
  _optima.py _basin.py _plateaus.py   # moved from algorithms/
  sequence.py boolean.py ordinal.py dna.py rna.py protein.py

neighbors/            # was _neighbors.py (1782 lines)
  __init__.py        # re-export build_edges, EdgeResult, generators
  generators.py      # NeighborGenerator protocol + Boolean/Sequence/Ordinal/Default (+ capability attr)
  edges.py           # build_edges, _select_strategy, EdgeResult, _normalize_edge_output, the invariant raise
  _kernels.py        # _build_active/_pairwise/_broadcast, _masked_grouping, _bytemap_*, _active_*
  _classify.py       # _classify_* helpers

analysis/epistasis/   # was epistasis.py (1733 lines)
  __init__.py        # re-export the public epistasis fns
  walsh_hadamard.py  # walsh_hadamard + _generate_interactions/_H_matrix/_V_matrix/_coefficient_to_sequence/_ensemble_encode_features
  motifs.py          # classify_epistasis, extradimensional_bypass, _assign_roles*, _calculate_pos_neg*, _seeded_igraph, _get_motif_node_indices
  gamma.py           # gamma, gamma_star, _gamma_statistics, _gamma_pair_via_dict, _gamma_position_pair_worker
  idiosyncrasy.py    # idiosyncratic_index, global_idiosyncratic_index, diminishing_returns_index, increasing_costs_index, _idiosyncratic_position_worker
  higher_order.py    # higher_order_epistasis
analysis/_utils.py    # shared _pythonize, _pack_rows, _coerce_seed/rng helper

_data/                # was _data.py (1307 lines)
  __init__.py        # re-export prepare_landscape_data, PreparedData, handlers
  handlers.py        # InputHandler protocol + 4 handlers, each owning parse+validate (folds _parse_*_input)
  pipeline.py        # prepare_landscape_data (filter→prepare→clean→encode, single data_types assignment); PreparedData
  _encode.py         # encode_data internals (display-frame moved toward get_data)
  _validation.py     # shared validators + _coerce_fitness/_align_index
```

### Walk API (algorithms/)
```python
@dataclass
class WalkResult:
    path: np.ndarray          # visited node ids incl. start
    @property
    def final(self) -> int: return int(self.path[-1])
    @property
    def n_steps(self) -> int: return len(self.path) - 1

class Walk:                                   # base
    def __init__(self, cache, *, seed=None): ...
    def run(self, start) -> WalkResult: ...   # validates start bounds

class HillClimb(Walk):                        # was local_search(step) + hill_climb
    def __init__(self, cache, *, strategy="best-improvement", seed=None): ...
class RandomWalk(Walk):                        # was random_walk
    def __init__(self, cache, *, length=100, neutral_neighbors=None, seed=None): ...
```
- One strategy param name `strategy` ∈ {"best-improvement","first-improvement"}; `seed` on all walks (fixes
  non-reproducible first-improvement); start-node bounds check in `run`.
- Drop the `attribute=` logging param of random_walk: callers gather `cache.fitness[result.path]` themselves.
- `local_search` (zero callers) folds into `HillClimb` as the private step; public export removed.
- Callers: `autocorrelation` → `RandomWalk(cache, length=L, seed=s).run(node)`; `basin_fitness_correlation` &
  `basin.py` → `HillClimb(cache).run(i)` (hoist the instance across a batch — reuse, perf-safe).
- Perf: object construction is per-walk, never per-step; inner loop byte-identical. Validate with `bench/`.

### Return-contract mapping (D2) — finalize exact fields against real returns during delivery
- **stay `float`**: fitness_flattening_index, local_optima_ratio, r_s_ratio, gradient_intensity, autocorrelation,
  neutrality, idiosyncratic_index, global_idiosyncratic_index, diminishing_returns_index, increasing_costs_index,
  gamma, gamma_star.
- **`@dataclass`** (fixed-schema multi-stat): fdc `(coefficient, p_value)`; classify_epistasis (counts/props per
  epistasis type — replaces space-containing prose keys); extradimensional_bypass; the per-GO path/distance
  summaries (mean + variance).
- **tidy DataFrame** (per-term / per-node / per-LO): **walsh_hadamard** (cols: `order, positions, alleles,
  coefficient` — kills the nested `dict[order][stringkey]`); single_mutation_effects; all_mutation_effects;
  evolvability_enhancing_mutations; local_optima_accessibility / mean_path_length_to_local_optima /
  mean_distance_to_local_optima (always the collection form — kills the dict-or-list[dict] polymorphism).
- **np.ndarray / Series** (a distribution): fitness_distribution, dfe.

### `kind` attribute (D1, value-preserving — verified)
- Add semantic `self.kind` ∈ {boolean, ordinal, dna, rna, protein, categorical}; subclasses set it; SequenceLandscape
  stores the concrete kind from DNA/RNA/Protein. Per-instance registries keyed on `kind` (drop `id(alphabet)`).
- Discriminators read `kind`: `_get_default_distance_metric`, WHT branch, `has_ordinal`. WHT alphabet reconstruction
  reads `self.alphabet` (no hardcoded letter lists). Update `landscape_type` serialization + api_regression test.

## Execution order (deliver only after user leaves; pytest + `bench/` gate after each phase)
0. **Foundational prep** (low-risk, unblocks the rest): add `from __future__ import annotations` to all in-scope
   modules; create `graphfla/exceptions.py` (`GraphFLAError`, `NotBuiltError(RuntimeError)`); fix the broken top-level
   `graphfla.Landscape` export + dead `__getattr__`/`setup_module` cleanup.
1. **Landscape foundation**: decompose into `_build.py`/`_io.py`/`landscape.py`; add semantic `kind`; **freeze-after-build**
   (privatize `_compute_*`, build-time only); **estimator protocol** (get_params/set_params, params-forward `__repr__`,
   `describe()`→structured report, `__eq__`/`__hash__`); upfront build validation + `NotBuiltError`; fix `verbose=None`.
2. **neighbors/** subpackage + generator-honesty invariant (raise on ordinal+Hamming) + polymorphic dispatch + typing.
3. **_data/** subpackage: handlers own parsing; single `prepare_landscape_data`; `len(X)==len(f)` + reserved-`fitness`
   guards; unify `get_data()` column prefix → `pos_i` + expose `feature_columns`.
4. **Walk API** (Walk/WalkResult; one `strategy` param; seed on all walks **incl. first-improvement rng [E2, value-
   changing]**) + move `basin`/`optima`/`plateaus` into `landscape/`; update internal callers.
5. **analysis/** per module: split `epistasis/`; apply return-contract (dataclasses + tidy DataFrames, **WHT→DataFrame**);
   privatize the `_compute_*` movers → `analysis/` becomes pure readers; unify is-built guard + validation.
6. **Value-changing slice [E1]** (isolated, regenerate goldens): non-finite-fitness build guard + NaN-normalize
   degenerate metric returns + guard divides. Kept separate so the golden-test regeneration is auditable.
7. **Naming sweep**: apply the full rename table across all modules + callers + tests (no aliases).
8. **Logging migration**: route ~115 `print()`s through the logger; `warnings.warn` for pseudo-warnings.
9. **Mechanical/polish batch**: dedup (`_pythonize`×6, alphabets×3), dead-code removal, docstring-drift fixes,
   type-annotation backfill, `Raises` sections.
10. **Final verification**: full pytest + full `bench/`; only then adapt out-of-scope modules when the user signals.

(Deferred, not in this effort: packaging/`pyproject [project]`/`py.typed`/version-mismatch — flagged for a later pass.)

## Open micro-items (resolve during delivery, not blocking)
- `classify_epistasis` keep verb vs `epistasis_classification` noun (leaning: keep, reads naturally).
- Exact `@dataclass` field names (derive from each function's real return).
- `_build.py` as free functions (chosen, lowest risk) vs a `LandscapeBuilder` class.

