# ProteinGym × GraphFLA

Annotate GraphFLA fitness-landscape features for every
[ProteinGym](https://proteingym.org) deep-mutational-scanning (DMS) substitution
assay, and merge them with ProteinGym's zero-shot and supervised model-performance
tables into a single CSV (one row per DMS).

Each assay is turned into a `ProteinLandscape`:

* **Only mutated positions are kept** — a genotype is the string of amino acids at
  the union of positions mutated anywhere in the assay; invariant positions are dropped.
* **Single-site saturation assays are removed** (average number of mutations == 1),
  because epistasis cannot be derived from them.

## Install

```bash
pip install graphfla            # or: pip install -e .  from the repo root
```

## Get the ProteinGym data

You provide the paths; nothing is downloaded for you. From ProteinGym v1.3:

* **DMS substitution datasets** (one CSV per assay) — unzip
  `DMS_ProteinGym_substitutions.zip`
  from <https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/>.
* **Reference metadata** — `reference_files/DMS_substitutions.csv` from the
  [ProteinGym GitHub repo](https://github.com/OATML-Markslab/ProteinGym)
  (provides each assay's `target_seq`).
* **Performance tables** (optional, for the merge) — the DMS-level Spearman CSVs:
  `benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv`
  and `benchmarks/DMS_supervised/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv`.

## Run (one command)

```bash
python proteingym_features.py \
    --dms-dir    /path/to/DMS_ProteinGym_substitutions \
    --reference  /path/to/reference_files/DMS_substitutions.csv \
    --zero-shot  /path/to/DMS_substitutions_Spearman_DMS_level.csv \
    --supervised /path/to/DMS_substitutions_Spearman_DMS_level.csv \
    --output     proteingym_graphfla_table.csv
```

The run is **resumable**: per-dataset results are cached as JSON under
`--cache-dir`, so re-running only processes datasets not yet done. Use `--force`
to recompute.

### Useful flags

| flag | meaning |
|---|---|
| `--n-jobs` | CPU cores for the multi-core epistasis features (default: all). |
| `--limit N` | Process at most `N` datasets (smoke testing). |
| `--skip-features a,b` | Skip named features. `gamma_statistic,gamma_star` are by far the most expensive (they sweep all position-pairs × allele-pairs) and can dominate runtime on assays with many mutated positions. |
| `--cache-dir` | Where per-dataset JSON results are cached (resumability). |

## Output

`proteingym_graphfla_table.csv`, one row per (multi-mutant) DMS, with:

* **Meta**: `seq_len`, `n_mutated_positions`, `n_variants`, `avg/median/max_n_mutations`,
  `frac_single/multi`, fitness summary, and landscape structure
  (`n_configs`, `n_edges`, `n_lo`, `n_lo_members`, `n_plateau`, `go_fitness`).
* **GraphFLA features** (scalar; dict-valued metrics expanded into `name.key` columns):
  ruggedness (`lo_ratio`, `autocorrelation`, `r_s_ratio`, `gradient_intensity`),
  correlation (`fitness_distance_corr`, `fitness_flattening_index`, `basin_fit_corr`,
  `neighbor_fit_corr`), navigability (`global_optima_accessibility`,
  `mean_path_lengths_go`, `mean_dist_go`), robustness (`neutrality`,
  `evol_enhance_mutations`), `fitness_distribution.*`, and epistasis
  (`diminishing_returns_index`, `increasing_costs_index`, `classify_epistasis.*`,
  `extradimensional_bypass.*`, `higher_order_epistasis`, `global_idiosyncratic_index`,
  `gamma_statistic`, `gamma_star`).
* **Model performance**: zero-shot (`zs_*`) and supervised (`sup_*`) Spearman per model.

Features that are undefined for a given landscape are left as `NaN` (e.g.
`basin_fit_corr` when there is a single optimum).
