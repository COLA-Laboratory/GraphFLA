# PPT performance optimization tables

Use these as copy-ready tables for the optimization-progress slides.

Notes:
- Runtime reduction is `baseline / optimized`.
- Construction memory is peak construction delta RSS (`construct_delta_mb` / `orig_MB` vs `final_MB` in the final campaign summary).
- Trajectory memory was monitored but not optimized as a primary target; the optimized cache path stayed essentially flat (`delta_mb` about 0-5 MB in the campaign notes).
- Analysis memory was sampled to catch regressions; only rows with useful before/after memory samples are shown.

## 1. Landscape construction runtime

| Dataset | Baseline runtime | Optimized runtime | Reduction |
|---|---:|---:|---:|
| GB1 protein-L | 5.10 s | 1.40 s | 3.6x |
| TrpB3I protein-S | 0.148 s | 0.061 s | 2.4x |
| Papkou DNA-L | 3.75 s | 0.88 s | 4.3x |
| Westmann DNA-S | 0.849 s | 0.043 s | 19.9x |
| CR9114 boolean-L | 0.707 s | 0.182 s | 3.9x |
| CR6261 boolean-S | 0.023 s | 0.011 s | 2.0x |
| WReOs ordinal-S | 0.004 s | 0.002 s | 1.5x |
| HPO ordinal-L | 3.25 s | 0.658 s | 4.9x |
| UBE4B sparse-S | 0.260 s | 0.051 s | 5.1x |
| D7PM05 sparse-M | 30.33 s | 0.414 s | 73x |
| GFP sparse-L | 117.1 s | 0.701 s | 167x |
| PHOT sparse-L | 60.04 s | 1.504 s | 40x |
| HIS7 sparse-XL | 298.7 s | 8.007 s | 37x |

## 2. Landscape construction memory

| Dataset | Baseline memory | Optimized memory | Reduction |
|---|---:|---:|---:|
| GB1 protein-L | 683 MB | 402 MB | 1.7x |
| TrpB3I protein-S | 248 MB | 14 MB | 17.7x |
| Papkou DNA-L | 371 MB | 85 MB | 4.4x |
| Westmann DNA-S | 1505 MB | ~0 MB | eliminated |
| CR9114 boolean-L | 80 MB | 48 MB | 1.7x |
| CR6261 boolean-S | ~0 MB | ~0 MB | flat |
| WReOs ordinal-S | ~0 MB | ~0 MB | flat |
| HPO ordinal-L | 370 MB | 127 MB | 2.9x |
| UBE4B sparse-S | 68 MB | ~0 MB | eliminated |
| D7PM05 sparse-M | 2988 MB | 4 MB | 747x |
| GFP sparse-L | 433 MB | 20 MB | 21.7x |
| PHOT sparse-L | 649 MB | 123 MB | 5.3x |
| HIS7 sparse-XL | 3454 MB | 1000 MB | 3.5x |

## 3. Trajectory optimizers runtime, GB1

| Function / workload | Baseline runtime | Optimized runtime | Reduction |
|---|---:|---:|---:|
| `local_search`, best-improvement @100k | 0.929 s | 0.140 s | 6.6x |
| `local_search`, first-improvement @100k | 0.100 s | 0.096 s | 1.04x |
| `hill_climb`, best-improvement @100k | 1.739 s | 0.328 s | 5.3x |
| `hill_climb`, first-improvement @100k | 0.752 s | 0.604 s | 1.24x |
| `random_walk` @10k | 2.152 s | 1.693 s | 1.27x |

## 4. Trajectory optimizers memory

| Function / workload | Baseline memory | Optimized memory | Reduction |
|---|---:|---:|---:|
| `local_search`, best-improvement | No extra cache | +0-5 MB | flat |
| `local_search`, first-improvement | No extra cache | +0-5 MB | flat |
| `hill_climb`, best-improvement | No extra cache | +0-5 MB | flat |
| `hill_climb`, first-improvement | No extra cache | +0-5 MB | flat |
| `random_walk` | Tiny streaming output | +0-5 MB | flat |

## 5. Landscape analysis runtime, HPO

| Feature / function | Baseline runtime | Optimized runtime | Reduction |
|---|---:|---:|---:|
| `mean_path_lengths_go` | error / 119 s on Papkou | 0.1-0.2 s | ~1000x |
| `all_mutation_effects` | 83.82 s | 0.97 s | 86x |
| `single_mutation_effects` | 44.17 s | 0.75 s | 59x |
| `global_idiosyncratic_index` | 30.87 s | 0.78 s | 40x |
| `diminishing_returns_index` / `increasing_costs_index` | ~1.66 s | ~0.06 s | 15-44x |
| `gamma_statistic` / `gamma_star` | 165.44 s | 8.5 s | 20x |
| `extradimensional_bypass_analysis` | 67.90 s | 64.0 s | 1.06x |

## 6. Landscape analysis memory, HPO sampled delta RSS

| Feature / function | Baseline memory | Optimized memory | Reduction |
|---|---:|---:|---:|
| `gamma_statistic` @1 | 388 MB | 12 MB | 33x |
| `global_idiosyncratic_index` @1 | 55 MB | <1 MB | >55x |
| `all_mutation_effects` @1 | 349 MB | 55 MB | 6.3x |
| `single_mutation_effects` @1 | 113 MB | 3 MB | 35x |
| `diminishing_returns_index` | 27 MB | 1 MB | 27x |
| `r_s_ratio` | 151 MB | 147 MB | flat |
| `walsh_hadamard` / `higher_order_epistasis` | guarded / skipped on wide landscapes | guarded / skipped | avoided multi-GB risk |

