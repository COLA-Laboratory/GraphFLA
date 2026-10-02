# Test suite

Run `python -m pytest`; install dependencies from `requirements-dev.txt`.
The default suite contains basic tests only and requires no network access.
Empirical literature tests, including the full Papkou graph, run separately:
`python -m validation.tests`. See the permanent [literature contract](../validation/TESTING.md).

| Tests | Contract |
|---|---|
| `test_construction_oracles.py` | Independent all-pairs distance oracle across classes, directions, epsilon, sparsity and strategies; forced lookup, search, overflow, chunking and hash-collision paths |
| `test_build_contracts.py` | Input types, numeric fitness overflow, Boolean aliases, field-name collisions, indices, thresholds, neutral graphs, genetic-background trimming and GraphML |
| `test_compact_sequences.py` | Complete compact/general-path output equivalence with backgrounds, duplicates, input formats, thresholds, direction and custom input handlers |
| `test_ee_mutations.py` | Independent equations, categorical focal exclusion, BH families, edge cases and public EE API |
| `test_literature_contract.py` | Evidence roles, citations, hash failures, collection filters and JUnit provenance |
| `test_utilities_contracts.py` | Distances, rule combinations, Cartesian sampling and graph walks |
| `test_benchmark_contracts.py` | Complete public analysis inventory, callable benchmark parameters, pinned DMS checksums and input dimensions |
| Existing metric/profile/golden suites | Retained numerical and API regression coverage; separate literature-validation work remains independent |

The oracle computes expected neighborhoods from input values, without using the
production neighbor helpers. Hypothesis uses deterministic generation and bounded
small examples. Existing random smoke fixtures are now seeded, and the golden
helper no longer suppresses warnings globally. Current behavior alone is not a
scientific oracle; the retained golden analysis values do not substitute for
independent literature validation.

Coverage uses both lines and branches. Coverage reports identify untested paths;
percentages alone cannot establish correctness. The suite's small cases cover
both common encodings and optimized fallback paths; the empirical test guards
full-scale remapping and filtering against an externally published graph.

## Literature-validation integration

The `sep26` validation records, fixtures and analysis fixes are integrated with
the optimized construction pipeline. `test_neighborhood_oracles.py` uses the
fitness-array kernel interface and the geometry-aware automatic dispatch. The
functional filter includes surviving neutral connections before component
selection, while vertex attributes are still attached only after pruning.

One inherited precision requirement remains unresolved. Construction converts
fitness to float64, so integers beyond its exact range can round across the
functional threshold. The two parametrizations of
`test_neutral_pair_filter_preserves_integer_fitness_dtype` retain their original
assertions as strict expected failures; they are not passing validation evidence.
Representable large integers and neutral bridges across the threshold have
separate passing regression coverage. Resolving the precision policy is deferred
to the next scope discussion, rather than changing numeric representation as
part of this integration.
