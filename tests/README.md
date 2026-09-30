# Test suite

Run `python -m pytest`; install dependencies from `requirements-dev.txt`.
The default suite includes the pinned Papkou integration case and requires no
network access. Use `-m 'not integration'` for small development checks.

| Tests | Contract |
|---|---|
| `test_construction_oracles.py` | Independent all-pairs distance oracle across classes, directions, epsilon, sparsity and strategies; forced lookup, search, overflow, chunking and hash-collision paths |
| `test_build_contracts.py` | Input types, numeric fitness overflow, Boolean aliases, field-name collisions, indices, thresholds, neutral graphs, genetic-background trimming and GraphML |
| `test_papkou_construction.py` | Full author input to exact author node/edge identities; original fitness and directed weights |
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
