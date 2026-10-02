# EE tests and validation audit

Scope: the finalized general EE API and its internal statistical calculation.
The scientific definition and public signatures are unchanged in this round.
Reference: Wagner, A. (2023), *Evolvability-enhancing mutations in the fitness
landscapes of an RNA and a protein*, Nature Communications 14, 3624,
[doi:10.1038/s41467-023-39321-8](https://doi.org/10.1038/s41467-023-39321-8).
See [the scientific review](EE_MUTATIONS_REVIEW.md) for the definition audit.

## Basic tests

`tests/test_ee_mutations.py` uses bounded synthetic landscapes and a literal
per-mutation SciPy oracle in `validation/oracles/ee.py`. The oracle imports no
production helper and loads no datasets. Moving it out of the empirical driver
keeps small tests independent of paper fixtures.

| Contract | Coverage retained or added |
| --- | --- |
| EE criterion and denominator | Beneficial, deleterious and neutral cases; strict excess; all ordered pairs, including untestable ones |
| Neighborhood definition | Exclude the entire focal position in multiallelic landscapes; sparse/unequal neighborhoods; one-position empty neighborhoods |
| Inference | Both endpoint variances; independent p-values; zero variance; insufficient samples; BH ranks, ties and untestable family members |
| Public API | Scalar/table agreement, effect filters without changing testing families, nullable schema, labels and joins, empty results |
| Invariance | Optimization direction, scale/offset, row and edge orientation/order, lazy-cache independence |
| Validation/migration | Invalid FDR/type/epsilon, missing/duplicate configurations, unbuilt inputs, shape/variance alignment, legacy warning and behavior |
| Optimized moments | Forced tiny blocks across wide/punctured inputs; preserving small non-focal variance next to a huge excluded focal effect |

The invalid-FDR Cartesian product across both wrappers was replaced by one
parameterized validator check plus representative rejection through each public
wrapper. The duplicate additive-zero check in `test_metrics.py` was removed:
stronger direct EE tests and the retained additive golden regression cover it.
Profile migration/selector tests remain in `test_profile.py`. No unrelated
metric's scientific assertions were removed or weakened.

An initial stress test demanded near-bitwise p-value equality after subtracting
means near 1e10. It failed because those tiny differences are ill-conditioned in
float64 (and the oracle used uncentered means). It was replaced by two meaningful
checks: tight p-value/decision agreement at ordinary scale, and an independent
local variance calculation in the cancellation-sensitive case. The latter catches
the unsafe optimization of subtracting large focal second moments from totals.
No literature tolerance was widened.

## Literature tests and scope of claims

The permanent [contract](TESTING.md), [templates](templates/) and
`validation.testing` extend the existing immutable case/event schema. The new
suite is opt-in, offline, filterable by study/case, and independently scheduled
in CI. Shared tests exercise false evidence claims, missing citations, unknown
IDs, filter intersection, JUnit identity, altered/missing inputs and incorrect
targets. Input verification occurs before module fixtures compute results.

Wagner has separate author-replay and general-estimator tests for each dataset.
Author replay checks all 456,448 signed significance flags and the four printed
counts. General tests check the public effects table's pair identities, raw
p-values, missingness and every EE decision against independent enumeration,
then compare actual public counts with new independent-equation cases. The
public fraction is checked too. The private RNA measurement-error calculation
remains an additional research check; it is not exposed as the public API.

| Dataset | Input configurations | Ordered pairs | Author beneficial/deleterious | General beneficial/deleterious/neutral |
| --- | ---: | ---: | ---: | ---: |
| Protein | 7,882 | 175,552 | 681 / 221 | 196 / 389 / 0 |
| RNA | 4,176 | 52,672 | 2,983 / 3,702 | 2 / 0 / 0 |

Existing Lyons and other empirical tests moved into the dedicated suite without
changing their scientific assertions. The full Papkou author-graph test now has
an explicit source-backed case in addition to its original exact edge/fitness
checks. Existing case files, input locations and historical fingerprints were
preserved. Old `pytest_node` paths in frozen cases remain historical; markers
supply the current test mapping. All 123 repository/external artifact references
and the existing 25 research events pass contract verification.

## Execution and resource review

```sh
python -m pytest
python -m validation.tests --literature-study Wagner2023 -q
python -m validation.tests -q --junitxml=literature.xml --durations=10
python -m validation --store /path/to/research-store check --artifacts
```

The first complete migrated literature run passed 19 tests in 53.20 seconds on
macOS arm64 / Python 3.13.11, with process peak RSS 1,748,402,176 bytes (1.63 GiB).
Papkou classification was the slowest call (21.50 s); complete Wagner reproduction
calls took 11.86 s and 3.70 s. These are execution observations, not cross-version
performance benchmarks. Large existing complete-population studies remain out of
the basic suite; CI allows ten minutes for the independent literature job.
The fixtures are unchanged; Wagner's input/author extracts occupy about 1.2 MB.
The full Papkou construction case retains 261,333 input observations and compares
135,178 vertices / 324,044 edges after the published filtering.

EE performance measurements use only 64–1,024 input configurations and run
separately from all correctness work. The vectorized implementation retains
centered two-pass variances and bounds temporary neighbor blocks. See
[the performance protocol and results](../benchmarks/EE_RESULTS.md).

Final verification after the public-table checks and harness revisions:
**2,027 basic tests passed; two pre-existing strict xfails remain**. The EE
statistical kernel has 135/135 covered statements and 32/32 covered branches.
The separate literature suite passed all 19 tests in 50.17 s; this run's process
peak RSS was 2,194,407,424 bytes (2.04 GiB). The variation from the earlier RSS
observation is retained rather than reporting only the smaller value. The final
run overlapped other correctness checks; it is not a timing benchmark. EE and
gamma ASV selectors both passed their bounded smoke runs; discovery contains
99 concrete benchmarks and no leaked base-class entries. Fatal-error lint,
Python 3.9 syntax checks for the new infrastructure, and diff checks passed.
Only Python 3.13 was executed locally; the existing CI matrix covers 3.9/3.11/3.13.
