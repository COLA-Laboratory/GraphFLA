# EE mutations: definition, author replay and corrected calculation

Reviewed 2026-10-02. Scope: Wagner (2023), *Evolvability-enhancing mutations in
the fitness landscapes of an RNA and a protein*,
[doi:10.1038/s41467-023-39321-8](https://doi.org/10.1038/s41467-023-39321-8).
The user clarified that this task concerns one paper, not the four papers from
the preceding idiosyncrasy review. Public API redesign remains a separate step.

## Sources and definition

The main text, Supplementary Methods (pp. 2–4), Figures S1/S2 and
[author repository](https://github.com/andreas-wagner-uzh/EE_mutations/tree/7b3071ea1dec1b91818bea69cdf6ccf219f47877)
were examined. The repository supplies `EE_funcs.py`, `prot_EEmutations.py`,
`RNA_EEmutations.py`, both per-mutation output tables, and the two adaptive-walk
notebooks that read those tables. Fixture provenance pins this commit and the
source hashes. These are two experimental landscapes within one study, not
two independent papers.

For each directed mutation from u to v at position j, exclude **every** change
at j from the neighborhoods of both endpoints. With delta_f = f(v)-f(u) and
delta_mean = mean(N_v)-mean(N_u), Supplementary Eq. S2 requires:

```
delta_mean > max(0, delta_f)
```

The beneficial case therefore requires more than merely increasing neighbor
fitness. The old GraphFLA test `delta_mean_neighbor_fit > epsilon` used
unrestricted means, omitted significance testing, and considered only stored
improving edges. On additive OneMax(4) it gave 1; the corrected definition gives
0. The old Nature Reviews Genetics citation was also incorrect and is removed.

The paper uses two-sided one-sample t tests and Benjamini–Hochberg FDR 0.01.
Author code applies separate corrections to the delta_f null and the zero null,
each over **all ordered pairs**, then selects positive-direction rejections.
The text describes beneficial/deleterious tests separately; with symmetric
variances on these non-tied data, testing each class or both orientations yields
the same BH thresholds because reverse p-values coincide.

## Reproduction findings: a published-code defect matters

The Methods describes the difference variance as V_u + V_v. Both released
scripts actually use V_v + V_v. This breaks the symmetry of the statistical
test under reversal. The protein helper supplies population neighborhood
variance (`ddof=0`). The RNA helper instead propagates experimental variances:
the script sets variance_i = 6*SE_i**2 and supplies sum(variance_i)/k**2 for
each neighborhood. Those are different uncertainty models.

The scripts pass n=min(k_u,k_v) to SciPy, producing df=n-1. The Methods calls
min(k_u,k_v) the degrees of freedom. We use the conventional one-sample df=n-1,
consistent with the executable scripts. The variance expression in the text
also has a squared-symbol/square-root typesetting inconsistency; the intended
sum of the two endpoint variances is explicit in the preceding argument.

The notebooks classify beneficial/deleterious/neutral effects **after** reading
the output tables, whose effects were formatted to four decimals. Thus the
reported 38 protein and 34 RNA neutral ordered pairs are rounding-created;
there are no exactly equal neighbor fitnesses in the recovered full-precision
inputs. We reproduce this output-table step explicitly, not by rounding the
production estimator.

| Landscape | Ordered pairs | Published beneficial EE | Author replay | Published deleterious EE | Author replay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Protein: 7,882 ParD3 variants | 175,552 | 681 (0.39%) | 681 | 221 (0.13%) | 221 |
| RNA: 4,176 tRNA variants | 52,672 | 2,983 (5.7%) | 2,983 | 3,702 (7.0%) | 3,702 |

Direct substitution enumeration reconstructs both full neighbor sets. Protein
substitutions are restricted to amino-acid pairs connected by at least one
single-nucleotide codon change, as the Methods requires. All 456,448 signed
test decisions (two tests for each of 228,224 ordered pairs) match the author
tables exactly. Before four-decimal classification, the same author procedure
gives protein 681/222 and RNA 2,990/3,705 beneficial/deleterious counts.

One RNA pair, GCAUAUCGGC ↔ GCAUCUCGGC, differs at the last printed effect digit:
our recovered magnitude is 0.14295000039940747, which formats as 0.1430, while
the table records 0.1429. This is about 4e-10 above a rounding boundary. The
exact input-file precision used by the author is not available; the cause is
not asserted. Both effect signs and all statistical decisions match. The
validation reports these two rows explicitly rather than claiming a bytewise
reproduction of the complete output files.

## Production calculation and its independent checks

Production uses V_u+V_v, never the duplicated-target expression. It keeps the
paper-code conventions of population variances, df=min(k_u,k_v)-1 and separate
BH families. Rejection must also satisfy Eq. S2 with a nonnegative effect-size
epsilon and a machine-roundoff guard. Zero difference and zero variance mean
no rejection; a nonzero difference with zero variance uses the limiting p=0.
At least two non-focal neighbors per endpoint are needed for a test. Untestable
pairs stay in the denominator and BH family (as nonrejections); if no pair is
testable, the scalar is NaN with a warning.

The independent oracle directly collects neighborhoods per directed mutation
and invokes SciPy's summary-statistic t test; production shares node/position
moments and uses the t survival function. Full per-pair p-values and all final
EE decisions agree for both studies under the symmetric variance formula.

| Calculation | Protein beneficial/deleterious | RNA beneficial/deleterious |
| --- | ---: | ---: |
| Author procedure, full precision | 681 / 222 | 2,990 / 3,705 |
| Symmetric variance, study-specific uncertainty | 196 / 389 | 2,583 / 3,535 |
| Current public API: neighborhood variation only | 196 / 389 | 2 / 0 |

The RNA difference is substantial: experimental measurement uncertainty cannot
be replaced silently by variation among neighbors. The private numeric kernel
accepts variances for validation, but the public API does not yet expose that
input. Its RNA result is verified against its own stated model, **not** called
a reproduction of the RNA empirical EE fraction.

The unchanged scalar return type now aggregates all three EE effect classes
over all represented ordered pairs. On the study graphs this is 585/175552 =
0.003332345971563981 for protein and 2/52672 = 0.00003797083839611178 for RNA
under the public neighborhood-variation model. These totals must not be compared
to the paper's beneficial-only percentages.

The public function follows the supplied graph and retained neutral adjacency,
deduplicating reciprocal graph edges. It rejects multi-site edges, negates
fitness for minimization, and is independent of unrestricted neighbor-mean
cache values. The existing `auto_calculate` cache-preparation behavior remains
for compatibility. Construction has not changed: dropped configurations or
discarded neutral pairs cannot be recovered, and the ordinary protein builder
does not impose Wagner's codon restriction. The reproduction imports the exact
study graph via the existing GraphML API.

Synthetic checks cover additivity, multiallelic site exclusion, unequal
neighborhoods, beneficial/deleterious/neutral mutations, denominator and missing
tests, FDR, zero variance, minimization, affine fitness transforms, relabeling,
reciprocal storage, cache independence and invalid inputs. All 30 golden-catalog
EE references were checked against the independent oracle before updating only
that metric's snapshots.

## Re-run and remaining API discussion

```
python -m validation.ee_mutations
pytest tests/test_ee_mutations.py tests/test_ee_literature.py -q
```

Inputs and compact author-output fixtures are hash-pinned and run offline.
The full reproduction report is `validation/ee_mutations_results.json`.
Matching headline numbers alone is insufficient: here it would preserve a
published implementation defect. The author replay is a validation tool, not
an optional production mode.

Proposals, not implemented interfaces: expose uncertainty data with explicit
SE/SD/variance and replicate-count semantics; return class counts, denominators
and per-mutation p-values; make FDR and neighborhood completeness explicit;
retire the unrelated `auto_calculate` dependency. The current fix neither adds
parameters nor changes the result type. Naming and result-schema decisions
remain for the user discussion.
