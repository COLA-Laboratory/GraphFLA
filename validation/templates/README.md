# Add a literature test

Copy `case.json.template` into `validation/cases/<case-id>.json` and
`test_metric.py.template` into `validation/tests/test_<metric>.py`. Replace every
placeholder, implement the input adapter and public call, choose the correct
evidence role, and complete the [contract](../TESTING.md) before promotion.
Templates are deliberately not collected or registered as runnable cases.

Keep acquisition separate. The fixture provenance document should contain:

- Full paper reference, target locator and author repository/archive version.
- Upstream URL, retrieval date, raw filename/bytes/SHA-256 and license/terms.
- Derived filenames/bytes/SHA-256, conversion script and exact command.
- Population size, column meanings/units and the calculation conventions.
- Independent origin of expected values and tolerance justification.
- Typical runtime/peak memory and the command/environment used to measure them.

Use `input_paths` / `literature_inputs` instead of a hard-coded external path.
Do not paste new GraphFLA results into the template as expected values. Without
an independent target, retain a research checkpoint rather than a dummy test.
The existing EE tests illustrate author replay and independent checks; the
Lyons and empirical modules include paper-result comparisons.

For full-population studies with separate paper replay and public-estimator conventions, start with `test_metric_details.py.template`. Share verified input, compare per-item identities/values explicitly, and keep the two evidence roles separate.

For pooled ratios/correlations, `test_pooled_metric.py.template` checks group
numerators and denominators as well as the scalar. Gamma illustrates this with
all ordered position pairs and a known fourfold orientation multiplicity.
Shared inputs are calculated once, while every selected case retains its own
hash verification and evidence role. Acceptance of a remaining reproduction
gap does not authorize widening frozen paper tolerances or replacing targets.

Use `test_additive_metric.py.template` for fitted statistics such as r/s. It
adds explicit encoding, rank, coefficient and residual checks to the existing
case contract, and separates author regularization from public OLS behavior.

For transform coefficients and regularized models, use
[test_coefficient_model.py.template](test_coefficient_model.py.template).
It separates printed coefficients, direct-transform oracles, author replay and
regularized estimates, with explicit intercept/basis/CV conventions.
