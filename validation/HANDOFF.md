# Validation handoff

Read this before doing any validation work. It records what two sessions
established, where the material lives, and the contract that lets a later
session **add** validations instead of restarting or redoing them.

Nothing here is a finished benchmark. It is reviewed raw material.

---

## 1. Where things live

| What | Where | Committed? |
| --- | --- | --- |
| Contract tooling, catalog, metric registry, versioned cases | `validation/` | yes |
| Per-study reproduction records (the integration material) | `validation/harvest/` | yes |
| Scope ledger and definition of done | `validation/harvest/LEDGER.md` | yes |
| Metric-vs-paper definition findings | `validation/metric_provenance_findings.md` | yes |
| Function-by-function validation plan | `tests/VALIDATION.md` | yes |
| Promoted empirical tests and fixtures | `tests/test_literature.py`, `tests/fixtures/` | yes |
| **Source PDFs, supplements, author code, per-study dossiers** | `~/Documents/GraphFLA-validation/2026-09-30/papers/` | **no — 1.7 GB, external by design** |
| Append-only research event store | `~/Documents/GraphFLA-validation/2026-09-30/events/` | no |

The external store is the source of truth for provenance: it holds the
downloaded artifacts, their SHA-256 hashes, and the append-only event log. The
repository holds the distilled records. Do not duplicate the 1.7 GB into git.

## 2. The contract — how to add a validation without redoing work

### Before touching anything

```bash
python -m validation check --artifacts
python -m validation --store ~/Documents/GraphFLA-validation/2026-09-30 queue
```

`check` verifies every fixture against its recorded hash.

**`queue` is stale and will tell you to redo finished work.** It reports Moulana
2022, Bendixsen, Poelwijk, Lite, Domingo and Baeza-Centurion as `queued` and
tells you to acquire sources before experimenting; all six are done or closed.
Ferretti's checkpoint still reads `blocked_artifact` although its csI half
reproduced. The second session recorded its conclusions in the harvest records
but did not write matching checkpoint events back to the store.

**`validation/harvest/LEDGER.md` is authoritative for what is done, not the
queue.** Reconcile before relying on `queue`: for each LEDGER row that is `done`
or `closed`, append a checkpoint event to the store, then re-run `queue` and
confirm it is empty of those studies.

### The three gates

A number is only scientific evidence if it passes all three, separately:

1. **Synthetic correctness** — hand-computable or analytically known case.
2. **Theory agreement** — matches the published *definition*, quoted verbatim.
3. **Published reproduction** — matches a published *number* on the published input.

Passing (1) tells you nothing about (3). A saved GraphFLA output is a regression
snapshot, never evidence for (2) or (3).

### Adding a study

1. Read its dossier in the external store first. Never re-acquire what is there.
2. Record the expected value, its locator and a verbatim quote **before** running
   GraphFLA. Deciding what counts as a match afterwards is how fabricated
   reproductions happen.
3. Write `validation/harvest/<StudyKey>/record.json` per
   `validation/harvest/README.md`, one `overlaps` entry per candidate statistic.
4. Only then promote to `validation/cases/` and a test in `tests/test_literature.py`.

### Outcome vocabulary

`reproduced_exact`, `reproduced_with_precision`, `mismatch`,
`definition_incompatible`, `input_unavailable`, `not_attempted`, `no_overlap`,
`regression_snapshot` (matched a number GraphFLA itself produced — drift
detection, never scientific evidence).
A `mismatch` or `definition_incompatible` with a stated reason is a **useful
result**. A fabricated match is the only real failure.

### Running the suite

```bash
rtk proxy "python -m pytest tests/ -q"
```

`rtk proxy` is required: a shell hook rewrites a bare `python -m pytest` in a way
that breaks `validation` imports and reports "no tests collected".

Agents must run GraphFLA as
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<repo> python script.py` from outside the
repository, or Python writes `__pycache__` into it.

## 3. Traps that have already cost time

- **Circularity.** *Huang et al. 2025, "Augmenting Biological Fitness Prediction
  Benchmarks with Landscape Features from GraphFLA"* reports landscape statistics
  that **GraphFLA itself produced**. Reproducing its Tables A2–A4 is a regression
  check, not validation. 15 of the 20 comparisons in
  `harvest/CoreLit_Epistasis/` are of this kind and are marked accordingly. The
  same applies to the supplied landscape-statistics table: it is a paper
  *discovery index*, never a source of expected values.
- **Supplied bibliography is unverified.** `KuoJCCLHWC20` has an entirely wrong
  author list; `MoulanaDP22` is truncated to 3 of 10 authors. Find papers by
  title and DOI.
- **`priority_size` is unreliable.** `corpus_registry.json` records `0` where the
  size was never parsed, so a mechanical `> 1024` filter silently drops studies.
  `DomingoDL18` is really 5,184. Resolve size from
  `landscapes[].theoretical_size_expression`.
- **Plot-only values are not targets.** Several studies print nothing and only
  plot the statistic. Digitising a plot converts an honest "no target" into a
  false-confidence one. Look for publisher Source Data instead.
- **NaN is not censoring.** In the Moulana inputs a NaN fitness mixes QC failures
  with true non-binders. Any promoted case must say which population it describes.
- **Directed graph.** GraphFLA's graph holds improving edges only. A paper
  counting undirected mutation pairs has roughly twice the edge count.
- **A cited DOI can be fabricated.** One docstring cited a Nature Reviews
  Genetics DOI that returns 404 and is unknown to Crossref. Verify citations.

## 4. State at handoff

Scope of the second session: **supplied-corpus** landscapes with **more than
1,024 variants**, plus core literature for metric families the corpus does not
cover. No supplied-corpus study at or below 1,024 variants was worked. The
cutoff scopes the supplied corpus only — it does not exclude a core-literature
paper whose landscape is small, and the Ferretti csI reproduction below uses 32
genotypes. Note that `harvest/CoreLit_Epistasis/` reached its negative search
conclusions under the cutoff anyway; those searches should be redone unrestricted
before any metric family is declared to have no published origin.

19 in-scope studies: **9 done, 6 closed, 1 blocked, 3 partially open**. Full
table and per-study reasons in `validation/harvest/LEDGER.md`.

Reproductions established and independently re-verified by the lead:

| Study | Result |
| --- | --- |
| Papkou 2023 | 514 peaks; 740,211 motifs incl. 85,203 reciprocal-sign |
| Westmann 2024 | 2,092 peaks (58 above WT, 2,034 below) |
| Wu 2016 | 30 peaks, 15 above WT; 92.638% vs published 93% |
| Johnston 2024 | 520 peaks among author-designated active candidates |
| Bendixsen 2019 | HDV 982 and ligase 68 peaks, exact |
| Poelwijk 2019 | order-2 R² 0.87335 vs printed 0.87 |
| Moulana 2022 | 0 monotone fitness-increasing Wuhan → BA.1 paths |
| **Ferretti 2016, A. niger csI** | **peaks 4, sinks 2, γ 0.3273→0.33, r/s 0.8928→0.89** |

The Ferretti csI row is the first independent empirical reproduction of **γ** and
**r/s**, the two metrics previously flagged as least evidenced. Its γ* differs
(0.2692 vs 0.25), plausibly because Ferretti's γ* carries an ε-tolerance
trichotomy that GraphFLA's plain `np.sign` does not implement — check before
treating γ* as reproducible.

The navigability family is, on current literature, largely **unvalidatable**: of
17 candidate comparisons, 8 were definition-incompatible, 4 had no overlap and 3
had unavailable inputs. Accessibility and path-length definitions differ between
essentially every paper.

## 4b. Known defects in the harvest records

An external review checked these records. Defects that make a **machine-readable
outcome misleading were repaired**, because `record.json` is consumed by code: in
every case the original claim is preserved in `original_agent_claim` and the
correction is explained in the entry's `reason`. Defects that only affect
presentation or portability are **recorded, not repaired**, and listed here for
the packaging session.

Repaired: two over-generous outcomes in the Moulana records are now `mismatch`;
ten circular comparisons in `CoreLit_Epistasis` are now `regression_snapshot`;
`MoulanaDPCRGSBD23` carries an authoritative `lead_correction.population_description`.
A consumer filtering on `reproduced_*` now receives only defensible reproductions.

Still outstanding:

- **Absolute paths.** Many `input.path` values are absolute
  (`/Users/arwen/Documents/GitHub/GraphFLA/...`) and will not resolve in another
  checkout. `harvest/README.md` now requires repository-relative paths; existing
  records predate that rule.
- **Hash semantics.** Several `sources[]` entries pair the hash of the
  *downloaded* file with the path of a *transformed* file (a column subset or a
  re-encoding), so byte verification fails. All four such entries in
  `MoulanaDPCRGSBD23` fail; `MoulanaDP22` has the same issue with the correct
  hash given only in prose. The schema now separates `sha256` from
  `saved_sha256`.
- **`notes.md` still reads as if the circular matches were reproductions.** The
  warning and the corrected outcome live in `record.json`; the prose companion in
  `CoreLit_Epistasis/` was not rewritten.
- **`definition_match` is overloaded.** Some entries use `"no"` to mean "GraphFLA
  has no metric for this quantity" rather than "the definitions differ".
  `MoulanaDP22` entry 0 now says `no_graphfla_metric` instead; others were left.
  Read `definition_divergence` before acting on `definition_match`.

## 5. Open items, in priority order

1. **Two metrics do not implement the papers they cite.** Details and evidence in
   `validation/metric_provenance_findings.md`. Deliberately unfixed — fixing them
   changes published metric values and needs a scope decision.
   - `evolvability_enhancing_mutations` tests `Δ mean-neighbour-fitness > 0`,
     which Wagner 2023 explicitly rejects as trivially satisfied under
     additivity; he requires `> max(0, Δw)`. On additive OneMax(4) GraphFLA
     returns 1.0 where Wagner's criterion gives 0. Its cited DOI does not exist.
   - `idiosyncratic_index` uses the analytic baseline `√2·σ(f)` where Lyons 2020
     resamples `n` genotype pairs matched to each mutation's background count.
2. **API gap.** Moulana 2022's headline result is the number of monotone
   fitness-increasing paths between two named genotypes. GraphFLA exposes no
   public metric for it; the reproduction had to use `ls.graph.subcomponent`.
3. **Data defect.** `data/BioSequence/Weinreich2006Tan2011_Weinreich2006.csv`
   stores fitness `15.32333629` for the all-ones genotype — 2^15.3233 = 41,000,
   where the Weinreich 2006 source MIC table prints 4,100. Ten-fold, on the
   global optimum. The other 31 rows were not contradicted.
4. **Blocked acquisition.** Podgornaia & Laub 2015 (PhoQ, Science) supplementary
   material is behind a robot check and needs manual download.
5. **Unresolved.** The TEM/β-lactamase half of Ferretti Figure 4c does not
   reproduce under ln, log10 or log2 of the averaged source MICs. The Weinreich
   Supporting Online Material and the MAGELLAN `.fl` inputs remain blockers.
6. **γ\* ε-tolerance**, per section 4.

## 6. Production changes made in these sessions

Four confirmed defects were fixed and are covered by tests; a review then found
two further defects in those fixes, also fixed. See `CHANGELOG.md`. Full suite:
**1565 passed, 0 failed**. Two of the fixes change returned values
(`single_mutation_effects` / `all_mutation_effects` sign, and `r_s_ratio` on
constant fitness), so they are release-noted as breaking.
