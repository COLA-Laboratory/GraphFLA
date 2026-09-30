# Reproduction harvest — raw material for the packaging session

This directory collects **one record per study** produced by the acquisition /
reproduction fleet. It is deliberately raw: nothing here is a promoted
benchmark case. The next session packages it.

Scope of this tranche: supplied-corpus landscapes with **more than 1,024
variants**, plus core-literature papers found for metric families the supplied
corpus does not cover.

## What each agent must produce

Exactly two files per study, in `harvest/<StudyKey>/`:

- `record.json` — the structured record, schema below. Machine-readable; this is
  what the packaging session consumes.
- `notes.md` — verbatim quotes with locators, and the reasoning. Human-readable
  backup for anything `record.json` cannot hold.

Source files (PDFs, supplements, author code) go in the study's existing dossier
under `papers/<Dossier>/sources/`, never here.

## record.json schema

```jsonc
{
  "study_key": "BendixsenCOH19",          // bib_key from corpus_registry.json
  "dossier": "papers/Bendixsen2019",
  "recorded_at": "2026-09-30T20:00:00Z",
  "agent": "codex-luna",

  "paper": {
    "title": "...",                        // as printed by the publisher
    "doi": "10.1371/journal.pbio.3000300",
    "year": 2019,
    "bibliography_verified_against": "crossref|publisher_page|pubmed",
    "supplied_bibliography_errors": "..."  // the supplied table is unverified;
                                           // record any mismatch you find
  },

  "sources": [                             // every file you obtained
    {"url": "...", "http_status": 200, "sha256": "...",
     "saved_path": "papers/X/sources/y_codex.pdf", "what": "main text",
     "saved_sha256": "...", "transformation": "none | column subset | gzip"}
  ],
  // `sha256` identifies the DOWNLOADED bytes; `saved_sha256` identifies the file
  // actually retained. They differ whenever a subset or re-encoding was stored,
  // and conflating them makes the record fail byte verification.

  "overlaps": [                            // ONE entry per candidate statistic
    {
      "metric": "n_lo",                    // GraphFLA name, or "none"
      "paper_value": 982,
      "locator": "Results, printed p.9",
      "quote": "The HDV landscape has 982 peaks, while the Ligase ...",
      "printed_or_plotted": "printed",     // NEVER read a value off a plot
      "definition_match": "yes|no|unknown",
      "definition_quote": "Peaks ... were defined as genotypes that were ...",
      "definition_divergence": "...",      // if not "yes", say exactly how
      "input": {"path": "data/BioSequence/Bendixsen2019_hdv.csv",
                "sha256": "...", "n_rows": 16384,
                "preprocessing": "none; fitness column used as-is"},
      "graphfla_call": "DNALandscape().build_from_data(seqs, fitness, epsilon=0); ls.n_lo",
      "graphfla_value": 982,
      "outcome": "reproduced_exact",
      "reason": "..."                      // REQUIRED when outcome is not a
                                           // reproduction; see vocabulary below
    }
  ],

  "blockers": [
    {"what": "Supplementary Table S1", "url": "...", "why": "paywalled",
     "needed_for": "gamma target"}
  ],

  "verdict": "targets_found|no_overlap|blocked|definition_incompatible",
  "verdict_reason": "one paragraph"
}
```

### `outcome` vocabulary (use exactly these)

| value | meaning |
| --- | --- |
| `reproduced_exact` | GraphFLA equals the printed value exactly |
| `reproduced_with_precision` | equal to the printed precision (e.g. paper rounds to 2 dp) |
| `mismatch` | ran successfully, value differs — `reason` must say what differs |
| `definition_incompatible` | the paper's statistic is not GraphFLA's, so no run was made |
| `input_unavailable` | overlap exists but the input could not be rebuilt |
| `not_attempted` | overlap exists, run deferred — `reason` must say why |
| `no_overlap` | searched, and the paper reports nothing this metric could match |
| `regression_snapshot` | matched a value GraphFLA itself produced. Detects drift; **not** scientific evidence |

A `mismatch` or `definition_incompatible` is a **useful result**, not a
failure. A fabricated match is the only real failure.

## Rules every agent follows

0. Paths in `input.path` are **repository-relative** (`data/BioSequence/x.csv`)
   or store-relative (`papers/X/sources/y.csv`). Absolute paths do not resolve in
   another checkout, and these fields are machine-consumed. Records written
   before this rule still carry absolute paths; see `validation/HANDOFF.md`.
1. Never write inside `/Users/arwen/Documents/GitHub/GraphFLA`. Read it freely.
2. Reuse the existing dossier before downloading anything. Never overwrite an
   existing file; add a `_codex` suffix.
3. Every definitional or numerical claim is a verbatim quote with a locator.
   If the paper does not say it, write `NOT STATED IN PAPER`.
4. Never read a value off a plotted point. Plot-only values are recorded as
   `printed_or_plotted: "plotted"` and are not reproduction targets.
5. The supplied bibliography is unverified — at least one entry has an entirely
   wrong author list. Find papers by title and DOI, not by the supplied authors.
6. Record failures with reasons. Silence about a failed comparison is worse
   than the failure.
