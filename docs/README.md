# Documentation development

The active website is `docs/mkdocs.yml` + `docs/content/`. It uses the selected
G styling, class tabs with expandable methods, and continuous analysis articles.
`docs/candidates/` is an archive of design prototypes, not a build dependency.

The current integration covers **all 7 public Landscape classes, 31 Analysis
functions, and all 9 Problems classes** from package revision
`c64262b28bf0f7f547c22010843a7b2a48953f1b`. `profile` has its own page, `analysis/profile.md`; the website intentionally
omits `list_metrics`. Result
types pages and the deleted EE entry are also removed. Analysis results use the
current source's ordinary dictionary contracts.
Landscapes, Analysis and Problems are in the active navigation; other old
template samples are excluded from builds.
Nine dataset tutorials are also available, generated from the canonical
executed notebooks in `tutorials/datasets/`. Their catalog is `docs/tutorials.yml`.
Catalog `title` names the optimization problem in navigation, the tutorial index
and the page heading; `study` identifies the dataset or experimental system beneath
it. The website applies this title hierarchy during rendering and preserves the
executed notebook and its downloads byte for byte.

The site root uses the landing page maintained in `home/`; read
[`home/README.md`](home/README.md) for its content, design tokens and figures.
`content/index.md` selects `overrides/landing.html`, which extends Material's
`main.html` and replaces only the homepage content and footer. The native
header, search, sticky navigation and mobile drawer belong to the same theme
on the homepage, tutorials and API pages. `overrides/partials/tabs.html` places
the shared navigation inside Material's header: Home, Metrics, Case studies,
Tutorials and API reference. Metrics and Case studies link to the corresponding
homepage sections, including from nested documentation pages. The repository
link uses Material's native `repo_url` / `repo_name` slot beside search.

`_support/landing.py` renders the homepage before search indexing, generates
assets under `assets/home/`, and resolves shared navigation URLs for every page.
API pages share one sidebar group containing Landscapes, Problems and Analysis.
Function chips link to exact API anchors, checked during every full site check.
`home/scripts/home.js` preserves old root-level Landscape bookmarks. Page motion
uses browser-native CSS smooth scrolling and respects reduced-motion preferences.
A normal build creates the complete site without a separate homepage build.

The homepage includes five illustrative scenarios: protein engineering, chemistry,
materials, software tuning and hyperparameter search. Compact demo tables and example
profiles live in `home/content.yml`; `home/figures/examples.py` reuses the original
authored three-peak surface. The separately downloadable, reproducible data examples
are described in `content/demo-methods.md`. `home/catalogue.py` generates the Datasets
page and its counts from the repository data collection.

The interactive prediction/optimization panels use real prepared research results
under `home/insights/`, separately from the illustrative workflow demos. They ship
with a pinned local D3 runtime and no server-side prediction or training. See
`home/README.md` for the data boundary and `content/insights-sources.md` for provenance.

The homepage's construction comparison is an independently recorded, bounded
run of `docs/scripts/benchmark_home.py`. Its naive baseline uses Python pairwise
Hamming comparisons and a dense float64 matrix inside the same full construction
pipeline. Each worker is limited to 45 seconds and 768 MiB; input and graph hashes
are compared. `content/benchmarks.md` explains the protocol and all workload
results. The downloaded script and raw JSON live in `content/assets/benchmarks/`.
Do not substitute old-version comparisons or theoretical allocation sizes for
the recorded runtime and peak-process RSS measurements.

Problems retains the original model introductions in two continuous articles:
Biological Models and Combinatorial Problems. `problems-migration.json` records
the original introduction hashes and their destinations. The calling convention
is shared through `includes/problem-interface.md`. The base class lives with the
biological models; all constructor fields, methods and API examples come from
the finalized source.

The six finalized metric introductions were integrated into their existing
topic pages. `migration.json` records their source hashes and destinations.
Introductions without a finalized replacement retain their existing text.
The EE introduction is shared through `includes/ee-mutations.md` at both of its
existing topic positions, using `--8<-- "ee-mutations.md"`; edit that one file
to update both pages. Includes contain narrative only, not API directives.
When displaying the same API in a second article, set `skip_local_inventory:
true` in that directive's options so cross-references retain one primary target.
The removed `higher_order_epistasis` API and obsolete copied examples are gone;
Walsh order summaries and EE functions use the current source. Source-owned
Examples are shown at each API.

## Build, preview, check

From the repository root, using Python 3.13:

```sh
python3.13 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r docs/requirements.txt
.venv-docs/bin/python docs/manage.py serve
```

Open <http://127.0.0.1:8818/>. Saving source docstrings, Markdown, templates or CSS
rebuilds the preview, including homepage copy, styles, icons and templates.
Restart `serve` after editing Python build modules in `docs/_support/` or
`docs/home/`, because MkDocs caches Python modules during a server session.

```sh
.venv-docs/bin/python docs/manage.py build  # strict build into docs/.build/site
.venv-docs/bin/python docs/manage.py check  # contracts, strict build, local links and assets
```

These commands also work from another directory when invoked with the absolute
path to `docs/manage.py`. They do not require installing or importing GraphFLA,
its scientific dependencies, or executing its examples.

The source defaults to **the same checkout as this documentation**. Another
session's package worktree is neither modified nor followed automatically.
After package changes are integrated into this checkout, the next build reads
them. To deliberately preview a different checkout, use:

```sh
.venv-docs/bin/python docs/manage.py serve --source-root /path/to/GraphFLA
```

The path is a repository root containing `graphfla/`; the override is read-only
and applies to that invocation. `GRAPHFLA_DOCS_SOURCE_ROOT` is the equivalent
explicit environment setting. Release builds should use the matching code and
documentation revision, without a source override.

## Author content once

**API article prose** lives in Markdown under `content/` and shared `includes/`. Keep the
existing conceptual thread, equations and integrated tutorial examples there.
The migration retained the existing conceptual text; scientific/content review
remains separate from this infrastructure work.

**API content** lives in the corresponding Python docstrings, in NumPy style.
Add one reference where that API belongs in the article:

```markdown
## Fitness Distance Correlation

The authored explanation stays here.

::: graphfla.analysis.fdc
```

The first paragraph of each function's docstring supplies its short summary
below the signature, followed by See Also. Longer source descriptions are kept
in Notes. This also applies to older pages marked `api_narrative: true`: that
legacy setting no longer suppresses the function's short summary. Authored
topic introductions remain separate and are not rewritten by the renderer.

A class page needs only metadata and its reference:

```markdown
---
title: Landscape
api_class: true
hide: [toc]
---

::: graphfla.landscape.Landscape
```

The template distributes source content across Overview, Constructor, Attributes,
Methods, and Examples & notes. Public methods, inherited methods and documented
properties are discovered automatically. No method lists, parameter copies,
per-function HTML or CSS are required. New standalone functions need a reference
in the appropriate article; adding a new article also needs a MkDocs nav entry.
Tabs without content are omitted, including constructor/method tabs for simple
result classes.

Attributes and properties use one compact name/type line followed by the source
description. They do not repeat the name in a separate signature block or show
the implementation's `property` badge. Source annotations/default values and
all docstring sections remain available; member anchors and cross-references
still use mkdocstrings' native inventory. Both class layouts share this renderer.

For an article containing several classes, set `api_grouped_classes: true` in
its frontmatter and place each `::: graphfla.problems.ClassName` reference after
its authored introduction. The shared inline class template renders a signature,
short summary and See Also, then Parameters, Attributes, Methods, Examples and
Notes tabs as applicable. It does not add a redundant Overview tab or visible
class heading. Inherited methods are discovered by the same renderer used for
standalone classes.

Articles containing multiple APIs automatically become collapsible groups in
the left navigation, with links to each API's anchor. Nested MkDocs nav entries
group the two Problems articles without opening both class lists. This uses
Material's native navigation. The right table of contents retains only authored
section headings; generated API anchors remain available for direct links.
All sidebar icons are configured in `extra.navigation_icons` in `mkdocs.yml`,
using Material's bundled SVGs and native `meta.icon` rendering. Page and group
icons have one central map; new API links automatically receive a class or
function icon from their source kind. No per-object CSS or copied SVGs are needed.
Sidebar labels stay on one line, with an ellipsis when space is insufficient.
The complete name stays in the DOM for accessibility and appears in a native
hover tooltip; neither public names nor anchor targets are shortened. This is
handled by the shared CSS and `api.js`, without per-function label overrides.

Functions and methods use at most five independent tabs: Parameters, Returns,
Examples, Notes, and Raises & Warns. Other Parameters and Receives join the input
panel; Yields joins the output panel. Notes contains the extended description,
Notes, References and any less common sections, retaining their own headings.
Raises and warning sections share a panel. Empty groups are omitted. The
signature, short summary, See Also and deprecation notices stay above the tabs.
Content comes directly from the parsed docstring; authors add no tab markup.
Class and function tabs share `templates/python/material/api/tabs.html.jinja`.
Citation and method deep links reveal the containing tabs automatically.

The shared renderer covers Parameters, Other Parameters, Attributes, Returns,
Yields, Receives, Raises, Warns, Warnings, See Also, Notes, References, Examples,
and deprecation/version notices. Signatures, annotations, argument markers and
defaults come from source via Griffe/mkdocstrings. Standard reST inline roles,
math, citations and notices are adapted centrally for Markdown. This adapter is
not a general Sphinx parser: unsupported directives/section types fail visibly
and should be supported once in the shared layer.

Existing malformed or incomplete docstrings are not repaired by rendering. For
example, a parameter absent from a NumPy Parameters section stays absent, while
an old prose-only docstring remains prose. Improve these in source when that
content work is authorized. Do not compensate by writing a second API contract
in Markdown.

## Shared implementation

| Concern | Single place to edit |
| --- | --- |
| Homepage and global header integration | `_support/landing.py`, `overrides/`, `home/` |
| Material shell styling | `content/assets/site.css` |
| Theme, navigation, global extraction defaults | `mkdocs.yml` |
| G colors, signature panels, field names and borders | `content/assets/api.css` |
| Class tabs and method disclosures | `templates/python/material/class.html.jinja`, `function.html.jinja` |
| Classes within tutorial articles / shared methods | `templates/python/material/api/inline_class.html.jinja`, `api/class_methods.html.jinja` |
| Compact attribute entries in both class layouts | `templates/python/material/attribute.html.jinja`, `api/class_attributes.html.jinja` |
| Shared tab markup / function section tabs | `templates/python/material/api/tabs.html.jinja`, `api/function_sections.html.jinja` |
| Short summary, See Also and notices | `templates/python/material/api/summary.html.jinja` |
| Summary extraction and the five section groups | `_support/function_layout.py` |
| All docstring sections and field layouts | `templates/python/material/docstring.html.jinja`, `api/macros.html.jinja` |
| Common reST presentation compatibility | `_support/markup.py` |
| Source binding, public exports, anchors, provenance | `_support/hooks.py` |
| Expandable article links and authored-only TOC | `_support/navigation.py` |
| Signature highlighting only | `_support/signatures.py` |
| Method filtering and revealing deep-linked tabs | `content/assets/api.js` |
| Notebook pages, navigation catalog and download bundles | `_support/notebooks.py`, `tutorials.yml` |
| Notebook output presentation | `templates/notebook.md.j2`, `content/assets/tutorials.css` |

The templates use mkdocstrings' public template/filter interfaces. Class layout
extends its base class template; native signature extraction/formatting and
Material copy controls remain shared. Export resolution follows Griffe's import
metadata, including functions whose name matches their module. It does not use
a per-function name registry. Material fonts, page width and overall theme are
unchanged.

Direct dependencies are pinned in `requirements.txt`. Upgrade them deliberately
and run `check`; template compatibility is part of the contract. No generated
HTML or extracted API JSON needs to be committed or edited.

## Verification and reports

`check` uses a disposable fixture package that raises if imported. Tests cover
all common sections, inherited members/properties, citations, positional-only
and keyword arguments, failure on missing objects/unsupported directives, and
source-only changes with unchanged authored Markdown. A live-server test also
confirms a docstring edit appears without restarting the server. It then builds
the real site strictly and checks local links, fragments and duplicate IDs.

Examples are **rendered, not executed**, during normal builds. The tests include
an explicitly executed fixture doctest; actual GraphFLA example execution and
scientific correctness belong to package verification. Synchronization alone
does not establish that a docstring describes correct behavior.
The two retained NK tutorial examples were also executed against the integrated
Problems revision during migration; this is not part of the normal site build.

Build outputs include:

- `_api/build-manifest.json`: source Git revision, referenced objects, resolved
  exports, parsed section names, docstring hashes and public API coverage.
- `_api/quality-issues.json`: existing missing/unknown parameter descriptions.
  These are advisory, so content cleanup does not block infrastructure work.

`extra.api_modules` in `mkdocs.yml` names the modules whose complete public
`__all__` must be documented. A missing public API fails the build; keep these
checks enabled when adding or removing exports. `extra.api_omissions` records
the deliberately excluded `list_metrics` with its scope reason. The build
report separates these from missing APIs and rejects stale exclusions or an
excluded API that is still rendered. This is independent of the advisory
content-quality report.

`.github/workflows/docs.yml` runs this documentation check and uploads a preview
artifact when code or docs change in a PR or on main. It does not deploy or
modify the package workflow.

`docs/scripts/migrate_candidate.py` is a one-time migration record from the frozen
G articles. Do not rerun it on edited articles: normal builds never regenerate or
overwrite authored Markdown. Old hand-copied API sections and per-function
examples/references were replaced by source references; integrated tutorial
Examples remain authored content.

`docs/scripts/import_problems.py` likewise records the one-time migration and
split of the original Problems article. It is not a build step: do not rerun it
over subsequently edited tutorials.

## Dataset tutorials

Edit narrative and code only in `tutorials/datasets/*.ipynb`; the accompanying
CSV inputs live in its `data/` directory. `tutorials.yml` is the one catalog of
page slugs, labels, icons and required inputs. It generates Tutorials navigation,
the index and each notebook's download bundle. The nine original introductions
and all other Markdown cells were retained; only Walsh dictionary access and
the current `profile` arguments needed mechanical API updates during migration.
`tutorials-migration.json` records the original hashes and changed code cells.

The build uses nbconvert's Markdown exporter, wrapped by one shared template,
and MkDocs' virtual files. It preserves Markdown, Python source and saved outputs,
then uses Material's code and table styles. Generated pages are never edited or
stored in `content/`. Builds do not execute notebooks or import GraphFLA.
Each page offers the exact rendered `.ipynb` and a ZIP containing that notebook
plus its required data at the original relative paths. No public Colab link is
claimed; this local integration does not publish the package or website.

To refresh outputs in an environment containing this checkout's GraphFLA and
scientific dependencies:

```sh
python -m pip install -r docs/requirements-notebooks.txt
python docs/scripts/execute_tutorials.py
# Or pass one or more notebook filenames to re-run only those tutorials.
```

The runner uses a fresh kernel per notebook, serial execution, one BLAS thread,
and a six-minute/5 GiB process-tree limit. It records source revision and a cell
source hash in the notebook; logs/resource reports go to `.build/notebook-execution/`.
A build fails if cells changed after the last execution, a cell is unexecuted,
an error output remains, or a required input is missing from the catalog. The
generated `_api/tutorial-manifest.json` records notebook/data hashes, code/output
counts and execution provenance. After editing narrative alone, re-run the same
verification workflow so the published source and recorded results stay aligned.
