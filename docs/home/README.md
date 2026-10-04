# Landing page

Source of the GraphFLA landing page and of the design system the rest of the site
should follow. The main MkDocs build uses these sources for the site root;
`build.py` also renders a standalone design preview.

```sh
.venv-docs/bin/python docs/home/build.py          # writes docs/home/dist/
open docs/home/dist/index.html                    # the landing page
open docs/home/dist/styleguide.html               # the design system, rendered
.venv-docs/bin/python -m unittest docs/tests/test_home.py
```

## Where things live

| To change | Edit | Notes |
|---|---|---|
| A colour, font, type size, spacing step, radius | `styles/tokens.css` | The only file with colour values. Figures pick the change up on the next build. |
| How a component looks (button, card, chip, legend, chart…) | `styles/components.css` | |
| How a section is laid out | `styles/sections.css` | |
| Wording, links and illustrative demo tables/profiles | `content.yml` | Inline markup: `*accent*`, `` `code` ``. |
| The markup of a component | `templates/macros.html` | One macro per component. |
| The markup or order of a section | `templates/sections/*.html`, `templates/home.html` | |
| A case-study card | `content.yml` → `cases.stories`, `docs/tutorials.yml`, `icons/cases/<slug>.svg` | Use the optimization problem as the title, with the dataset or study context above it. |
| The benchmark numbers and charts | `docs/content/assets/benchmarks/construction.json` | Re-record with `docs/scripts/benchmark_home.py`; never typed by hand. |
| A landscape drawing | `figures/scenes.py` | Shapes in `figures/surfaces.py`, line style in `figures/terrain.py`. |
| The five interactive illustrations | `content.yml`, `figures/examples.py` | Keep the original `surfaces.workflow` three-peak shape, compact tables and profile bars. Do not replace them with real-data projections. |
| The dataset catalogue | `catalogue.py` | Counts and links come from repository CSV files and data cards. |
| Research scatter plots | `insights/`, `insight_data.py`, `scripts/insights.js`, `styles/insights.css` | Real prepared benchmark data, independent of the illustrative How it works demos. |

`page.py` joins the copy with the data it describes (tutorial catalog, benchmark
record, the worked example) and `build.py` renders the templates.

## Design system

`dist/styleguide.html` shows every token and component. The rules behind it:

**Colour**
- One page colour (`--gfl-color-page`) and one raised surface (`--gfl-color-card`).
  A card may contain a darker inset (`--gfl-color-well`); cards never nest.
- One accent. It marks the best variant in a figure and the primary action, and
  nothing else. A second highlight colour is not available.
- Four text levels, `text` to `text-faint`. Body copy on a card is `text-muted`;
  `text-faint` is for prompts, comments and fixed residues only.
- Lines are white at low opacity (`line`, `line-strong`) so they work on both surfaces.

**Type**
- Bricolage Grotesque for headings and large numbers, IBM Plex Sans for text,
  IBM Plex Mono for code and measured values.
- Sizes come from the scale `--gfl-text-2xs` … `--gfl-text-lg` and the three
  heading sizes. No other font sizes.

**Space and shape**
- Spacing is a multiple of 4px, taken from `--gfl-space-*`.
- Three radii: `sm` for chips and tags, `md` for controls, `lg` for cards.

**Figures**
- Line art only: contour lines plus fall lines down the slopes. No fills, no
  lighting, no colour ramps.
- Markers and walks are white; the one that reaches the global optimum is the accent.
- Text never goes inside the SVG. A scene registers label positions
  (`Figure.label`) and the page lays the wording from `content.yml` over the image.

## Conventions the tests enforce

`docs/tests/test_home.py` fails when:

- a colour appears outside `tokens.css`, or a template sets a style inline
  (inline custom properties such as `--x` or `--value` carry data and are allowed);
- a `var(--gfl-…)` reference has no definition;
- a `gfl-` class is used without a style rule, or a style rule is used by no page;
- a function or class named on the page is not in the package's public API;
- a number on the page disagrees with its source: headline counts, scenario input
  hashes, plotted values, actual neighbor pairs, and case-study table sizes.
  The quick start loads and analyzes a tabular reaction dataset when GraphFLA's
  runtime dependencies are installed.

## Extending

**A new case study** needs an optimization-problem title, study subtitle, short description and count
unit in `cases.stories`,
an entry in `docs/tutorials.yml`, and `icons/cases/<slug>.svg` on the 48×48 grid of the others (`class="gfl-icon"`, parts
marked with the `gfl-icon__*` classes).

**A new section**: add its copy to `content.yml`, a partial in
`templates/sections/`, one `{% include %}` in `templates/home.html`, and its layout
in `styles/sections.css`. Build it from the macros in `templates/macros.html`
before writing new markup.

**A new figure**: write a scene in `figures/scenes.py` that returns a `Figure`,
register it in `SCENES` with the surface it sits on (`page` or `card`), and
reference it from `content.yml` as `figure: {name: …, alt: …}`.

**A new component**: one macro in `templates/macros.html`, one block in
`styles/components.css`, and an example in `templates/styleguide.html`.

## Website integration

Run `.venv-docs/bin/python docs/manage.py serve` for the complete site at
<http://127.0.0.1:8818/>, or `docs/manage.py check` to validate it. The standalone
preview above is for design work; use the full site to follow tutorial/API links.

`content/index.md` selects `overrides/landing.html`, which extends Material's
`main.html`, following [Material's homepage template](https://github.com/squidfunk/mkdocs-material/blob/master/src/overrides/home.html).
The native header, search and mobile drawer stay available. The shared global
navigation is rendered from `templates/sections/header.html` inside Material's
sticky tabs slot on every page; only the homepage body uses `templates/home.html`.
`templates/standalone.html` remains a design-preview shell, not the website shell.

The `_support/landing.py` hook calls the shared `build_assets` / `page.load_page`
path, registers generated assets and renders the homepage before search indexing.
The four stylesheets and shared menu/copy script are loaded by MkDocs. API and
tutorial layout stays in Material, with the small shell adjustments in
`content/assets/site.css`. No generated sources need to be committed or copied.
Restart `serve` after changes to Python build modules.

Navigation links, exact function anchors and the four research references live in
`content.yml`. Case titles name the optimization problem; subtitles identify the dataset or
study context. Descriptions explain the choices and objective in a concise
academic style. The footer count is generated from the first CSV in each
tutorial catalog entry (the configuration table; later files are lookups).
Set `data_has_header: false` for a headerless table, as used by perovskites.
Metrics and Case studies are plain links to homepage sections. Scrolling uses
CSS `scroll-behavior`, with `prefers-reduced-motion` respected. The GitHub link
beside search is configured through Material's native repository settings.

The quick start loads `experiments.csv`, selects variable and response columns,
builds a general Landscape and calls profile. It is not restricted to sequences.

## Homepage illustrations and reproducible examples

The homepage switcher uses deliberately illustrative data and profiles, labeled
as such. Tables show recognizable choices: amino-acid substitutions, catalysts and
solvents and temperatures, three-component alloy proportions, named On/Off build
settings and three random-forest parameters. Use at least three variable choices
per scenario (three mutable sites for proteins); compact tabs use an underline,
not large bordered cards. Their purpose is to explain the workflow. Use the original authored
three-peak surface with small shifts between scenes and a sparse projected mesh;
the user explicitly rejected interpolated experimental surfaces and dense
force-directed graphs. Keep the table rows compact and retain the numeric bars.
The diagrams and example profiles are not computed from the short demo tables.

Run `.venv-ci39/bin/python docs/home/prepare_scenarios.py` in the existing scientific
environment to reproduce the separately downloadable data examples. It uses measured GB1, reaction, alloy and LLVM
data, plus a bounded random-forest grid evaluated on scikit-learn's diabetes data.
The CSVs, source hashes, seeds, graph coordinates and computed features are saved
under `examples/`. This preparation is single-threaded and time bounded; ordinary
builds never fit models. Read `content/demo-methods.md` for sampling, normalization,
neighborhood and neutrality-tolerance conventions for those archived computations.
They are not the data source for the homepage illustrations.

## Interactive research results

The two research panels load prepared JSON from `insights/proteingym/data.json`
and `insights/evolution/data.json`. `insight_data.py` validates the numeric ranges,
unique dataset IDs and feature/model labels before packaging a small browser
payload. Partial preparation files are excluded from the main website; isolated
development previews can request them explicitly. Never use the synthetic How it
works values or UI test fixtures in these plots.

`scripts/insights.js` uses the locally bundled D3 7.9.0 distribution (ISC license
in `vendor/D3-LICENSE`) for scales, axes and motion. Each plot has independent
feature/outcome selectors, an ordinary least-squares line over finite paired
observations, and accessible citation cards. Tom Select provides compact searchable
menus; Floating UI keeps study cards beside their points and flips them at boundaries.
All three libraries are pinned and bundled locally. No confidence band is drawn. Missing
results are omitted for that selection, with the displayed dataset count updated.
The source page is `content/insights-sources.md`; preparation and provenance belong
there and in the data files, not in long homepage notes. Data work is an explicit
offline preparation step; building or browsing the site never trains models.
