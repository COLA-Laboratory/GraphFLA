# GraphFLA documentation preview

> **Archived design prototypes.** The active source-synchronized website now uses
> [`../README.md`](../README.md) and `python docs/manage.py serve` on port 8818.
> The instructions below describe the historical comparisons only. Do not edit
> these snapshots to maintain current API documentation.

The selected version is **G**, at <http://127.0.0.1:8817/>, using D's layout:

- Classes use Material's native section tabs and expandable methods.
- Analysis stays in authored topic pages: Epistasis, Navigability, Robustness
  (including Neutrality), and Ruggedness. All 28 function placements, tutorial
  explanations, formulas, examples and references are retained.
- Only API components have custom styling. The original Material font, content
  width, general spacing, black header and purple accent are unchanged.

A–F remain available as earlier comparisons on ports 8811–8816. They are not
rebuilt by default.

## Three visual styles on the same D layout

| Style | URL | Treatment |
| --- | --- | --- |
| E | http://127.0.0.1:8815/ | Violet parameter names and pale lavender panels |
| F | http://127.0.0.1:8816/ | White parameter names on dark labels, with white panels |
| G | http://127.0.0.1:8817/ | Blue input fields and teal return values |

All three render exactly the same article HTML as D, including classes and the
four complete analysis narratives. The only additions are stylesheets and the
site letter in the header/footer. The original D remains available for comparison.

```sh
.venv-docs/bin/python docs/candidates/build.py --variant new-styles
.venv-docs/bin/python docs/candidates/serve.py --variants efg
```

`assets/api-skin.css` contains the shared visual rules. Each file in
`assets/skins/{e,f,g}.css` only supplies color tokens. Names are emphasized in both
authored function documentation and generated class methods; nothing needs to be
restyled in individual pages. Fonts, page widths, navigation and content are shared.

## Build and preview

From the worktree root:

```sh
uv venv .venv-docs
uv pip install --python .venv-docs/bin/python -r docs/candidates/requirements.txt
.venv-docs/bin/python docs/candidates/build.py
.venv-docs/bin/python docs/candidates/serve.py
```

The build extracts class content from source without importing GraphFLA, copies
analysis Markdown, generates the MkDocs configuration, and builds with strict
mode. Generated files live in `.generated/` and `.build/`, both excluded by the
repository `.gitignore`. The server serves independent local sites.
If it is already running, rebuild and refresh the browser; no restart is needed.
Build and serve both default to G. Use `--variant` and `--variants` to select
other previews explicitly.

## Author once, style centrally

| Concern | Edit |
| --- | --- |
| Authored analysis prose, equations, examples and API descriptions | `narratives/*.md` |
| Signature/parameter/return/disclosure appearance | `assets/api.css` |
| Shared visual style rules / each style's palette | `assets/api-skin.css` / `assets/skins/*.css` |
| Signature rendering and consistent field name/type markup | `api_hooks.py` |
| Class tabs, inherited methods and generated API sections | `build.py` |
| Class source content | Package docstrings, owned by the package session |

The pages use ordinary Markdown and three semantic building blocks:

````markdown
```api
def graphfla.analysis.neutrality(landscape, threshold: float = 0.01) -> float
```

Your explanation and equations stay here as ordinary Markdown.

!!! api-parameters "Parameters"

    **landscape** : `Landscape`
    :   The fitness landscape object.

!!! api-returns "Returns"

    **float**
    :   Your return-value description.
````

`api` is a SuperFences extension configured by a small MkDocs hook. The field
blocks are native Markdown admonitions containing native definition lists.
Authors write no CSS, token-color spans, per-function HTML layouts, or JavaScript.
One stylesheet changes all instances, including class methods. The theme itself
has no fork or template override. Native Material tabs, search, navigation,
copy buttons and admonitions remain in use.

All API panels share one thin outer border. Their titles use the same inset token
as the parent, so a smaller title font cannot create a gap at the corners. The
parent clips the title background to its own radius. There are no inner accent
bars or special extra left borders for Returns, and no per-function overrides.

Signature fences are parsed by the shared formatter: only the final class/function
name receives bold emphasis, the module prefix stays muted, and a return annotation
uses the output color. The signature surface and native copy button are styled by
the shared CSS. This does not restyle ordinary Python examples, add labels to copied
code, or require authors to annotate identifiers individually.

## Original narrative migration

`import_narratives.py` performed a **one-time** migration from the original local
`docs/site/content/docs/api/analysis/` files. The migrated Markdown is now the
editable source for this preview; it is not overwritten during normal builds.

The importer replaces legacy colored signature spans with semantic `api` fences,
changes parameter bullets into definition lists, and makes the previous Epistasis
Description/Parameters/Returns/References/Example panels visible in their original
sequence. It preserves every rendered word, including existing naming/content
inconsistencies. It compares the complete rendered text (ignoring whitespace) and
section-label counts before writing. `narratives/provenance.json` records original
and rendered-text hashes for checking this migration.

To import a fresh original set deliberately, supply its directory explicitly:

```sh
.venv-docs/bin/python docs/candidates/import_narratives.py /path/to/original/analysis
```

This replaces the migrated narrative files. Normal editing should happen directly
in `narratives/`; changes there do not require importing again.

The preview does not rename APIs, revise scientific explanations, execute examples,
change package logic, or deploy the public documentation.

References: [SuperFences custom blocks](https://facelessuser.github.io/pymdown-extensions/extensions/superfences/#custom-fences),
[Material content tabs](https://squidfunk.github.io/mkdocs-material/reference/content-tabs/).
