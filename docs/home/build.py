"""Build the GraphFLA landing page and its design-system reference.

Run ``python docs/home/build.py``; the pages are written to ``docs/home/dist``.
"""
import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

from figures import render
from figures import load_tokens
from figures.examples import render_examples
from figures.palette import token_groups
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup, escape
from page import HOME, load_page

DIST = HOME / "dist"
STYLESHEETS = ("tokens.css", "base.css", "components.css", "sections.css", "insights.css")
# Must provide the families named by the --gfl-font-* tokens.
FONTS = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800"
         "&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap")

_INLINE = re.compile(r"\*(.+?)\*|`(.+?)`")


def inline(text):
    """Render the inline markup of ``content.yml``: ``*accent*`` and `` `code` ``."""
    def markup(match):
        accent, code = match.groups()
        if accent:
            return f'<span class="gfl-accent">{accent}</span>'
        return f'<code class="gfl-inline-code">{code}</code>'

    return Markup(_INLINE.sub(markup, str(escape(text))))


def icon(name):
    """Return the SVG source of ``icons/<name>.svg`` for inlining."""
    return Markup((HOME / "icons" / f"{name}.svg").read_text().strip())


def environment(links, assets="assets/"):
    """Return the template environment.

    Parameters
    ----------
    links : dict
        Named URLs; the ``href`` filter resolves a name and passes anything else through.
    assets : str, default="assets/"
        URL prefix of stylesheets, figures and scripts.
    """
    env = Environment(loader=FileSystemLoader(HOME / "templates"), autoescape=True,
                      undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    env.filters["inline"] = inline
    env.filters["href"] = lambda target: links.get(target, target)
    env.tests["external"] = lambda url: url.startswith(("http://", "https://"))
    env.globals["icon"] = icon
    env.globals["asset"] = lambda path: assets + path
    return env


def build_assets(assets):
    """Write shared assets and return the page context for either build entry point."""
    assets = Path(assets)
    figures = render(assets / "figures")
    figures.update(render_examples(assets / "figures", load_tokens()))
    page = load_page(figures)
    (assets / "data").mkdir(parents=True, exist_ok=True)
    for path in (HOME / "examples").glob("*"):
        shutil.copyfile(path, assets / "data" / path.name)
    (assets / "css").mkdir(parents=True, exist_ok=True)
    for sheet in (*STYLESHEETS, "styleguide.css"):
        shutil.copyfile(HOME / "styles" / sheet, assets / "css" / sheet)
    shutil.copyfile(HOME / "scripts" / "home.js", assets / "home.js")
    shutil.copyfile(HOME / "scripts" / "insights.js", assets / "insights.js")
    shutil.copytree(HOME / "vendor", assets / "vendor", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("README.md"))
    data = json.dumps(page["insights"]["panels"], ensure_ascii=False, allow_nan=False)
    (assets / "insights-data.js").write_text("window.graphflaInsights = " + data.replace("<", "\\u003c") + ";\n")
    (assets / "data" / "insights.json").write_text(data + "\n")
    if page["insights"]["panels"]:
        folders = [HOME / "insights" / panel["id"] for panel in page["insights"]["panels"]]
        folders.append(HOME / "insights" / "references")
        with zipfile.ZipFile(assets / "data" / "insights-preparation.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
            for folder in folders:
                for path in sorted(folder.rglob("*")):
                    if path.is_file() and path.suffix in (".py", ".json", ".csv", ".md"):
                        bundle.write(path, path.relative_to(HOME.parent.parent))
    return page


def build(out=DIST):
    """Write the landing page, the design-system page and their assets to ``out``."""
    out = Path(out)
    page = build_assets(out / "assets")

    shell = environment(page["links"]).get_template("standalone.html")
    (out / "index.html").write_text(shell.render(
        page, title=page["meta"]["document_title"], body="home.html", fonts=FONTS, stylesheets=STYLESHEETS))
    (out / "styleguide.html").write_text(shell.render(
        page, title=f"{page['meta']['title']} design system", body="styleguide.html", fonts=FONTS,
        stylesheets=(*STYLESHEETS, "styleguide.css"), tokens=token_groups()))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DIST, help="output directory (default: docs/home/dist)")
    print("Built", build(parser.parse_args().out))
