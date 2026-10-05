"""Integrate the authored landing page into the normal MkDocs build."""

import json
from pathlib import Path
import sys
import tempfile

from jinja2 import ChoiceLoader, FileSystemLoader
from markupsafe import Markup
from mkdocs.plugins import event_priority
from mkdocs.structure.files import File
from mkdocs.utils import get_relative_url

HOME = Path(__file__).resolve().parents[1] / "home"
sys.path.insert(0, str(HOME))
import build as home_build
from catalogue import markdown as dataset_catalogue

ASSETS = "assets/home/"
PAGE = None


def on_files(files, config):
    global PAGE
    # Virtual files keep generated SVGs/CSS out of the authored content tree.
    # Read bytes before the temporary directory is removed.
    with tempfile.TemporaryDirectory(prefix="graphfla-home-assets-") as directory:
        root = Path(directory)
        PAGE = home_build.build_assets(root)
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "styleguide.css":
                files.append(File.generated(
                    config, ASSETS + path.relative_to(root).as_posix(),
                    content=path.read_bytes(),
                ))
    files.append(File.generated(config, "datasets.md", content=dataset_catalogue()))
    return files


@event_priority(-100)
def on_page_content(html, page, **kwargs):
    if page.meta.get("template") != "landing.html":
        return html
    # The landing copy also owns the homepage's search-result title and snippet.
    page.meta.update(document_title=PAGE["meta"]["document_title"], description=PAGE["meta"]["description"])
    # Rendering before on_page_context also puts the homepage in the search index.
    return home_build.environment(PAGE["links"], ASSETS).get_template("home.html").render(PAGE, embedded=True)


def on_env(env, **kwargs):
    env.loader = ChoiceLoader([env.loader, FileSystemLoader(HOME / "templates")])
    return env


def on_page_context(context, page, config, **kwargs):
    context = navigation_context(context, page.url, page.is_homepage)
    if page.is_homepage:
        context["gfl_structured_data"] = structured_data(config)
    return context


def on_template_context(context, **kwargs):
    return navigation_context(context, "404.html")


def structured_data(config):
    """Return the schema.org JSON-LD that describes GraphFLA to search engines."""
    # Google shows the WebSite name above the search result.
    website = {"@type": "WebSite", "name": PAGE["meta"]["title"], "url": config.site_url}
    software = {
        "@type": "SoftwareSourceCode",
        "name": PAGE["meta"]["title"],
        "description": PAGE["meta"]["description"],
        "url": config.site_url,
        "codeRepository": config.repo_url,
        "programmingLanguage": "Python",
        "sameAs": [config.repo_url, PAGE["links"]["pypi"]],
        "citation": [
            {"@type": "ScholarlyArticle", "name": paper["title"], "url": paper["link"]["href"],
             "author": [{"@type": "Person", "name": name} for name in paper["authors"].split(", ")]}
            for paper in PAGE["closing"]["publications"]["papers"]
        ],
    }
    data = {"@context": "https://schema.org", "@graph": [website, software]}
    # An escaped "<" cannot close the surrounding script element.
    return Markup(json.dumps(data, ensure_ascii=False).replace("<", "\\u003c"))


def navigation_context(context, page_url, is_homepage=False):
    env = home_build.environment(PAGE["links"], ASSETS)

    def href(target):
        url = PAGE["links"].get(target, target)
        if url.startswith(("https://", "http://")):
            return url
        path, separator, anchor = url.partition("#")
        if not path and is_homepage:
            return url
        return get_relative_url(path or ".", page_url) + (separator + anchor if separator else "")

    env.filters["href"] = href
    active_section = ("Home" if is_homepage else "Tutorials" if page_url.startswith("tutorials/")
                      else "API reference" if page_url not in
                      ("404.html", "benchmarks/", "datasets/", "demo-methods/", "insights-sources/") else "")
    context.update(
        gfl_fonts=home_build.FONTS,
        gfl_logo=home_build.icon("logo"),
        global_navigation=env.get_template("sections/header.html").render(
            PAGE, navigation_only=True, active_section=active_section),
    )
    return context
