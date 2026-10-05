"""Validate built local navigation without fetching external websites."""

from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml
from bs4 import BeautifulSoup

CONFIG = Path(__file__).resolve().parents[1] / "mkdocs.yml"


def deployment_path(config=CONFIG):
    """Return the URL path the site is served under, e.g. ``/GraphFLA/``."""
    site_url = yaml.safe_load(config.read_text()).get("site_url") or "/"
    return urlsplit(site_url).path.rstrip("/") + "/"


def site_errors(site, base_path=None):
    """Return broken local links, anchors and duplicate ids in a built site.

    Root-relative links (as in ``404.html``) include the deployment path,
    which is removed before resolving them against ``site``.
    """
    base_path = deployment_path() if base_path is None else base_path
    site = site.resolve()
    pages = {
        p: BeautifulSoup(p.read_text(), "html.parser") for p in site.rglob("*.html")
    }
    errors = []
    for path, soup in pages.items():
        location = str(path.relative_to(site))
        for key, count in Counter(e["id"] for e in soup.select("[id]")).items():
            if count > 1:
                errors.append(f"{location}: duplicate id {key}")
        for link in soup.select(
            "a[href], link[href], img[src], script[src], source[src]"
        ):
            target = link.get("href", link.get("src"))
            url = urlsplit(target)
            if url.scheme or url.netloc or not (url.path or url.fragment):
                continue
            local = url.path
            if local.startswith("/"):
                if not local.startswith(base_path):
                    errors.append(f"{location}: link outside the site path {target}")
                    continue
                local = local[len(base_path) :]
            base = site if url.path.startswith("/") else path.parent
            dest = (base / unquote(local)).resolve() if url.path else path
            if dest.is_dir():
                dest = dest / "index.html"
            if not dest.exists():
                errors.append(f"{location}: missing link or asset {target}")
            elif (
                url.fragment
                and dest in pages
                and not pages[dest].find(id=unquote(url.fragment))
            ):
                errors.append(f"{location}: missing anchor {target}")
        if soup.find("autoref"):
            errors.append(f"{location}: unresolved reference markup")
    return errors
