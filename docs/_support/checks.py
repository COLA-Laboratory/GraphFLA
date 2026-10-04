"""Validate built local navigation without fetching external websites."""

from collections import Counter
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup


def site_errors(site):
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
        for link in soup.select("a[href], link[href], img[src], script[src], source[src]"):
            target = link.get("href", link.get("src"))
            url = urlsplit(target)
            if url.scheme or url.netloc or not (url.path or url.fragment):
                continue
            base = site if url.path.startswith("/") else path.parent
            dest = (
                (base / unquote(url.path).lstrip("/")).resolve() if url.path else path
            )
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
