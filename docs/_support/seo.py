"""Search-result snippets and search-engine verification files for every page."""

import re

from bs4 import BeautifulSoup
from markupsafe import escape
from mkdocs.structure.files import File

# Search results show about this many characters of a description.
SNIPPET_LENGTH = 155
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def on_files(files, config):
    # IndexNow confirms that a submission comes from the site owner by fetching this key from the site.
    key = config.extra.get("indexnow_key")
    if key:
        files.append(File.generated(config, f"{key}.txt", content=key))
    return files


def on_page_context(context, page, **kwargs):
    # Material writes the description into an attribute unescaped.
    page.meta["description"] = escape(page.meta.get("description") or summary(page.content))
    return context


def summary(html, length=SNIPPET_LENGTH):
    """Return the opening sentences of the first prose paragraph of a page.

    Parameters
    ----------
    html : str
        Rendered page content.
    length : int, default=SNIPPET_LENGTH
        Maximum number of characters. A longer first sentence is cut at a word boundary.

    Returns
    -------
    str
        The summary, or an empty string when the page has no prose paragraph.
    """
    for paragraph in BeautifulSoup(html, "html.parser").find_all("p"):
        if paragraph.find_parent(class_="admonition"):
            continue
        # Inline math reads as plain text without its MathJax delimiters.
        text = re.sub(r"\\[()]", "", " ".join(paragraph.get_text().split()))
        # Captions, labels and download links do not end like a sentence.
        if not re.search(r"[.!?][\"”’)]?$", text):
            continue
        sentences = SENTENCE_BREAK.split(text)
        result = sentences[0]
        if len(result) > length:
            return result[: length - 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"
        for sentence in sentences[1:]:
            if len(result) + 1 + len(sentence) > length:
                break
            result += " " + sentence
        return result
    return ""
