"""Shared highlighting for signatures already extracted by mkdocstrings."""

import re
from html import escape

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


def _highlight_signature_fragment(source):
    return highlight(
        source, PythonLexer(stripnl=False, ensurenl=False), HtmlFormatter(nowrap=True)
    ).rstrip("\n")


def format_signature(source):
    """Emphasize the declared API name without changing its text or copy value."""
    match = re.match(
        r"(\s*(?:(?:async\s+)?def\s+|class\s+)?)"
        r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?=\s*\()",
        source,
    )
    if match:
        lead, qualified = match.groups()
        module, dot, name = qualified.rpartition(".")
        code = (
            '<span class="api-sig-keyword">' + escape(lead) + "</span>" if lead else ""
        )
        if module:
            code += '<span class="api-sig-module">' + escape(module + dot) + "</span>"
        code += '<strong class="api-sig-name">' + escape(name) + "</strong>"
        rest = source[match.end() :]
        closing = rest.rfind(")")
        if closing >= 0 and re.match(r"\s*(?:->|→)", rest[closing + 1 :]):
            code += _highlight_signature_fragment(rest[: closing + 1])
            code += (
                '<span class="api-sig-return">'
                + _highlight_signature_fragment(rest[closing + 1 :])
                + "</span>"
            )
        else:
            code += _highlight_signature_fragment(rest)
    else:
        code = _highlight_signature_fragment(source)
    return (
        '<div class="api-signature highlight"><pre><code>'
        + code
        + "</code></pre></div>"
    )
