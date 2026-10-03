"""Small MkDocs hook: one reusable formatter for semantic API signature fences."""
import re
from html import escape

from bs4 import BeautifulSoup
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


def _highlight_signature_fragment(source):
    return highlight(source, PythonLexer(stripnl=False, ensurenl=False),
                     HtmlFormatter(nowrap=True)).rstrip('\n')


def format_signature(source, language, css_class, options, md, **kwargs):
    """Emphasize the declared API name without changing its text or copy value."""
    match = re.match(r'(\s*(?:(?:async\s+)?def\s+|class\s+)?)'
                     r'([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?=\s*\()', source)
    if match:
        lead, qualified = match.groups()
        module, dot, name = qualified.rpartition('.')
        code = '<span class="api-sig-keyword">'+escape(lead)+'</span>' if lead else ''
        if module:
            code += '<span class="api-sig-module">'+escape(module+dot)+'</span>'
        code += '<strong class="api-sig-name">'+escape(name)+'</strong>'
        rest = source[match.end():]
        closing = rest.rfind(')')
        if closing >= 0 and re.match(r'\s*(?:->|→)', rest[closing+1:]):
            code += _highlight_signature_fragment(rest[:closing+1])
            code += '<span class="api-sig-return">'+_highlight_signature_fragment(rest[closing+1:])+'</span>'
        else:
            code += _highlight_signature_fragment(rest)
    else:
        code = _highlight_signature_fragment(source)
    return '<div class="api-signature highlight"><pre><code>'+code+'</code></pre></div>'


def on_config(config):
    config.mdx_configs.setdefault('pymdownx.superfences', {})['custom_fences'] = [
        {'name': 'api', 'class': 'api-signature', 'format': format_signature}
    ]
    return config


def on_page_content(html, **kwargs):
    """Give equivalent Markdown field labels the same name/type treatment."""
    soup = BeautifulSoup(html, 'html.parser')
    for term in soup.select('.admonition[class*="api-"] dt'):
        # Generated class references already supply a separate type span.
        if term.select_one('.field-type'):
            continue
        text = term.get_text()
        match = re.match(r'^(\w+)(\s*:.*)$', text, flags=re.S)
        name, annotation = (match[1], match[2]) if match else (text, '')
        term.clear()
        code = soup.new_tag('code', attrs={'class': 'api-field-name'})
        code.string = name
        term.append(code)
        if annotation:
            typ = soup.new_tag('span', attrs={'class': 'api-field-type'})
            typ.string = annotation
            term.append(typ)
    return str(soup)
