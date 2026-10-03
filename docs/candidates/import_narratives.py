"""One-time, text-preserving migration of the original analysis Markdown.

Only presentation syntax changes: old colored HTML signatures become `api`
fences, old tab panels become sequential sections, and API field groups become
native MkDocs admonitions. Explanations, examples, links and math stay authored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from collections import Counter
from pathlib import Path

import markdown
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
NAMES = ('epistasis', 'navigability', 'robustness', 'ruggedness')
FIELDS = ('Parameters', 'Returns', 'Raises', 'Other Parameters')
EXTENSIONS = ['admonition', 'attr_list', 'md_in_html', 'def_list', 'tables',
              'pymdownx.superfences', 'pymdownx.tabbed', 'pymdownx.arithmatex']


def body(source):
    return re.sub(r'\A---\n.*?\n---\n', '', source, count=1, flags=re.S)


def rendered_words(source):
    """Ignore only HTML, Markdown delimiters and whitespace, never prose."""
    rendered = markdown.markdown(body(source), extensions=EXTENSIONS,
        extension_configs={'pymdownx.tabbed': {'alternate_style': True},
                           'pymdownx.arithmatex': {'generic': True}})
    soup = BeautifulSoup(rendered, 'html.parser')
    labels = []
    for tabs in soup.select('.tabbed-labels'):
        labels.extend(x.get_text(strip=True) for x in tabs.select('label'))
        tabs.decompose()
    for title in list(soup.select('p')):
        label = title.get_text(strip=True)
        if label in (*FIELDS, 'Description', 'References', 'Example') and (
            'admonition-title' in title.get('class', []) or title.find('strong')
        ):
            labels.append(label)
            title.decompose()
    return re.sub(r'\s+', '', soup.get_text()), Counter(labels)


def definition_list(source):
    # Existing definition lists (Epistasis) already have semantic field markup.
    if re.search(r'^:\s', source, flags=re.M):
        return source
    lines = source.splitlines()
    result = []
    i = 0
    while i < len(lines):
        match = re.match(r'^-\s+(.+?)\s*<br\s*/?>\s*$', lines[i])
        if not match:
            result.append(lines[i])
            i += 1
            continue
        label = match[1]
        i += 1
        desc = []
        while i < len(lines) and not re.match(r'^-\s+', lines[i]):
            desc.append(lines[i])
            i += 1
        text = textwrap.dedent('\n'.join(desc)).strip('\n')
        # A list inside a definition needs a paragraph break in Python-Markdown.
        text = re.sub(r'(?m)([^\n])\n(?=-\s)', r'\1\n\n', text)
        result.extend([label, ':   '+text.replace('\n', '\n    '), ''])
    return '\n'.join(result)


def migrate(source):
    # These wrappers only implemented the old gallery styling.
    source = re.sub(r'^<div class="api-card" markdown>\s*$', '', source, flags=re.M)
    source = re.sub(r'^</div>\s*$', '', source, flags=re.M)
    lines = source.splitlines()
    output = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r'<span class="(?:sig-block|method-signature)"', line):
            signature = BeautifulSoup(line, 'html.parser').get_text()
            output.extend(['```api', signature, '```'])
            i += 1
            continue
        tab = re.match(r'^=== "([^"]+)"\s*$', line)
        if tab:
            title = tab[1]
            i += 1
            block = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith('    ')):
                block.append(lines[i][4:] if lines[i].startswith('    ') else lines[i])
                i += 1
            text = '\n'.join(block).strip('\n')
            if title in FIELDS:
                output.extend([f'!!! api-{title.lower().replace(" ", "-")} "{title}"', '',
                               textwrap.indent(definition_list(text), '    '), ''])
            else:
                output.extend(['**'+title+'**', '', text, ''])
            continue
        field = re.match(r'^\*\*(Parameters|Returns|Raises|Other Parameters)\*\*\s*$', line)
        if field:
            title = field[1]
            i += 1
            block = []
            while i < len(lines):
                next_line = lines[i]
                if re.match(r'^(?:\*\*[A-Za-z ]+\*\*\s*$|#{1,6} |!!! |---\s*$|<span class=)', next_line):
                    break
                block.append(next_line)
                i += 1
            text = definition_list('\n'.join(block).strip('\n'))
            output.extend([f'!!! api-{title.lower().replace(" ", "-")} "{title}"', '',
                           textwrap.indent(text, '    '), ''])
            continue
        output.append(line)
        i += 1
    return '\n'.join(output).rstrip()+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Original analysis Markdown directory')
    parser.add_argument('--output', type=Path, default=HERE/'narratives')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    manifest = {}
    for name in NAMES:
        original_path = args.source/(name+'.md')
        original = original_path.read_text()
        migrated = migrate(original)
        (before, before_labels), (after, after_labels) = rendered_words(original), rendered_words(migrated)
        if before != after or before_labels != after_labels:
            import difflib
            diff = list(difflib.SequenceMatcher(None, before, after).get_opcodes())
            changes = [(tag, before[a:b][:160], after[c:d][:160]) for tag,a,b,c,d in diff if tag!='equal']
            raise ValueError(f'{name}: rendered text changed: {changes[:12]}; labels {before_labels} -> {after_labels}')
        (args.output/(name+'.md')).write_text(migrated)
        manifest[name] = {
            'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'rendered_text_sha256': hashlib.sha256(before.encode()).hexdigest(),
            'signature_count': migrated.count('```api\n'),
            'source': str(original_path),
        }
        print(f'{name}: all rendered text preserved; {manifest[name]["signature_count"]} functions')
    (args.output/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
