"""Build four independent MkDocs API layout studies from identical source snapshots."""
from __future__ import annotations

import html
import argparse
import inspect
import io
import json
import re
import shutil
import subprocess
import sys
import textwrap
import tokenize
from pathlib import Path

import markdown
import yaml
from numpydoc.docscrape import NumpyDocString

HERE = Path(__file__).resolve().parent
STYLES = {'a': 'Continuous reference', 'b': 'Structured reference',
          'c': 'Focused pages', 'd': 'Progressive disclosure',
          'e': 'Violet emphasis', 'f': 'Ink labels', 'g': 'Blue inputs, teal outputs'}
SKINS = ('e', 'f', 'g')
CATEGORIES = ['Landscapes', 'Problems', 'Analysis', 'Algorithms', 'Plotting',
              'Filters', 'Sampling', 'Networks']


def inline(s):
    s = re.sub(r':(?:class|func|meth|attr|mod|obj):`([^`]+)`', r'`\1`', s)
    return re.sub(r'``([^`]+)``', r'`\1`', s)


def prose(lines):
    s = inline('\n'.join(lines)).strip()
    s = re.sub(r'(?m)([^\n])\n(?=(?:- |\d+\. ))', r'\1\n\n', s)
    s = re.sub(r'^\.\. \[([^]]+)\]\s*', r'- **\1** ', s, flags=re.M)
    s = re.sub(r'^\[(\d+)\]\s*', r'- [\1] ', s, flags=re.M)
    # Display docstring math as math, retaining its original expression.
    s = re.sub(r'^\.\. math::\s*\n((?:[ \t]+[^\n]*\n?)+)',
               lambda m: '\n$$\n' + textwrap.dedent(m[1]).strip() + '\n$$\n', s, flags=re.M)
    return s


def parse(doc):
    raw = inspect.cleandoc(doc or '')
    # The sampling module uses a legacy bullet syntax; normalize delimiters only.
    if '\nParameters:\n' in raw:
        raw = re.sub(r'^(Parameters|Returns):$', lambda m: m[1]+'\n'+'-'*len(m[1]), raw, flags=re.M)
        raw = re.sub(r'^- (\w+): (.+)$', r'\1 : \2', raw, flags=re.M)
    return NumpyDocString(raw)


def anchor(s):
    return re.sub(r'[^a-z0-9_-]+', '-', s.lower()).strip('-')


def heading(title, level, prefix=''):
    return f'{"#" * level} {title} {{ #{prefix}{anchor(title)} }}\n\n'


def signature(s, semantic=False):
    s = re.sub(r'\s+', ' ', s).strip()
    if len(s) > 88 and '(' in s:
        start, rest = s.split('(', 1)
        inner, end = rest.rsplit(')', 1)
        parts, depth, last = [], 0, 0
        for token in tokenize.generate_tokens(io.StringIO(inner).readline):
            if token.type != tokenize.OP:
                continue
            if token.string in '([{':
                depth += 1
            elif token.string in ')]}':
                depth -= 1
            elif token.string == ',' and depth == 0:
                parts.append(inner[last:token.start[1]].strip())
                last = token.end[1]
        if inner[last:].strip():
            parts.append(inner[last:].strip())
        s = start+'(\n'+''.join('    '+p+',\n' for p in parts)+')'+end
    fence = 'api' if semantic else 'python'
    return f'```{fence}\n{s}\n```\n\n'


def field_rows(items, style, prefix):
    if not items:
        return ''
    if style == 'b':
        rows = ['<div class="field-table-wrap"><table class="field-table"><thead><tr><th>Field / type</th><th>Description</th></tr></thead><tbody>']
        for item in items:
            name, typ, desc = item
            label = name or typ
            typ = typ if name else ''
            rows.append(f'<tr id="{prefix}{anchor(label)}"><td><code>{html.escape(label)}</code>'
                        f'<span class="field-type">{html.escape(typ)}</span></td><td>'
                        + markdown.markdown(prose(desc), extensions=['fenced_code', 'tables']) + '</td></tr>')
        return '\n'.join(rows)+'</tbody></table></div>\n\n'
    out = ''
    for name, typ, desc in items:
        label = name or typ
        typ = typ if name else ''
        out += f'`{label}`'
        if typ:
            out += f' <span class="field-type">{html.escape(typ)}</span>'
        out += '\n:   '+prose(desc).replace('\n', '\n    ')+'\n\n'
    return out


def examples(lines):
    # Keep doctest prompts and outputs intact; prose remains outside code blocks.
    out, code = [], []
    def flush():
        if code:
            out.append('```pycon\n'+'\n'.join(code).rstrip()+'\n```\n')
            code.clear()
    for line in lines:
        if line.startswith(('>>>', '...')) or (code and line.strip()):
            code.append(line)
        else:
            flush()
            out.append(inline(line))
    flush()
    return '\n'.join(out)+'\n\n'


def sections(d, style, level=2, prefix='', keys=None):
    keys = keys or ['Parameters', 'Other Parameters', 'Returns', 'Yields', 'Raises',
                    'Warns', 'Warnings', 'Notes', 'Examples', 'References', 'See Also']
    out = ''
    for key in keys:
        value = d[key]
        if not value:
            continue
        if style == 'd' and key in ('Parameters', 'Other Parameters', 'Returns', 'Raises', 'Attributes'):
            fields = field_rows(value, style, prefix+anchor(key)+'-')
            out += f'<span id="{prefix}{anchor(key)}"></span>\n\n'
            out += f'!!! api-{anchor(key)} "{key}"\n\n'+textwrap.indent(fields.strip(), '    ')+'\n\n'
            continue
        out += heading(key, level, prefix)
        if key in ('Parameters', 'Other Parameters', 'Returns', 'Yields', 'Raises', 'Warns', 'Attributes', 'Methods'):
            out += field_rows(value, style, prefix+anchor(key)+'-')
        elif key == 'Examples':
            out += examples(value)
        elif key == 'See Also':
            for names, desc in value:
                out += ', '.join('`'+name[0]+'`' for name in names)+' — '+prose(desc)+'\n\n'
        else:
            out += prose(value)+'\n\n'
    return out


def method_order(o):
    preferred = ['build_from_data', 'build_from_graph', 'get_data', 'describe', 'get_lon', 'to_graph',
                 'evaluate', 'run', 'apply']
    return sorted(o['methods'], key=lambda m: preferred.index(m['name']) if m['name'] in preferred else len(preferred))


def member_index(o, style, split=False):
    methods = method_order(o)
    if not methods:
        return ''
    out = heading('Methods at a glance', 2)
    if style != 'd':
        out += '<div class="member-index" markdown="1">\n\n'
    for m in methods:
        link = f'{m["name"]}.md' if split else '#method-'+m['name']
        summary = prose(parse(m['docstring'])['Summary'])
        out += f'[`{m["name"]}()`]({link})\n:   {summary}\n\n'
    return out+('</div>\n\n' if style != 'd' else '')


def member(m, style, level=3):
    prefix = 'method-'+m['name']
    d = parse(m['docstring'])
    body = f'<span class="member-anchor" id="{prefix}"></span>\n\n'
    if style != 'd':
        body += heading('`'+m['name']+'`', level, '')
    if m.get('inherited_from'):
        body += '*Inherited from* `'+m['inherited_from']+'`\n\n'
    if style != 'd':
        body += prose(d['Summary'])+'\n\n'
    body += signature(m['signature'], style=='d')
    body += prose(d['Extended Summary'])+'\n\n'
    body += sections(d, style, level+1, prefix+'-')
    if style == 'd':
        summary = ' '.join(prose(d['Summary']).split())
        title = f'`{m["name"]}()` <span>{html.escape(summary)}</span>'
        body = '??? method-detail "'+title+'"\n\n'+textwrap.indent(body.strip(),'    ')+'\n\n'
    return body


def attributes(o, style, level=2):
    out = sections(parse(o['docstring']), style, level, keys=['Attributes'])
    if o['properties']:
        out += heading('Properties', level)
        for m in o['properties']:
            d = parse(m['docstring'])
            out += heading('`'+m['name']+'`', level+1, 'property-')
            out += prose(d['Summary'])+'\n\n'+prose(d['Extended Summary'])+'\n\n'
            out += sections(d, style, level+2, 'property-'+m['name']+'-')
    return out


def tabs(items):
    return '\n\n'.join('=== "'+title+'"\n\n'+textwrap.indent(body.strip(), '    ') for title, body in items if body.strip())+'\n\n'


def page_header(o, style, title=None):
    d = parse(o['docstring'])
    meta = f'<div class="api-meta">{o["kind"].upper()} <span> / </span> <code>{o["module"]}</code></div>\n\n'
    return (meta+f'# {title or o["name"]}\n\n'+prose(d['Summary'])+'\n\n'+signature(o['signature'], style=='d'))


def render(o, style, target):
    d = parse(o['docstring'])
    out = ('---\nhide:\n  - toc\n---\n\n' if style == 'd' else '')+page_header(o, style)
    extras = sections(d, style, keys=['Warnings', 'Notes', 'Examples', 'References', 'See Also'])
    description = prose(d['Extended Summary'])+'\n\n' if d['Extended Summary'] else ''
    if o['kind'] == 'function':
        if style == 'd':
            out += tabs([('Parameters & returns', description+sections(d, style, keys=['Parameters','Other Parameters','Returns','Yields'])),
                         ('Notes & errors', sections(d, style, keys=['Raises','Warns','Warnings','Notes'])),
                         ('Examples & references', sections(d, style, keys=['Examples','References','See Also']))])
        else:
            out += description+sections(d, style)
    elif style == 'c':
        links = [('[Constructor](constructor.md)', 'Signature and parameters')]
        if d['Attributes'] or o['properties']:
            links.append(('[Attributes & properties](attributes.md)', 'State available on the object'))
        if extras:
            links.append(('[Examples & notes](notes.md)', 'Usage, notes and references'))
        out += '<div class="section-links" markdown="1">\n\n'
        out += '\n\n'.join(a+'\n:   '+b for a,b in links)+'\n\n</div>\n\n'
        out += member_index(o, style, split=True)+description
        common = '[← '+o['name']+'](index.md)\n\n'
        (target/'constructor.md').write_text(common+page_header(o,style,o['name']+' constructor')+description+sections(d,style,keys=['Parameters','Other Parameters','Raises']))
        if d['Attributes'] or o['properties']:
            (target/'attributes.md').write_text(common+f'# {o["name"]} attributes\n\n'+attributes(o,style))
        if extras:
            (target/'notes.md').write_text(common+f'# {o["name"]} examples & notes\n\n'+extras)
        for m in method_order(o):
            md = parse(m['docstring'])
            content = common+f'<div class="api-meta">METHOD / <code>{o["module"]}.{o["name"]}</code></div>\n\n'
            content += f'# {m["name"]}\n\n'+prose(md['Summary'])+'\n\n'+signature(m['signature'])
            if m.get('inherited_from'):
                content += '<p class="api-meta">Inherited from <code>'+html.escape(m['inherited_from'])+'</code></p>\n\n'
            content += prose(md['Extended Summary'])+'\n\n'+sections(md,style)
            (target/(m['name']+'.md')).write_text(content)
    elif style == 'd':
        methods = '<label>Find a method <input type="search" class="method-filter" placeholder="e.g. build, data, graph" aria-label="Filter methods"></label> <button type="button" class="expand-methods">Expand all</button>\n{ .method-tools }\n\n'
        methods += ''.join(member(m,style,2) for m in method_order(o))
        out += tabs([('Overview', description+member_index(o,style)),
                     ('Constructor', sections(d,style,keys=['Parameters','Other Parameters','Raises'])),
                     ('Attributes', attributes(o,style)), ('Methods', methods),
                     ('Examples & notes', extras)])
    else:
        out += '<nav class="api-jumps" aria-label="Page sections"><a href="#constructor">Constructor</a>'
        if d['Attributes'] or o['properties']:
            out += '<a href="#attributes-and-properties">Attributes</a>'
        out += '<a href="#methods">Methods</a></nav>\n\n'
        out += member_index(o,style)
        out += description+heading('Constructor',2)+sections(d,style,3,'constructor-',keys=['Parameters','Other Parameters','Raises'])
        if d['Attributes'] or o['properties']:
            out += heading('Attributes and properties',2)+attributes(o,style,3)
        out += heading('Methods',2)+''.join(member(m,style) for m in method_order(o))
        out += extras
    out += '\n---\n\n<span class="source-note">'+html.escape(o['source'])+':'+str(o['lineno'])+'</span>\n'
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=[*STYLES, 'all', 'new-styles'], default='g',
                        help='G is the selected appearance on the D layout; other variants remain comparisons.')
    args=parser.parse_args()
    objects=[]
    for fname in ['landscapes','classes','functions']:
        subprocess.run([sys.executable,str(HERE/f'extract_{fname}.py')], check=True)
        objects.extend(json.loads((HERE/'content'/f'{fname}.json').read_text()))
    for style, label in STYLES.items():
        if args.variant not in ('all', style) and not (args.variant=='new-styles' and style in SKINS):
            continue
        layout = 'd' if style in SKINS else style
        generated=HERE/'.generated'/style
        generated.mkdir(parents=True,exist_ok=True)
        content=generated/'content'
        if content.exists():
            shutil.rmtree(content)
        content.mkdir()
        shutil.copytree(HERE/'assets',content/'assets')
        nav=[]
        for category in CATEGORIES:
            entries=[]
            if layout=='d' and category=='Analysis':
                narrative_dir=content/'analysis'
                narrative_dir.mkdir()
                for name in ('epistasis','navigability','robustness','ruggedness'):
                    shutil.copyfile(HERE/'narratives'/f'{name}.md', narrative_dir/f'{name}.md')
                    entries.append({name.title():f'analysis/{name}.md'})
                nav.append({category:entries})
                continue
            for o in objects:
                if o['category'] != category: continue
                # Open each site directly on the complete Landscape page.
                rel=Path('.') if o['key']=='landscape' else Path(o['key'])
                target=content/rel
                target.mkdir(exist_ok=True)
                (target/'index.md').write_text(render(o,layout,target))
                page=(rel/'index.md').as_posix()
                if layout=='c' and o['kind']=='class':
                    pages=[{'Overview':page},{'Constructor':(rel/'constructor.md').as_posix()}]
                    for name,title in [('attributes','Attributes & properties'),('notes','Examples & notes')]:
                        if (target/(name+'.md')).exists(): pages.append({title:(rel/(name+'.md')).as_posix()})
                    pages.extend({m['name']:(rel/(m['name']+'.md')).as_posix()} for m in method_order(o))
                    entries.append({o['name']:pages})
                else:
                    entries.append({o['name']:page})
            if entries: nav.append({category:entries})
        port=8811+list(STYLES).index(style)
        config={
            'site_name':f'GraphFLA · {style.upper()}',
            'site_description':f'GraphFLA API documentation — {label}',
            'site_url':f'http://127.0.0.1:{port}/',
            'docs_dir':'content','site_dir':str(HERE/'.build'/style),
            'nav':nav,
            'theme':{'name':'material','font':False,'language':'en',
                     'palette':{'scheme':'default','primary':'black','accent':'purple'},
                     'features':['navigation.sections','navigation.top','search.suggest','search.highlight','content.code.copy','toc.follow']},
            'plugins':['search'],
            'markdown_extensions':['admonition','attr_list','md_in_html','def_list',{'toc':{'permalink':True,'toc_depth':3}},
               {'pymdownx.highlight':{'use_pygments':True}},'pymdownx.superfences','pymdownx.inlinehilite','pymdownx.details',
               {'pymdownx.tabbed':{'alternate_style':True}},{'pymdownx.arithmatex':{'generic':True}}],
            'extra_css':['assets/reference.css'],
            'extra_javascript':['assets/reference.js','https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'],
            'extra':{'generator':False},
            'copyright':f'Candidate {style.upper()} · {label} · Real API content · Layout review',
        }
        if layout=='d':
            # Keep Material's original typography and page grid; style API blocks only.
            config['theme'].pop('font', None)
            config['extra_css']=['assets/api.css']
            config['hooks']=[str(HERE/'api_hooks.py')]
            config['site_name']='GraphFLA'
            config['site_description']='GraphFLA API documentation — narrative analysis and tabbed classes'
            config['copyright']='GraphFLA · Documentation preview'
            if style in SKINS:
                config['extra_css'] += ['assets/api-skin.css', f'assets/skins/{style}.css']
                config['site_name']=f'GraphFLA · {style.upper()}'
                config['copyright']=f'Style {style.upper()} · {label} · Documentation preview'
        path=generated/'mkdocs.yml'
        path.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
        subprocess.run([sys.executable,'-m','mkdocs','build','--strict','-f',str(path)],check=True)
    print('Built requested documentation preview(s).')


if __name__=='__main__':
    main()
