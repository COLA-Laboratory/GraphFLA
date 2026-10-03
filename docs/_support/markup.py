"""Small presentation adapter for NumPy docstrings' common reST markup.

This never edits package docstrings. Griffe owns section/API parsing; this adapter
only translates inline roles, math, references and standard notices to Markdown.
"""

from html import escape
import re
import textwrap

from numpydoc.docscrape import NumpyDocString


def split_notices(text):
    """Separate API notices from prose without rewriting either one."""
    lines, body, result, i = text.splitlines(), [], [], 0
    while i < len(lines):
        line = lines[i]
        if re.match(
            r"^\s*(?:\.\. (?:note|warning|deprecated|versionadded|versionchanged|versionremoved)::|!!! (?:note|warning|danger|tip))",
            line,
        ):
            block = [line]
            indent = len(line) - len(line.lstrip())
            i += 1
            while i < len(lines) and (
                not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent
            ):
                block.append(lines[i])
                i += 1
            result.append("\n".join(block))
        else:
            body.append(line)
            i += 1
    return "\n".join(body).strip(), "\n\n".join(result)


def notices(text):
    """Keep API notices even when a narrative page hides a repeated summary."""
    return split_notices(text)[1]


def citation_id(label):
    return "ref-" + re.sub(r"[^\w-]+", "-", label).strip("-")


def ref_id(object_id, label):
    return object_id + "--" + citation_id(label)


def inline(text, object_id):
    text = re.sub(r":math:`([^`]+)`", lambda m: "$" + m[1] + "$", text)

    def role(match):
        value = match[2].lstrip("~")
        label = value.rsplit(".", 1)[-1] if match[2].startswith("~") else value
        if value.startswith("graphfla."):
            return f'<autoref identifier="{escape(value, quote=True)}" optional><code>{escape(label)}</code></autoref>'
        return "`" + label + "`"

    text = re.sub(r":(class|func|meth|attr|mod|obj|exc):`([^`]+)`", role, text)
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    # convert_markdown prefixes Markdown fragment links automatically; raw HTML
    # anchors below are namespaced here because the Markdown parser stashes them.
    text = re.sub(r"\[([^]\n]+)\]_", lambda m: f"[{m[1]}](#{citation_id(m[1])})", text)
    text = re.sub(r"`([^`<>]+)\s+<([^<>]+)>`_", r"[\1](\2)", text)
    return text


def adapt(text, object_id, title=None, public_path=None):
    if not text:
        return ""
    if title and title.lower() == "see also":
        parsed = NumpyDocString("See Also\n--------\n" + text)["See Also"]
        parts = []
        for names, desc in parsed:
            labels = []
            for name, role, *rest in names:
                target = (
                    name
                    if "." in name
                    else (public_path or object_id).rsplit(".", 1)[0] + "." + name
                )
                labels.append(
                    f'<autoref identifier="{escape(target, quote=True)}" optional><code>{escape(name)}</code></autoref>'
                )
            parts.append(
                "- "
                + ", ".join(labels)
                + (" — " + inline(" ".join(desc), object_id) if desc else "")
            )
        return "\n".join(parts)
    lines, out, i, fence = text.splitlines(), [], 0, None
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*(`{3,}|~{3,})", line):
            marker = line.lstrip()[0]
            fence = None if fence == marker else marker
        if fence or re.match(r"^\s*(>>>|\.\.\.) ", line):
            out.append(line)
            i += 1
            continue
        directive = re.match(r"^(\s*)\.\. ([\w-]+)::\s*(.*)$", line)
        if directive:
            indent, kind, tail = directive.groups()
            i += 1
            block = []
            while i < len(lines) and (
                not lines[i].strip()
                or len(lines[i]) - len(lines[i].lstrip()) > len(indent)
            ):
                block.append(lines[i])
                i += 1
            contents = textwrap.dedent("\n".join(block)).strip()
            if kind == "math":
                out.extend(
                    [
                        "",
                        "$$",
                        tail + ("\n" if tail and contents else "") + contents,
                        "$$",
                        "",
                    ]
                )
            elif kind in (
                "note",
                "warning",
                "deprecated",
                "versionadded",
                "versionchanged",
                "versionremoved",
            ):
                label = {
                    "deprecated": "Deprecated",
                    "versionadded": "Added in",
                    "versionchanged": "Changed in",
                    "versionremoved": "Removed in",
                }.get(kind, kind.title())
                heading = label + (" " + tail if tail else "")
                style = (
                    "warning"
                    if kind in ("warning", "deprecated", "versionremoved")
                    else "note"
                )
                out.extend(
                    [
                        "",
                        f'!!! {style} "{escape(heading, quote=True)}"',
                        "",
                        textwrap.indent(adapt(contents, object_id), "    "),
                        "",
                    ]
                )
            else:
                raise ValueError(
                    f"Unsupported docstring directive {kind!r} in {object_id}; add shared support instead of dropping it."
                )
            continue
        reference = re.match(r"^\s*\.\. \[([^]]+)\]\s*(.*)$", line)
        if reference:
            out.extend(
                [
                    "",
                    f'<span id="{ref_id(object_id, reference[1])}"></span>',
                    "",
                    f"**[{reference[1]}]** " + inline(reference[2], object_id),
                ]
            )
        else:
            # Keep narrative lists as lists when converting from reST.
            if re.match(r"^\s*(?:- |\d+\. )", line) and out and out[-1].strip():
                out.append("")
            out.append(inline(line, object_id))
        i += 1
    return "\n".join(out)
