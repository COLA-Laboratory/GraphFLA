"""Arrange source docstring sections without copying or rewriting their content."""

import re

from markup import split_notices


def function_layout(sections):
    buckets = {
        name: []
        for name in ("parameters", "returns", "examples", "notes", "exceptions")
    }
    summary = ""
    descriptions, notices, related, deprecated = [], [], [], []
    for section in sections:
        kind = section.kind.value
        if kind == "text":
            body, notice = split_notices(section.value)
            if notice:
                notices.append(notice)
            if body and not summary:
                parts = re.split(r"\n\s*\n", body, maxsplit=1)
                summary = parts[0]
                if len(parts) > 1:
                    descriptions.append(parts[1])
            elif body:
                descriptions.append(body)
        elif kind == "deprecated":
            deprecated.append(section)
        elif kind == "admonition" and section.value.kind == "see-also":
            related.append(section)
        elif kind in ("parameters", "other parameters", "type parameters", "receives"):
            buckets["parameters"].append(section)
        elif kind in ("returns", "yields"):
            buckets["returns"].append(section)
        elif kind == "examples":
            buckets["examples"].append(section)
        elif kind in ("raises", "warns") or (
            kind == "admonition"
            and section.value.kind in ("warning", "warnings", "danger")
        ):
            buckets["exceptions"].append(section)
        else:
            # Notes, references and less common structured sections retain their
            # own headings inside one panel. Unknown types still fail in the renderer.
            buckets["notes"].append(section)

    groups = []
    for key, label in [
        ("parameters", "Parameters"),
        ("returns", "Returns"),
        ("examples", "Examples"),
        ("notes", "Notes"),
        ("exceptions", "Raises & Warns"),
    ]:
        description = "\n\n".join(descriptions) if key == "notes" else ""
        if not buckets[key] and not description:
            continue
        kinds = {s.kind.value for s in buckets[key]}
        if key == "returns" and kinds == {"yields"}:
            label = "Yields"
        if key == "exceptions":
            if kinds == {"raises"}:
                label = "Raises"
            elif "raises" not in kinds:
                label = "Warnings"
        groups.append(
            {
                "key": key,
                "label": label,
                "sections": buckets[key],
                "description": description,
            }
        )
    return {
        "summary": summary,
        "notices": "\n\n".join(notices),
        "deprecated": deprecated,
        "related": related,
        "groups": groups,
    }
