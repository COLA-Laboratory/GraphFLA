#!/usr/bin/env python3
"""Extract full Landscape API records from source without importing GraphFLA."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    "graphfla/landscape/landscape.py",
    "graphfla/landscape/sequence.py",
    "graphfla/landscape/protein.py",
    "graphfla/landscape/_io.py",
    "graphfla/landscape/_build.py",
]
TARGETS = [
    ("landscape", "Landscape", "graphfla.landscape"),
    ("protein-landscape", "ProteinLandscape", "graphfla.landscape"),
]


def signature(node):
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    rendered = []
    for arg, default in zip(positional, defaults):
        if arg.arg in {"self", "cls"} and not rendered:
            continue
        part = arg.arg
        if arg.annotation:
            part += ": " + ast.unparse(arg.annotation)
        if default is not None:
            part += " = " + ast.unparse(default)
        rendered.append(part)
    if args.posonlyargs:
        count = len(args.posonlyargs)
        rendered.insert(count, "/")
    if args.vararg:
        part = "*" + args.vararg.arg
        if args.vararg.annotation:
            part += ": " + ast.unparse(args.vararg.annotation)
        rendered.append(part)
    elif args.kwonlyargs:
        rendered.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        part = arg.arg
        if arg.annotation:
            part += ": " + ast.unparse(arg.annotation)
        if default is not None:
            part += " = " + ast.unparse(default)
        rendered.append(part)
    if args.kwarg:
        part = "**" + args.kwarg.arg
        if args.kwarg.annotation:
            part += ": " + ast.unparse(args.kwarg.annotation)
        rendered.append(part)
    result = f"{node.name}({', '.join(rendered)})"
    if node.returns:
        result += " -> " + ast.unparse(node.returns)
    return result


def decorators(node):
    return [ast.unparse(d) for d in node.decorator_list]


def main():
    classes = {}
    trees = {}
    for rel in SOURCES:
        path = ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        trees[rel] = tree
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = (node, rel)

    def bases(class_node):
        return [ast.unparse(base).split(".")[-1] for base in class_node.bases]

    def linearize(name, active=()):
        if name in active or name not in classes:
            return []
        node = classes[name][0]
        chain = [name]
        for base in bases(node):
            chain.extend(n for n in linearize(base, (*active, name)) if n not in chain)
        return chain

    output = []
    for key, name, module in TARGETS:
        cls, source = classes[name]
        members, properties = {}, {}
        for owner in linearize(name):
            owner_node, owner_source = classes[owner]
            for member in owner_node.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                member_name = member.name
                if member_name.startswith("_") and member_name != "__init__":
                    continue
                if member_name in members or member_name in properties:
                    continue
                decs = decorators(member)
                is_property = "property" in decs or any(d.endswith(".setter") or d.endswith(".deleter") for d in decs)
                record = {
                    "name": member_name,
                    "signature": signature(member),
                    "docstring": ast.get_docstring(member, clean=False),
                    "source": owner_source,
                    "lineno": member.lineno,
                }
                if owner != name:
                    record["inherited_from"] = owner
                (properties if is_property else members)[member_name] = record

        init = members.get("__init__")
        if init is None:
            raise RuntimeError(f"No effective constructor found for {name}")
        record = {
            "key": key,
            "category": "Landscapes",
            "name": name,
            "module": module,
            "kind": "class",
            "signature": init["signature"].replace("__init__", name, 1),
            "docstring": ast.get_docstring(cls, clean=False),
            "source": source,
            "lineno": cls.lineno,
            "methods": [member for member_name, member in members.items() if member_name != "__init__"],
            "properties": list(properties.values()),
        }
        if init["docstring"]:
            record["constructor_docstring"] = init["docstring"]
        output.append(record)

    destination = Path(__file__).resolve().parent / "content" / "landscapes.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(output)} classes to {destination}")
    for record in output:
        print(f"{record['name']}: {len(record['methods'])} methods, {len(record['properties'])} properties")


if __name__ == "__main__":
    main()
