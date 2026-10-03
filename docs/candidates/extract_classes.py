#!/usr/bin/env python3
"""Extract selected public class APIs from GraphFLA source using only the AST."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "content" / "classes.json"

TARGETS = (
    ("nk", "Problems", "graphfla.problems", "graphfla/problems/biological.py", "NK"),
    (
        "optimization-problem",
        "Problems",
        "graphfla.problems",
        "graphfla/problems/base_problem.py",
        "OptimizationProblem",
    ),
    (
        "hill-climb",
        "Algorithms",
        "graphfla.algorithms",
        "graphfla/algorithms/walk.py",
        "HillClimb",
    ),
    ("landscape-filter", "Filters", "graphfla", "graphfla/filters.py", "LandscapeFilter"),
)


def _annotation(node: ast.expr | None) -> str:
    return ast.unparse(node) if node is not None else ""


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str | None = None) -> str:
    """Render the definition's parameters while omitting its receiver."""
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parts: list[str] = []
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        if index == 0 and arg.arg in {"self", "cls"}:
            continue
        item = arg.arg
        annotation = _annotation(arg.annotation)
        if annotation:
            item += f": {annotation}"
        if default is not None:
            item += f"{' = ' if annotation else '='}{ast.unparse(default)}"
        parts.append(item)
        if args.posonlyargs and index == len(args.posonlyargs) - 1:
            parts.append("/")

    if args.vararg:
        item = "*" + args.vararg.arg
        annotation = _annotation(args.vararg.annotation)
        if annotation:
            item += f": {annotation}"
        parts.append(item)
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        item = arg.arg
        annotation = _annotation(arg.annotation)
        if annotation:
            item += f": {annotation}"
        if default is not None:
            item += f"{' = ' if annotation else '='}{ast.unparse(default)}"
        parts.append(item)

    if args.kwarg:
        item = "**" + args.kwarg.arg
        annotation = _annotation(args.kwarg.annotation)
        if annotation:
            item += f": {annotation}"
        parts.append(item)
    return f"{name or node.name}({', '.join(parts)})"


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = set()
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            names.add(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            names.add(decorator.attr)
    return names


def _is_property(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return bool(_decorator_names(node) & {"property", "cached_property"})


def _doc(node: ast.AST) -> str:
    return ast.get_docstring(node, clean=False) or ""


def _load_sources() -> dict[str, ast.Module]:
    result = {}
    for path in (ROOT / "graphfla").rglob("*.py"):
        result[path.relative_to(ROOT).as_posix()] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return result


def _class_index(sources: dict[str, ast.Module]) -> dict[str, tuple[str, ast.ClassDef]]:
    result = {}
    for source, module in sources.items():
        for node in module.body:
            if isinstance(node, ast.ClassDef):
                result[node.name] = (source, node)
    return result


def _bases(node: ast.ClassDef) -> list[str]:
    return [base.id for base in node.bases if isinstance(base, ast.Name)]


def _members(
    source: str,
    cls: ast.ClassDef,
    class_index: dict[str, tuple[str, ast.ClassDef]],
    seen: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    seen = set() if seen is None else seen
    methods: dict[str, dict] = {}
    properties: dict[str, dict] = {}

    def add(owner_source: str, owner: ast.ClassDef, inherited_from: str | None) -> None:
        for member in owner.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if member.name.startswith("_"):
                continue
            is_prop = _is_property(member)
            record = {
                "name": member.name,
                "signature": _signature(member),
                "docstring": _doc(member),
                "source": owner_source,
                "lineno": member.lineno,
            }
            if inherited_from:
                record["inherited_from"] = inherited_from
            target = properties if is_prop else methods
            target.setdefault(member.name, record)

    add(source, cls, None)
    for base_name in _bases(cls):
        if base_name in seen or base_name == "object":
            continue
        seen.add(base_name)
        base = class_index.get(base_name)
        if base:
            base_source, base_cls = base
            add(base_source, base_cls, base_name)
            # Include members inherited by the base, retaining the declaring class.
            deeper_methods, deeper_properties = _members(base_source, base_cls, class_index, seen)
            for member in deeper_methods:
                methods.setdefault(member["name"], member)
            for member in deeper_properties:
                properties.setdefault(member["name"], member)
    return list(methods.values()), list(properties.values())


def extract() -> list[dict]:
    sources = _load_sources()
    class_index = _class_index(sources)
    records = []
    for key, category, module_name, source, name in TARGETS:
        cls = next(
            node for node in sources[source].body
            if isinstance(node, ast.ClassDef) and node.name == name
        )
        methods, properties = _members(source, cls, class_index)
        constructor = next(
            (node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"),
            None,
        )
        record = {
            "key": key,
            "category": category,
            "name": name,
            "module": module_name,
            "kind": "class",
            "signature": _signature(constructor, name=name) if constructor else f"{name}()",
            "docstring": _doc(cls),
            "source": source,
            "lineno": cls.lineno,
            "methods": methods,
            "properties": properties,
        }
        # Constructor docstrings are useful only when they add information not
        # already expressed by the class-level parameter documentation.
        if constructor:
            ctor_doc = _doc(constructor)
            if ctor_doc.strip() and not _parameters_covered(_doc(cls), constructor):
                record["constructor_docstring"] = ctor_doc
        records.append(record)
    return records


def _parameters_covered(class_doc: str, constructor: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check that class docs already cover the constructor's named parameters."""
    class_params = _parameter_names(class_doc)
    args = constructor.args
    ctor_params = {
        arg.arg
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if arg.arg not in {"self", "cls"}
    }
    if args.vararg:
        ctor_params.add("*" + args.vararg.arg)
    if args.kwarg:
        ctor_params.add("**" + args.kwarg.arg)
    return bool(ctor_params) and ctor_params <= class_params


def _parameter_names(doc: str) -> set[str]:
    # Numpy-style Parameters sections are used by the current class definitions.
    in_parameters = False
    result = set()
    for line in textwrap.dedent(doc).splitlines():
        stripped = line.strip()
        if stripped == "Parameters":
            in_parameters = True
            continue
        if in_parameters and stripped and set(stripped) == {"-"}:
            continue
        if in_parameters and stripped and not line.startswith((" ", "\t")) and " :" in stripped:
            result.add(stripped.split(" :", 1)[0].split(",", 1)[0].strip())
        elif in_parameters and stripped and not line.startswith((" ", "\t")):
            break
    return result


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(extract(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
