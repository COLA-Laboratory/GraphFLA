"""Shared MkDocs presentation, source binding and build provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from bs4 import BeautifulSoup
from mkdocs.plugins import event_priority
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from markup import adapt, notices
from signatures import format_signature
from function_layout import function_layout
from navigation import expand_article_navigation, remove_api_toc_entries
from notebooks import configure as configure_tutorials, add_tutorials

REFERENCES = {}
RESOLVED = {}
SECONDARY_REFERENCES = set()
SOURCE_ROOT = None
NAVIGATION = None
TUTORIAL_CATALOG = None
TUTORIAL_REPORT = []


def unsupported(kind):
    raise ValueError(
        f"Unsupported docstring section {kind!r}; extend the shared template."
    )


@event_priority(100)
def on_config(config):
    global SOURCE_ROOT, NAVIGATION, TUTORIAL_CATALOG
    NAVIGATION = None
    REFERENCES.clear()
    RESOLVED.clear()
    SECONDARY_REFERENCES.clear()
    TUTORIAL_CATALOG = configure_tutorials(config)
    default = Path(config.config_file_path).resolve().parent.parent
    SOURCE_ROOT = Path(
        os.environ.get("GRAPHFLA_DOCS_SOURCE_ROOT")
        or config.extra.get("api_source_root")
        or default
    ).resolve()
    if not SOURCE_ROOT.is_dir():
        raise ValueError(f"API source directory does not exist: {SOURCE_ROOT}")
    plugin = config.plugins["mkdocstrings"]
    plugin.config.handlers["python"]["paths"] = [str(SOURCE_ROOT)]
    package = SOURCE_ROOT / "graphfla"
    if package.is_dir() and str(package) not in config.watch:
        config.watch.append(str(package))
    if "pymdownx.snippets" in config.mdx_configs:
        config.mdx_configs["pymdownx.snippets"]["base_path"] = [
            str(HERE.parent / "includes")
        ]
    return config


def on_files(files, config):
    global TUTORIAL_REPORT
    TUTORIAL_REPORT = add_tutorials(files, config, TUTORIAL_CATALOG)
    return files


def on_pre_build(config):
    handler = config.plugins["mkdocstrings"].get_handler("python")
    handler.env.filters["api_markup"] = adapt
    handler.env.filters["api_notices"] = notices
    handler.env.filters["api_unsupported"] = unsupported
    handler.env.filters["api_function_layout"] = function_layout


def resolve_export(identifier, handler):
    """Follow Griffe's recorded import edges, including module/name collisions."""
    seen = set()
    options = handler.get_options({})
    while identifier not in seen and "." in identifier:
        seen.add(identifier)
        scope, name = identifier.rsplit(".", 1)
        parent = handler.collect(scope, options)
        target = parent.imports.get(name) if parent.is_module else None
        if not target or target == identifier:
            return identifier
        identifier = target
    if identifier in seen:
        raise ValueError(f"Cyclic API export: {identifier}")
    return identifier


def on_page_markdown(markdown, page, config, **kwargs):
    # A page-level preset keeps prose outside the API while retaining Notes,
    # Examples and warnings. Per-object options can still explicitly override it.
    def directive(match):
        name, block = match[1], match[2]
        REFERENCES.setdefault(page.file.src_uri, []).append(name)
        canonical = resolve_export(
            name, config.plugins["mkdocstrings"].get_handler("python")
        )
        RESOLVED[name] = canonical
        local = yaml.safe_load(block) or {}
        options = local.setdefault("options", {})
        if options.get("skip_local_inventory"):
            SECONDARY_REFERENCES.add((page.file.src_uri, name))
        options.setdefault("extra", {}).setdefault("public_path", name)
        if page.meta.get("api_narrative"):
            options.setdefault("extra", {}).setdefault("narrative", True)
        if page.meta.get("api_class"):
            options.setdefault("heading_level", 1)
        if page.meta.get("api_grouped_classes"):
            options.setdefault("extra", {}).setdefault("inline_class", True)
            options.setdefault("show_root_heading", False)
        return (
            "::: "
            + canonical
            + "\n"
            + "".join(
                "    " + line + "\n"
                for line in yaml.safe_dump(local, sort_keys=False).splitlines()
            )
            + "\n"
        )

    # Directives in literal code examples are deliberately left alone.
    chunks = re.split(r"(^```[^\n]*\n.*?^```\s*$)", markdown, flags=re.M | re.S)
    for i in range(0, len(chunks), 2):
        chunks[i] = re.sub(
            r"^::: ([\w.]+)\s*\n((?:(?: {4}|\t)[^\n]*\n|[ \t]*\n)*)",
            directive,
            chunks[i],
            flags=re.M,
        )
    return "".join(chunks)


def on_page_content(html, page, config, **kwargs):
    soup = BeautifulSoup(html, "html.parser")
    for block in soup.select(".doc-signature"):
        code = block.select_one("code")
        if not code:
            continue
        # mkdocstrings owns extraction, defaults, positional markers and wrapping.
        # The shared formatter changes only highlighting of the resulting text.
        rendered = BeautifulSoup(
            format_signature(code.get_text().rstrip("\n")), "html.parser"
        )
        block.replace_with(rendered.div)
    # Include method headings in their disclosure for reliable anchor navigation.
    for wrapper in soup.select(".doc-function"):
        details = wrapper.find("details", recursive=False)
        heading = wrapper.find(
            lambda e: (
                e.name in ("a", "h1", "h2", "h3", "h4", "h5", "h6") and e.has_attr("id")
            ),
            recursive=False,
        )
        if details and heading:
            heading["class"] = list(heading.get("class", [])) + ["member-anchor"]
            details.insert(1, heading.extract())
    for public in REFERENCES.get(page.file.src_uri, []):
        canonical = RESOLVED[public]
        for target in soup.select("[id]"):
            key = target["id"]
            if key == canonical or key.startswith(canonical + "."):
                alias = public + key[len(canonical) :]
                config.plugins["autorefs"].register_anchor(
                    page,
                    alias,
                    key,
                    primary=(page.file.src_uri, public) not in SECONDARY_REFERENCES,
                )
    remove_api_toc_entries(
        page.toc, [RESOLVED[name] for name in REFERENCES.get(page.file.src_uri, [])]
    )
    return str(soup)


def on_nav(nav, **kwargs):
    global NAVIGATION
    NAVIGATION = nav
    return nav


def on_env(env, config, **kwargs):
    # Markdown/API references are ready here, before any page (including 404)
    # is rendered. Build the native navigation once for the entire site.
    handler = config.plugins["mkdocstrings"].get_handler("python")
    options = handler.get_options({})
    api_kinds = {
        public: handler.collect(canonical, options).kind.value
        for public, canonical in RESOLVED.items()
    }
    expand_article_navigation(
        NAVIGATION, REFERENCES, RESOLVED, config.extra.get("navigation_icons", {}), api_kinds
    )
    return env


def _revision():
    result = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


@event_priority(100)
def on_post_build(config):
    handler = config.plugins["mkdocstrings"].get_handler("python")
    options = handler.get_options({})
    objects, issues = {}, []
    for page, identifiers in REFERENCES.items():
        for identifier in identifiers:
            obj = handler.collect(RESOLVED[identifier], options)
            candidates = [obj]
            if obj.is_class:
                candidates += [
                    m for n, m in obj.all_members.items() if not n.startswith("_")
                ]
            for item in candidates:
                if item.path in objects:
                    continue
                doc = item.docstring
                sections = doc.parsed if doc else []
                objects[item.path] = {
                    "canonical_path": item.canonical_path,
                    "kind": item.kind.value,
                    "docstring_sha256": hashlib.sha256(
                        (doc.value if doc else "").encode()
                    ).hexdigest(),
                    "sections": [s.title or s.kind.value for s in sections],
                }
                function = (
                    item.all_members.get("__init__")
                    if item.is_class
                    else item
                    if item.is_function
                    else None
                )
                if function:
                    real = {
                        p.name
                        for p in function.parameters
                        if p.name not in ("self", "cls")
                    }
                    contract_sections = sections
                    if item.is_class and function.docstring:
                        contract_sections = sections + function.docstring.parsed
                    contract_kinds = {"parameters"}
                    if item.is_class and "dataclass" in item.labels:
                        # Dataclass constructor fields are conventionally
                        # documented once in the class Attributes section.
                        contract_kinds.add("attributes")
                    described = {
                        p.name.lstrip("*")
                        for s in contract_sections
                        if s.kind.value in contract_kinds
                        for p in s.value
                    }
                    for name in sorted(described - real):
                        issues.append(
                            {
                                "object": item.path,
                                "kind": "unknown_parameter",
                                "name": name,
                            }
                        )
                    for name in sorted(real - described):
                        issues.append(
                            {
                                "object": item.path,
                                "kind": "undocumented_parameter",
                                "name": name,
                            }
                        )
    prefixes = {name.split(".")[0] for names in REFERENCES.values() for name in names}
    imported = sorted(name for name in prefixes if name in sys.modules)
    if imported:
        raise RuntimeError(
            "Documented package was imported during a static build: "
            + ", ".join(imported)
        )
    # Compare against the package's public exports, so new public APIs cannot
    # silently disappear from the website as the package evolves.
    documented = {name for names in REFERENCES.values() for name in names}
    coverage = {}
    omissions = config.extra.get("api_omissions", {})
    if any(not isinstance(reason, str) or not reason.strip() for reason in omissions.values()):
        raise ValueError("Each API omission must have an explicit scope reason.")
    public_objects = set()
    for module_name in config.extra.get("api_modules", []):
        module = handler.collect(module_name, options)
        expected = {module_name + "." + name for name in module.exports}
        public_objects.update(expected)
        omitted = {name: omissions[name] for name in sorted(expected & omissions.keys())}
        missing = sorted(expected - documented - omissions.keys())
        coverage[module_name] = {
            "public_objects": sorted(expected),
            "documented": sorted(expected & documented),
            "omitted": omitted,
            "missing": missing,
        }
        if missing:
            raise ValueError(
                "Public APIs missing from the website: " + ", ".join(missing)
            )
    stale_omissions = omissions.keys() - public_objects
    if stale_omissions:
        raise ValueError("API omissions no longer exported: " + ", ".join(sorted(stale_omissions)))
    if omissions.keys() & documented:
        raise ValueError("Omitted APIs must not also appear in the website.")
    output = Path(config.site_dir) / "_api"
    output.mkdir(exist_ok=True)
    (output / "build-manifest.json").write_text(
        json.dumps(
            {
                "source_revision": _revision(),
                "source_mode": "static",
                "pages": REFERENCES,
                "exports": RESOLVED,
                "objects": objects,
                "coverage": coverage,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "quality-issues.json").write_text(json.dumps(issues, indent=2) + "\n")
    (output / "tutorial-manifest.json").write_text(json.dumps(TUTORIAL_REPORT, indent=2) + "\n")
