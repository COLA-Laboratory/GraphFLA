"""Build tutorial pages and downloads from the canonical executed notebooks."""

import hashlib
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import nbformat
import yaml
from bs4 import BeautifulSoup
from jinja2 import FileSystemLoader
from mkdocs.structure.files import File
from nbconvert import MarkdownExporter
from notebook_sources import source_hash

HERE = Path(__file__).resolve().parent


def clean_output_html(html):
    """Use the site's shared table styling, without pandas' embedded CSS."""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.select("style, script"):
        element.decompose()
    for element in soup.select("[style], [border]"):
        element.attrs.pop("style", None)
        element.attrs.pop("border", None)
    for table in soup.select("table.dataframe"):
        # Material's native table styles apply to tables without a custom class.
        table.attrs.pop("class", None)
    return str(soup)


def configure(config):
    name = config.extra.get("tutorial_catalog")
    if not name:
        return None
    path = Path(config.config_file_path).parent / name
    catalog = yaml.safe_load(path.read_text())
    catalog["source_dir"] = (path.parent / catalog["source_dir"]).resolve()
    config.watch.extend([str(path), str(catalog["source_dir"])])
    config.nav.insert(
        1 if config.nav and config.nav[0] == {"Home": "index.md"} else 0,
        {
            "Tutorials": [{"Overview": "tutorials/index.md"}]
            + [
                {item["title"]: f"tutorials/{item['slug']}.md"}
                for item in catalog["notebooks"]
            ]
        },
    )
    return catalog


def add_tutorials(files, config, catalog):
    if not catalog:
        return []
    source = catalog["source_dir"]
    exporter = MarkdownExporter(
        template_file="notebook.md.j2",
        extra_loaders=[FileSystemLoader(HERE.parent / "templates")],
    )
    exporter.register_filter("notebook_html", clean_output_html)
    report, rows = [], []
    for item in catalog["notebooks"]:
        path = source / item["notebook"]
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        code = [c for c in nb.cells if c.cell_type == "code"]
        provenance = nb.metadata.get("graphfla", {})
        if provenance.get("executed_source_sha256") != source_hash(nb):
            raise ValueError(
                f"Tutorial source changed since execution: {path.name}; run docs/scripts/execute_tutorials.py"
            )
        if [c.execution_count for c in code] != list(range(1, len(code) + 1)):
            raise ValueError(f"Tutorial has unexecuted cells: {path.name}")
        if any(o.output_type == "error" for c in code for o in c.outputs):
            raise ValueError(f"Tutorial contains an error output: {path.name}")
        data_names = set(item["data"])
        referenced_data = set(
            re.findall(r'["\']data/([^"\']+)["\']', "\n".join(c.source for c in code))
        )
        if referenced_data != data_names:
            raise ValueError(
                f"Tutorial data manifest differs from its code: {path.name}"
            )
        resources = {
            "unique_key": item["slug"],
            "output_files_dir": "assets/" + item["slug"],
            "notebook_download": "downloads/" + path.name,
            "bundle_download": "downloads/" + item["slug"] + ".zip",
        }
        markdown, resources = exporter.from_notebook_node(nb, resources=resources)
        # Site-facing titles describe the task; the executed notebook stays intact.
        heading = "# " + item["title"]
        if item.get("study"):
            heading += "\n\n*" + item["study"] + "*"
        markdown = re.sub(r"(?m)^# [^\n]+", lambda _: heading, markdown, count=1)
        metadata = yaml.safe_dump({"title": item["title"], "icon": item["icon"]})
        files.append(
            File.generated(
                config,
                f"tutorials/{item['slug']}.md",
                content="---\n" + metadata + "---\n\n" + markdown,
            )
        )
        files.append(
            File.generated(
                config, "tutorials/downloads/" + path.name, abs_src_path=str(path)
            )
        )
        for name, content in resources.get("outputs", {}).items():
            files.append(File.generated(config, "tutorials/" + name, content=content))
        data_hashes = {}
        archive = BytesIO()
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
            bundle.writestr(path.name, path.read_bytes())
            for name in sorted(data_names):
                data = (source / "data" / name).read_bytes()
                data_hashes[name] = hashlib.sha256(data).hexdigest()
                bundle.writestr("data/" + name, data)
            bundle.writestr(
                "README.txt",
                (
                    f"{item['title']} — GraphFLA tutorial\n\n"
                    "Keep this notebook beside its data/ folder. Run cells in order in an\n"
                    "environment with GraphFLA and pandas installed. Data sources and\n"
                    "interpretation limits are documented in the notebook.\n\n"
                    f"Validated GraphFLA source revision: {provenance['source_revision']}\n"
                ),
            )
        files.append(
            File.generated(
                config,
                "tutorials/downloads/" + item["slug"] + ".zip",
                content=archive.getvalue(),
            )
        )
        rows.append(
            f"| [{item['title']}]({item['slug']}.md) | {item.get('study', item['topic'])} | {item['variables']} | {item['objective']} |"
        )
        report.append(
            {
                "notebook": path.name,
                "page": f"tutorials/{item['slug']}.md",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "code_cells": len(code),
                "outputs": sum(len(c.outputs) for c in code),
                "data_sha256": data_hashes,
                **provenance,
            }
        )
    index = (
        "---\nicon: material/book-open-page-variant-outline\n---\n\n# Tutorials\n\n"
        "Explore GraphFLA with nine datasets, from chemical reactions and materials to "
        "neural architectures and compiler settings. Each tutorial takes you from "
        "preparing the data to building a landscape and interpreting its analysis.\n\n"
        "Choose an optimization problem below. Every page includes Python code, saved results, and "
        "a downloadable notebook with its data.\n\n"
        "| Tutorial | Study / dataset | Variables | Objective |\n| --- | --- | --- | --- |\n"
        + "\n".join(rows)
        + "\n"
    )
    files.append(File.generated(config, "tutorials/index.md", content=index))
    return report
