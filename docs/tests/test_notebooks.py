"""Notebook rendering preserves saved evidence and never executes code on build."""

import re
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import nbformat
from mkdocs.structure.files import Files

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from notebook_sources import source_hash
from notebooks import add_tutorials


class NotebookRendering(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        (self.root / "data/input.csv").write_text("x,y\n1,2\n")
        self.nb = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    "# Example\n\nNARRATIVE_SENTINEL with $x^2$."
                ),
                nbformat.v4.new_code_cell(
                    'raise RuntimeError("must not execute")\n# read "data/input.csv"',
                    execution_count=1,
                    outputs=[
                        nbformat.v4.new_output(
                            "stream", name="stdout", text="SAVED_OUTPUT\n"
                        ),
                        nbformat.v4.new_output(
                            "display_data",
                            data={
                                "text/html": '<style>.dataframe {color:red}</style><table class="dataframe"><tr><td>TABLE_SENTINEL</td></tr></table>',
                                "text/plain": "TABLE_FALLBACK",
                            },
                        ),
                    ],
                ),
            ]
        )
        self.nb.metadata.language_info = {"name": "python"}
        self.nb.metadata.graphfla = {
            "executed_source_sha256": source_hash(self.nb),
            "source_revision": "fixture",
        }
        self.path = self.root / "example.ipynb"
        nbformat.write(self.nb, self.path)
        self.catalog = {
            "source_dir": self.root,
            "notebooks": [
                {
                    "notebook": self.path.name,
                    "slug": "example",
                    "title": "Chemical reaction optimization",
                    "study": "Example chemistry dataset",
                    "icon": "material/flask-outline",
                    "data": ["input.csv"],
                    "topic": "Example",
                    "variables": "x",
                    "objective": "y",
                }
            ],
        }
        self.config = SimpleNamespace(
            site_dir=str(self.root / "site"),
            use_directory_urls=True,
            plugins=SimpleNamespace(_current_plugin="fixture"),
        )

    def test_saved_content_and_downloads_share_the_notebook(self):
        files = Files([])
        report = add_tutorials(files, self.config, self.catalog)
        page = files.get_file_from_path("tutorials/example.md").content_string
        self.assertIn("# Chemical reaction optimization\n\n*Example chemistry dataset*", page)
        self.assertNotIn("# Example\n", page)
        for text in [
            "NARRATIVE_SENTINEL",
            "$x^2$",
            "raise RuntimeError",
            "SAVED_OUTPUT",
            "TABLE_SENTINEL",
        ]:
            self.assertIn(text, page)
        self.assertNotIn("TABLE_FALLBACK", page)
        self.assertNotIn("<style>", page)
        self.assertEqual(report[0]["code_cells"], 1)
        self.assertEqual(report[0]["outputs"], 2)
        with ZipFile(
            BytesIO(
                files.get_file_from_path(
                    "tutorials/downloads/example.zip"
                ).content_bytes
            )
        ) as bundle:
            self.assertEqual(bundle.read("example.ipynb"), self.path.read_bytes())
            self.assertEqual(
                bundle.read("data/input.csv"),
                (self.root / "data/input.csv").read_bytes(),
            )

    def test_stale_outputs_missing_data_and_errors_fail_visibly(self):
        self.nb.cells[1].source += "\n# changed"
        nbformat.write(self.nb, self.path)
        with self.assertRaisesRegex(ValueError, "changed since execution"):
            add_tutorials(Files([]), self.config, self.catalog)
        self.nb.metadata.graphfla.executed_source_sha256 = source_hash(self.nb)
        self.nb.cells[1].outputs.append(
            nbformat.v4.new_output(
                "error", ename="ValueError", evalue="failed", traceback=[]
            )
        )
        nbformat.write(self.nb, self.path)
        with self.assertRaisesRegex(ValueError, "error output"):
            add_tutorials(Files([]), self.config, self.catalog)
        self.nb.cells[1].outputs.pop()
        nbformat.write(self.nb, self.path)
        self.catalog["notebooks"][0]["data"] = []
        with self.assertRaisesRegex(ValueError, "data manifest"):
            add_tutorials(Files([]), self.config, self.catalog)


class ColabSetup(unittest.TestCase):
    def test_install_and_data_pins_match_the_package_version(self):
        root = Path(__file__).resolve().parents[2]
        version = re.search(
            r'^__version__ = "([^"]+)"',
            (root / "graphfla/__init__.py").read_text(),
            re.M,
        ).group(1)
        for path in sorted((root / "tutorials/datasets").glob("*.ipynb")):
            code = "\n".join(
                c.source
                for c in nbformat.read(path, as_version=4).cells
                if c.cell_type == "code"
            )
            with self.subTest(notebook=path.name):
                self.assertEqual(re.findall(r"graphfla==([\w.]+)", code), [version])
                self.assertEqual(
                    re.findall(r"/GraphFLA/v([\w.]+)/tutorials/", code), [version]
                )
