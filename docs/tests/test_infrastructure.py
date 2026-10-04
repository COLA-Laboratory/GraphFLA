"""Documentation contracts, using a disposable package that cannot be imported."""

import ast
import doctest
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen
from urllib.error import URLError

from bs4 import BeautifulSoup
import yaml

DOCS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOCS / "_support"))
from markup import adapt
from checks import site_errors


def assert_links(test, site):
    test.assertEqual(site_errors(site), [])


class DocumentationInfrastructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="graphfla-docs-contract-")
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "source"
        shutil.copytree(DOCS / "tests/fixtures", cls.source)
        cls.content = cls.root / "content"
        cls.content.mkdir()
        (cls.content / "index.md").write_text(
            "---\napi_narrative: true\n---\n\n# Authored narrative\n\nNARRATIVE_MUST_NOT_CHANGE.\n\n## Meaning\n\n::: docfixture.metric\n\n::: docfixture.stream\n"
        )
        (cls.content / "box.md").write_text(
            "---\napi_class: true\n---\n\n::: docfixture.Box\n"
        )
        (cls.content / "grouped.md").write_text(
            "---\napi_grouped_classes: true\n---\n\n# Problems\n\n## Box introduction\n\nPRESERVED_CLASS_INTRO.\n\n::: docfixture.Box\n    options:\n      skip_local_inventory: true\n\n## Base introduction\n\n::: docfixture.models.Base\n"
        )
        cfg = yaml.safe_load((DOCS / "mkdocs.yml").read_text())
        cfg.update(
            docs_dir=str(cls.content),
            site_dir=str(cls.root / "site"),
            watch=[str(cls.source)],
            nav=[
                {"Narrative": "index.md"},
                {"Box": "box.md"},
                {"Problems": "grouped.md"},
            ],
            hooks=[str(DOCS / "_support/hooks.py")],
            extra_css=[],
            extra_javascript=[],
        )
        cfg["extra"]["api_source_root"] = str(cls.source)
        cfg["theme"]["custom_dir"] = str(DOCS / "overrides")
        cfg["extra"]["api_modules"] = ["docfixture"]
        cfg["extra"]["api_omissions"] = {}
        cfg["extra"]["tutorial_catalog"] = None
        cfg["plugins"][1]["mkdocstrings"]["custom_templates"] = str(DOCS / "templates")
        cls.config = cls.root / "mkdocs.yml"
        cls.config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cls.build()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def build(cls, success=True):
        env = os.environ.copy()
        env.pop("GRAPHFLA_DOCS_SOURCE_ROOT", None)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "-f",
                str(cls.config),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        if success and result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result

    def page(self, name="index"):
        file = (
            self.root
            / "site"
            / ("index.html" if name == "index" else name + "/index.html")
        )
        return BeautifulSoup(file.read_text(), "html.parser")

    def test_complete_sections_and_narrative_boundary(self):
        page = self.page()
        text = page.get_text()
        for sentinel in [
            "NARRATIVE_MUST_NOT_CHANGE",
            "SOURCE_PARAMETER_V1",
            "SOURCE_RETURN_V1",
            "WARNING_SENTINEL",
            "NOTE_SENTINEL",
            "REFERENCE_SENTINEL",
            "YIELD_SENTINEL",
            "RECEIVE_SENTINEL",
            "Deprecated 2.0",
            "Use a newer entry point",
            ">>> 2 + 3",
        ]:
            self.assertIn(sentinel, text)
        self.assertIn("A summary that belongs to the API.", text)
        self.assertIn("EXTENDED_DESCRIPTION_SENTINEL", text)
        for kind in [
            "parameters",
            "other-parameters",
            "returns",
            "raises",
            "warns",
            "warning",
            "note",
            "references",
            "see-also",
            "examples",
            "yields",
            "receives",
        ]:
            self.assertTrue(page.select(f'[data-api-section="{kind}"]'), kind)
        self.assertGreaterEqual(len(page.select(".arithmatex")), 2)
        self.assertTrue(page.select('a[href="#docfixture.metric.metric--ref-1"]'))
        self.assertTrue(page.select('a[href="#docfixture.metric.stream"]'))

    def test_static_collection_and_shadowed_export(self):
        data = json.loads((self.root / "site/_api/build-manifest.json").read_text())
        self.assertEqual(data["source_mode"], "static")
        self.assertEqual(
            data["exports"]["docfixture.metric"], "docfixture.metric.metric"
        )
        self.assertEqual(
            data["objects"]["docfixture.metric.metric"]["kind"], "function"
        )
        self.assertNotIn("docfixture", sys.modules)
        self.assertEqual(data["coverage"]["docfixture"]["missing"], [])
        self.assertEqual(len(data["coverage"]["docfixture"]["documented"]), 3)

    def test_public_api_omission_fails_build(self):
        page = self.content / "index.md"
        original = page.read_text()
        try:
            page.write_text(original.replace("::: docfixture.stream\n", ""))
            result = self.build(success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "Public APIs missing from the website: docfixture.stream",
                result.stdout + result.stderr,
            )
        finally:
            page.write_text(original)
            self.build()

    def test_explicit_scope_omission_is_reported_and_validated(self):
        package = self.source / "docfixture/__init__.py"
        original_package = package.read_text()
        original_config = self.config.read_text()
        reason = "Outside the requested website scope."
        try:
            package.write_text(
                original_package.replace('"Box"]', '"Box", "scoped_out"]')
                + '\n\ndef scoped_out():\n    """An intentionally unlisted API."""\n'
            )
            cfg = yaml.safe_load(original_config)
            cfg["extra"]["api_omissions"] = {"docfixture.scoped_out": reason}
            self.config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            self.build()
            report = json.loads((self.root / "site/_api/build-manifest.json").read_text())
            coverage = report["coverage"]["docfixture"]
            self.assertEqual(coverage["omitted"], {"docfixture.scoped_out": reason})
            self.assertEqual(coverage["missing"], [])
            self.assertNotIn("docfixture.scoped_out", report["objects"])
            package.write_text(original_package)
            result = self.build(success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("API omissions no longer exported: docfixture.scoped_out", result.stdout + result.stderr)
        finally:
            package.write_text(original_package)
            self.config.write_text(original_config)
            self.build()

    def test_class_inheritance_properties_constructor_and_tabs(self):
        page = self.page("box")
        self.assertEqual(len(page.select(".api-class-tabs > input[type=radio]")), 5)
        text = page.get_text()
        for sentinel in [
            "INITIAL_SIZE",
            "INHERITED_PARAMETER",
            "INHERITED_RETURN",
            "PROPERTY_SENTINEL",
            "DIRECT_PARAMETER",
            "CLASS_NOTE",
            "CONSTRUCTOR_NOTE",
            "Inherited from",
        ]:
            self.assertIn(sentinel, text)
        self.assertEqual(
            len(page.select('[data-api-path="docfixture.models.Box.inherited"]')), 1
        )
        target = page.find(id="docfixture.models.Box.inherited")
        self.assertIsNotNone(target.find_parent("details"))

    def test_grouped_classes_keep_introductions_and_inline_apis(self):
        page = self.page("grouped")
        article = page.select_one("article")
        self.assertEqual(len(article.select("h1")), 1)
        self.assertEqual(len(article.select(".api-inline-class")), 2)
        self.assertIn("PRESERVED_CLASS_INTRO.", article.get_text())
        box = article.select_one(
            '.api-inline-class[data-api-path="docfixture.models.Box"]'
        )
        labels = [
            e.get_text(strip=True)
            for e in box.select(".api-inline-class-tabs > .tabbed-labels label")
        ]
        self.assertEqual(
            labels, ["Parameters", "Attributes", "Methods", "Examples", "Notes"]
        )
        self.assertNotIn("Methods at a glance", box.get_text())
        self.assertIn("docfixture.Box(", box.select_one(".api-signature").get_text())
        self.assertTrue(box.select(".method-detail .api-function-tabs"))

    def test_sidebar_members_and_editorial_toc_are_separate(self):
        page = self.page()
        primary = page.select_one(".md-nav--primary")
        self.assertTrue(primary.select('a[href="#docfixture.metric.metric"]'))
        secondary = page.select_one(".md-sidebar--secondary")
        self.assertIn("Meaning", secondary.get_text())
        self.assertFalse(secondary.select('a[href*="#docfixture."]'))
        # Every article's member menu is also available from other pages.
        other = self.page("box").select_one(".md-nav--primary")
        self.assertTrue(other.select('a[href="../#docfixture.metric.metric"]'))
        self.assertTrue(other.select('a[href="../grouped/#docfixture.models.Box"]'))
        grouped = self.page("grouped").select_one(".md-sidebar--secondary")
        self.assertIn("Box introduction", grouped.get_text())
        self.assertNotIn("docfixture", grouped.get_text())

    def test_function_sections_are_independent_tabs_with_visible_notices(self):
        page = self.page()
        metric = page.select_one('[data-api-path="docfixture.metric.metric"]')
        tabs = metric.select_one(".api-function-tabs")
        labels = tabs.select(":scope > .tabbed-labels label")
        panels = tabs.select(":scope > .tabbed-content > .tabbed-block")
        self.assertEqual(len(labels), len(panels))
        self.assertEqual(labels[0].get_text(strip=True), "Parameters")
        self.assertEqual(
            [label.get_text(strip=True) for label in labels],
            ["Parameters", "Returns", "Examples", "Notes", "Raises & Warns"],
        )
        summary = metric.select_one(".api-function-summary")
        related = metric.select_one(".api-function-related")
        self.assertEqual(
            summary.get_text(strip=True), "A summary that belongs to the API."
        )
        self.assertIsNone(summary.find_parent(class_="api-function-tabs"))
        self.assertIsNone(related.find_parent(class_="api-function-tabs"))
        self.assertEqual(summary.find_next_sibling(), related)
        self.assertIn(
            "EXTENDED_DESCRIPTION_SENTINEL",
            tabs.select_one(".api-extended-description").get_text(),
        )
        self.assertEqual(
            metric.select_one('[data-api-section="note"]').find_parent(
                class_="tabbed-block"
            ),
            metric.select_one('[data-api-section="references"]').find_parent(
                class_="tabbed-block"
            ),
        )
        self.assertEqual(
            metric.select_one('[data-api-section="raises"]').find_parent(
                class_="tabbed-block"
            ),
            metric.select_one('[data-api-section="warns"]').find_parent(
                class_="tabbed-block"
            ),
        )
        for label, panel in zip(labels, panels):
            control = tabs.find("input", id=label["for"])
            self.assertIsNotNone(control)
            self.assertTrue(panel.select("[data-api-section]"), label.get_text())
        notice = metric.select_one(".admonition.warning")
        self.assertIsNotNone(notice)
        self.assertIsNone(notice.find_parent(class_="api-function-tabs"))
        stream = page.select_one('[data-api-path="docfixture.metric.stream"]')
        self.assertNotEqual(
            tabs.find("input")["name"],
            stream.select_one(".api-function-tabs input")["name"],
        )
        references = page.find(id="docfixture.metric.metric--ref-1")
        self.assertIsNotNone(references.find_parent(class_="tabbed-block"))
        method = self.page("box").select_one(
            '[data-api-path="docfixture.models.Box.inherited"]'
        )
        self.assertIsNotNone(method.select_one("details .api-function-tabs"))

    def test_links_ids_and_native_copy_content(self):
        assert_links(self, self.root / "site")
        signature = self.page().select_one(".api-signature code").get_text()
        self.assertIn("docfixture.metric(", signature)
        self.assertIn("threshold: float = 0.5", signature)
        self.assertIn("/", signature)
        self.assertIn("**kwargs", signature)
        self.assertIn("-> float", signature)

    def test_example_execution_is_a_separate_explicit_check(self):
        tree = ast.parse((self.source / "docfixture/metric.py").read_text())
        doc = ast.get_docstring(tree.body[0])
        example = doctest.DocTestParser().get_doctest(doc, {}, "metric", "metric.py", 0)
        runner = doctest.DocTestRunner()
        result = runner.run(example)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.attempted, 1)

    def test_unsupported_directives_are_not_silently_discarded(self):
        with self.assertRaisesRegex(ValueError, "Unsupported docstring directive"):
            adapt(".. unknown-directive::\n    detail", "fixture.fn")

    def test_live_reload_reads_changed_source_without_restart(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        address = f"127.0.0.1:{port}"
        path = self.source / "docfixture/metric.py"
        original = path.read_text()
        env = os.environ.copy()
        env.pop("GRAPHFLA_DOCS_SOURCE_ROOT", None)
        with (self.root / "serve.log").open("w+") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mkdocs",
                    "serve",
                    "-f",
                    str(self.config),
                    "--dev-addr",
                    address,
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

            def wait_for(sentinel):
                deadline = time.monotonic() + 20
                while process.poll() is None and time.monotonic() < deadline:
                    try:
                        with urlopen("http://" + address + "/", timeout=1) as response:
                            if sentinel in response.read().decode():
                                return
                    except (URLError, TimeoutError):
                        pass
                    time.sleep(0.1)
                log.seek(0)
                self.fail(f"Live reload did not render {sentinel}:\n" + log.read())

            try:
                wait_for("SOURCE_PARAMETER_V1")
                path.write_text(
                    original.replace("SOURCE_PARAMETER_V1", "SOURCE_LIVE_V2")
                )
                wait_for("SOURCE_LIVE_V2")
            finally:
                process.terminate()
                process.wait(timeout=5)
                path.write_text(original)

    def test_z_source_change_is_reflected_without_page_edits(self):
        path = self.source / "docfixture/metric.py"
        original = path.read_text()
        page_before = (self.content / "index.md").read_bytes()
        try:
            path.write_text(
                original.replace("threshold: float = 0.5", "cutoff: float = 2.0")
                .replace("threshold : float", "cutoff : float")
                .replace("SOURCE_PARAMETER_V1", "SOURCE_PARAMETER_V2")
                .replace("SOURCE_RETURN_V1", "SOURCE_RETURN_V2")
            )
            self.build()
            text = self.page().get_text()
            self.assertIn("cutoff: float = 2.0", text)
            self.assertIn("SOURCE_PARAMETER_V2", text)
            self.assertIn("SOURCE_RETURN_V2", text)
            self.assertNotIn("SOURCE_PARAMETER_V1", text)
            self.assertEqual((self.content / "index.md").read_bytes(), page_before)
            self.assertIn("NARRATIVE_MUST_NOT_CHANGE", text)
        finally:
            path.write_text(original)
            self.build()

    def test_zz_missing_object_fails_build(self):
        path = self.content / "index.md"
        original = path.read_text()
        try:
            path.write_text(original + "\n::: docfixture.does_not_exist\n")
            result = self.build(success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does_not_exist", result.stdout + result.stderr)
        finally:
            path.write_text(original)
            self.build()


if __name__ == "__main__":
    unittest.main()
