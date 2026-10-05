"""The real website must ship the landing page, tutorials and source-owned APIs together."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
import yaml

DOCS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOCS / "_support"))
from checks import site_errors


class LandingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="graphfla-integrated-site-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.site = Path(cls.temp.name) / "site"
        env = os.environ.copy()
        env.pop("GRAPHFLA_DOCS_SOURCE_ROOT", None)
        result = subprocess.run(
            [sys.executable, "-m", "mkdocs", "build", "--strict", "-f",
             str(DOCS / "mkdocs.yml"), "--site-dir", str(cls.site)],
            env=env, capture_output=True, text=True,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.home = BeautifulSoup((cls.site / "index.html").read_text(), "html.parser")

    def test_homepage_is_the_authored_design_with_working_links_and_assets(self):
        self.assertEqual(len(self.home.select(".gfl-home")), 1)
        self.assertEqual(len(self.home.select("header")), 1)
        self.assertEqual(len(self.home.select("h1")), 1)
        self.assertIsNotNone(self.home.select_one(".md-sidebar--primary"))
        self.assertTrue(self.home.select('.gfl-site-navigation a[href="tutorials/"]'))
        self.assertTrue(self.home.select('.gfl-site-navigation a[href="landscape/"]'))
        self.assertEqual(site_errors(self.site), [])
        # A project-site prefix must survive every homepage link and asset URL.
        for tag in self.home.select("a[href], link[href], img[src], script[src]"):
            target = tag.get("href", tag.get("src"))
            url = urlsplit(target)
            if not url.scheme and not url.netloc:
                path = urlsplit(urljoin("https://example.org/GraphFLA/", target)).path
                self.assertTrue(path.startswith("/GraphFLA/"), target)

    def test_global_header_and_search_are_shared_by_home_tutorials_and_apis(self):
        expected = ["Home", "Metrics", "Case studies", "Tutorials", "API reference"]
        for path in ("index.html", "landscape/index.html", "analysis/epistasis/index.html",
                     "tutorials/suzuki/index.html", "404.html"):
            soup = BeautifulSoup((self.site / path).read_text(), "html.parser")
            nav = soup.select_one(".md-header .gfl-site-navigation .gfl-nav")
            self.assertIsNotNone(nav, path)
            labels = [tag.get_text(" ", strip=True).replace(" ↗", "")
                      for tag in nav.select(":scope > a")]
            self.assertEqual(labels, expected, path)
            self.assertTrue(soup.select('.md-header input[aria-label="Search"]'), path)
            self.assertFalse(nav.select("details"), path)
            self.assertTrue(soup.select('.md-header__source a[href="https://github.com/COLA-Laboratory/GraphFLA"]'), path)
            for label, anchor in (("Metrics", "metrics"), ("Case studies", "cases")):
                link = next(a for a in nav.select("a") if a.get_text(strip=True) == label)
                base = "https://example.org/GraphFLA/" + path.removesuffix("index.html")
                self.assertEqual(urljoin(base, link["href"]), "https://example.org/GraphFLA/#" + anchor)

    def test_metric_chips_link_to_their_exact_source_owned_api_anchors(self):
        for chip in self.home.select(".gfl-chip"):
            url = urlsplit(chip["href"])
            destination = BeautifulSoup((self.site / url.path / "index.html").read_text(), "html.parser")
            self.assertTrue(url.fragment.startswith("graphfla.analysis."))
            self.assertIsNotNone(destination.find(id=url.fragment), chip.get_text())
            self.assertTrue(url.fragment.endswith("." + chip.get_text(strip=True)))

    def test_every_case_opens_a_tutorial_with_notebook_and_data_downloads(self):
        catalog = yaml.safe_load((DOCS / "tutorials.yml").read_text())["notebooks"]
        self.assertEqual(
            [card["href"] for card in self.home.select(".gfl-case")],
            [f"tutorials/{item['slug']}/" for item in catalog],
        )
        for item in catalog:
            page = self.site / "tutorials" / item["slug"] / "index.html"
            soup = BeautifulSoup(page.read_text(), "html.parser")
            self.assertTrue(soup.select('a[href$=".ipynb"]'), item["slug"])
            self.assertTrue(soup.select('a[href$=".zip"]'), item["slug"])
            self.assertTrue(soup.select(".md-nav--primary"))

    def test_api_coverage_profile_and_search_survive_integration(self):
        manifest = json.loads((self.site / "_api/build-manifest.json").read_text())
        for module in manifest["coverage"].values():
            self.assertEqual(module["missing"], [])
        self.assertIn("graphfla.analysis.profile", manifest["coverage"]["graphfla.analysis"]["documented"])
        search = json.loads((self.site / "search/search_index.json").read_text())["docs"]
        locations = {item["location"] for item in search}
        self.assertIn("tutorials/", locations)
        self.assertIn("analysis/profile/", locations)
        self.assertTrue(any("Understand the" in item["text"] for item in search if item["location"] == ""))
