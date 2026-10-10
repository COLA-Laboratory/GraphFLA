"""Contracts of the landing page: one source for every colour, and no number that disagrees with its data."""

import ast
import contextlib
import csv
import io
import hashlib
import json
from unittest.mock import patch
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from bs4 import BeautifulSoup

DOCS = Path(__file__).resolve().parents[1]
REPO = DOCS.parent
HOME = DOCS / "home"
sys.path.insert(0, str(HOME))
import build as home_build
import page as home_page
from charts import Series, line_chart
from figures import load_tokens, render
from figures.camera import Camera
from figures.palette import Palette
from figures.scenes import hero
from figures import surfaces
from figures import examples as workflow_figures

COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(")
STYLED_SOURCES = [path for pattern in ("styles/*.css", "templates/**/*.html", "icons/**/*.svg")
                  for path in HOME.glob(pattern) if path.name != "tokens.css"]


def public_names(module):
    """Return ``__all__`` of a GraphFLA subpackage without importing it."""
    tree = ast.parse((REPO / "graphfla" / module / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", "") == "__all__" for target in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError(f"graphfla.{module} has no __all__")


class LandingPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="graphfla-home-")
        out = home_build.build(Path(cls.temp.name))
        cls.pages = {name: BeautifulSoup((out / f"{name}.html").read_text(), "html.parser")
                     for name in ("index", "styleguide")}
        cls.content = yaml.safe_load((HOME / "content.yml").read_text())
        cls.tutorials = yaml.safe_load((DOCS / "tutorials.yml").read_text())["notebooks"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    # -- design system ---------------------------------------------------------

    def test_colours_are_defined_only_in_the_tokens_file(self):
        offenders = {str(path.relative_to(HOME)): COLOUR.findall(path.read_text()) for path in STYLED_SOURCES}
        self.assertEqual({name: found for name, found in offenders.items() if found}, {})

    def test_templates_style_nothing_inline(self):
        # Only custom properties may be set inline: they carry data (a position, a share), not styling.
        for path in HOME.glob("templates/**/*.html"):
            self.assertEqual(re.findall(r'style="(?!--)[^"]*"', path.read_text()), [], path.name)

    def test_every_referenced_token_is_defined(self):
        sources = "\n".join(path.read_text() for path in STYLED_SOURCES)
        declared = set(load_tokens()) | set(re.findall(r"(--[\w-]+)\s*:", sources))
        self.assertEqual(set(re.findall(r"var\((--[\w-]+)", sources)) - declared, set())

    def test_classes_used_and_classes_styled_are_the_same_set(self):
        css = "\n".join(path.read_text() for path in HOME.glob("styles/*.css"))
        styled = set(re.findall(r"\.(gfl-[\w-]+)", css))
        used = {name for page in self.pages.values() for tag in page.find_all(class_=True)
                for name in tag["class"] if name.startswith("gfl-")}
        self.assertEqual(used - styled, set(), "classes without a style rule")
        self.assertEqual(styled - used, set(), "style rules that no page uses")

    def test_figures_take_their_colours_from_the_tokens(self):
        tokens = load_tokens()
        accent = tokens["--gfl-color-accent"]
        with tempfile.TemporaryDirectory() as out:
            render(out, {**tokens, "--gfl-color-accent": "#123456"}, names=["neutrality"])
            drawing = (Path(out) / "neutrality.svg").read_text()
        self.assertIn("#123456", drawing)
        self.assertNotIn(accent, drawing)

    # -- page ------------------------------------------------------------------

    def test_external_link_arrows_render_as_text_on_ios(self):
        # Without U+FE0E, iOS draws "↗" as a colour emoji.
        page = str(self.pages["index"])
        self.assertIn("↗\ufe0e", page)
        self.assertEqual(page.count("↗"), page.count("↗\ufe0e"))

    def test_page_has_every_section_and_a_footer(self):
        page = self.pages["index"]
        for anchor in ("top", "how", "metrics", "start", "performance", "cases"):
            self.assertIsNotNone(page.find(id=anchor), anchor)
        self.assertIsNotNone(page.find("footer"))

    def test_every_tutorial_has_a_cover_card_and_accessible_html_copy(self):
        cards = self.pages["index"].select(".gfl-case")
        self.assertEqual([card["href"] for card in cards], [f"tutorials/{item['slug']}/" for item in self.tutorials])
        for item, card in zip(self.tutorials, cards):
            self.assertEqual(card.h3.get_text(strip=True), item["title"])
            self.assertEqual(card.select_one(".gfl-case__topic").get_text(strip=True), item["study"])
            self.assertEqual(card["aria-labelledby"], card.h3["id"])
            self.assertEqual(card["aria-describedby"], card.p["id"])
            image = card.find("img")
            self.assertEqual(image["alt"], "")
            self.assertEqual((image["width"], image["height"]), ("1536", "1024"))
            self.assertTrue((Path(self.temp.name) / image["src"]).is_file())
            self.assertIsNone(card.find("svg"))
            self.assertIsNone(card.select_one(".gfl-case__description"))
            self.assertIsNone(card.select_one(".gfl-case__source"))

    def test_cover_style_configuration_builds_each_preserved_collection(self):
        self.assertIn(self.content["cases"]["cover_style"], ("a", "b", "c"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = yaml.safe_load((HOME / "content.yml").read_text())
            config = root / "content.yml"
            for style in ("a", "b", "c"):
                with self.subTest(style=style):
                    content["cases"]["cover_style"] = style
                    config.write_text(yaml.safe_dump(content))
                    with patch.object(home_page, "CONTENT", config):
                        out = home_build.build(root / "reused-output")
                    page = BeautifulSoup((out / "index.html").read_text(), "html.parser")
                    images = page.select(".gfl-case__image")
                    self.assertEqual(len(images), len(self.tutorials))
                    for item, image in zip(self.tutorials, images):
                        self.assertEqual(image["src"], f"assets/covers/{style}/{item['slug']}.webp")
                        self.assertTrue((out / image["src"]).is_file())
                    self.assertEqual({p.name for p in (out / "assets/covers").iterdir()}, {style})
        with self.assertRaisesRegex(ValueError, "cases.cover_style"):
            home_page.case_cards("tutorials/", self.content["cases"]["stories"], "unknown")
        with patch.object(home_page, "HOME", Path(self.temp.name) / "missing"):
            with self.assertRaisesRegex(FileNotFoundError, "Missing case cover"):
                home_page.case_cards("tutorials/", self.content["cases"]["stories"], "c")

    def test_all_four_research_papers_have_individual_links(self):
        papers = self.pages["index"].select(".gfl-citation")
        self.assertEqual(len(papers), 4)
        self.assertEqual([p.select_one(".gfl-eyebrow").get_text(strip=True).split()[0]
                          for p in papers], ["NeurIPS", "ISSTA", "KDD", "IJCAI"])
        self.assertEqual(len({p.select_one("a")["href"] for p in papers}), 4)

    # -- content agrees with the package and the data ----------------------------

    def test_listed_api_names_exist(self):
        functions = {name for card in self.content["metrics"]["cards"] for name in card["functions"]}
        self.assertEqual(functions - public_names("analysis"), set())
        classes = {entry["name"] for entry in self.content["start"]["classes"]["entries"]}
        self.assertEqual(classes, public_names("landscape"))

    def test_headline_counts_match_what_they_count(self):
        stats = [stat["value"] for stat in self.content["hero"]["stats"]]
        generators = public_names("problems") - {"OptimizationProblem"}
        self.assertEqual(stats[1:4], [str(len(public_names("landscape"))), str(len(generators)),
                                     str(len(self.tutorials))])

    def test_archived_data_examples_match_their_recorded_inputs(self):
        recorded = json.loads((HOME / "examples/results.json").read_text())["scenarios"]
        self.assertEqual(set(recorded), {"protein", "chemistry", "materials", "software", "hpo"})
        for name, result in recorded.items():
            path = HOME / "examples" / (name + ".csv")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), result["csv_sha256"])
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), result["count"])
            for node in result["nodes"]:
                self.assertAlmostEqual(float(rows[node["id"]][result["outcome"]]), node["value"])
            for a, b in result["edges_shown"]:
                left, right = (rows[result["nodes"][i]["id"]] for i in (a, b))
                if name == "protein":
                    self.assertEqual(sum(x != y for x, y in zip(left["sequences"], right["sequences"])), 1)
                else:
                    self.assertEqual(sum(left[c] != right[c] for c in result["columns"]), 1)
            self.assertAlmostEqual(sum(result["interactions"].values()), 1.0)
        self.assertEqual(recorded["hpo"]["count"], 4 * 4 * 3)
        for section in ("#how", "#start"):
            tabs = self.pages["index"].select(f'{section} .gfl-scenario-tabs [role="tab"]')
            self.assertEqual(len(tabs), 5)
            self.assertEqual(sum(t["aria-selected"] == "true" for t in tabs), 1)
        self.assertFalse(self.pages["index"].select('.gfl-hero .gfl-badge'))
        self.assertIn("Illustrative data and profiles", self.pages["index"].select_one('#how').get_text())
        for panel in self.pages["index"].select('.gfl-scenario'):
            self.assertEqual(len(panel.select('tbody tr')), 8)
            self.assertEqual(len(panel.select('.gfl-report .gfl-meter')), 4)

    def test_hero_best_path_ends_at_the_highest_peak(self):
        figure = hero(Palette.from_tokens("page", load_tokens()))
        label = figure.labels["global_peak"]
        endpoint = (label["x"] * figure.width / 100 - 23,
                    label["y"] * figure.height / 100 + 14)
        camera = Camera(385, 338, 505, 38, 27, 265)
        peak = camera.project(0.40, 0.36, surfaces.hero(0.40, 0.36))
        self.assertLess(sum((a - b) ** 2 for a, b in zip(endpoint, peak)), 9)

    def test_workflow_layers_share_projection_and_keep_every_neighbor_edge(self):
        upper = Camera(**workflow_figures.UPPER)
        lower = Camera(**workflow_figures.LOWER)
        offsets = []
        for u, v in ((0, 0), (1, 0), (1, 1), (0, 1), (0.37, 0.62)):
            top, bottom = upper.project(u, v), lower.project(u, v)
            self.assertEqual(top[0], bottom[0])
            offsets.append(bottom[1] - top[1])
        self.assertLess(max(offsets) - min(offsets), 1e-9)
        self.assertGreater(offsets[0], 0)
        for key, ((nodes, edges), _, _) in workflow_figures.SCENARIOS.items():
            with self.subTest(scenario=key):
                figure, stats = workflow_figures.draw(key, load_tokens())
                elements = list(ET.fromstring(figure.svg()).iter())
                drawn_nodes = {int(e.attrib["data-node"]) for e in elements if "data-node" in e.attrib}
                drawn_edges = {tuple(map(int, e.attrib["data-edge"].split("-")))
                               for e in elements if "data-edge" in e.attrib}
                self.assertEqual(drawn_nodes, set(range(len(nodes))))
                self.assertEqual(drawn_edges, set(edges))
                guides = [e for e in elements if e.attrib.get("stroke-dasharray") == "2 6"]
                self.assertEqual(len(guides), stats["local_optima"])
                for guide in guides:
                    coordinates = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", guide.attrib["d"])))
                    self.assertAlmostEqual(coordinates[0], coordinates[2], places=2)
                self.assertEqual(figure.labels["surface"]["x"], figure.labels["graph"]["x"])

    def test_quick_start_examples_load_and_analyze_their_datasets(self):
        try:
            import graphfla.analysis  # noqa: F401
        except ImportError as error:
            self.skipTest(f"GraphFLA's runtime dependencies are not installed: {error}")
        import pandas as pd
        reader = pd.read_csv
        suzuki = reader(DOCS.parent / "tutorials/datasets/data/suzuki.csv", keep_default_na=False)
        suzuki = suzuki.loc[(suzuki.base == "NaOH") & (suzuki.ligand != "None"), ["ligand", "solvent", "response_uv_pct"]]
        recorded = lambda name: reader(HOME / "examples" / (name + ".csv"))
        # Small real datasets, renamed to the columns each example reads.
        frames = {
            "variants.csv": recorded("protein").rename(columns={"sequences": "sequence", "fitness": "activity"}),
            "reactions.csv": suzuki.rename(columns={"ligand": "catalyst", "response_uv_pct": "yield"}),
            "alloys.csv": recorded("materials").rename(columns={"H1000_HV": "hardness"}),
            "builds.csv": recorded("software").rename(columns={"compile_time_raw": "time"}),
            "grid_search.csv": recorded("hpo").rename(
                columns={"max_depth": "depth", "min_samples_leaf": "leaf", "max_features": "features"}),
        }
        examples = self.content["start"]["code"]["examples"]
        self.assertEqual([e["id"] for e in examples], [s["id"] for s in self.content["how"]["scenarios"]])
        for example in examples:
            with self.subTest(example=example["id"]):
                *statements, last = example["source"].strip().splitlines()
                namespace = {}
                with patch("pandas.read_csv", side_effect=lambda name: frames[name].copy()), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec("\n".join(statements), namespace)
                    profile = eval(last, namespace)
                landscape = namespace["landscape"]
                self.assertEqual(landscape.n_configs, len(namespace["df"]))
                self.assertEqual(profile["local_optima_ratio"], landscape.n_lo / landscape.n_configs)


class BenchmarkChart(unittest.TestCase):
    def chart(self, slow, fast):
        return line_chart(["a", "b"], [Series("slow", slow), Series("fast", fast, emphasis=True)],
                          [(0.0, "0"), (10.0, "10")], lambda value: f"{value:.0f}", x_title="x", title="t")

    def test_points_that_read_the_same_share_one_label(self):
        svg = self.chart((2.0, 8.0), (2.1, 3.0))
        self.assertEqual(svg.count(">both 2<"), 1)
        self.assertEqual(svg.count("<circle"), 3)

    def test_close_points_with_different_values_keep_their_own_labels(self):
        svg = self.chart((2.6, 8.0), (2.4, 3.0))
        self.assertNotIn("both", svg)
        self.assertEqual(svg.count("<circle"), 4)


if __name__ == "__main__":
    unittest.main()
