"""Page descriptions are taken from the opening prose of a page and fit in a search result."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from seo import summary


class Summary(unittest.TestCase):
    def test_labels_admonitions_and_math_delimiters_are_left_out(self):
        html = ('<p><em>Study label</em></p><div class="admonition note"><p>A note.</p></div>'
                "<p>Each solution has \\(n\\) binary variables. Fitness is maximized.</p>")
        self.assertEqual(summary(html), "Each solution has n binary variables. Fitness is maximized.")

    def test_whole_sentences_are_kept_while_they_fit(self):
        html = "<p>First sentence here. Second one. A third sentence that no longer fits.</p>"
        self.assertEqual(summary(html, length=40), "First sentence here. Second one.")

    def test_a_long_first_sentence_is_cut_at_a_word_boundary(self):
        result = summary("<p>" + "landscape " * 30 + "ends here.</p>", length=50)
        self.assertLessEqual(len(result), 50)
        self.assertTrue(result.endswith("landscape…"), result)

    def test_a_page_without_prose_has_no_summary(self):
        self.assertEqual(summary("<h1>Title</h1><p>Download notebook</p>"), "")
