"""Assemble what the landing-page templates render: site copy joined with the data it describes."""
import csv
import json
from html import escape
from pathlib import Path

import yaml
from charts import Series, format_bytes, format_duration, line_chart
from catalogue import inventory
from insight_data import load_insights
from markupsafe import Markup
from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Name, Number

HOME = Path(__file__).resolve().parent
DOCS = HOME.parent
CONTENT = HOME / "content.yml"
TUTORIALS = DOCS / "tutorials.yml"
BENCHMARK = DOCS / "content" / "assets" / "benchmarks" / "construction.json"

# Tones of a part-to-whole bar, from its first part to its last.
SHARE_TONES = ("data-low", "data-mid", "data")
CODE_CLASSES = ((Keyword, "keyword"), (Name.Function, "call"), (Number, "number"), (Comment, "comment"))


def load_page(figures):
    """Return the template context.

    Parameters
    ----------
    figures : dict
        Metadata returned by ``figures.render``.
    """
    page = yaml.safe_load(CONTENT.read_text())
    page["figures"] = figures
    page["insights"]["panels"] = load_insights()

    how = page["how"]
    for number, step in enumerate(how["steps"], start=1):
        step["number"] = number
    for scenario in how["scenarios"]:
        data = scenario["profile"]
        # Local optima and fitness-distance correlation are computed from the drawn graph.
        drawn = figures["example-" + scenario["id"]]
        scenario["figure"] = {
            "name": "example-" + scenario["id"], "labels": how["figure_labels"],
            "alt": (f"Illustrative {scenario['title'].lower()} landscape with {drawn['local_optima']} peaks, "
                    f"drawn above its neighbor graph of {drawn['nodes']} candidates."),
        }
        values = [row[-1] for row in scenario["rows"]]
        best = (max if scenario["maximize"] else min)(values)
        scenario["rows"] = [{"cells": row[:-1], "value": f"{row[-1]:g}{scenario['outcome_suffix']}",
                            "best": row[-1] == best,
                            "share": row[-1] / max(values)} for row in scenario["rows"]]
        if scenario.get("sequence"):
            baseline = scenario["rows"][0]["cells"][0]
            for row in scenario["rows"]:
                row["residues"] = [{"letter": aa, "variable": i in (4, 6, 9), "changed": aa != baseline[i]}
                                   for i, aa in enumerate(row["cells"][0])]
        labels = how["reports"]
        scenario["entries"] = [
            {"label": labels["peaks"], "value": f"{drawn['local_optima']} / {drawn['nodes']}",
             "share": drawn["local_optima"] / drawn["nodes"]},
            {"label": labels["rs"] if scenario["id"] == "protein" else labels["autocorrelation"],
             "value": f"{data['roughness']:.2f}", "share": data["roughness"]},
            {"label": labels["neutrality"], "value": f"{data['neutrality']:.0%}", "share": data["neutrality"]},
            {"label": labels["fdc"], "value": f"{drawn['fdc']:.2f}".replace("-", "\u2212"), "share": abs(drawn["fdc"])},
        ]
        scenario["interactions"] = [
            {"percent": 100 * share, "label": label, "value": f"{share:.0%}", "tone": tone}
            for share, label, tone in zip(data["interactions"], labels["parts"], SHARE_TONES)]
    page["hero"]["stats"][-1]["value"] = str(sum(group["count"] for group in inventory()))

    # Each quick-start example shares the icon of its "How it works" scenario.
    icons = {scenario["id"]: scenario["icon"] for scenario in how["scenarios"]}
    for example in page["start"]["code"]["examples"]:
        example.update(icon=icons[example["id"]], lines=highlight(example["source"]))

    page["performance"].update(benchmark(page["performance"]))
    page["cases"]["cards"] = case_cards(page["links"]["tutorials"], page["cases"]["stories"])
    return page


def highlight(source):
    """Return Python source as one HTML string per line, with syntax classes on the tokens."""
    tokens = list(PythonLexer().get_tokens(source))
    lines, current = [], ""
    for index, (kind, text) in enumerate(tokens):
        # Pygments has no token for calls: a name directly followed by "(" is one.
        if kind in Name and index + 1 < len(tokens) and tokens[index + 1][1].startswith("("):
            kind = Name.Function
        name = next((name for parent, name in CODE_CLASSES if kind in parent), None)
        for piece_index, piece in enumerate(text.split("\n")):
            if piece_index:
                lines.append(current)
                current = ""
            if piece:
                current += f'<span class="gfl-code__{name}">{escape(piece)}</span>' if name else escape(piece)
    if current:
        lines.append(current)
    # A line holding one space keeps its height where an empty one would collapse.
    return [Markup(line or " ") for line in lines]


def benchmark(spec):
    """Return the headline numbers, legend and charts of the recorded construction benchmark."""
    rows = json.loads(BENCHMARK.read_text())["summary"]
    largest = rows[-1]
    columns = [f"{row['configurations']:,}" for row in rows]

    def series(measure):
        return [Series(spec["series"][method], tuple(row[method][measure] for row in rows),
                       emphasis=method == "graphfla") for method in ("naive", "graphfla")]

    seconds, peak = series("seconds"), series("peak_rss_bytes")
    time, memory = spec["charts"]["time"], spec["charts"]["memory"]
    charts = [
        {**time, "svg": Markup(line_chart(
            columns, seconds, _ticks(seconds, TIME_TICKS), format_duration, log=True, x_title=spec["x_title"],
            title=f"{time['title']} against number of {spec['x_title']}, {time['unit']}"))},
        {**memory, "svg": Markup(line_chart(
            columns, peak, _ticks(peak, MEMORY_TICKS), format_bytes, log=True, x_title=spec["x_title"],
            title=f"{memory['title']} against number of {spec['x_title']}, {memory['unit']}"))},
    ]
    stats = [
        {"value": f"{largest['speedup']:,.0f}×",
         "label": spec["headline"]["speedup"].format(configurations=columns[-1])},
        {"value": f"{largest['peak_rss_ratio']:,.0f}×", "label": spec["headline"]["memory_saved"]},
    ]
    legend = [{"swatch": "line", "tone": "accent", "label": spec["series"]["graphfla"]},
              {"swatch": "line", "tone": "data-mid", "label": spec["series"]["naive"]}]
    return {"stats": stats, "legend": legend, "charts": charts}


# Gridlines at readable units. Each chart spans its values plus one gridline below, which
# keeps labels under the lowest points clear of the x axis.
TIME_TICKS = [(1e-4, "0.1 ms"), (1e-3, "1 ms"), (1.0, "1 s"), (60.0, "1 min"), (3600.0, "1 h"), (86400.0, "1 day"),
              (604800.0, "1 week")]
MEMORY_TICKS = [(2.0 ** 20 * 10 ** k, f"{10 ** k} MiB") for k in (1, 2)] + [
    (2.0 ** 30 * 10 ** k, f"{10 ** k} GiB") for k in range(3)] + [
    (2.0 ** 40 * 10 ** k, f"{10 ** k} TiB") for k in range(3)]


def _ticks(series, candidates):
    """Return the candidate gridlines from one below the smallest value to just above the largest."""
    values = [value for item in series for value in item.values]
    first = max(k for k, (tick, _) in enumerate(candidates) if tick <= min(values)) - 1
    last = min(k for k, (tick, _) in enumerate(candidates) if tick >= max(values))
    return candidates[first:last + 1]


def case_cards(base_url, stories):
    """Join optimization problems with their tutorial inputs, scale and destination."""
    cards = []
    catalog = yaml.safe_load(TUTORIALS.read_text())
    source = TUTORIALS.parent / catalog["source_dir"] / "data"
    for item in catalog["notebooks"]:
        # The first data file is the configuration table; others are lookups.
        with (source / item["data"][0]).open(newline="") as handle:
            rows = csv.reader(handle)
            if item.get("data_has_header", True):
                next(rows)
            count = sum(1 for _ in rows)
        cards.append({**stories[item["slug"]], "slug": item["slug"],
                      "count": count,
                      "href": f"{base_url}{item['slug']}/"})
    return cards
