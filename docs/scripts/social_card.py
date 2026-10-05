"""Render the link-preview image shown when the website is shared.

Run ``python docs/scripts/social_card.py`` after changing the hero title, the hero figure or the
design tokens; it rewrites ``docs/content/assets/social-card.png`` with headless Chrome.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

DOCS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOCS / "home"))
import build as home_build

OUTPUT = DOCS / "content" / "assets" / "social-card.png"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# The 1.91:1 size that Open Graph and X large-image cards display without cropping.
WIDTH, HEIGHT = 1200, 630

CARD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="{fonts}">
{stylesheets}
<style>
  body {{ margin: 0; width: {width}px; height: {height}px; overflow: hidden; }}
  .card {{ position: relative; width: 100%; height: 100%; }}
  .card__copy {{
    position: absolute; top: 64px; left: 72px; width: 540px;
    display: flex; flex-direction: column; gap: var(--gfl-space-8);
  }}
  .card .gfl-logo {{ font-size: 34px; }}
  .card .gfl-logo .gfl-icon {{ width: 40px; height: 40px; }}
  .card .gfl-heading--1 {{ font-size: 58px; }}
  .card__url {{ font-family: var(--gfl-font-mono); font-size: 24px; color: var(--gfl-color-text-muted); }}
  .card__figure {{ position: absolute; right: 32px; bottom: 56px; width: 540px; }}
</style>
</head>
<body class="gfl-home">
<div class="card">
  <div class="card__copy">
    <p class="gfl-logo">{logo}<span>{name}</span></p>
    <h1 class="gfl-heading gfl-heading--1">{title}</h1>
    <p class="card__url">{url}</p>
  </div>
  <img class="card__figure" src="figures/hero.svg" alt="">
</div>
</body>
</html>
"""


def render(out=OUTPUT, chrome=CHROME):
    """Write the card to ``out`` and return its path."""
    site = yaml.safe_load((DOCS / "mkdocs.yml").read_text())["site_url"]
    with tempfile.TemporaryDirectory(prefix="graphfla-social-card-") as directory:
        root = Path(directory)
        page = home_build.build_assets(root)
        html = root / "card.html"
        html.write_text(CARD.format(
            fonts=home_build.FONTS, width=WIDTH, height=HEIGHT,
            stylesheets="\n".join(f'<link rel="stylesheet" href="css/{sheet}">' for sheet in home_build.STYLESHEETS),
            logo=home_build.icon("logo"), name=page["meta"]["title"],
            title=home_build.inline(page["hero"]["title"]), url=site.split("://", 1)[-1].rstrip("/"),
        ))
        # The time budget lets the web fonts load before the capture.
        subprocess.run([chrome, "--headless=new", "--hide-scrollbars", "--force-device-scale-factor=1",
                        f"--window-size={WIDTH},{HEIGHT}", "--virtual-time-budget=10000",
                        f"--screenshot={out}", html.as_uri()], check=True, capture_output=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUTPUT, help="output PNG (default: %(default)s)")
    parser.add_argument("--chrome", default=CHROME, help="Chrome or Chromium executable")
    args = parser.parse_args()
    print("Wrote", render(args.out, args.chrome))
