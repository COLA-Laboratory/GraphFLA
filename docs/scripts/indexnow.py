"""Tell IndexNow search engines (Bing, Yandex, Seznam, Naver) that the website's pages changed.

Run ``python docs/scripts/indexnow.py`` after a deployment. Page URLs come from the live sitemap; the
key, served by the website as ``<key>.txt``, comes from ``mkdocs.yml``. Google does not use IndexNow.
"""

import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "mkdocs.yml"
ENDPOINT = "https://api.indexnow.org/indexnow"
LOC = "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"


def submit(config=CONFIG):
    """Submit every URL of the deployed sitemap and return the HTTP status."""
    settings = yaml.safe_load(config.read_text())
    site, key = settings["site_url"], settings["extra"]["indexnow_key"]
    with urlopen(site + "sitemap.xml", timeout=30) as response:
        urls = [loc.text for loc in ElementTree.parse(response).iter(LOC)]
    # A key file under the site path authorizes URLs under that path only.
    body = {"host": urlsplit(site).hostname, "key": key, "keyLocation": f"{site}{key}.txt", "urlList": urls}
    request = Request(ENDPOINT, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json; charset=utf-8"})
    with urlopen(request, timeout=30) as response:
        print(f"IndexNow: submitted {len(urls)} URLs, HTTP {response.status}")
        return response.status


if __name__ == "__main__":
    submit()
