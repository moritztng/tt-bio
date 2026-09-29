#!/usr/bin/env python3
"""Every local link on the published site resolves to something GitHub Pages will serve.

Moritz, 2026-09-29, clicking the logo on /benchmarks/: "i get error. doesnt direct me back to
landing page". The brand link read `../assets/`, a directory with no index, and the 404 page
carried the same link plus two "Overview" links to `/`, the root of moritztng.github.io, which
is not this site. Nothing checked an href, so a visitor found it first.

For every `href` and `src` in every site/**/*.html: a relative path resolves against the page's
directory, a root-absolute path must sit under BASE, the target must be a file or a directory
with an index.html, and a `#fragment` must name an id on the target page. External URLs, mailto:
and in-page anchors to ids on the same page are checked only for the id. Exit 1 names each bad
link. `pages.yml` runs this before upload, so a broken link blocks the publish.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
BASE = "/tt-bio/"  # where GitHub Pages serves site/: https://moritztng.github.io/tt-bio/
LINK = re.compile(r'\b(?:href|src)="([^"]*)"')
ID = re.compile(r'\bid="([^"]+)"')


def target(site: Path, page: Path, path: str) -> Path | None:
    """The file a link lands on, or None if Pages would answer 404."""
    if path.startswith("/"):
        if not path.startswith(BASE):
            return None
        hit = site / path[len(BASE):]
    else:
        hit = page.parent / path if path else page
    hit = hit.resolve()
    if site.resolve() not in (hit, *hit.parents):
        return None
    if hit.is_dir():
        hit = hit / "index.html"
    return hit if hit.is_file() else None


def check(site: Path) -> list[str]:
    bad = []
    for page in sorted(site.rglob("*.html")):
        text = page.read_text(errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            for url in LINK.findall(line):
                u = urlsplit(url)
                if u.scheme or u.netloc:
                    continue
                where = f"{page.relative_to(site)}:{n} {url}"
                hit = target(site, page, u.path)
                if hit is None:
                    bad.append(f"{where}: resolves to nothing Pages serves")
                elif u.fragment and hit.suffix == ".html" and \
                        u.fragment not in ID.findall(hit.read_text(errors="replace")):
                    bad.append(f"{where}: no id=\"{u.fragment}\" on {hit.relative_to(site.resolve())}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site", nargs="?", type=Path, default=REPO / "site")
    bad = check(ap.parse_args().site)
    for line in bad:
        print(line, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
