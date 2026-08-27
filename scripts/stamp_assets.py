#!/usr/bin/env python3
"""Content-stamp the assets the HTML references, so a deploy is never invisible.

`assets/site.css` was served under a fixed URL with an hour of browser cache, and the
icons with thirty days. A redeploy changed the bytes and nothing changed the URL, so a
returning visitor kept the old stylesheet and the change looked like it had not shipped.
Stamping the URL with a hash of the file makes a changed file a changed URL, which is what
lets `_headers` cache these hard instead of guessing at a TTL.

  python3 scripts/stamp_assets.py           rewrite the links
  python3 scripts/stamp_assets.py --check   exit 1 if any link is stale
"""
import hashlib
import pathlib
import re
import sys

SITE = pathlib.Path(__file__).resolve().parent.parent / "site"
PAGES = ("index.html", "benchmarks/index.html", "404.html")
STAMPED = ("site.css", "favicon.ico", "icon-500.png", "apple-touch-icon.png", "mark.svg")


def digest(name: str) -> str:
    return hashlib.sha256((SITE / "assets" / name).read_bytes()).hexdigest()[:8]


def stamp(check: bool) -> int:
    want = {n: digest(n) for n in STAMPED}
    stale = []
    for page in PAGES:
        p = SITE / page
        text = original = p.read_text()
        for name, d in want.items():
            # any href to this asset, with or without an existing ?v=
            pat = re.compile(r'((?:\.\./|/tt-bio/|/)?assets/%s)(\?v=[0-9a-f]{8})?' % re.escape(name))
            text = pat.sub(lambda m: "%s?v=%s" % (m.group(1), d), text)
        if text != original:
            stale.append(page)
            if not check:
                p.write_text(text)
    if check and stale:
        print("stale asset stamps in: %s\nRun scripts/stamp_assets.py." % ", ".join(stale))
        return 1
    for name, d in want.items():
        print("  %-22s %s" % (name, d))
    return 0


if __name__ == "__main__":
    sys.exit(stamp("--check" in sys.argv))
