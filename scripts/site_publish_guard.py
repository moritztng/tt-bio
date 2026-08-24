#!/usr/bin/env python3
"""A model reaches tt-bio.com only when every processor column is measured.

Moritz, 2026-08-24: "if not all processors are measured yet dont add the model to
tt-bio.com". A row with a blank column still renders: the bar chart just draws the
platforms it has, so a partially-measured model looks like a published claim while the
column that might not flatter us is simply absent. This is the gate that stops that.

Default mode verifies `site/data/perf-512aa.json` and exits 1 naming any row with a cell
that is not `measured`. `--strip` moves those rows out of the published file into
`perf/page_rows_pending.json`, where they wait for their missing cells; `--restore` puts
back every pending row that has since become complete. The JSON subtitle and the page's
meta description are both regenerated from the rows that survive, so neither direction
leaves stale prose, and the default mode fails if either has drifted from the row counts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "site" / "data" / "perf-512aa.json"
PAGE = REPO / "site" / "benchmarks" / "index.html"
PENDING = REPO / "perf" / "page_rows_pending.json"

CATEGORIES = ("models", "design", "affinity", "embed")
WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
         13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
         18: "eighteen", 19: "nineteen", 20: "twenty"}
NOUNS = {"models": "structure-prediction model", "design": "binder-design model",
         "affinity": "binding-affinity model", "embed": "protein-embedding model"}
# What the meta description calls each category. Shorter than NOUNS because the sentence
# already says "models", and it is a search result, not a caption.
SHORT = {"models": "folding", "design": "binder design", "affinity": "binding affinity",
         "embed": "protein embedding"}
META_RE = re.compile(r'(<meta name="description" content=")([^"]*)(">)')


def rows(doc: dict, cat: str) -> list:
    block = doc.get(cat)
    if isinstance(block, list):
        return block
    return (block or {}).get("models", [])


def set_rows(doc: dict, cat: str, new: list) -> None:
    if isinstance(doc.get(cat), list):
        doc[cat] = new
    else:
        doc[cat]["models"] = new


def missing(row: dict) -> list[str]:
    return [k for k, c in (row.get("cells") or {}).items() if c.get("status") != "measured"]


def counts(doc: dict) -> list[tuple[str, int]]:
    """Rows the page actually draws, which is not the same as rows in the file.

    A row can leave the page two ways. `--strip` moves it to perf/page_rows_pending.json,
    and these counts followed that from the start. `"hidden": true` leaves it in the file
    and the renderer skips it, and these counts did not: the day RF3 and RFdiffusion3 were
    hidden the subtitle went on claiming eight folding and two design models over a page
    drawing seven and one, and the meta description said seventeen over sixteen. Both ways
    out have to subtract here or the prose overstates the page.
    """
    return [(cat, len([r for r in rows(doc, cat) if not r.get("hidden")]))
            for cat in CATEGORIES]


def join(parts: list[str]) -> str:
    return ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]


def subtitle_for(doc: dict) -> str:
    """The subtitle's counts, from the rows that are actually published."""
    parts = [f"{WORDS.get(n, n)} {NOUNS[cat]}{'s' if n != 1 else ''}"
             for cat, n in counts(doc) if n]
    listed = join(parts)
    return (listed[0].upper() + listed[1:] +
            " at 512 residues, measured on Tenstorrent Blackhole and on "
            "NVIDIA H200, B200 and A100.")


def meta_for(doc: dict) -> str:
    """The page's meta description, from the same counts.

    Two strings state the row counts and only one of them lives in the JSON. The strip that
    created this file regenerated the subtitle and left the meta reading "Eighteen open
    biomolecular models" on a page drawing fifteen, because nothing owned it. Now the same
    counts produce both and the default mode fails if either drifts.
    """
    live = [(cat, n) for cat, n in counts(doc) if n]
    total = sum(n for _, n in live)
    listed = join([f"{WORDS.get(n, n)} {SHORT[cat]}" for cat, n in live])
    return (f"{WORDS.get(total, total).capitalize()} open biomolecular models measured on "
            f"Tenstorrent: {listed}. Seconds per prediction and throughput per dollar against "
            "NVIDIA H200, B200 and A100 at 512 residues, with every number's provenance on "
            "the page.")


def retitle(doc: dict) -> list[str]:
    """Rewrite both count-bearing strings and report what moved."""
    moved = []
    want = subtitle_for(doc)
    if doc.get("subtitle") != want:
        doc["subtitle"] = want
        moved.append(f"subtitle: {want}")
    page = PAGE.read_text()
    want_meta = meta_for(doc)
    fixed, n = META_RE.subn(lambda m: m[1] + want_meta + m[3], page, count=1)
    if not n:
        raise SystemExit(f"no <meta name=\"description\"> in {PAGE}")
    if fixed != page:
        PAGE.write_text(fixed)
        moved.append(f"meta: {want_meta}")
    return moved


def stale(doc: dict) -> list[str]:
    """Count-bearing prose that no longer matches the published rows."""
    bad = []
    if doc.get("subtitle") != subtitle_for(doc):
        bad.append(f"subtitle does not match the published row counts, want: {subtitle_for(doc)}")
    found = META_RE.search(PAGE.read_text())
    if not found:
        bad.append(f"no <meta name=\"description\"> in {PAGE}")
    elif found[2] != meta_for(doc):
        bad.append(f"meta description does not match the published row counts, "
                   f"want: {meta_for(doc)}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strip", action="store_true", help="move incomplete rows to the pending file")
    ap.add_argument("--restore", action="store_true", help="publish pending rows that are complete")
    args = ap.parse_args()

    doc = json.loads(DATA.read_text())
    pending = json.loads(PENDING.read_text()) if PENDING.exists() else {"rows": []}
    changed = []

    # Both directions move a row between two files, so both can duplicate it. A `--strip`
    # against a data file that was reverted after an earlier strip left three rows twice in
    # perf/page_rows_pending.json, and --restore would then have published each of them twice.
    # Key on (category, id) and refuse to hold or publish one that is already there.
    held_ids = {(h["category"], h["row"]["id"]) for h in pending["rows"]}

    if args.restore:
        keep = []
        for held in pending["rows"]:
            cat, rid = held["category"], held["row"]["id"]
            if missing(held["row"]):
                keep.append(held)
                continue
            if rid in [r["id"] for r in rows(doc, cat)]:
                changed.append(f"dropped duplicate held {cat}/{rid}, already published")
                continue
            set_rows(doc, cat, rows(doc, cat) + [held["row"]])
            changed.append(f"restored {cat}/{rid}")
        pending["rows"] = keep

    if args.strip:
        for cat in CATEGORIES:
            keep = []
            for row in rows(doc, cat):
                if not missing(row):
                    keep.append(row)
                    continue
                if (cat, row["id"]) in held_ids:
                    changed.append(f"dropped {cat}/{row['id']}, already held pending")
                else:
                    pending["rows"].append({"category": cat, "row": row})
                    changed.append(f"held {cat}/{row['id']} (missing {', '.join(missing(row))})")
            set_rows(doc, cat, keep)

    if args.strip or args.restore:
        changed += retitle(doc)
        DATA.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        PENDING.parent.mkdir(parents=True, exist_ok=True)
        PENDING.write_text(json.dumps(pending, indent=2, ensure_ascii=False) + "\n")
        for line in changed or ["nothing to move"]:
            print(line)
        return 0

    bad = [f"{cat}/{row['id']} missing {', '.join(missing(row))}"
           for cat in CATEGORIES for row in rows(doc, cat) if missing(row)]
    for line in bad:
        print(line, file=sys.stderr)
    if bad:
        print("A published row needs every processor column measured. "
              "Run scripts/site_publish_guard.py --strip.", file=sys.stderr)
    drift = stale(doc)
    for line in drift:
        print(line, file=sys.stderr)
    if drift:
        print("Run scripts/site_publish_guard.py --strip to regenerate the counts.",
              file=sys.stderr)
    return 1 if bad or drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
