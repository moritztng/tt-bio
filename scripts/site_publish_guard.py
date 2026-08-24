#!/usr/bin/env python3
"""A model reaches tt-bio.com only when every processor column is measured.

Moritz, 2026-08-24: "if not all processors are measured yet dont add the model to
tt-bio.com". A row with a blank column still renders: the bar chart just draws the
platforms it has, so a partially-measured model looks like a published claim while the
column that might not flatter us is simply absent. This is the gate that stops that.

Default mode verifies `site/data/perf-512aa.json` and exits 1 naming any row with a cell
that is not `measured`. `--strip` moves those rows out of the published file into
`perf/page_rows_pending.json`, where they wait for their missing cells; `--restore` puts
back every pending row that has since become complete. The subtitle's model counts are
regenerated from the rows that survive, so neither direction leaves stale prose.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "site" / "data" / "perf-512aa.json"
PENDING = REPO / "perf" / "page_rows_pending.json"

CATEGORIES = ("models", "design", "affinity", "embed")
WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
NOUNS = {"models": "structure-prediction model", "design": "binder-design model",
         "affinity": "binding-affinity model", "embed": "protein-embedding model"}


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


def retitle(doc: dict) -> None:
    """Rebuild the subtitle's counts from the rows that are actually published."""
    counts = [(cat, len(rows(doc, cat))) for cat in CATEGORIES]
    parts = [f"{WORDS.get(n, n)} {NOUNS[cat]}{'s' if n != 1 else ''}" for cat, n in counts if n]
    listed = ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]
    doc["subtitle"] = (listed[0].upper() + listed[1:] +
                       " at 512 residues, measured on Tenstorrent Blackhole and on "
                       "NVIDIA H200, B200 and A100.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strip", action="store_true", help="move incomplete rows to the pending file")
    ap.add_argument("--restore", action="store_true", help="publish pending rows that are complete")
    args = ap.parse_args()

    doc = json.loads(DATA.read_text())
    pending = json.loads(PENDING.read_text()) if PENDING.exists() else {"rows": []}
    changed = []

    if args.restore:
        keep = []
        for held in pending["rows"]:
            if missing(held["row"]):
                keep.append(held)
                continue
            set_rows(doc, held["category"], rows(doc, held["category"]) + [held["row"]])
            changed.append(f"restored {held['category']}/{held['row']['id']}")
        pending["rows"] = keep

    if args.strip:
        for cat in CATEGORIES:
            keep = []
            held_ids = {(h["category"], h["row"]["id"]) for h in pending["rows"]}
            for row in rows(doc, cat):
                if missing(row):
                    # A second --strip over a tree whose row is already held used to append it
                    # again, and main carried two copies of all three held rows for it. A
                    # duplicated hold restores twice: the row draws twice and the subtitle counts
                    # it twice.
                    if (cat, row["id"]) not in held_ids:
                        pending["rows"].append({"category": cat, "row": row})
                        held_ids.add((cat, row["id"]))
                        changed.append(f"held {cat}/{row['id']} (missing {', '.join(missing(row))})")
                    else:
                        changed.append(f"already held, not duplicated: {cat}/{row['id']}")
                else:
                    keep.append(row)
            set_rows(doc, cat, keep)

    if args.strip or args.restore:
        retitle(doc)
        DATA.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        PENDING.parent.mkdir(parents=True, exist_ok=True)
        PENDING.write_text(json.dumps(pending, indent=2, ensure_ascii=False) + "\n")
        for line in changed:
            print(line)
        print(f"subtitle: {doc['subtitle']}")
        return 0

    bad = [f"{cat}/{row['id']} missing {', '.join(missing(row))}"
           for cat in CATEGORIES for row in rows(doc, cat) if missing(row)]
    counts = Counter((h["category"], h["row"]["id"]) for h in pending["rows"])
    bad += [f"{cat}/{rid} is held {n} times in {PENDING.name}; --restore would publish it {n} times"
            for (cat, rid), n in counts.items() if n > 1]
    for line in bad:
        print(line, file=sys.stderr)
    if bad:
        print("A published row needs every processor column measured. "
              "Run scripts/site_publish_guard.py --strip.", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
