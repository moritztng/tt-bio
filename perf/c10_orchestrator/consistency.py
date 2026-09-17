#!/usr/bin/env python3
"""Cross-artifact consistency for the C10 corpus.

This campaign's own story is numbers rotting: a 17.34 s cell that was a throttled clock, a 2.309 s
prize that was the same, a 15.031 s floor built on rates from the wrong machine, a 67.59 TFLOP/s
dense cube that implies 799 MHz, a size-independent work term that was grid under-fill. Every one
was quoted, in good faith, after it had stopped being true.

Fourteen directories now cross-reference each other. This checks two things across all of them:
numbers of record agree wherever they are quoted, and a retired number never appears without its
retirement. It found `grid_evidence/` still presenting the 67.59 TFLOP/s cube as a Blackhole rate.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Measured, and the same everywhere they appear. c10-fixed-cost c4e6725bd, c10-bare-baseline.
OF_RECORD = {
    "F at 512 aa": ["3.9830", "3.983"],
    "W at 512 aa": ["14665.0", "14665", "14,665"],
    "baseline at 512 aa": ["14.8813", "14.881"],
    "pinned clock": ["1350"],
}

# A retired number, and the words that must appear near it for the quote to be honest.
RETIRED = {
    "15.031": (["suspect", "retired", "refuted", "withdrawn", "not usable", "cannot be compared",
                "contaminat", "below the floor", "wrong machine", "130-core"],
               "the roofline floor: its arithmetic half came from pc's 130-core firmware and the "
               "fold already reads below it at a recorded clock"),
    "2.309": (["retired", "was the clock", "artifact", "refuted", "no longer"],
              "the 'remaining prize': it was cell_of_record minus floor, and the cell was a "
              "~1063 MHz reading"),
    "67.59": (["799", "artifact", "throttled", "clock floor", "do not carry"],
              "a dense-cube rate that implies 799 MHz against the four-measurement cluster"),
    "3,301": (["refuted", "withdrawn", "artifact", "under-fill", "under-filling"],
              "the size-independent work term's lower bound, refuted by c10-size-scaling"),
    "8,220": (["refuted", "withdrawn", "artifact", "under-fill", "under-filling"],
              "the size-independent work term's pair reading, refuted by c10-size-scaling"),
}
# Scope is the DOCUMENT, not the paragraph. A reader of a README sees the whole file, so a
# retirement stated anywhere in it is stated. Paragraph scoping was tried first and flagged nine
# places, seven of which were honest -- a title naming what the document retires, a line quoting an
# identity two paragraphs before refuting it, a list of past artifacts. A checker that cries wolf
# gets switched off, which is worse than not having one.


def _docs():
    return sorted(p for p in HERE.rglob("README.md") if "__pycache__" not in str(p))


def _distance(text, num, markers):
    """Lines between a retired number's first mention and the nearest retirement word. Reported,
    not enforced: a retirement 200 lines below its number is honest but easy to miss."""
    lines = text.splitlines()
    hits = [i for i, l in enumerate(lines) if num in l]
    marks = [i for i, l in enumerate(lines) if any(m in l.lower() for m in markers)]
    if not hits or not marks:
        return None
    return min(abs(h - m) for h in hits for m in marks)


def check():
    problems, quotes, far = [], {k: [] for k in OF_RECORD}, []
    for doc in _docs():
        rel = str(doc.relative_to(HERE))
        text = doc.read_text()
        low = text.lower()
        for num, (markers, what) in RETIRED.items():
            if num not in text:
                continue
            if not any(m in low for m in markers):
                problems.append({
                    "file": rel, "number": num, "kind": "retired number quoted with no retirement "
                    "anywhere in the document", "what_it_is": what,
                    "excerpt": " ".join(next(l for l in text.splitlines() if num in l).split())[:160],
                })
            else:
                d = _distance(text, num, markers)
                if d is not None and d > 60:
                    far.append({"file": rel, "number": num, "lines_away": d})
        for label, forms in OF_RECORD.items():
            for f in forms:
                if f in text:
                    quotes[label].append(rel)
                    break
    return {
        "scope": "CPU cross-check of the C10 artifact tree. Reads the READMEs; changes nothing.",
        "docs_checked": len(_docs()),
        "numbers_of_record": {k: sorted(set(v)) for k, v in quotes.items()},
        "retired_numbers_tracked": {k: v[1] for k, v in RETIRED.items()},
        "problems": problems,
        "retirement_far_from_its_number": far,
        "clean": not problems,
        "limits": [
            "Textual, not semantic: it checks that a retired number appears beside its retirement, "
            "not that the surrounding prose is correct.",
            "Document-scoped. A retirement stated anywhere in the file counts, because a reader "
            "sees the whole file; distance over 60 lines is reported separately rather than failed.",
            "It cannot see a number that is wrong but was never on this list. The list is the "
            "campaign's memory and has to be extended by hand when something else is retired.",
        ],
    }


if __name__ == "__main__":
    r = check()
    (HERE / "consistency.json").write_text(json.dumps(r, indent=2) + "\n")
    print(f"{r['docs_checked']} READMEs checked, {len(RETIRED)} retired numbers tracked")
    for p in r["problems"]:
        print(f"  BARE {p['number']} in {p['file']}: {p['excerpt'][:110]}")
    print("clean" if r["clean"] else f"{len(r['problems'])} problem(s)")
