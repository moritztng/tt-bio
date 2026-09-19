#!/usr/bin/env python3
"""Flag a retired TRIX figure quoted as a live claim in state/trix/LEDGER.md.

The ledger is append-only: its oldest material is at the top and the corrections are at the bottom,
and over one night six KNOWN rows were superseded, two fold targets were voided and one lever was
retracted. A reader landing mid-file can pick up a number that has since been replaced, so the rule
this campaign adopted is that a retired figure must be marked WHERE IT APPEARS and not only in an
appendix. This checks that rule mechanically.

It is deliberately noisy: a hit is "this line quotes a retired figure with no marker ON THE LINE",
which is usually a table row or a historical comparison and fine. Read the hits, do not just count
them. The signal it exists to catch is a retired figure standing alone in a paragraph that reads as
a current claim -- which is how the 0.693 s target survived three rows.
"""
import re
import sys
from pathlib import Path

LEDGER = Path("/home/moritz/.coworker/state/trix/LEDGER.md")

RETIRED = {
    "30.939": "module wall, pre-E1/E2/E3",
    "9.65 TFLOP": "rate, standalone and pre-E",
    "8.8 %": "% of roof, pre-E",
    "6.61 ms": "floor that was never arithmetic",
    "2.82x": "ratio, standalone and post-E",
    "28.485": "standalone wall, pre-E",
    "65.4 %": "scaffolding share, pre-E6/F1",
    "109.56": "roof read at a governor-set ~1213 MHz",
    "137.1": "roof, asserted not measured",
    "71.9 s": "fold wall, never on today's tree",
    "18.617": "module standalone, superseded",
    "0.693 s": "void target, wrong kernel",
    "0.546 s": "void target, wrong kernel",
    "1.2894x": "not an out_block_h measurement",
}

MARKER = re.compile(
    r"superseded|retired|void|stale|was |used to|no longer|refut|wrong|earlier|prior|asserted|"
    r"corrected|instead of|rather than|not an?\b|~~|before this campaign|decompos|governor|pre-E|"
    r"campaign carried|quoted at|for recognition|understat|Aug 2026|then\b", re.I)


def main() -> int:
    lines = LEDGER.read_text().split("\n")
    hits = [(i, fig, why, ln.strip()[:100])
            for i, ln in enumerate(lines, 1)
            for fig, why in RETIRED.items()
            if fig in ln and not MARKER.search(ln)]
    print(f"{LEDGER}: {len(lines)} lines, {len(hits)} unmarked quotation(s) of a retired figure\n")
    for i, fig, why, ln in hits:
        print(f"  L{i}  [{fig} — {why}]\n      {ln}\n")
    print("Each hit needs a human read: a table row under an 'Aug 2026' header is fine; a bare\n"
          "paragraph claim is not. Mark it on the line, do not delete it -- deleting a retracted\n"
          "claim hides how long it stood.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
