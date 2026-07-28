#!/usr/bin/env python3
"""Assert the dataset card's PSBench numbers are the ones the committed artifact actually supports.

The card's wave-averaging comparison is its central methodological claim and the most-quoted thing
in it. It has now been wrong twice, both times by quoting a number this repo had already superseded:

  * first as a mix-dependent ratio ("overstates by 2.2x", "100x down to 1.4x"), retracted because the
    multiplier depends on which targets are in the mix;
  * then, in the very pass that diagnosed that and wrote down the lesson ("every quantitative claim
    in the card had to be traced back to the measurement that currently supports it"), corrected to
    the **n=200, four-antigen** figures -- while leg (i) had already finished at **n=350 on seven
    antigens**. The correction traced the claim back to a passage instead of to the data, which is
    exactly the failure it named. 200/200 vs 106/200 became 350/350 vs 216/350, and the
    wave-averaged dynamic range 1.7x became 2.6x once 9DM7's 0.339 median joined the set.

Prose cannot be trusted to stay in step with a growing measurement, so this recomputes the numbers
from `docs/implementation-parity-data/abag-xm-psbench-legi-pilot.json` (350 rows, the artifact the
leg wrote) and greps the card for each one. If the leg is ever re-run on more targets, this fails
until the card is updated, which is the whole point.

    python3 scripts/abag_xm_card_claims_selftest.py     # exit 0 = card matches the artifact
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARD = ROOT / "docs" / "abag-xm-dataset-card.md"
ARTIFACT = ROOT / "docs" / "implementation-parity-data" / "abag-xm-psbench-legi-pilot.json"
ACCEPTABLE = 0.23


def main():
    if not CARD.exists():
        print(f"FAIL: card missing at {CARD}")
        return 1
    if not ARTIFACT.exists():
        print(f"FAIL: leg (i) artifact missing at {ARTIFACT}")
        return 1

    rows = json.loads(ARTIFACT.read_text())
    rows = [r for r in rows if r.get("HA") is not None and r.get("dockq_wave") is not None]
    ours = [float(r["HA"]) for r in rows]
    wave = [float(r["dockq_wave"]) for r in rows]
    per_t_ours, per_t_wave = {}, {}
    for r in rows:
        per_t_ours.setdefault(r["target"], []).append(float(r["HA"]))
        per_t_wave.setdefault(r["target"], []).append(float(r["dockq_wave"]))
    mo = {t: st.median(v) for t, v in per_t_ours.items()}
    mw = {t: st.median(v) for t, v in per_t_wave.items()}

    n = len(rows)
    facts = {
        "n_models": n,
        "n_targets": len(mo),
        "ours_acceptable": sum(1 for v in ours if v >= ACCEPTABLE),
        "wave_acceptable": sum(1 for v in wave if v >= ACCEPTABLE),
        "ours_lo": min(mo.values()), "ours_hi": max(mo.values()),
        "wave_lo": min(mw.values()), "wave_hi": max(mw.values()),
    }
    facts["ours_ratio"] = facts["ours_hi"] / facts["ours_lo"]
    facts["wave_ratio"] = facts["wave_hi"] / facts["wave_lo"]

    print(f"artifact: {n} rows, {len(mo)} targets")
    print(f"  acceptable (>= {ACCEPTABLE}): ours {facts['ours_acceptable']}/{n}, "
          f"wave {facts['wave_acceptable']}/{n}")
    print(f"  median span ours {facts['ours_lo']:.3f}-{facts['ours_hi']:.3f} "
          f"({facts['ours_ratio']:.0f}x), wave {facts['wave_lo']:.3f}-{facts['wave_hi']:.3f} "
          f"({facts['wave_ratio']:.1f}x)")

    # Collapse whitespace before matching: the card is hard-wrapped prose, so a required phrase
    # legitimately spans a line break ("a factor of\n152,"). Matching raw text made this fail on a
    # card that was correct, which is the failure mode that gets a guard deleted rather than fixed.
    text = " ".join(CARD.read_text().split())
    # Numbers the card must state. Written as the exact substrings the card uses, because a card
    # that states the right value in the wrong sentence is still a card someone will misread.
    required = [
        (f"{facts['n_models']} models", "total model count"),
        (f"{facts['wave_acceptable']} of {n}", "wave acceptable count"),
        (f"{facts['ours_acceptable']} of {n}", "per-interface acceptable count"),
        (f"{facts['ours_lo']:.3f} to {facts['ours_hi']:.3f}", "per-interface median span"),
        (f"{facts['wave_lo']:.3f} to {facts['wave_hi']:.3f}", "wave-averaged median span"),
        (f"factor of {facts['ours_ratio']:.0f}", "per-interface dynamic range"),
        (f"factor of {facts['wave_ratio']:.1f}", "wave-averaged dynamic range"),
    ]
    fails = [f"card does not state the {what}: expected {want!r}"
             for want, what in required if want not in text]

    # And it must not still claim a smaller sample than the artifact holds.
    for stale in re.findall(r"\b(\d+) of (\d+)\b", text):
        if int(stale[1]) != n:
            fails.append(f"card quotes 'x of {stale[1]}' but the artifact has {n} rows")

    if fails:
        print("\nFAIL")
        for f in sorted(set(fails)):
            print(f"  - {f}")
        return 1
    print("\nPASS: the card's PSBench numbers match the artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
