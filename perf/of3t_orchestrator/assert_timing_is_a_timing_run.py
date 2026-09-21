#!/usr/bin/env python3
"""A seconds figure quoted as the campaign's timing must come from a run that was timing.

D164, pass 304. The campaign's headline cross-stack number -- 870.75 s on a p300c against 7-8 s
on an H200, "~116x" -- is `perf/of3t_l1/out/r3_384.json`, whose own docstring says what it was
for: *"Does the taped OF3 trunk fit the card at a crop their recipe trains at?"* It is D14's
MEMORY ladder. `ladder.py --probe-every` defaults to 1 ("allocator reads per verb"), the recorded
`argv` never passes the flag, and the artifact counts 82,610 + 163,900 = 246,510 verb calls inside
the timed window -- 3.53 ms of wall clock per call. The campaign's own later clean timing of the
SAME backward graph (2,473 tape nodes), same crop, same host, reads 222.48 s. The quoted figure
was 3.91x too large, in the direction that overstates our own gap.

`audit_evidence.py` already recomputes 172.43 + 698.32 = 870.75 and asserts `forward.cycles == 1`,
so the sentence cannot drift from its artifact. That is not the same guarantee: a guard that binds
a sentence to an artifact cannot notice the artifact is the wrong INSTRUMENT.

WHAT THIS ASSERTS. For every artifact that looks probe-instrumented, its seconds literals may
appear in the campaign's LIVE prose only inside a paragraph that also names D164. Discussing the
number as a corrected one is fine; quoting it as the campaign's timing is not.

WHAT IT DOES NOT ASSERT. It cannot tell a slow op from a slow instrument. `ms_per_call` is a
heuristic and is reported, not trusted alone: an artifact is flagged only when it counts verb
calls AND its argv did not subsample them AND the per-call wall clock is above THRESHOLD_MS. A
run that legitimately counts its verbs cheaply is not flagged, and a probe-instrumented run that
happens to be fast per call would be missed.

usage: assert_timing_is_a_timing_run.py [--self-test]
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STATE = pathlib.Path("/home/moritz/.coworker/state")

THRESHOLD_MS = 0.5          # wall clock per verb call above which the instrument dominates
MIN_CALLS = 1000            # below this the per-call figure is noise

# The live prose. The state doc is scanned only ABOVE `PASSLOG:` -- PASSLOG is history by A30 and
# a pass that quoted the old number correctly at the time must stay quotable.
DOCS = [
    (STATE / "of3t-orchestrator.md", "live fields"),
    (STATE / "of3t" / "EVIDENCE.md", "whole file"),
    (STATE / "of3t" / "LEDGER.md", "whole file"),
]
EXEMPT_MARK = "D164"


def _walk(node, path=""):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def instrumented(doc):
    """(is_flagged, seconds_literals, why) for one loaded artifact."""
    argv = " ".join(str(x) for x in (doc.get("argv") or []))
    m = re.search(r"--probe-every[= ]+(\d+)", argv)
    subsampled = bool(m) and int(m.group(1)) > 1
    calls, secs = 0, []
    for _p, d in _walk(doc):
        if not isinstance(d, dict) or "verb_calls" not in d:
            continue
        try:
            c = int(d["verb_calls"])
        except (TypeError, ValueError):
            continue
        calls += c
        if isinstance(d.get("s"), (int, float)):
            secs.append(float(d["s"]))
    if calls < MIN_CALLS or not secs:
        return False, [], ""
    total = sum(secs)
    ms = total / calls * 1000.0
    if subsampled or ms < THRESHOLD_MS:
        return False, [], ""
    lits = [f"{v:.2f}" for v in secs] + [f"{total:.2f}"]
    why = (f"{calls} verb calls with no --probe-every subsample, {ms:.2f} ms of wall clock "
           f"per call over {total:.2f} s")
    return True, sorted(set(lits)), why


def live_text(path, mode):
    t = path.read_text(errors="replace")
    if mode == "live fields":
        cut = t.find("\nPASSLOG:")
        if cut > 0:
            t = t[:cut]
    return t


def paragraphs(t):
    off = 0
    for block in re.split(r"\n\s*\n", t):
        yield off, block
        off += len(block) + 2


def scan(docs_text, literals):
    hits = []
    for label, t in docs_text:
        for _off, block in paragraphs(t):
            if EXEMPT_MARK in block:
                continue
            for lit in literals:
                if re.search(rf"(?<![\d.]){re.escape(lit)}(?![\d])", block):
                    line = block.strip().splitlines()[0][:110]
                    hits.append((label, lit, line))
    return hits


def main():
    if "--self-test" in sys.argv:
        return self_test()
    flagged = []
    for p in sorted(ROOT.glob("perf/**/*.json")):
        try:
            doc = json.loads(p.read_text())
        except Exception:                                                   # noqa: BLE001
            continue
        if not isinstance(doc, dict):
            continue
        yes, lits, why = instrumented(doc)
        if yes:
            flagged.append((p.relative_to(ROOT), lits, why))
    if not flagged:
        print("assert_timing_is_a_timing_run: no probe-instrumented artifact found in perf/ -- "
              "nothing to police (this is a PASS only if the tree really has none; "
              "perf/of3t_l1/out/r3_384.json is the known one and lives on wk/of3t)")
        return 0
    docs_text = [(lbl, live_text(p, lbl)) for p, lbl in DOCS if p.is_file()]
    bad = []
    for rel, lits, why in flagged:
        print(f"probe-instrumented: {rel}  [{why}]")
        print(f"  seconds literals it must not be quoted by: {', '.join(lits)}")
        for hit in scan(docs_text, lits):
            bad.append((rel, *hit))
    for rel, label, lit, line in bad:
        print(f"FAIL {label}: quotes {lit} s from {rel} in a paragraph that does not name "
              f"{EXEMPT_MARK}\n     {line}")
    if bad:
        print(f"\n{len(bad)} quotation(s) of a memory-fit run's wall clock as a timing figure.")
        return 1
    print(f"OK: {len(flagged)} probe-instrumented artifact(s); none of their seconds is quoted "
          f"in live prose outside a paragraph naming {EXEMPT_MARK}")
    return 0


def self_test():
    """Negative control: the real sentence, with and without its correction marker."""
    doc = {"argv": ["--tokens", "384", "--backward"],
           "forward": {"s": 172.43, "verb_calls": 82610, "cycles": 1},
           "backward": {"s": 698.32, "verb_calls": 163900}}
    yes, lits, why = instrumented(doc)
    assert yes, "the real artifact's shape must be flagged"
    assert "870.75" in lits and "172.43" in lits, lits
    print(f"  probe detector fires on the real shape: {why}")

    naked = ("Both sides now have a number at crop 384: **870.75 s on a p300c against 7-8 s\n"
             "on an H200, ~116x.** TT is one taped trunk cycle at num_recycles = 0.")
    assert scan([("control", naked)], lits), "MUST fire on the uncorrected sentence"
    print("  fires on the pre-correction sentence")

    marked = naked + "\n\nSuperseded by D164: the clean re-time of the same graph is 222.48 s."
    # one paragraph still lacks the marker, so the control proves per-PARAGRAPH granularity
    assert scan([("control", marked)], lits), "still fires -- the quoting paragraph is unmarked"
    joined = naked.replace("Both sides", "D164 corrects this. Both sides")
    assert not scan([("control", joined)], lits), "must NOT fire once the paragraph names D164"
    print("  silent once the quoting paragraph itself names D164")

    clean = {"argv": ["--probe-every", "500"],
             "forward": {"s": 3.41, "verb_calls": 82610}}
    assert not instrumented(clean)[0], "a subsampled run must not be flagged"
    slow = {"argv": [], "forward": {"s": 0.2, "verb_calls": 82610}}
    assert not instrumented(slow)[0], "a cheap per-call run must not be flagged"
    print("  does not fire on a subsampled run or a cheap per-call run")
    print("SELF-TEST PASSES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
