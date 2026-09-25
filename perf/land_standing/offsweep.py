#!/usr/bin/env python3
"""Every default-OFF env flag on main, with the evidence its own source carries.

`merged-lever-defaults-off-is-not-a-landed-win` is this row's charge, and until now its candidate
list was hand-written prose that turned out to be stale in five of six entries. This enumerates
the pool mechanically instead: a flag defaulting False is a lever a user does not receive, so the
question for each is whether the code says it was held off ON PURPOSE.

A screen, not a verdict, and its first run proved why. It reads only the comment block above each
definition, so a flag whose reason to be off lives somewhere else -- a decision doc, a commit
message, docs/tuning-flags.md -- reads as a held win. Three of the six it first flagged were like
that: `TT_BIO_TRIATT_B8` is a closed card-dependence stop recorded in `825f18772`,
`TT_BIO_UNFUSED_SILU`'s approval was withdrawn in `state/ask-8879-decision.md`, and
`TT_BIO_TRIMUL_OUT_L1` had a fold A/B that killed it and was not written down where the flag is.
Check each hit against git and the perf artifacts before believing it.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
DEF = re.compile(r'^(\s*)(_?\w+)\s*(?::[^=]+)?=\s*env_flag\(\s*"([A-Z_0-9]+)"\s*,\s*False\s*\)')
# Words that say "off on purpose" -- a kill switch, an escape hatch, a diagnostic, or a stop.
# Widened after the first run misread three flags. "MEASURED, AND IT LOSES" and "Diagnosis only"
# both say off-on-purpose in words none of the original patterns matched, and a lever whose
# comment quotes a LOSING ratio reads the same as one quoting a winning one.
HELD = re.compile(r"kill switch|escape hatch|hard stop|card-dependen|diagnos|legacy|fall ?back|"
                  r"opt-in|debug|NO-GO|not bit-exact|regress|refus|off by default and|"
                  r"A/B|instrument|probe|it loses|stays off|worth nothing|does not reach|"
                  r"slower|withdraw|release-gated|negative control", re.I)
# A measured claim: a ratio, a percentage, or a time.
WIN = re.compile(r"\b\d+\.\d+ ?x\b|\b\d+\.\d+ ?%|\b\d+\.\d+ ?(?:ms|s)\b|\b\d+ ?%")

rows = []
for f in sorted((ROOT / "tt_bio").rglob("*.py")):
    lines = f.read_text().splitlines()
    for i, line in enumerate(lines):
        m = DEF.match(line)
        if not m:
            continue
        # the contiguous comment block immediately above, plus any docstring right below
        j = i - 1
        block = []
        while j >= 0 and (lines[j].lstrip().startswith("#") or not lines[j].strip()):
            if lines[j].strip():
                block.append(lines[j].strip().lstrip("#").strip())
            elif block:
                break
            j -= 1
        ctx = " ".join(reversed(block))
        rows.append({"flag": m.group(3), "file": str(f.relative_to(ROOT)), "line": i + 1,
                     "held": bool(HELD.search(ctx)), "win": bool(WIN.search(ctx)),
                     "ctx_len": len(ctx), "ctx": ctx[:220]})

cand = [r for r in rows if r["win"] and not r["held"]]
held = [r for r in rows if r["held"]]
silent = [r for r in rows if not r["win"] and not r["held"]]
print(f"default-OFF flags on this tree: {len(rows)}")
print(f"  says why it is off (kill switch / stop / diagnostic / opt-in): {len(held)}")
print(f"  carries a measured number and NO stated reason to be off:      {len(cand)}")
print(f"  carries neither:                                               {len(silent)}\n")
for tag, group in (("CANDIDATES", cand), ("NO EVIDENCE EITHER WAY", silent)):
    print(f"--- {tag} ---")
    for r in sorted(group, key=lambda r: -r["ctx_len"]):
        print(f"  {r['flag']:38s} {r['file']}:{r['line']}")
        print(f"      {r['ctx'][:200] or '(no comment block)'}")
    print()
