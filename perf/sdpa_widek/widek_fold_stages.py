#!/usr/bin/env python3
"""Per-stage wall of each fold leg, from the predict log's own timestamps.

The whole-fold wall cannot resolve this lever: it carries model load, MSA staging, prep and 200
diffusion steps, none of which the lever touches, and the OFF arm's own seed-to-seed spread on that
wall is wider than the effect. `trunk` is the stage that runs the pairformer, so it is the only
denominator the lever can move.
"""
import re, sys, json
from datetime import datetime
from pathlib import Path

WD = Path(sys.argv[1])
MARKS = [("load", r"loading model"), ("msa", r"\bmsa\b"), ("prep", r"\bprep\b"),
         ("trunk0", r"trunk 0/"), ("diff0", r"diffusion 0/"), ("diffend", r"diffusion 199/")]

def stamps(p):
    out = {}
    for line in p.read_text(errors="ignore").splitlines():
        m = re.match(r"^(\d\d:\d\d:\d\d)\s", line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%H:%M:%S")
        for name, pat in MARKS:
            if name not in out and re.search(pat, line):
                out[name] = t
    return out

rows = {}
for p in sorted(WD.glob("*.log")):
    s = stamps(p)
    if "trunk0" not in s or "diff0" not in s:
        continue
    d = lambda a, b: round((s[b] - s[a]).total_seconds(), 1) if a in s and b in s else None
    rows[p.stem] = {"trunk_s": d("trunk0", "diff0"), "prep_s": d("prep", "trunk0"),
                    "msa_s": d("msa", "prep"),
                    "diffusion_s": d("diff0", "diffend")}
print(json.dumps(rows, indent=1))
off = [v["trunk_s"] for k, v in rows.items() if k.startswith("off") and "repeat" not in k]
rep = [v["trunk_s"] for k, v in rows.items() if "repeat" in k]
on = [v["trunk_s"] for k, v in rows.items() if k.startswith("on")]
print(f"trunk off {off} mean {sum(off)/len(off):.1f}s" if off else "no off")
print(f"trunk rep {rep}")
print(f"trunk on  {on} mean {sum(on)/len(on):.1f}s" if on else "no on")
if off and on:
    print(f"TRUNK SPEEDUP {(sum(off)/len(off))/(sum(on)/len(on)):.4f}x   "
          f"off spread {max(off)-min(off):.1f}s  on spread {max(on)-min(on):.1f}s")
