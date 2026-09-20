#!/usr/bin/env python3
"""One line per recorded p150a model: engine freshness and what each cell carries.

The re-record's whole point is that a fragment recorded at an engine we no longer ship is
stale however complete it looks, so freshness is read off the artifact's own commit through
the gate's own comparator, not off a marker and not off which rungs are present.

`clocked` is the count of rungs that carry an AICLK sample taken during the fold. A runtime
with no clock beside it is not a measurement on this part: the same 512 aa cell reads 21.90 s
at 800 MHz and 14.69 s at 1350.
"""
import glob
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("rg", "scripts/release_gate.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)

CARD = sys.argv[1] if len(sys.argv) > 1 else "p150a"
for f in sorted(glob.glob("docs/size_ladder_baseline.d/*.json")):
    d = json.load(open(f))
    for m, ent in sorted((d.get("cards", {}).get(CARD, {}).get("models", {}) or {}).items()):
        rt = ent.get("runtime_s", {}) or {}
        lev = ent.get("levers", {}) or {}
        fresh = rg._size_ladder_same_engine(ent.get("commit"), "HEAD")
        clocked = sum(1 for v in (ent.get("aiclk", {}) or {}).values() if v)
        reps = len(ent.get("runtime_reps_s", {}) or {})
        struct = len(ent.get("structure", {}) or {})
        wide = {str((lev.get(r, {}) or {}).get("SDPA_WIDE_K", {}).get("resolved"))
                for r in lev if isinstance(lev.get(r), dict)}
        print(f"{m:14s} {str(ent.get('commit'))[:9]} {'FRESH' if fresh else 'stale ':6s} "
              f"rungs={len(rt):2d} clocked={clocked:2d} reps={reps:2d} struct={struct:2d} "
              f"top={max(rt, key=int) if rt else '-':>5s} "
              f"SDPA_WIDE_K={'/'.join(sorted(wide)) if wide else '-'}")
