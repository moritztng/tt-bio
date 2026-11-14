#!/usr/bin/env python3
"""Fold ab/rows.jsonl and the arm censuses into one table: per rung, per arm, digest, warm runtime, AICLK.

Rep 0 of every arm pays its tree's kernel compile, so the runtime column is the last rep only.
"""
import json
from pathlib import Path

AB = Path(__file__).resolve().parent / "ab"
FLAGS = ("REBLOCK_PERMUTE", "REBLOCK_PERMUTE_GATED", "PAIR_TRANSPOSE_VIA_ROW_MAJOR", "TRANSPOSE_L1_RESIDENT")


def census(label):
    p = AB / f"census_{label}.json"
    if not p.exists():
        return {}
    return {r["flag"]: r for r in json.loads(p.read_text())["rows"] if r["flag"] in FLAGS}


rows = [json.loads(l) for l in (AB / "rows.jsonl").read_text().splitlines() if l.strip()]
out = {}
for r in rows:
    out.setdefault(r["rung"], {}).setdefault(r["arm"], []).append(r)
for rung, arms in sorted(out.items()):
    print(f"== {rung}")
    for arm, rs in sorted(arms.items()):
        last = max(rs, key=lambda r: r["rep"])
        c = census(f"{arm}-{rung}-r{last['rep']}") or census(f"{arm}-{rung}-r0")
        lev = "; ".join(f"{f} {v['served']}/{v['declined']} {v.get('rejects') or ''}" for f, v in c.items()
                        if v["served"] or v["declined"])
        print(f"  {arm} digest {last['digest'][:12]} runtime {[r['runtime_s'] for r in rs]} "
              f"AICLK median {[r['clock']['median'] for r in rs]} | {lev}")
