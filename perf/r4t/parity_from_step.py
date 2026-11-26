#!/usr/bin/env python3
"""Turn r4t-step trunk-step logs into a ``grade_parity.py --parity`` file, same-weights pairs only.

r4t-step prints, per step, our loss terms beside upstream's recorded terms for the same batch.
Upstream's terms were recorded once, at the released weights (``loss_parity_reference.py``). So a
line is a parity pair only when OUR weights were also the released ones when it ran: the first
step of a fresh run (``step == 1``) or a line that says ``updates_before == 0``. Every later step
compares our once-updated weights with upstream's never-updated ones; that gap is training, not
parity, and this script refuses to grade it (it lists what it dropped).

A batch seen in several logs is kept once (the first), so repeating an arm never inflates n.
Total is the sum of the terms on both sides, which is how upstream's total is formed.

    parity_from_step.py qb2_A.log qb2_B.log ... --dtype bfloat16 --out parity.json
    grade_parity.py --parity parity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def same_weights(rec: dict) -> bool:
    if "updates_before" in rec:
        return rec["updates_before"] == 0
    return rec.get("step") == 1 and not rec.get("resumed", False)


def pairs(lines) -> tuple[list[dict], list[str]]:
    kept, dropped, seen = [], [], set()
    for src, rec in lines:
        if "upstream_terms" not in rec:
            continue
        tag = f"{src}:step{rec.get('step')}:{rec.get('batch')}"
        if not same_weights(rec):
            dropped.append(f"{tag} (weights updated before this step)")
            continue
        if rec["batch"] in seen:
            dropped.append(f"{tag} (batch already graded)")
            continue
        seen.add(rec["batch"])
        kept.append(dict(batch=rec["batch"], src=tag, winner=rec.get("winner"),
                         upstream_winner=rec.get("upstream_winner"),
                         ours=dict(rec["terms"]), ref=dict(rec["upstream_terms"])))
    for p in kept:
        for side in ("ours", "ref"):
            p[side]["total"] = sum(float(v) for v in p[side].values())
    return kept, dropped


def read(paths):
    for p in paths:
        for ln in Path(p).read_text(errors="replace").splitlines():
            if ln.startswith("{"):
                try:
                    yield p, json.loads(ln)
                except json.JSONDecodeError:
                    continue


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    kept, dropped = pairs(read(a.logs))
    doc = dict(dtype=a.dtype, ours=[p["ours"] for p in kept], ref=[p["ref"] for p in kept],
               provenance=dict(pairs=[{k: p[k] for k in ("batch", "src", "winner", "upstream_winner")}
                                      for p in kept], dropped=dropped))
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(f"{len(kept)} same-weights pairs, {len(dropped)} lines dropped -> {a.out}")
    for d in dropped:
        print("  dropped " + d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
