#!/usr/bin/env python3
"""Compare today's quiet region-T A/B against the contended record it repeats.

Prints the two sessions side by side, a permutation p over block medians, and the digest identity
check. Reads JSON only -- opens no device, so it is free to run while the box is busy.
"""
import itertools, json, statistics as st, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = json.loads((HERE / "regiont_quiet_ab.json").read_text())
OLD = json.loads((HERE / "region_t_ab.json").read_text())


def arms(d, size="512"):
    out = {}
    for b in d.get("blocks", []):
        if b["size"] != size or not b.get("result"):
            continue
        out.setdefault(b["arm"], []).append([f["fold_s"] for f in b["result"]["folds"]])
    return out


def digests(d, size="512"):
    out = {}
    for b in d.get("blocks", []):
        if b["size"] != size or not b.get("result"):
            continue
        for f in b["result"]["folds"]:
            g = f.get("digest") or f.get("cif_digest") or b["result"].get("digest")
            if g:
                out.setdefault(b["arm"], set()).add(g[:16])
    return out


def perm_p(base_m, on_m):
    """Two-sided-in-sign permutation over block medians: how often does a random relabelling of
    the 8 block medians into two groups separate them at least this far in this direction?"""
    allm = base_m + on_m
    obs = st.median(base_m) - st.median(on_m)
    k = len(base_m)
    hits = tot = 0
    for combo in itertools.combinations(range(len(allm)), k):
        a = [allm[i] for i in combo]
        b = [allm[i] for i in range(len(allm)) if i not in combo]
        tot += 1
        if st.median(a) - st.median(b) >= obs:
            hits += 1
    return hits, tot


for tag, d in (("RECORD (contended, bfp8-region-t-fold)", OLD), ("TODAY (quiet, c14-land-tail)", NEW)):
    a = arms(d)
    if "base" not in a or "on" not in a:
        print(f"{tag}: incomplete"); continue
    flat = lambda bs: [x for b in bs for x in b]
    bm = [st.median(b) for b in a["base"]]
    om = [st.median(b) for b in a["on"]]
    mb, mo = st.median(flat(a["base"])), st.median(flat(a["on"]))
    aa = max(max(x, y) / min(x, y) for i, x in enumerate(bm) for y in bm[i + 1:])
    oo = max(max(x, y) / min(x, y) for i, x in enumerate(om) for y in om[i + 1:])
    hits, tot = perm_p(bm, om)
    sep = min(flat(a["base"])) - max(flat(a["on"]))
    print(f"\n{tag}")
    print(f"  base folds {sorted(round(x,3) for x in flat(a['base']))}")
    print(f"  on   folds {sorted(round(x,3) for x in flat(a['on']))}")
    print(f"  base median {mb:.3f} s  on median {mo:.3f} s  delta {mb-mo:+.4f} s  "
          f"ratio {mb/mo:.5f}")
    print(f"  A/A floor (base blocks) {aa:.5f}   on-arm block spread {oo:.5f}")
    print(f"  block medians base {[round(x,3) for x in bm]}  on {[round(x,3) for x in om]}")
    print(f"  complete separation gap {sep:+.4f} s (positive = every on fold beat every base fold)")
    print(f"  permutation p = {hits}/{tot} = {hits/tot:.4f}")
    print(f"  verdict: {'WIN' if mb/mo > aa else 'NULL, inside its own A/A floor'}")
    cl = [f["clock"] for b in d.get("blocks", []) if b.get("result")
          for f in b["result"]["folds"] if f.get("clock", {}).get("aiclk_n")]
    if cl:
        print(f"  clock {min(c['aiclk_min'] for c in cl)}-{max(c['aiclk_max'] for c in cl)} MHz "
              f"over {sum(c['aiclk_n'] for c in cl)} during-fold samples")

print("\nDIGESTS (16 hex chars)")
for tag, d in (("record", OLD), ("today", NEW)):
    print(f"  {tag}: {json.dumps({k: sorted(v) for k, v in digests(d).items()})}")
