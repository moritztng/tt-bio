#!/usr/bin/env python3
"""One table per A/B JSON: fold seconds, the A/A floor over the reference arms, the CIF digest,
the trunk bodies, and K2's served/declined so a dark gate cannot pass as a fired one.

Run from the worktree root:  python3 perf/b2sizes/summarize_qb2_ab.py perf/b2sizes/*.json
"""
import json
import statistics
import sys


def rows(path):
    d = json.load(open(path))
    head = (f"{path}\n  host={d.get('host')} card={d.get('card')} ttnn={d.get('ttnn')} "
            f"grid={d.get('grid')} rec={d.get('recycling_steps')} "
            f"steps={d.get('sampling_steps')}")
    out = [head]
    refs = [r["fold_s"] for r in d["runs"] if r.get("arm") == "on" and "fold_s" in r]
    ref = statistics.median(refs) if refs else None
    out.append(f"  reference arms n={len(refs)} {refs}")
    if len(refs) >= 2:
        out.append(f"  A/A floor (max-min over the reference arms) = {max(refs) - min(refs):.3f} s")
    out.append(f"  {'arm':8s} {'fold_s':>8s} {'vs on':>9s} {'ratio':>7s} {'block ms':>10s} "
               f"{'triatt ms':>10s} {'trimul ms':>10s}  K2 srv/dec  loadavg  cif")
    for i, r in enumerate(d["runs"]):
        if "error" in r:
            out.append(f"  {r.get('arm','?'):8s} ERROR {r['error'][:120]}")
            continue
        f = r["fold_s"]
        pm = r.get("persistent_mask", {})
        dl = f"{f - ref:+.3f}" if ref else "-"
        rt = f"{ref / f:.4f}x" if ref else "-"
        cif = ",".join(sorted(r.get("cif_sha256", {}).values()))
        out.append(f"  {r['arm']:8s} {f:8.3f} {dl:>9s} {rt:>7s} "
                   f"{r.get('block_wall_ms', 0):10.1f} {r.get('triatt_wall_ms', 0):10.1f} "
                   f"{r.get('trimul_wall_ms', 0):10.1f}  "
                   f"{pm.get('served')}/{pm.get('declined')}  "
                   f"{r.get('loadavg', ['?'])[0]}  {cif}")
        if pm.get("rejects"):
            out.append(f"           rejects {pm['rejects']}")
    digests = {tuple(sorted(r.get("cif_sha256", {}).items())) for r in d["runs"] if "fold_s" in r}
    out.append(f"  bit-exact across arms: {len(digests) == 1}")
    return "\n".join(out)


for p in sys.argv[1:]:
    print(rows(p))
    print()
