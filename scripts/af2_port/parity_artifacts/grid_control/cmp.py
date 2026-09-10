"""Compare two af2ig tap reports field by field, ignoring only the arm bookkeeping."""
import json
import sys

SKIP = {"_arm"}


def load(p):
    d = json.load(open(p))
    return {k: v for k, v in d.items() if k not in SKIP}


a, b = load(sys.argv[1]), load(sys.argv[2])
na, nb = sys.argv[3], sys.argv[4]
keys = sorted(set(a) | set(b))
diffs = []
for k in keys:
    if a.get(k, "<absent>") != b.get(k, "<absent>"):
        diffs.append(k)
print(f"{na} vs {nb}: {len(keys)} fields, {len(diffs)} differ")
if not diffs:
    print("BIT-FOR-BIT IDENTICAL")
for k in diffs:
    va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
    if k == "rows" and isinstance(va, list) and isinstance(vb, list):
        ia = {r.get("name", r.get("tap")): r for r in va}
        ib = {r.get("name", r.get("tap")): r for r in vb}
        moved = [n for n in sorted(set(ia) | set(ib)) if ia.get(n) != ib.get(n)]
        print(f"  rows: {len(ia)} vs {len(ib)} taps, {len(moved)} taps differ")
        for n in moved[:200]:
            ra, rb = ia.get(n, {}), ib.get(n, {})
            fa = {kk: ra.get(kk) for kk in sorted(set(ra) | set(rb)) if ra.get(kk) != rb.get(kk)}
            print(f"    {n}: " + ", ".join(
                f"{kk}={ra.get(kk)!r}->{rb.get(kk)!r}" for kk in fa))
    elif k == "scalars" and isinstance(va, (list, dict)):
        print(f"  scalars:\n    {na}: {json.dumps(va, default=str)}\n    {nb}: {json.dumps(vb, default=str)}")
    elif isinstance(va, list) and isinstance(vb, list) and len(va) + len(vb) < 60:
        print(f"  {k}:\n    {na}: {va}\n    {nb}: {vb}")
    else:
        sa, sb = json.dumps(va, default=str), json.dumps(vb, default=str)
        print(f"  {k}: {sa[:400]} -> {sb[:400]}")
