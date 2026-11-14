"""Design-by-design comparison of two runs' CIFs for the same seed: byte-equal apart from the _entry.time stamp, else CA RMSD (no superposition).

    python perf/rfd4/l1decline/compare.py <dirA> <dirB>     # e.g. .../b16/compile-s1000 .../b32/compile-s1000
Designs are matched by their `_dN.cif` index; only indices present in both are compared.
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path


def designs(d):
    return {int(re.search(r"_d(\d+)\.cif$", p.name).group(1)): p for p in Path(d).glob("*_d*.cif")}


def body(p):
    """The CIF without its _entry.time stamp, the one line two runs of the same design always differ in."""
    return b"".join(l for l in p.read_bytes().splitlines(True) if not l.startswith(b"_entry.time"))


def ca(p):
    out = []
    for line in p.read_text().splitlines():
        f = line.split()
        if f and f[0] in ("ATOM", "HETATM") and f[2] == "CA":
            out.append(tuple(float(v) for v in f[-5:-2]))  # Cartn_x/y/z, before the model and id columns
    return out


a, b = designs(sys.argv[1]), designs(sys.argv[2])
rows = []
for i in sorted(set(a) & set(b)):
    ha, hb = (hashlib.sha256(body(p)).hexdigest()[:16] for p in (a[i], b[i]))
    row = {"design": i, "sha_a": ha, "sha_b": hb, "equal": ha == hb}
    if ha != hb:
        xa, xb = ca(a[i]), ca(b[i])
        if len(xa) == len(xb) and xa:
            row["ca_rmsd_A"] = round(math.sqrt(sum(sum((p - q) ** 2 for p, q in zip(u, v))
                                                   for u, v in zip(xa, xb)) / len(xa)), 4)
    rows.append(row)
print(json.dumps({"a": sys.argv[1], "b": sys.argv[2], "n": len(rows),
                  "equal": sum(r["equal"] for r in rows), "rows": rows}, indent=1))
