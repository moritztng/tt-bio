"""Turn perf/mgx_matrix/out_np/ into the measured table for embed, saprot, affinity and design.

Per case: exit code, wall time, the refusal or error line, and what came out. An embedding counts
only when every vector is finite and not all zero; an affinity only when the scalar is finite; a
design only when a structure file was written.

Usage: analyze_np.py <out_np>  -> JSON on stdout
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
ERR = re.compile(r"(Error|error:|Exception|refus|cannot|Traceback|FATAL|not supported|invalid)", re.I)


def tail(log: Path):
    lines = [l.rstrip() for l in log.read_text(errors="replace").splitlines() if l.strip()]
    m = re.search(r"EXIT=(\d+) WALL=(\d+)s", "\n".join(lines[-3:]))
    body = [l for l in lines if not l.startswith("EXIT=") and "device bring-up lock" not in l
            and "unauthenticated requests" not in l]
    err = next((l for l in reversed(body) if ERR.search(l)), None)
    return (int(m.group(1)) if m else None), (int(m.group(2)) if m else None), err, body[-1:] or [""]


def embed_out(d: Path):
    man = d / "manifest.json"
    if not man.is_file():
        return None
    seqs = json.loads(man.read_text()).get("sequences", [])
    out = []
    for s in seqs:
        z = np.load(d / s["file"])
        per = z["per_residue"] if "per_residue" in z else z[z.files[0]]
        out.append({"id": s["id"], "length": s["length"], "shape": list(per.shape),
                    "finite": bool(np.isfinite(per).all()),
                    "nonzero": bool(np.abs(per).sum() > 0),
                    "logits": "logits" in z.files})
    return out


def affinity_out(d: Path):
    rows = []
    for f in sorted(d.glob("*_affinity.json")):
        v = json.loads(f.read_text())
        nums = {k: x for k, x in v.items() if isinstance(x, (int, float))}
        rows.append({"file": f.name, "values": nums,
                     "finite": all(math.isfinite(x) for x in nums.values()) and bool(nums)})
    return rows


table = {}
for log in sorted(OUT.rglob("*.log")):
    rel = log.relative_to(OUT).with_suffix("")
    parts = rel.parts
    if len(parts) != 3:
        continue
    surface, model, case = parts
    code, wall, err, last = tail(log)
    d = OUT / rel
    v = {"exit": code, "wall_s": wall, "error": err if code else None, "last": last[0][:300]}
    if surface in ("embed", "saprot"):
        v["embeddings"] = embed_out(d)
    elif surface == "affinity":
        v["affinity"] = affinity_out(d)
    elif surface == "design":
        v["structures"] = sorted(str(p.relative_to(d)) for p in d.rglob("*")
                                 if p.suffix in (".cif", ".pdb") and "input" not in p.parts)[:12]
    table.setdefault(surface, {}).setdefault(model, {})[case] = v
json.dump(table, sys.stdout, indent=1, default=str)
