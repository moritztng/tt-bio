"""Does the chunk width move a sample? Compares plan-width folds sample by sample.

    python perf/mgx_sample_width/width.py <default tag> <tag> [<tag> ...]

Structures are written best-first by confidence, so a file's index is a rank, not a sample id.
Each sample of a width run is matched to its nearest default-width sample by Kabsch CA-RMSD; the
match is unambiguous when that distance is far below the distance to the second nearest, which
is printed as the margin. The floor is the default run's own spread: each sample's Kabsch
CA-RMSD to its nearest sibling, what one more noise draw moves a structure. Bit-identity comes
from the per-sample digests in runs.jsonl, where boltz2 writes them.
"""
import json
import re
import sys
from pathlib import Path

import gemmi
import numpy as np

HERE = Path(__file__).resolve().parent


def ca(path):
    st = gemmi.read_structure(str(path))
    return np.array([a.pos.tolist() for r in st[0].all() for a in [r.atom] if a.name == "CA"])


def kabsch(p, q):
    p, q = p - p.mean(0), q - q.mean(0)
    u, _s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt(((p @ (u @ np.diag([1, 1, d]) @ vt) - q) ** 2).sum(1).mean()))


def samples(tag):
    got = {}
    for f in (HERE / "out" / tag / "pred").rglob("*.cif"):
        m = re.search(r"_model_(\d+)\.cif$", f.name)
        got[int(m.group(1)) if m else 0] = ca(f)
    return [got[k] for k in sorted(got)]


def digests(rec):
    """Per-sample hashes in sample order; the line also names the width, which is not compared."""
    return [l.split()[-1] for l in rec.get("digest") or [] if l.strip()]


def main():
    recs = {}
    for line in (HERE / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("ok"):
            recs[r["tag"]] = r
    base, *others = sys.argv[1:]
    ref = samples(base)
    sib = [min(kabsch(x, y) for j, y in enumerate(ref) if j != i) for i, x in enumerate(ref)]
    print(f"{base}: {len(ref)} samples, card {recs[base]['card']}; floor (nearest sibling) "
          f"min {min(sib):.3f} median {float(np.median(sib)):.3f} A")
    for tag in others:
        got = samples(tag)
        rows = [sorted(kabsch(x, y) for y in ref)[:2] for x in got]
        same = sum(a == b for a, b in zip(digests(recs[tag]), digests(recs[base])))
        print(f"  {tag} (card {recs[tag]['card']}): worst nearest {max(r[0] for r in rows):.3f} A, "
              f"median {float(np.median([r[0] for r in rows])):.3f} A, smallest margin "
              f"{min(r[1] for r in rows):.3f} A; {same}/{len(ref)} digests identical")


if __name__ == "__main__":
    main()
