#!/usr/bin/env python3
"""All-atom RMSD between two mmCIF folds, with the atom identities checked first.

Written to score `TT_BIO_APB_CONCAT_HEADS` at the FOLD from the CIFs its own parity run
left behind (perf/roof_concat/parity/), because that run's timings are unusable -- two
identical-arm folds read 25.12 s and 19.444 s at loadavg 27-31 -- while its structures are
exact and still on disk. No card, no ttnn.

Two numbers per pair, and the difference between them matters:

  direct    per-atom coordinate difference, no superposition. For two arms folded from the
            SAME seed and input this is the honest reading: nothing rotated the molecule, so
            a rigid-body term here would be a real divergence and not a frame choice.
  kabsch    after optimal superposition. Quoted as a CONTROL: if it is far below direct, the
            pair differs by a rigid-body move rather than by local geometry, which is a
            different claim and must not be reported as agreement.

Atom identity is compared before any coordinate is read. Two CIFs of the same fixture
should agree atom-for-atom; if they do not, an RMSD over index-aligned rows is comparing
different atoms and is meaningless rather than large.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

# Cartn_x is column 10 in this writer's _atom_site loop (0-based), and the identity of an
# atom is (label_atom_id, label_comp_id, label_seq_id, label_asym_id).
COL_X, COL_ATOM, COL_COMP, COL_SEQ, COL_ASYM = 10, 3, 5, 6, 9


def read(path: Path):
    ids, xyz = [], []
    for line in path.read_text().splitlines():
        if not (line.startswith("ATOM ") or line.startswith("HETATM ")):
            continue
        f = line.split()
        ids.append((f[COL_ATOM], f[COL_COMP], f[COL_SEQ], f[COL_ASYM]))
        xyz.append((float(f[COL_X]), float(f[COL_X + 1]), float(f[COL_X + 2])))
    return ids, np.asarray(xyz, dtype=np.float64)


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    ac, bc = a - a.mean(0), b - b.mean(0)
    v, _, wt = np.linalg.svd(ac.T @ bc)
    d = np.sign(np.linalg.det(v @ wt))
    r = v @ np.diag([1.0, 1.0, d]) @ wt
    return float(np.sqrt(((ac @ r - bc) ** 2).sum(1).mean()))


def compare(p: Path, q: Path):
    ip, a = read(p)
    iq, b = read(q)
    if ip != iq:
        n = sum(1 for x, y in zip(ip, iq) if x != y)
        return None, None, f"atom identities differ in {n} of {min(len(ip), len(iq))} rows"
    if len(a) != len(b):
        return None, None, f"atom counts differ: {len(a)} vs {len(b)}"
    direct = float(np.sqrt(((a - b) ** 2).sum(1).mean()))
    return direct, kabsch_rmsd(a, b), f"{len(a)} atoms"


def main(argv):
    base = Path(argv[1]) if len(argv) > 1 else Path("perf/roof_concat/parity")
    print(f"{'pair':<46}{'direct A':>10}{'kabsch A':>10}   note")
    for size in (298, 512):
        arms = {t: base / f"cdk2x2_{size}_{t}.cif" for t in ("off", "off2", "offseed", "on")}
        if not all(p.is_file() for p in arms.values()):
            continue
        for label, x, y in (("A/A control        off vs off2", "off", "off2"),
                            ("seed floor         off vs offseed", "off", "offseed"),
                            ("LEVER              off vs on", "off", "on")):
            d, k, note = compare(arms[x], arms[y])
            tag = f"{size} aa  {label}"
            if d is None:
                print(f"{tag:<46}{'-':>10}{'-':>10}   {note}")
            else:
                print(f"{tag:<46}{d:>10.5f}{k:>10.5f}   {note}")
        # sha256 is the cheapest bit-exactness signal and it is free here
        shas = {t: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for t, p in arms.items()}
        same = "off==off2 BIT-EXACT" if shas["off"] == shas["off2"] else "off!=off2"
        print(f"{'':<46}{'':>10}{'':>10}   {same}, on={shas['on']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
