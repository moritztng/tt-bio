#!/usr/bin/env python3
"""Release-gate design quality of every CIF in each DIR (host only).

    gap_quality.py N_DESIGN DIR [DIR ...]

Per directory: CA-CA steps in band, chain breaks, heavy-atom clashes and distinct amino acids over the
last N_DESIGN residues, with perf/wh-correctness/check_structure.py's thresholds. Used to read a batch
of CLI designs (e.g. 8 at --batch_size 8) where there is no same-seed reference to grade against.
"""
import importlib.util
import pathlib
import sys

import gemmi
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cs", ROOT / "perf" / "wh-correctness" / "check_structure.py")
geom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geom)
N = int(sys.argv[1])
for d in sys.argv[2:]:
    rows = []
    for p in sorted(pathlib.Path(d).glob("*.cif")):
        st = gemmi.read_structure(str(p))
        st.setup_entities()
        ca, names = [], []
        for ch in st[0]:
            for r in ch:
                a = r.find_atom("CA", "*")
                if a is not None:
                    ca.append([a.pos.x, a.pos.y, a.pos.z])
                    names.append(r.name)
        ca = np.asarray(ca)[-N:]
        step = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
        lo, hi = geom.CA_CA_BAND
        seq = [geom._one_letter(n) for n in names[-N:]]
        rows.append((float(((step >= lo) & (step <= hi)).mean()), int((step > geom.CA_CA_BREAK).sum()),
                     geom.clashes(st)[0], len({c for c in seq if c != "X"})))
    a = np.array(rows, float)
    print("%s  n=%d  in_band %s  breaks %s  clashes %s (mean %.1f)  distinct %s" % (
        d, len(rows), a[:, 0].round(3).tolist(), a[:, 1].astype(int).tolist(),
        a[:, 2].astype(int).tolist(), a[:, 2].mean(), a[:, 3].astype(int).tolist()))
