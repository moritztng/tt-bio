#!/usr/bin/env python3
"""Score already-folded arm CIFs. Separate from the folding so a scorer bug costs no device time."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path("/home/ttuser/.coworker/wt/ttx-eltwise-fusion-sweep")
sys.path.insert(0, str(REPO / "perf" / "other512"))
from cif_rmsd import read_atoms, kabsch_rmsd

for j in sorted((REPO / "perf/eltwise_fusion").glob("parity*_*.json")):
    d = json.loads(j.read_text())
    rows = {r["arm"]: r for r in d.get("arms", [])}
    if "off" not in rows:
        print(f"{j.name}: no off arm"); continue
    bk, bx = read_atoms(Path(rows["off"]["cif"]))
    bi = {k: i for i, k in enumerate(bk)}
    scores = {}
    for arm in ("on", "off2"):
        if arm not in rows:
            continue
        ck, cx = read_atoms(Path(rows[arm]["cif"]))
        ci = {k: i for i, k in enumerate(ck)}
        common = sorted(set(bk) & set(ck))
        P, Q = bx[[bi[k] for k in common]], cx[[ci[k] for k in common]]
        ca = [k for k in common if "CA" in k]
        s = dict(all_atom_rmsd=float(kabsch_rmsd(P, Q)), n_atoms=len(common),
                 max_atom_dev=float(np.abs(P - Q).max()),
                 ca_rmsd=(float(kabsch_rmsd(bx[[bi[k] for k in ca]], cx[[ci[k] for k in ca]]))
                          if ca else None), n_ca=len(ca),
                 bit_exact=rows[arm]["cif_sha256"] == rows["off"]["cif_sha256"])
        scores[arm] = s
        print(f"{d['env']['model']:12s} {rows[arm]['fixture']} {arm:4s} vs off: "
              f"all-atom {s['all_atom_rmsd']:.6f} A  CA {s['ca_rmsd']:.6f} A  "
              f"maxdev {s['max_atom_dev']:.4f} A  n={s['n_atoms']}/{s['n_ca']}CA  "
              f"bit-exact={s['bit_exact']}")
    d["scores"] = {rows["off"]["fixture"]: scores}
    j.write_text(json.dumps(d, indent=1))
