"""Score every rung a break-run wrote, and say WHERE each break is.

`struct_signal.py` answers "is this backbone continuous"; this answers "and if not, at which
residue, and is that residue in the designed binder or in the target". A break at the binder's
first residue and a break in the middle of the target are the same number there and different
findings here.

    python perf/ceilrfd3/break_score.py perf/ceilrfd3/breakrun/pcbh.jsonl [...]
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceilings.struct_signal import clashes, read_cif  # noqa: E402


def breaks_with_index(atom, asym, seq, xyz):
    ca = atom == "CA"
    out, worst = [], 0.0
    for ch in np.unique(asym[ca]):
        m = ca & (asym == ch)
        s, p = seq[m], xyz[m]
        o = np.argsort(s)
        s, p = s[o], p[o]
        adj = np.diff(s) == 1
        if not adj.any():
            continue
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        worst = max(worst, float(d[adj].max()))
        for i in np.nonzero(adj & (d > 4.5))[0]:
            out.append((str(ch), int(s[i]), int(s[i + 1]), round(float(d[i]), 2)))
    return worst, out


def main(paths):
    rows = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if not r.get("ok"):
                rows.append({**{k: r.get(k) for k in
                                ("spec_id", "target", "target_res", "binder", "total_res", "seed")},
                             "err": (r.get("error") or {}).get("type", "no-cif")})
                continue
            atom, comp, asym, seq, elem, xyz = read_cif(r["cif"])
            worst, brk = breaks_with_index(atom, asym, seq, xyz)
            frac, _ = clashes(asym, seq, elem, xyz)
            rows.append({"spec_id": r["spec_id"], "target": pathlib.Path(r["target"]).stem,
                         "target_res": r["target_res"], "binder": r["binder"],
                         "total_res": r["total_res"], "seed": r["seed"],
                         "wall_s": r["wall_s"], "atoms": r["atoms"],
                         "worst_ca_ca": round(worst, 2), "n_breaks": len(brk),
                         "clash_frac": round(frac, 5),
                         # The design CLI writes ONE chain numbered 0..N-1, target then design,
                         # so seq < crop is target, seq >= crop is the designed segment and the
                         # pair (crop-1, crop) is the junction between them. Getting that
                         # off by one would report a junction break as the design's first bond.
                         "breaks": [{"at": "%d->%d" % (a, b), "d": d,
                                     "where": "target" if b < r["target_res"] else
                                              ("junction" if b == r["target_res"] else "design"),
                                     "design_pos": (b - r["target_res"] + 1)
                                                   if b >= r["target_res"] else None}
                                    for _, a, b, d in brk]})
    print(json.dumps(rows, indent=2))
    hdr = "%-18s %-10s %5s %5s %5s %5s %8s %7s %9s  %s"
    print("\n" + hdr % ("spec", "target", "crop", "bind", "total", "seed", "worstCA", "breaks",
                        "clash", "where"))
    for r in rows:
        if "err" in r:
            print(hdr % (r["spec_id"], "-", r["target_res"], r["binder"], r["total_res"],
                         r["seed"], "-", "-", "-", r["err"]))
            continue
        where = ", ".join("%s@%s%s" % (b["where"], b["at"],
                                       "" if b["design_pos"] is None else
                                       " (design res %d)" % b["design_pos"])
                          for b in r["breaks"]) or "-"
        print(hdr % (r["spec_id"], r["target"], r["target_res"], r["binder"], r["total_res"],
                     r["seed"], r["worst_ca_ca"], r["n_breaks"], r["clash_frac"], where))


if __name__ == "__main__":
    main(sys.argv[1:])
