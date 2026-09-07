"""Structural signal for one fold's output dir: backbone continuity, clashes, confidence.

An OOM that stops happening is not the bar. A ceiling pass has already shipped a 1024 aa fold
that came back at pLDDT 0.90 on a backbone with 19-22 A breaks and ~7% of heavy atoms clashing,
and every host-side check and the parity gate passed on it, because none of them read the
geometry. So every rung a ceiling task calls PASS gets scored here, on the CIF.

Model-agnostic: reads `<out_dir>/*results_*/structures/*.cif` and the sibling `results.json`,
which every tt-bio structure model writes.

Both arms carry a negative control and the two are independent, measured on a 1095 aa RF3 fold:
shifting every residue above 500 by 20 A moves `worst_ca_ca` 4.20 -> 18.56 and `ca_breaks`
0 -> 1 and leaves `clash_frac` alone; superposing one chain onto another chain's centroid moves
`clash_frac` 0.0071 -> 0.1288 and leaves continuity alone. For scale, the deposited experimental
9SAT reads `clash_frac` 0.00098 and `worst_ca_ca` 3.91 A through this same code.

    python perf/ceilings/struct_signal.py <out_dir>
"""
import glob
import json
import sys

import numpy as np


def read_cif(path):
    """Return (atom_name, comp_id, asym_id, seq_id, xyz) columns, header-driven."""
    cols, rows, in_loop = {}, [], False
    for line in open(path):
        s = line.strip()
        if s.startswith("_atom_site."):
            cols[s.split(".", 1)[1]] = len(cols)
            in_loop = True
            continue
        if in_loop and (s.startswith("ATOM") or s.startswith("HETATM")):
            rows.append(s.split())
        elif in_loop and rows:
            break
    def col(name):
        return [r[cols[name]] for r in rows]
    xyz = np.array([[float(v) for v in (r[cols["Cartn_x"]], r[cols["Cartn_y"]],
                                        r[cols["Cartn_z"]])] for r in rows])
    return (np.array(col("label_atom_id")), np.array(col("label_comp_id")),
            np.array(col("label_asym_id")),
            np.array([int(v) if v not in ".?" else -1 for v in col("label_seq_id")]),
            np.array(col("type_symbol")), xyz)


def continuity(atom, asym, seq, xyz):
    """Max and count of CA-CA breaks between sequence-adjacent residues, per chain."""
    ca = atom == "CA"
    worst, breaks, n = 0.0, 0, 0
    for ch in np.unique(asym[ca]):
        m = ca & (asym == ch)
        s, p = seq[m], xyz[m]
        o = np.argsort(s)
        s, p = s[o], p[o]
        adj = np.diff(s) == 1
        if not adj.any():
            continue
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)[adj]
        n += len(d)
        worst = max(worst, float(d.max()))
        breaks += int((d > 4.5).sum())
    return worst, breaks, n


def clashes(asym, seq, elem, xyz, cutoff=2.0, sep=2):
    """Fraction of heavy atoms with a heavy-atom neighbour under `cutoff` A that is at
    least `sep` residues away in sequence (or on another chain). Bonded pairs excluded by
    the separation rule, so a clean structure scores ~0."""
    heavy = elem != "H"
    p, a, s = xyz[heavy], asym[heavy], seq[heavy]
    hit = np.zeros(len(p), bool)
    step = 2048
    for i in range(0, len(p), step):
        d = np.linalg.norm(p[i:i + step, None, :] - p[None, :, :], axis=-1)
        far = (a[i:i + step, None] != a[None, :]) | (
            np.abs(s[i:i + step, None] - s[None, :]) >= sep)
        near = (d < cutoff) & far
        hit[i:i + step] |= near.any(1)
        hit |= near.any(0)
    return float(hit.sum()) / len(p), len(p)


def main(out_dir):
    cif = sorted(glob.glob(f"{out_dir}/*results_*/structures/*.cif"))
    res = sorted(glob.glob(f"{out_dir}/*results_*/results.json"))
    if not cif:
        print(json.dumps({"scored": 0}))
        return
    atom, comp, asym, seq, elem, xyz = read_cif(cif[0])
    worst, breaks, nadj = continuity(atom, asym, seq, xyz)
    frac, nheavy = clashes(asym, seq, elem, xyz)
    r = json.load(open(res[0]))[0] if res else {}
    print(json.dumps({
        "scored": 1, "cif": cif[0], "n_atoms": len(atom), "n_heavy": nheavy,
        "ca_pairs": nadj, "worst_ca_ca": round(worst, 2), "ca_breaks": breaks,
        "clash_frac": round(frac, 5), "plddt": r.get("plddt"), "ptm": r.get("ptm"),
        "iptm": r.get("iptm"), "has_clash": r.get("has_clash"),
        "n_tokens": r.get("n_tokens"), "runtime_s": r.get("runtime_s")}))


if __name__ == "__main__":
    main(sys.argv[1])
