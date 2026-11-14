"""Build the positive control for the RFD4 refold oracle: the NATIVE interface, same oracle, same MSA.

The RFD4 in-silico success rate is `ppi.pass` from an of3 refold of each design. A 0 % is only a
measurement if the oracle can reproduce an interface on this target AT ALL -- otherwise it is the
11-04 failure in a new costume. Global target RMSD does not settle that: on 9TH6 the OpenDDE abag
checkpoint folds chain A to 8.8 A from the crystal and still scores DockQ 0.84-0.90, because DockQ
only needs the epitope to be locally right. So the control has to be an actual interface.

The control is the deposited complex expressed in the design's own frame and numbering:
  chain A  the design's target chain, unchanged -- so the staged a3m (keyed by sequence hash) still
           hits and chain A keeps its 419-row alignment, exactly as in the graded run;
  chain B  the deposited VHH, rigid-body moved by the transform that superposes the deposited
           target onto the design's target (0.96 A over 428 CA on 9TH6, so the move is well posed).

Feed the result to refold.py as a one-design set. If it comes back ppi.pass=1 the oracle grades this
target and 0/128 is a real result about the designs; if the native interface also fails, the oracle
cannot grade this target and no success rate may be quoted from it.

    python3 rfd4_control.py --ref ref/9TH6.cif --ref-target A --ref-binder B \
        --design designs/tt-bh-msa/9TH6_A_b120_s1234_d0_generated.cif --out designs/control-native/
"""

import argparse
from pathlib import Path

import gemmi
import numpy as np


def ca_by_seqid(chain):
    out = {}
    for r in chain:
        a = r.find_atom("CA", "*")
        if a:
            code = gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
            out[r.seqid.num] = (np.array([a.pos.x, a.pos.y, a.pos.z]), code)
    return out


def get_chain(st, name):
    for ch in st[0]:
        if ch.name == name:
            return ch
    raise SystemExit(f"no chain {name}")


def superpose(P, Q):
    """Rotation+translation taking P onto Q, plus the RMSD it achieves."""
    pc, qc = P.mean(0), Q.mean(0)
    V, _, W = np.linalg.svd((P - pc).T @ (Q - qc))
    R = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ W))]) @ W
    rmsd = float(np.sqrt(((((P - pc) @ R) - (Q - qc)) ** 2).sum() / len(P)))
    return R, qc - pc @ R, rmsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-target", required=True)
    ap.add_argument("--ref-binder", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--design-target", default="A")
    ap.add_argument("--design-binder", default="B")
    ap.add_argument("--out", required=True, help="directory; the control cif is written inside")
    ap.add_argument("--max-rmsd", type=float, default=2.0,
                    help="refuse if target superposition is worse than this")
    args = ap.parse_args()

    ref = gemmi.read_structure(args.ref)
    ref.setup_entities()
    ref.remove_alternative_conformations()
    des = gemmi.read_structure(args.design)
    des.setup_entities()

    rt = ca_by_seqid(get_chain(ref, args.ref_target))
    dt = ca_by_seqid(get_chain(des, args.design_target))

    best = None
    for off in range(-200, 201):
        common = [i for i in dt if i + off in rt]
        if len(common) < 100:
            continue
        ident = sum(rt[i + off][1] == dt[i][1] for i in common) / len(common)
        if best is None or (ident, len(common)) > (best[1], best[2]):
            best = (off, ident, len(common))
    if best is None or best[1] < 0.98:
        raise SystemExit(f"target chains do not align: {best}")
    off, ident, _ = best
    common = sorted(i for i in dt if i + off in rt)
    P = np.array([rt[i + off][0] for i in common])
    Q = np.array([dt[i][0] for i in common])
    R, t, rmsd = superpose(P, Q)
    print(f"target superposition: {len(common)} CA, offset {off}, identity {ident:.0%}, RMSD {rmsd:.2f} A")
    if rmsd > args.max_rmsd:
        raise SystemExit(f"RMSD {rmsd:.2f} A over the {args.max_rmsd} A bar: the deposited target is "
                         "not the conformation the design was built on, so the graft would invent a pose")

    # design target as-is + deposited binder moved into the design frame
    out = gemmi.Structure()
    out.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    tgt = get_chain(des, args.design_target).clone()
    tgt.name = args.design_target
    model.add_chain(tgt)
    # The deposited chain carries its waters and heteroatoms; left in, they reach the oracle as
    # ligands and the control stops being "the same two polymer chains as a design".
    src = get_chain(ref, args.ref_binder)
    dropped = sorted({r.name for r in src if not r.find_atom("CA", "*")})
    binder = gemmi.Chain(args.design_binder)
    for r in src:
        if r.find_atom("CA", "*"):
            binder.add_residue(r)
    if dropped:
        print(f"dropped non-polymer residues from the binder chain: {dropped}")
    n = 0
    for r in binder:
        for a in r:
            v = np.array([a.pos.x, a.pos.y, a.pos.z]) @ R + t
            a.pos = gemmi.Position(*v)
            n += 1
    model.add_chain(binder)
    out.add_model(model)
    out.setup_entities()

    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    stem = Path(args.design).stem
    p = d / f"{stem}_CONTROLNATIVE.cif"
    out.make_mmcif_document().write_file(str(p))
    print(f"moved {n} binder atoms ({len(binder)} res) into the design frame")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
