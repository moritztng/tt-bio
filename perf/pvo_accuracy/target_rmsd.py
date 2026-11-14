"""CA RMSD of a predicted chain against its deposited counterpart, on the shared numbering.

Why this exists: a refold oracle reports plDDT, and plDDT is a LOCAL confidence. On 9TH6 the of3
oracle returns overall plDDT 77.6 while its target chain sits 10.6 A from the crystal -- confident
secondary structure, wrong domain packing. Any interface metric computed on top of that is not a
measurement, and the in-silico success rate it produces is a clean 0 % that means nothing. So the
gate on an oracle run is this number, not its confidence.

The numbering offset between a construct sequence and a deposited entry is unknown (9TH6's chain A
is deposited 25-524 and modelled 1-500 in the design), so the offset is found by maximising
one-letter identity over the overlap and the alignment is then required to be fully identical.

    python3 target_rmsd.py --ref ref/9TH6.cif --ref-chain A --chain A pred1.cif pred2.cif ...
"""

import argparse
import statistics as st

import gemmi
import numpy as np


def ca_map(path, chain):
    """{seqid -> (xyz, one-letter)} for the CA atoms of `chain`."""
    s = gemmi.read_structure(str(path))
    s.setup_entities()
    s.remove_alternative_conformations()
    for ch in s[0]:
        if ch.name != chain:
            continue
        out = {}
        for r in ch:
            a = r.find_atom("CA", "*")
            if a:
                code = gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
                out[r.seqid.num] = (np.array([a.pos.x, a.pos.y, a.pos.z]), code)
        return out
    return {}


def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, _, W = np.linalg.svd(Pc.T @ Qc)
    R = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ W))]) @ W
    return float(np.sqrt((((Pc @ R) - Qc) ** 2).sum() / len(P)))


def best_offset(pred, ref, min_overlap=100):
    """Numbering shift that maximises identity, as (offset, identity, n)."""
    best = None
    for off in range(-200, 201):
        common = [i for i in pred if i + off in ref]
        if len(common) < min_overlap:
            continue
        ident = sum(ref[i + off][1] == pred[i][1] for i in common) / len(common)
        if best is None or (ident, len(common)) > (best[1], best[2]):
            best = (off, ident, len(common))
    return best


def rmsd_to_ref(pred_path, chain, ref, min_identity=0.98):
    pred = ca_map(pred_path, chain)
    if not pred:
        return None, f"no chain {chain}"
    got = best_offset(pred, ref)
    if got is None:
        return None, "no numbering overlap"
    off, ident, n = got
    if ident < min_identity:
        return None, f"identity {ident:.0%} at best offset {off}, not the same molecule"
    common = sorted(i for i in pred if i + off in ref)
    P = np.array([pred[i][0] for i in common])
    Q = np.array([ref[i + off][0] for i in common])
    return kabsch_rmsd(P, Q), f"{len(common)} CA, offset {off}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="deposited structure")
    ap.add_argument("--ref-chain", required=True)
    ap.add_argument("--chain", required=True, help="chain to score in each prediction")
    ap.add_argument("--quiet", action="store_true", help="summary line only")
    ap.add_argument("preds", nargs="+")
    args = ap.parse_args()

    ref = ca_map(args.ref, args.ref_chain)
    if not ref:
        raise SystemExit(f"{args.ref}: no chain {args.ref_chain}")

    vals = []
    for p in args.preds:
        v, note = rmsd_to_ref(p, args.chain, ref)
        if v is None:
            print(f"SKIP {p}: {note}")
            continue
        vals.append(v)
        if not args.quiet:
            print(f"{v:8.2f} A  {p}  ({note})")
    if vals:
        print(
            f"n={len(vals)}  median {st.median(vals):.2f} A  "
            f"min {min(vals):.2f}  max {max(vals):.2f}"
        )
    else:
        raise SystemExit("nothing scored")


if __name__ == "__main__":
    main()
