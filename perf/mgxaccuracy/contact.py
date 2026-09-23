#!/usr/bin/env python3
"""Is the delivered binder in contact with the target it was designed against?

`fit_rmsd` reads 95 A at a 1536-residue pxdesign target against 0.08 A at 512, and that alone
does not say what a user gets. Two different things produce a large fit residual:

  * the model's reconstruction of the target is wrong, so the conditioning failed and the
    binder was never designed against this surface; or
  * the fit that RECOVERS the frame is misaligned, so the design may be fine and the number
    lies.

They have the same user-visible consequence and this script measures that consequence rather
than guessing which layer it came from: `tt_bio/pxdesign/write.py` places the binder into the
input structure's frame **using that fit**, so a bad fit ships a binder sitting in space. A
binder that binds is within a few angstrom of the target; one that does not is tens away.

    python3 perf/mgxaccuracy/contact.py --target perf/bhdesign/targets/big_1831.cif \\
        --crop 1-512 --binder DESIGN.cif [DESIGN.cif ...]
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def atoms(path: pathlib.Path):
    """(chain, seq, atom_name, xyz) for every ATOM row, read from the mmCIF loop header.

    Column positions are taken from the file's own `_atom_site.` order rather than assumed:
    the fixtures here are gemmi-written and the designs are written by tt-bio's own writer,
    and the two do not agree on column order."""
    import numpy as np
    cols, rows = [], []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".", 1)[1].split()[0])
        elif s.startswith(("ATOM", "HETATM")):
            f = s.split()
            if len(f) == len(cols):
                rows.append(f)
    if not rows:
        raise SystemExit(f"{path}: no ATOM rows parsed against {len(cols)} columns")
    ix = {c: i for i, c in enumerate(cols)}
    ch = ix.get("label_asym_id", ix.get("auth_asym_id"))
    sq = ix.get("label_seq_id", ix.get("auth_seq_id"))
    xi = ix["Cartn_x"]
    xyz = np.array([[float(r[xi]), float(r[xi + 1]), float(r[xi + 2])] for r in rows])
    meta = [(r[ch], r[sq], r[ix["label_atom_id"]]) for r in rows]
    return meta, xyz


def crop_mask(meta, spec: str):
    """Keep residues named by `spec`, a comma list of `CHAIN:LO-HI` or a bare `LO-HI`.

    A bare range means "the first chain in the file", which is what the pxdesign fixture's
    `crop: ["1-512"]` means for a single-chain crop."""
    keep = []
    wants = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        chain = None
        if ":" in part:
            chain, part = part.split(":", 1)
        lo, _, hi = part.partition("-")
        wants.append((chain, int(lo), int(hi or lo)))
    first = meta[0][0]
    for c, s, _ in meta:
        try:
            n = int(s)
        except ValueError:
            keep.append(False)
            continue
        keep.append(any((ch == c if ch else c == first) and lo <= n <= hi
                        for ch, lo, hi in wants))
    return keep


# pxdesign sees the target only as a 64-bin distogram over 2-22 A
# (`tt_bio/pxdesign/featurize.py:40`), so 22 A is the distance beyond which two conditioned
# tokens are indistinguishable to it.
TEMPL_TOP_A = 22.0


def conditioning_graph(meta, xyz, top_a: float = TEMPL_TOP_A) -> dict:
    """Components of the graph on conditioned tokens whose pair distance the model can resolve.

    The saturated FRACTION does not predict a conditioning failure -- the 512 crops that give
    fit_rmsd 0.0755 A are already 73-76 % saturated. What predicts it is CONNECTIVITY: a target
    whose sub-`top_a` graph has two components carries no information at all about where one
    component sits relative to the other, so the rigid fit that recovers the output frame has no
    determined answer and lands at the scale of the separation.

    One CA per residue stands in for the distogram representative atom; the two differ by a
    couple of angstrom and the components do not.
    """
    import numpy as np
    ca = [(c, q) for (c, _s, a), q in zip(meta, xyz) if a == "CA"]
    if not ca:
        raise SystemExit("conditioning_graph: no CA atoms")
    ch = np.array([c for c, _ in ca])
    P = np.array([q for _, q in ca])
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    adj = (d <= top_a) & ~np.eye(len(P), dtype=bool)
    seen = np.zeros(len(P), bool)
    sizes = []
    for start in range(len(P)):
        if seen[start]:
            continue
        stack, cnt, seen[start] = [start], 0, True
        while stack:
            u = stack.pop()
            cnt += 1
            for v in np.flatnonzero(adj[u] & ~seen):
                seen[v] = True
                stack.append(v)
        sizes.append(cnt)
    inter = ch[:, None] != ch[None, :]
    return {"n_token": len(P), "components": sorted(sizes, reverse=True),
            "inter_chain_edges": int((adj & inter).sum() // 2),
            "saturated_frac": float(((d > top_a) & ~np.eye(len(P), dtype=bool)).sum()
                                    / max(1, (~np.eye(len(P), dtype=bool)).sum()))}


def main() -> int:
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--crop", default=None,
                    help="residue ranges of the target that were conditioned on; default all")
    ap.add_argument("--binder", nargs="+", required=True)
    ap.add_argument("--contact-a", type=float, default=5.0,
                    help="an atom pair this close or closer is a contact")
    args = ap.parse_args()

    tmeta, txyz = atoms(pathlib.Path(args.target))
    if args.crop:
        m = np.array(crop_mask(tmeta, args.crop))
        txyz = txyz[m]
        if not len(txyz):
            raise SystemExit(f"crop {args.crop!r} selected no target atoms")
    tcen = txyz.mean(0)
    print(f"target {pathlib.Path(args.target).name} crop={args.crop or 'all'}: "
          f"{len(txyz)} atoms, centroid {tcen.round(1).tolist()}, "
          f"radius {np.linalg.norm(txyz - tcen, axis=1).max():.1f} A")
    print(f"\n{'binder':<18}{'atoms':>6}{'min d':>9}{'median d':>10}"
          f"{'contacts':>10}   verdict")
    for b in args.binder:
        bmeta, bxyz = atoms(pathlib.Path(b))
        # Full pairwise distance: an 80-residue binder against a 12k-atom target is 4M pairs.
        d = np.linalg.norm(bxyz[:, None, :] - txyz[None, :, :], axis=-1)
        dmin = d.min(1)
        n_contact = int((d <= args.contact_a).sum())
        verdict = ("IN CONTACT" if dmin.min() <= args.contact_a else
                   "NOT DOCKED" if dmin.min() > 10 else "marginal")
        print(f"{pathlib.Path(b).name:<18}{len(bxyz):>6}{dmin.min():>9.2f}"
              f"{float(np.median(dmin)):>10.2f}{n_contact:>10}   {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
