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


def split_complex(path: pathlib.Path, binder_res: int = None, target_chains=None):
    """(binder xyz, target xyz, binder chain, residues per chain, target chain labels).

    BoltzGen writes the design and the target it was designed against into a single file, so
    the docking question cannot be asked with a separate --binder.

    Pass `binder_res` -- the binder's residue count, which the refold knows because it holds
    the binder alone -- and the chain with exactly that many residues is taken, the same rule
    `scrmsd.py:designed_chain` uses for every quality number in this row. Without it the chain
    with the FEWEST residues is taken, and THAT RULE IS WRONG WHENEVER A TARGET FRAGMENT IS
    SHORTER THAN THE BINDER: on GroEL 512/off560 (A 80 binder, B 488, C 24) it silently picked
    the 24-residue target fragment and measured the target against itself. Both rules refuse a
    tie rather than guess."""
    import numpy as np
    meta, xyz = atoms(path)
    per: dict = {}
    for c, s_, a in meta:
        if a == "CA":
            per[c] = per.get(c, 0) + 1
    if len(per) < 2:
        raise SystemExit(f"{path.name}: one chain — nothing to measure a contact against")
    if binder_res is not None:
        hits = [c for c, n in per.items() if n == binder_res]
        if len(hits) != 1:
            raise SystemExit(f"{path.name}: {len(hits)} chains with {binder_res} residues "
                             f"{per} — cannot identify the binder")
        b = hits[0]
    else:
        lo = min(per.values())
        small = [c for c, n in per.items() if n == lo]
        if len(small) != 1:
            raise SystemExit(f"{path.name}: {len(small)} chains tie at {lo} residues {per} — "
                             "cannot identify the binder")
        b = small[0]
    m = np.array([c == b for c, _, _ in meta])
    tmeta = [r for r, keep in zip(meta, m) if not keep]
    tch = np.array([c for c, _, _ in tmeta])
    if target_chains:
        # BoltzGen's design writer MERGES adjacent target chains: the GroEL 1536/off0 fixture
        # is 524 + 524 + 488 and the design is written as 1048 + 488. Counting chains off the
        # output therefore undercounts, so the fixture's own sizes are imposed here, in file
        # order, and the total is checked rather than trusted.
        bounds, acc = [], 0
        for n in target_chains:
            acc += n
            bounds.append(acc)
        lab, seen, idx = [], None, 0
        for c, sq, _a in tmeta:
            if (c, sq) != seen:
                seen = (c, sq)
                idx += 1
            lab.append(f"c{next(j for j, bd in enumerate(bounds) if idx <= bd) + 1}")
        if idx != bounds[-1]:
            raise SystemExit(f"{path.name}: --target-chains sums to {bounds[-1]} but the "
                             f"target has {idx} residues")
        tch = np.array(lab)
        per = {**{c: n for c, n in zip((f"c{i+1}" for i in range(len(target_chains))),
                                       target_chains)}, b: per[b]}
    return np.asarray(xyz)[m], np.asarray(xyz)[~m], b, per, tch


def per_chain_report(paths, contact_a: float, binder_res: int = None,
                     target_chains=None) -> int:
    """How many TARGET CHAINS does the delivered binder actually touch?

    `results/chain_census_all_cells.txt` sorts every cell in this row by chain count and finds
    every multi-chain crop at 0 % under the 4 A bar. That headline has a dial in it: GroEL
    512/off560 is B 488 + C 24, and it is the row's BEST cell at 87.5 % only because a
    24-residue fragment is called not-a-chain by a 100-residue threshold that is not derived
    from anything.

    A chain the binder never comes near is not part of the design problem, whatever its length.
    So this counts the chains the binder contacts instead of the chains present in the file --
    a property of the delivered structure rather than of a threshold chosen by hand. It can
    equally refute the headline: if a failing multi-chain cell also has its binders on one
    chain, interface chain count is not the carrier either, and the census keeps its caveat.
    """
    import numpy as np
    print(f"\ncontact = any atom pair within {contact_a} A. n_touched counts TARGET chains "
          f"with >= 1 contact.\n")
    touched_hist: dict = {}
    for f in paths:
        f = pathlib.Path(f)
        bxyz, txyz, b, per, tch = split_complex(f, binder_res, target_chains)
        d = np.linalg.norm(bxyz[:, None, :] - txyz[None, :, :], axis=-1)
        near = d <= contact_a
        chains = sorted(set(tch.tolist()))
        counts = {c: int(near[:, tch == c].sum()) for c in chains}
        dmins = {c: float(d[:, tch == c].min()) for c in chains}
        n_touched = sum(1 for c in chains if counts[c] > 0)
        touched_hist[n_touched] = touched_hist.get(n_touched, 0) + 1
        cells = "  ".join(f"{c}({per[c]}):{counts[c]}@{dmins[c]:.1f}" for c in chains)
        print(f"{f.name:<20} binder {b}({per[b]:>3})  n_touched={n_touched}   {cells}")
    total = sum(touched_hist.values())
    summary = ", ".join(f"{n} chain{'s' if n != 1 else ''}: {k}/{total}"
                        for n, k in sorted(touched_hist.items()))
    print(f"\nINTERFACE CHAIN COUNT over {total} design(s) -- {summary}")
    return 0


def main() -> int:
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complex", nargs="+", default=None, metavar="CIF",
                    help="design files that already contain the target; the binder is the "
                         "chain with the fewest residues")
    ap.add_argument("--target", required=False)
    ap.add_argument("--crop", default=None,
                    help="residue ranges of the target that were conditioned on; default all")
    ap.add_argument("--binder", nargs="+", default=None)
    ap.add_argument("--contact-a", type=float, default=5.0,
                    help="an atom pair this close or closer is a contact")
    ap.add_argument("--binder-res", type=int, default=None,
                    help="binder residue count; picks that chain instead of the smallest one, "
                         "which is wrong when a target fragment is shorter than the binder")
    ap.add_argument("--target-chains", default=None,
                    help="comma list of the FIXTURE's target chain sizes, imposed in file "
                         "order; the design writer merges adjacent chains and undercounts")
    ap.add_argument("--per-chain", action="store_true",
                    help="with --complex: break the contacts down per target chain and count "
                         "how many chains the binder actually touches")
    args = ap.parse_args()

    if args.complex and args.per_chain:
        return per_chain_report(args.complex, args.contact_a, args.binder_res,
                                [int(x) for x in args.target_chains.split(",")]
                                if args.target_chains else None)

    if args.complex:
        print(f"\n{'design':<18}{'binder':>7}{'chain':>7}{'min d':>9}{'median d':>10}"
              f"{'contacts':>10}   verdict")
        for f in args.complex:
            f = pathlib.Path(f)
            bxyz, txyz, ch, per, _tch = split_complex(f, args.binder_res)
            d = np.linalg.norm(bxyz[:, None, :] - txyz[None, :, :], axis=-1)
            dmin = d.min(1)
            verdict = ("IN CONTACT" if dmin.min() <= args.contact_a else
                       "NOT DOCKED" if dmin.min() > 10 else "marginal")
            print(f"{f.name:<18}{len(bxyz):>7}{ch:>7}{dmin.min():>9.2f}"
                  f"{float(np.median(dmin)):>10.2f}{int((d <= args.contact_a).sum()):>10}"
                  f"   {verdict}")
        return 0

    if not args.target or not args.binder:
        raise SystemExit("need --complex, or both --target and --binder")
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
