"""CA-RMSD of a predicted CIF against a deposited one, matched by residue, not by index.

The instrument a ceiling pass needs when its change is not bit-exact: a fix-vs-base RMSD cannot
tell "different" from "worse", because these models are deterministic at a fixed seed, so the
run-to-run floor is 0 and every non-bit-exact change reads as large. Only a reference outside both
arms ranks them.

Index alignment is the trap. The deposited 9SAT has 627 modelled CA and a prediction can have more
or fewer, so zipping two coordinate arrays superposes residue i onto residue i+k and reports a
phantom RMSD that looks like a real regression. Residues are matched on (chain, seq_id), chains are
mapped positionally because a prediction names them from the input yaml (A/H/L) while the deposit
uses label_asym_id (A/B/C), and the run REFUSES to print a number unless the residue NAMES agree
too -- without that last check any 3-chain prediction matched any 3-chain deposit at 100% key
overlap and reported 39.92 A as if it were a measurement.

Controls, both live: the same file against itself reads 0.0000 A over 627 CA, and a different
target refuses with "residue names agreeing on 68".

KNOWN LIMIT, and it is why this refuses more often than it answers. Matching on seq_id assumes the
two files NUMBER their residues the same way. A deposited mmCIF frequently does not -- 9SAT's
prediction and deposit share 602 of 627 seq_ids and agree on only 162 residue names even after
best-of chain matching, because the deposit numbers by its own entity sequence. Doing that properly
needs a per-chain sequence alignment, not a seq_id lookup, and this file does not have one. It
refuses instead of guessing.

For a crystal-referenced accuracy number on a model tt-bio ships, prefer the committed harness:
`scripts/release_gate.py --model <model>` scores the model's ground-truth anchor with a mapping
that is already vetted and floors the release already trusts. Use this script for
prediction-vs-prediction geometry and for deposits whose numbering you have checked.

    python perf/ceilings/ref_ca_rmsd.py <predicted.cif> <deposited.cif>
"""
import sys

import numpy as np


def ca(path):
    """{(asym, seq_id): xyz} for every CA, plus the chain order of first appearance.

    Header-driven so column order cannot drift.
    """
    cols, out, in_loop, hit, order = {}, {}, False, False, []
    for line in open(path):
        s = line.strip()
        if s.startswith("_atom_site."):
            cols[s.split(".", 1)[1]] = len(cols)
            in_loop = True
            continue
        if in_loop and (s.startswith("ATOM") or s.startswith("HETATM")):
            f = s.split()
            hit = True
            if f[cols["label_atom_id"]] != "CA":
                continue
            sid = f[cols["label_seq_id"]]
            if sid in (".", "?"):
                continue
            # Take the first altloc/model only.
            asym = f[cols["label_asym_id"]]
            if asym not in order:
                order.append(asym)
            out.setdefault((asym, int(sid)),
                           ([float(f[cols[c]]) for c in ("Cartn_x", "Cartn_y", "Cartn_z")],
                            f[cols["label_comp_id"]]))
        elif in_loop and hit:
            break
    return out, order


def _match_chains(a, ao, b, bo):
    """{predicted chain -> deposited chain}, each pair the best residue-name agreement left."""
    score = {}
    for x in ao:
        for y in bo:
            shared = [(a[(x, i)][1], b[(y, i)][1]) for (cx, i) in a if cx == x and (y, i) in b]
            if shared:
                score[(x, y)] = sum(p == q for p, q in shared) / len(shared)
    out, used = {}, set()
    for (x, y), _ in sorted(score.items(), key=lambda kv: -kv[1]):
        if x not in out and y not in used:
            out[x], _ = y, used.add(y)
    return out


def kabsch_rmsd(a, b):
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((ac @ r - bc) ** 2).sum(1).mean()))


def selftest():
    p = np.random.RandomState(0).randn(200, 3) * 10
    rot = np.linalg.qr(np.random.RandomState(1).randn(3, 3))[0]
    assert kabsch_rmsd(p, p @ rot + 7.0) < 1e-8, "Kabsch identity test FAILED"
    q = p.copy()
    q[100:] += 5.0
    assert kabsch_rmsd(p, q) > 1.0, "Kabsch negative control FAILED"


def main(pred, dep):
    selftest()
    (a, ao), (b, bo) = ca(pred), ca(dep)
    # A prediction names its chains from the input yaml's `id:` (A/H/L for an antibody-antigen
    # complex); the deposit names them by label_asym_id (A/B/C). Match them by SEQUENCE, not by
    # position: 9SAT's deposited chain order is not its yaml order, and a positional map there
    # scored 597 of 627 residues as matched with only 45 residue NAMES agreeing -- a phantom
    # waiting to be printed. Each predicted chain takes the deposited chain it agrees with best
    # on shared seq_ids, greedily, which is exact for the handful of chains a complex has.
    if set(ao) != set(bo):
        a = {(ren, k[1]): v for k, v in a.items()
             for ren in (_match_chains(a, ao, b, bo).get(k[0], k[0]),)}
    common = sorted(set(a) & set(b))
    small = min(len(a), len(b))
    # The keys alone are not enough. Renaming chains positionally makes ANY 3-chain prediction
    # match a 3-chain deposit on (chain, seq_id), so a wrong target passed a key-overlap guard at
    # 100% and reported 39.92 A as if it were a measurement. The residue NAMES have to agree too.
    same = sum(1 for k in common if a[k][1] == b[k][1])
    if not small or len(common) < 0.9 * small or same < 0.95 * max(1, len(common)):
        print(f"REFUSED overlap {len(common)} of pred {len(a)} / dep {len(b)}, residue names "
              f"agreeing on {same}: these two files are not the same target, so any RMSD would "
              f"be a phantom. pred chains {ao}, dep chains {bo}")
        return 2
    x = np.array([a[k][0] for k in common])
    y = np.array([b[k][0] for k in common])
    print(f"n_ca={len(common)} (pred {len(a)}, dep {len(b)}, {same} residue names agree)  "
          f"ca_rmsd={kabsch_rmsd(x, y):.4f} A")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
