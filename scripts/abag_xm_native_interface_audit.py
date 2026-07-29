#!/usr/bin/env python3
"""Is the interface each fold target declares one our scorers can actually score?

Every label in this dataset is defined against one antibody-antigen interface, named per target by
the manifest's `fold_auth_chain_id_1`/`_2`. Two things had never been checked: that those chains
touch at all, and that the residues carrying the contact are ones DockQ and interface lDDT will
accept. The second is the one that bites, and it is invisible to a plain contact test.

So this reports two counts per target and compares them:

  * structural contacts -- every heavy atom gemmi reads, i.e. what is physically there.
  * scorable contacts   -- only the atoms DockQ's own loader keeps, i.e. what a scorer can see.
    Measured with DockQ's `load_PDB` rather than a guessed list of standard residue names, so it
    is the scorer's real behaviour and not an approximation of it.

A target where the first is positive and the second is zero has an interface that exists and
cannot be scored. Found via the label QC sweep: 9ly2 and 9lz2 failed DockQ (`run_on_chains`
returned None) and interface lDDT for all 50 samples of all three generators, and the cause is the
same in both -- the whole interface is mediated by **SEP (phosphoserine)** and nothing else:

    9ly2  E-R  structural 71 contacts / 2.44 A   scorable 0   carried by SEP 522, SEP 524
    9lz2  L-R  structural 48 contacts / 2.74 A   scorable 0   carried by SEP 1362, SEP 1364

These are anti-phosphoepitope antibodies. The modification is also absent from the fold input --
the YAML declares plain serine -- so no generator can reproduce the chemistry the native antibody
recognises, which makes the target unrankable rather than merely hard.

The failing labels were therefore correct to fail. Loosening the chain matcher would have replaced
a loud failure with a confident meaningless number, which is the worse outcome; the decision this
audit supports is about which targets belong in the dataset.

    python3 scripts/abag_xm_native_interface_audit.py [--contact_a 5.0] [--json out.json]

Exit 1 if any target's declared interface is absent or unscorable, so it can gate a release.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs" / "implementation-parity-data" / "abag-xm-targets.parquet"
NATIVES = Path.home() / "abag_xm" / "ground_truth"


def _gemmi_chains(path):
    """{auth_chain_id: (heavy-atom coords, {(seqid, resname): coords})} -- everything present."""
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    out = {}
    for model in st:
        for ch in model:
            res = {}
            for r in ch.first_conformer():
                xyz = [[a.pos.x, a.pos.y, a.pos.z] for a in r]
                if xyz:
                    res[(r.seqid.num, r.name.strip().upper())] = np.array(xyz, dtype=float)
            if res:
                out[ch.name] = (np.concatenate(list(res.values())), res)
        break
    return out


def _dockq_chains(path):
    """{chain_id: heavy-atom coords} as DockQ's loader sees them -- what a scorer can score."""
    from DockQ.DockQ import load_PDB
    out = {}
    for ch in load_PDB(str(path)):
        xyz = np.array([a.coord for r in ch for a in r], dtype=float)
        if len(xyz):
            out[ch.id] = xyz
    return out


def _contacts(a, b, cutoff):
    """(number of heavy-atom pairs within cutoff, minimum distance)."""
    if not len(a) or not len(b):
        return 0, float("inf")
    ta, tb = cKDTree(a), cKDTree(b)
    n = sum(len(hit) for hit in ta.query_ball_tree(tb, cutoff))
    return n, float(ta.query(b)[0].min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact_a", type=float, default=5.0,
                    help="heavy-atom contact cutoff; 5.0 is DockQ's own fnat definition")
    ap.add_argument("--json", default=None, help="write the full per-target report here")
    a = ap.parse_args()

    df = pd.read_parquet(MANIFEST)
    rows, bad, missing = [], [], []
    for r in df.itertuples():
        pdb = r.pdb_id
        cif = NATIVES / f"{pdb}.cif"
        if not cif.exists():
            missing.append(pdb)
            continue
        c1, c2 = str(r.fold_auth_chain_id_1), str(r.fold_auth_chain_id_2)
        g = _gemmi_chains(cif)
        if c1 not in g or c2 not in g:
            rows.append({"target": pdb, "declared": [c1, c2], "status": "chain_absent",
                         "native_chains": sorted(g)})
            bad.append(pdb)
            continue
        n_str, mind = _contacts(g[c1][0], g[c2][0], a.contact_a)
        d = _dockq_chains(cif)
        n_sco = (_contacts(d[c1], d[c2], a.contact_a)[0]
                 if c1 in d and c2 in d else 0)
        status = ("no_interface" if not n_str else
                  "unscorable_interface" if not n_sco else "ok")
        rec = {"target": pdb, "declared": [c1, c2], "structural_contacts": n_str,
               "scorable_contacts": n_sco, "min_dist_a": round(mind, 2), "status": status}
        if status != "ok":
            # Name the residues that carry the contact. "Unscorable" is only actionable once you
            # can see it is two phosphoserines rather than something recoverable.
            carriers = collections.Counter()
            for side, other in ((c1, c2), (c2, c1)):
                for (num, name), xyz in g[side][1].items():
                    k, _ = _contacts(xyz, g[other][0], a.contact_a)
                    if k:
                        carriers[f"{side}:{name}{num}"] += k
            rec["contact_residues"] = dict(carriers.most_common())
            bad.append(pdb)
        rows.append(rec)

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"targets in manifest      : {len(df)}")
    print(f"natives on disk          : {len(rows)}  (missing {len(missing)})")
    print(f"declared interface OK    : {len(ok)}")
    print(f"absent or unscorable     : {len(bad)}")
    if ok:
        cs = sorted(r["scorable_contacts"] for r in ok)
        print(f"scorable contacts on the good ones: min {cs[0]}, median {cs[len(cs)//2]}, "
              f"max {cs[-1]}")
    for r in rows:
        if r["status"] == "ok":
            continue
        print(f"\n  {r['target']}  declared {r['declared']}  {r['status']}")
        if "structural_contacts" in r:
            print(f"      structural {r['structural_contacts']} contacts at "
                  f"{r['min_dist_a']} A, scorable {r['scorable_contacts']}")
        for res, n in (r.get("contact_residues") or {}).items():
            print(f"      carried by {res}: {n} atom pairs")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.json}")
    if bad:
        print(f"\n{len(bad)} target(s) cannot yield a meaningful interface label: {sorted(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
