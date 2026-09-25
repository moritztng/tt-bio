#!/usr/bin/env python3
"""What a design fixture actually IS, so two runs that name the same file can be compared.

This exists because of a real confound, found 2026-09-24. The upstream CPU reference and the
device runs both had a design spec saying `path: bgt1536.cif`, so the two jobs read as the
same job. The files differed: the reference's carried chains A=1008/B=528 with 162.40 A
between them -- the disconnected rung this row had disqualified -- while the device's carried
A=823/B=713 at 1.88 A, a contacting dimer. A whole "upstream does not degrade with size"
headline rested on setting those two side by side, and had to be withdrawn.

A spec names its input; it does not pin it. So pin it here, and note that the plain byte hash
of the mmCIF is NOT enough on its own: gemmi derives the data block name from the filename, so
two identical crops written to different filenames differ in bytes while being the same
structure. `coord_sha` below is over content only and is the one to compare.

    python3 perf/mgxaccuracy/fixture_id.py FIXTURE.cif [FIXTURE2.cif ...]
    python3 perf/mgxaccuracy/fixture_id.py --json a.cif b.cif      # machine-readable
    python3 perf/mgxaccuracy/fixture_id.py --compare a.cif b.cif   # exit 1 if they differ

`--compare` is the one to put in front of a cross-host reference comparison: it answers "are
these two runs about the same target" before any number is read off them.
"""
import argparse
import hashlib
import json
import sys

import gemmi
import numpy as np


def identify(path: str) -> dict:
    st = gemmi.read_structure(path)
    st.setup_entities()
    model = st[0]
    rows, chains = [], {}
    for ch in model:
        chains[ch.name] = len(ch)
        for res in ch:
            for atom in res:
                rows.append(
                    f"{ch.name}|{res.label_seq}|{res.name}|{atom.name}"
                    f"|{atom.pos.x:.3f}|{atom.pos.y:.3f}|{atom.pos.z:.3f}"
                )
    out = {
        "path": path,
        "chains": chains,
        "residues": sum(chains.values()),
        "atoms": len(rows),
        "coord_sha": hashlib.sha256("\n".join(rows).encode()).hexdigest()[:20],
    }
    # The separation is what decides whether a multi-chain target poses a joint problem at all.
    # A binder designed against two chains 162 A apart is really designed against one of them.
    names = list(chains)
    if len(names) >= 2:
        def xyz(ch):
            return np.array([[a.pos.x, a.pos.y, a.pos.z] for r in ch for a in r])

        a, b = xyz(model[names[0]]), xyz(model[names[1]])
        best = min(
            float(np.linalg.norm(a[i:i + 2000, None, :] - b[None, :, :], axis=-1).min())
            for i in range(0, len(a), 2000)
        )
        out["min_interchain_a"] = round(best, 2)
        out["centroid_sep_a"] = round(float(np.linalg.norm(a.mean(0) - b.mean(0))), 2)
        out["contacting"] = best < 5.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fixtures", nargs="+")
    ap.add_argument("--json", action="store_true", help="one JSON object per fixture")
    ap.add_argument("--compare", action="store_true",
                    help="exit 1 unless every fixture has the same coord_sha")
    args = ap.parse_args()

    ids = [identify(f) for f in args.fixtures]
    for d in ids:
        if args.json:
            print(json.dumps(d))
        else:
            sep = ""
            if "min_interchain_a" in d:
                sep = (f"  min_interchain {d['min_interchain_a']} A"
                       f"  centroid_sep {d['centroid_sep_a']} A"
                       f"  {'CONTACTING' if d['contacting'] else 'NOT CONTACTING'}")
            print(f"{d['path']}\n  chains {d['chains']}  residues {d['residues']}  "
                  f"atoms {d['atoms']}\n  coord_sha {d['coord_sha']}{sep}")

    if args.compare:
        shas = {d["coord_sha"] for d in ids}
        if len(shas) > 1:
            print("\nDIFFERENT TARGETS -- do not compare numbers measured on these.",
                  file=sys.stderr)
            return 1
        print("\nsame target (coord_sha matches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
