#!/usr/bin/env python3
"""How many Angstrom the device atom-featurization legs move an OpenFold3 structure.

Same four quantities `perf/of3t_foldab/fold_ab.py` established for the trunk bias flag, because
a bare Angstrom says nothing on its own:

  lever       OFF against ON at a MATCHED seed on a MATCHED card -- what the flag does;
  seed floor  the SAME arm at a different seed. Below the floor a lever is noise, above it a
              finding. ubiquitin is 76 aa and gets its own floor; the 512 aa cell's 1.84 A is
              not this target's;
  A/A         the same arm, same seed, re-run in a separate process. The instrument's zero,
              measured (A16), not assumed;
  accuracy    each arm against the 1UBQ experimental structure, so a move toward or away from
              the deposited coordinates can be told apart from a move.

Plus a BREAK control: the same pair with one side's residue order reversed, which must be large
or the metric is not discriminating.

  python3 perf/of3t_hostleg/fold_move.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import gemmi

ROOT = Path("/tmp/of3t/of3t-hostleg/fold")
GT = "examples/ground_truth_structures/ubiquitin.pdb"


def ca(path):
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    return {(c.name, r.seqid.num): r.find_atom("CA", "*").pos
            for c in st[0] for r in c if r.find_atom("CA", "*") is not None}


def rmsd(a, b, reverse=False):
    keys = sorted(set(a) & set(b))
    pa = [a[k] for k in keys]
    pb = [b[k] for k in (keys[::-1] if reverse else keys)]
    return gemmi.superpose_positions(pa, pb).rmsd, len(keys)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--gt", default=GT)
    ap.add_argument("--out", default=str(Path(__file__).with_name("FOLD_MOVE.json")))
    a = ap.parse_args()

    runs = {}
    for d in sorted(a.root.glob("*")):
        cif = d / "openfold3_results_ubq/structures/ubq.cif"
        if cif.is_file():
            runs[d.name] = cif
    if not runs:
        raise SystemExit(f"no folds under {a.root}")

    digests = {k: sha(v) for k, v in runs.items()}
    coords = {k: ca(v) for k, v in runs.items()}
    out = {
        "instrument": "of3t-hostleg fold_move.py -- the Angstrom move of the device "
                      "atom-featurization legs on an OpenFold3 fold, against the seed floor",
        "host": "qb1 (tt-quietbox) card 1, Blackhole p150a",
        "target": "examples/ubq.yaml, 76 aa, 602 atoms, --single_sequence, 1 sample, 20 steps",
        "runs": {k: {"path": str(v), "sha256": digests[k], "n_ca": len(coords[k])}
                 for k, v in runs.items()},
        "distinct_digests": sorted(set(digests.values())),
        "pairs": {},
    }

    def pair(label, x, y, **extra):
        if x not in coords or y not in coords:
            return
        r, n = rmsd(coords[x], coords[y])
        out["pairs"][label] = dict(
            a=x, b=y, ca_rmsd_A=r, n_ca=n,
            bit_identical=digests[x] == digests[y], **extra)

    # A/A: the instrument's zero, on the arm that must not have moved at all
    pair("AA_off_a_vs_off_b", "off_s0_a", "off_s0_b", what="A/A floor, flag OFF")
    pair("AA_off_a_vs_off_c", "off_s0_a", "off_s0_c", what="A/A floor, flag OFF")
    pair("AA_off_a_vs_off_r1", "off_s0_a", "off_s0_r1", what="A/A floor, flag OFF")
    pair("AA_on_a_vs_on_b", "on_s0_a", "on_s0_b", what="A/A floor, flag ON")
    pair("AA_on_a_vs_on_c", "on_s0_a", "on_s0_c", what="A/A floor, flag ON")
    # the lever, matched seed and card
    for r in ("a", "b", "c", "d"):
        pair(f"LEVER_off_vs_on_s0_{r}", f"off_s0_{r}", f"on_s0_{r}",
             what="the flag, matched seed 0 and card 1")
    # the seed floor on the shipped arm
    pair("SEEDFLOOR_off_s0_vs_s1", "off_s0_a", "off_s1_a",
         what="same arm, seed 0 vs seed 1")
    # zero and break controls
    if "off_s0_a" in coords:
        r0, n0 = rmsd(coords["off_s0_a"], coords["off_s0_a"])
        rb, nb = rmsd(coords["off_s0_a"], coords["off_s0_a"], reverse=True)
        out["controls"] = {
            "ZERO_self_vs_self_A": r0,
            "BREAK_residue_order_reversed_A": rb,
            "n_ca": n0,
        }
    # accuracy against the deposited structure
    if a.gt != "none" and Path(a.gt).is_file():
        gt = ca(a.gt)
        out["vs_experimental"] = {}
        for k in sorted(coords):
            r, n = rmsd(coords[k], gt)
            out["vs_experimental"][k] = {"ca_rmsd_A": r, "n_ca": n}
        out["vs_experimental_source"] = a.gt
    else:
        out["vs_experimental"] = f"absent: {a.gt}"

    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
