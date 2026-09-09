"""Cut one chain of a PDB entry to its own single-chain CIF, renumbered 1..N.

    python perf/ceilrfd3/make_target.py --cif 1gpb.cif --chain A --name gpb --out-dir perf/ceilrfd3/targets

`A1-N` in an RFD3 contig then selects the first N residues for any N on a ladder, which is the
only property the ceiling ladder needs of its target and the same recipe `laczc_1008.cif` was cut
with. Renumbering is by SEQUENTIAL INDEX, and the script refuses a chain with a numbering gap or a
CA-CA break: a source discontinuity renumbered into 1..N would read as a backbone break in the
target and get scored as one.
"""
import argparse
import pathlib

import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_structure, set_structure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True)
    ap.add_argument("--chain", default="A")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    arr = get_structure(CIFFile.read(a.cif), model=1)
    sub = arr[(arr.chain_id == a.chain) & ~arr.hetero].copy()
    sub = sub[sub.altloc_id == "."] if "altloc_id" in sub.get_annotation_categories() else sub
    resids = np.unique(sub.res_id)
    assert (np.diff(resids) == 1).all(), "numbering gap in %s chain %s" % (a.cif, a.chain)

    ca = sub[sub.atom_name == "CA"]
    d = np.linalg.norm(np.diff(ca.coord, axis=0), axis=1)
    assert ca.array_length() == len(resids), "missing CA"
    assert d.max() < 4.5, "CA-CA break %.2f A in the source chain" % d.max()

    sub.res_id = sub.res_id - int(resids.min()) + 1
    sub.chain_id = np.full(sub.array_length(), "A")
    label = "%s_%d" % (a.name, len(resids))
    out = pathlib.Path(a.out_dir) / ("%s.cif" % label)
    f = CIFFile()
    set_structure(f, sub, data_block=label)
    f.write(out)
    print("%s: %d residues, %d atoms, worst source CA-CA %.2f A -> %s"
          % (label, len(resids), sub.array_length(), d.max(), out))


if __name__ == "__main__":
    main()
