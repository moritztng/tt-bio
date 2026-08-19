"""Write each ladder rung as its own single-chain CIF, so the crop is the target file.

    python perf/pxdesign/make_cropped_cif.py --cif 1dp0.cif --chain A --hotspots 526,527,528 \
        --sizes 128,256,512,768 --binder-length 80 --out-dir /work/targets2

PXDesign applies `target.chains.<id>.crop` to the STRUCTURE the generator conditions on, but the
sequence it hands to Protenix comes from the entity, uncropped. Measured on this ladder: the
generator produced a 128-residue target chain while the Protenix target-template probe was handed
the full 1023-residue chain, so a per-crop MSA no longer matched and Protenix skipped the sample.
`pxdesign prepare-msa` has the same blind spot -- it searches the full chain too.

The way round it is to stop using `crop`: write the cropped residues out as their own structure and
point the yaml at that. Then the entity sequence, the MSA and the conditioning structure are the
same 128/256/512/768 residues, and the rungs differ in target length and nothing else.
"""

import argparse
import pathlib

import numpy as np
from biotite.structure import AtomArray  # noqa: F401
from biotite.structure.io.pdbx import CIFFile, get_structure, set_structure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True)
    ap.add_argument("--chain", default="A")
    ap.add_argument("--hotspots", required=True)
    ap.add_argument("--sizes", default="128,256,512,768")
    ap.add_argument("--binder-length", type=int, default=80)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="laczc")
    a = ap.parse_args()

    hotspots = sorted(int(x) for x in a.hotspots.split(","))
    centre = hotspots[len(hotspots) // 2]
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    arr = get_structure(CIFFile.read(a.cif), model=1)
    arr = arr[(arr.chain_id == a.chain) & arr.hetero.__invert__()]
    resids = np.unique(arr.res_id)
    print("chain %s: %d residues, %d..%d" % (a.chain, len(resids), resids.min(), resids.max()))

    for n in [int(x) for x in a.sizes.split(",")]:
        start = max(int(resids.min()), centre - n // 2)
        end = start + n - 1
        if end > int(resids.max()):
            end, start = int(resids.max()), int(resids.max()) - n + 1
        assert all(start <= h <= end for h in hotspots), (n, start, end, hotspots)

        sub = arr[(arr.res_id >= start) & (arr.res_id <= end)].copy()
        # renumber to 1..N so hotspots are crop-local and label_seq_id stays sequential
        sub.res_id = sub.res_id - start + 1
        sub.chain_id = np.full(sub.array_length(), "A")

        label = "%s_%d" % (a.name, n)
        cif_path = out / ("%s.cif" % label)
        f = CIFFile()
        set_structure(f, sub, data_block=label)
        f.write(cif_path)

        local = [h - start + 1 for h in hotspots]
        (out / ("%s.yaml" % label)).write_text(
            "target:\n"
            "  file: \"%s\"\n"
            "  chains:\n"
            "    A:\n"
            "      hotspots: %s\n"
            "binder_length: %d\n" % (cif_path.resolve(), local, a.binder_length))
        n_res = len(np.unique(sub.res_id))
        print("%-12s residues %d..%d -> 1..%d (%d present) hotspots %s"
              % (label, start, end, end - start + 1, n_res, local))


if __name__ == "__main__":
    main()
