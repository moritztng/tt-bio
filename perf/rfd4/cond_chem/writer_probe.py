"""Does the CIF writer survive, and preserve chemistry for, a non-protein input? No card needed.

    python perf/rfd4/cond_chem/writer_probe.py <annotated.cif> --out <dir> [--seed 0]

`write_design` copies the generated coordinates onto the *input* atom array, so a ligand's
elements, charges and bonds reach the output file from the input rather than from the model.
That is the mechanism the chemistry grade depends on, and it can be checked without designing
anything: hand the writer a stand-in design with the right shapes and see what comes out.

The stand-in is **not a design and its coordinates mean nothing**. Every designed token is
written as ALA on a straight backbone; every atomized token keeps the input's own position.
What is being graded is the writer's contract -- the assertions in
`process_single_sample_to_atom_array` about atomized tokens, and whether the ligand or
nucleotide chemistry survives the round trip -- not the geometry.

Run `chem_check.py --input <annotated.cif> --generated <out>/<id>_written.cif` afterwards.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

#: `<A>`, the atomized-token class; the writer wants -1 in its place and the atom in slot 1.
ATOMIZED_CLASS = 32
#: ALA, and the atom37 slots it occupies (N, CA, C, CB, O).
ALA_CLASS, ALA_SLOTS = 1, (0, 1, 2, 3, 4)


def stand_in_design(features):
    """A `DesignOutput` with the right shapes: ALA everywhere designed, input atoms kept."""
    import torch

    from tt_bio.rfd4.engine import DesignOutput

    padding = features["padding_mask"].to(torch.bool)
    n_designs, n_tokens = padding.shape
    input_mask = features["atom37_mask"].to(torch.bool)
    input_nm = features["coords"] / 10.0
    rt_in = features["residue_type"]

    # A token the input already has atoms for is held (the ligand, the duplex, the glycan);
    # the model returns those as they came in. Only the tokens with nothing in them are
    # designed, and those are the ones this stand-in makes up.
    held = input_mask.any(-1) & padding
    atomized = (rt_in == ATOMIZED_CLASS) & padding

    rt = torch.where(held, rt_in, torch.full_like(rt_in, ALA_CLASS))
    rt = torch.where(atomized | ~padding, torch.full_like(rt_in, -1), rt)

    mask = input_mask.clone()
    for slot in ALA_SLOTS:
        mask[..., slot] |= padding & ~held
    mask &= padding[..., None]

    # Designed tokens get a straight 3.8 A CA trace so nothing is NaN; held tokens keep the
    # input's own coordinates, which is what the writer is being graded on.
    coords = torch.zeros(n_designs, n_tokens, 37, 3)
    step = torch.arange(n_tokens, dtype=torch.float32)[None, :, None] * 0.38  # nanometres
    for slot, offset in zip(ALA_SLOTS, (0.1, 0.0, 0.2, 0.3, 0.4)):
        coords[..., slot, :] = step + offset
    coords = torch.where(held[..., None, None], input_nm, coords)

    logits = torch.zeros(n_designs, n_tokens, n_tokens)
    return DesignOutput(coords_nm=coords, residue_type=rt, atom_mask=mask, insertion_logits=logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from tt_bio.rfd4 import featurize, write_design

    features, atom_array = featurize(a.cif, seed=a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(a.cif).stem}_written.cif"
    if path.exists():
        path.unlink()
    written = write_design(stand_in_design(features), features, atom_array, path)
    print(f"tokens={int(features['padding_mask'].shape[1])} wrote {written}")


if __name__ == "__main__":
    main()
