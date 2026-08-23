"""Designed binders to mmCIF, in the frame of the target structure they were designed against."""
from __future__ import annotations

from pathlib import Path

import torch

# mmCIF is the output format because a designed binder carries no sequence, and PDB's fixed
# columns have nowhere to say so without inventing a residue name per position.
_CIF_COLS = ("group_PDB", "id", "type_symbol", "label_atom_id", "label_comp_id",
             "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z")

#: `restype` row of the `xpb` binder placeholder. Every token carrying it is designed.
BINDER_RESTYPE = 32


def _kabsch(a: torch.Tensor, b: torch.Tensor):
    """The rigid map taking `a` onto `b`: x -> (x - a.mean) @ R + b.mean.

    Host fp64 with an exact SVD. Alignment is a scoring and output step, not a device op, and a
    device-side approximation here would move every coordinate we write (`ttnn-host-kabsch`).
    """
    ca, cb = a.double().mean(0), b.double().mean(0)
    u, _, vt = torch.linalg.svd((a.double() - ca).T @ (b.double() - cb))
    d = torch.sign(torch.det(u @ vt))
    r = u @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64)) @ vt
    return r, ca, cb


def _atom_names(feats: dict, sel: torch.Tensor) -> list[str]:
    """`ref_atom_name_chars` back to PDB atom names, for the selected atoms."""
    chars = feats["ref_atom_name_chars"]
    return ["".join(chr(int(chars[i, j].argmax()) + 32) for j in range(4)).strip()
            for i in torch.nonzero(sel).flatten().tolist()]


def write_design_cifs(coords: torch.Tensor, feats: dict, outdir: Path | str,
                      stem: str = "design") -> list[dict]:
    """One mmCIF per sample: the designed binder, placed against the input target.

    `coords` is `(n_sample, N_atom, 3)` straight out of `ProtenixDesign.design`. Only the binder
    is written. The diffusion output also contains the model's reconstruction of the target, but
    the user already has the target full-atom and exact in the file they passed in, and a
    reconstruction is strictly worse than that -- so the binder is moved into the input
    structure's own frame instead, and the two files open together.

    The frame is recovered by fitting the generated distogram-representative atoms of the
    CONDITIONED tokens onto the coordinates the featurizer conditioned on. `fit_rmsd` is that
    fit's residual and it is the end-to-end correctness signal worth reading: PXDesign sees only
    a 64-bin distogram of the target, so a correct run reproduces the target's own fold while the
    binder is free, and a broken conditioning path lands in the tens of angstroms.

    The binder is written as GLY because that is what PXDesign generates: a backbone with no
    sequence and exactly N/CA/C/O per residue, which is precisely GLY's atom set. Anything else
    would be inventing side chains that do not exist.
    """
    cond = feats.get("condition")
    if cond is None:
        raise ValueError("feats has no `condition`; build it with "
                         "tt_bio.pxdesign.inputs.design_inputs_from_yaml")
    conditioned = (torch.tensor([n != "xpb" for n in cond["res_name"]])
                   & cond["is_resolved"].bool())
    if not bool(conditioned.any()):
        raise ValueError("no conditioned target tokens; nothing to align the binder against")
    disto = feats["distogram_rep_atom_mask"].bool()

    a2t = feats["atom_to_token_idx"].long()
    at_binder = (feats["restype"].argmax(-1) == BINDER_RESTYPE)[a2t]
    if not bool(at_binder.any()):
        raise ValueError("no binder atoms in the feature dict")
    names = _atom_names(feats, at_binder)
    btok = a2t[at_binder]
    btok = btok - int(btok.min())               # binder residues renumbered from 1 on write

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ref = cond["coord"][conditioned].double()
    out = []
    for s in range(coords.shape[0]):
        rep = coords[s][disto].double()
        r, ca, cb = _kabsch(rep[conditioned], ref)
        rmsd = float(((rep[conditioned] - ca) @ r + cb - ref).pow(2).sum(-1).mean().sqrt())
        binder = (coords[s][at_binder].double() - ca) @ r + cb
        path = outdir / (f"{stem}.cif" if coords.shape[0] == 1 else f"{stem}_{s}.cif")
        with path.open("w") as fh:
            fh.write(f"data_{stem}_{s}\nloop_\n")
            for c in _CIF_COLS:
                fh.write(f"_atom_site.{c}\n")
            for i, (name, xyz, tok) in enumerate(zip(names, binder.tolist(), btok.tolist()), 1):
                fh.write("ATOM %d %s %s GLY A %d %.3f %.3f %.3f\n"
                         % (i, name[0], name, tok + 1, *xyz))
        out.append({"sample": s, "cif": str(path), "fit_rmsd": rmsd,
                    "binder_atoms": len(names), "binder_residues": int(btok.max()) + 1,
                    "conditioned_tokens": int(conditioned.sum())})
    return out
