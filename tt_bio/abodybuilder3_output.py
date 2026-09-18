"""Turn the device model's atom14 output into what the evaluation instrument reads, and nothing more.

`tt_bio/antibody_rmsd.py` is `train-b1-instrument`'s, validated by recomputing ABodyBuilder3's own
released per-structure numbers to 0.00843 A over 1,236 evaluations. This module's only job is to
hand it a PDB and call it. **It deliberately contains no RMSD arithmetic, no superposition and no
region logic**, because a second scorer is how a reproduction ends up measuring a different quantity
than the paper it is reproducing.

That failure mode is specific and it is already documented: the published per-region "backbone
RMSD" is over **N, CA, C, CB**, not the conventional N, CA, C, O. Our atom14 layout's first four
slots are N, CA, C, O -- `restype_name_to_atom14_names["ALA"]` is
`['N', 'CA', 'C', 'O', 'CB', ...]` -- so slicing `[..., :4]` off our own output and scoring it would
take the carbonyl and drop CB, which moves CDR-H3 by ~0.07 A against a 0.20 A accuracy bar and
stays entirely plausible. Writing named ATOM records and letting the instrument select by name makes
that structurally impossible rather than a thing to remember.
"""

from __future__ import annotations

from pathlib import Path

import torch

from . import antibody_rmsd
from ._vendor.esm.utils import residue_constants as _rc

#: Atom names per restype in atom14 order, indexed by the aatype the model consumes. Unknown
#: residues take UNK, which carries the backbone only.
ATOM14_NAMES: list[list[str]] = [
    list(_rc.restype_name_to_atom14_names[_rc.restype_1to3[aa]]) for aa in _rc.restypes
] + [list(_rc.restype_name_to_atom14_names["UNK"])]

#: Their inference stage writes the heavy chain 1..n_h and the light chain from 501, and
#: `antibody_rmsd.residue_indices` uses that jump to find the chain break. Matching it is what lets
#: the instrument pair our residues with their truth.
LIGHT_RESSEQ_START = antibody_rmsd.LIGHT_RESSEQ_START


def write_fv_pdb(path: str | Path, aatype: torch.Tensor, atom14: torch.Tensor,
                 atom_mask: torch.Tensor, n_heavy: int) -> Path:
    """Write one Fv as a PDB: chain H numbered 1..n_heavy, chain L from 501.

    `aatype` is `[N]`, `atom14` is `[N, 14, 3]` and `atom_mask` is `[N, 14]`. Masked slots are not
    written at all rather than written as zeros, because the instrument keys on what is present and
    a zeroed atom would be scored as a real one at the origin.

    The chain identifier occupies exactly one column (21). A two-character chain silently corrupts
    the residue-number field that follows it, which has killed a finished fold in this repo before.
    """
    aatype = aatype.detach().cpu().long()
    atom14 = atom14.detach().cpu().double()
    atom_mask = atom_mask.detach().cpu()
    n_res = int(aatype.shape[0])
    assert 0 < n_heavy <= n_res, (n_heavy, n_res)

    lines, serial = [], 1
    for i in range(n_res):
        names = ATOM14_NAMES[int(aatype[i])]
        chain = "H" if i < n_heavy else "L"
        resseq = i + 1 if i < n_heavy else LIGHT_RESSEQ_START + (i - n_heavy)
        resname = _rc.restype_1to3.get(_rc.restypes[int(aatype[i])], "UNK") \
            if int(aatype[i]) < len(_rc.restypes) else "UNK"
        for slot, name in enumerate(names):
            if not name or not bool(atom_mask[i, slot]):
                continue
            x, y, z = (float(v) for v in atom14[i, slot])
            lines.append(
                f"ATOM  {serial:>5} {name:<4}{resname:>4} {chain}{resseq:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {name[0]:>2}")
            serial += 1
    lines.append("END")
    out = Path(path)
    out.write_text("\n".join(lines) + "\n")
    return out


def score_fv(true_pdb: str | Path, pred_pdb: str | Path, regions,
             atoms=antibody_rmsd.BACKBONE_ATOMS) -> dict[str, float]:
    """Per-region RMSDs, straight from `train-b1-instrument`'s instrument.

    One line on purpose. Every number this row reports about structure accuracy comes through here,
    so there is exactly one place that could ever drift, and it is not ours.
    """
    return antibody_rmsd.score_pdb_pair(true_pdb, pred_pdb, regions, atoms)
