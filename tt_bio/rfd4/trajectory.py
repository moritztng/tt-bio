"""Write a design's trajectory to disk, CA-only, as a multi-model mmCIF.

Upstream writes `save_trajectory_every` snapshots as a CA-only XTC or an all-atom multi-model
CIF. XTC needs mdtraj, which tt-bio does not ship, and the all-atom form needs the autoencoder
decoder run on every snapshot rather than only on the final state, which multiplies the device
cost of a run by the number of frames. So the port writes the CA-only form in the CIF container:
one model per snapshot, loadable in PyMOL or Mol* beside the design itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_trajectory(snapshots, features: dict, path: str | Path, *, row: int = 0) -> Path:
    """One design's CA trajectory as a multi-model CIF. Returns the path written.

    `snapshots` is `SamplerOutput.trajectory`; `row` picks the design. Coordinates are the clean
    prediction at each saved step, converted from nanometres, and only emitted tokens are
    written, which is the same set the design's CIF carries.
    """
    from biotite.structure import AtomArray, stack
    from biotite.structure.io.pdbx import CIFFile, set_structure

    from tt_bio.rfd4.sampler import BB_CA

    if not snapshots:
        raise ValueError("no trajectory was saved; pass save_trajectory_every > 0")

    emit = _emitted(features)
    chain_ids = _chain_ids(features, emit)
    frames = []
    for snapshot in snapshots:
        coords = snapshot["x_1"][BB_CA]
        coords = coords[row] if coords.dim() == 3 else coords
        frames.append(_frame(np.asarray(coords.detach().cpu(), dtype=np.float32)[emit] * 10.0,
                             chain_ids, AtomArray))

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    cif = CIFFile()
    set_structure(cif, stack(frames))
    cif.write(str(path))
    return path


def _emitted(features: dict) -> np.ndarray:
    """The tokens the flow moves, as a numpy mask over the example's token axis."""
    from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_flow_emit_masks
    from tt_bio.rfd4.sampler import BB_CA

    mask = compute_flow_emit_masks(features)[BB_CA]
    mask = mask[0] if mask.dim() > 1 else mask
    return np.asarray(mask.detach().cpu(), dtype=bool)


def _chain_ids(features: dict, emit: np.ndarray) -> np.ndarray:
    """Upstream's chain index per emitted token, as the letters a viewer expects."""
    chain = features.get("chain_iid")
    if chain is None:
        return np.full(int(emit.sum()), "A")
    chain = chain[0] if chain.dim() > 1 else chain
    index = np.asarray(chain.detach().cpu(), dtype=int)[emit]
    letters = np.array([chr(ord("A") + (i % 26)) for i in range(index.max() + 1)])
    return letters[index]


def _frame(coords: np.ndarray, chain_ids: np.ndarray, AtomArray) -> object:
    """One model: a CA per emitted token, numbered from 1 within its chain."""
    atoms = AtomArray(len(coords))
    atoms.coord = coords
    atoms.chain_id = chain_ids
    atoms.res_name = np.full(len(coords), "GLY")
    atoms.atom_name = np.full(len(coords), "CA")
    atoms.element = np.full(len(coords), "C")
    res_id = np.zeros(len(coords), dtype=int)
    for chain in np.unique(chain_ids):
        here = chain_ids == chain
        res_id[here] = np.arange(1, int(here.sum()) + 1)
    atoms.res_id = res_id
    return atoms
