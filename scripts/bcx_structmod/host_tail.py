"""Turn the device module's frames and angles into atoms, and grade them in Angstrom.

The per-tensor grades say how far the device arm's `traj` and `unnormalized_angles` sit from
float64. They are not the number this campaign decides on. BindCraft 2's kill bar is on the
STRUCTURE, in Angstrom, against a seed floor: re-running with a different seed moves the
structure 1.84 A at 512 aa, so a lever that moves it less than that is smaller than variation
the campaign already accepts.

So this runs the module's own host tail -- `l2_normalize`, `torsion_angles_to_frames`,
`frames_and_literature_positions_to_atom14_pos`, `atom14_to_atom37` -- on the DEVICE outputs and
on the float64 reference's outputs, with the same `aatype`, and reports the all-atom RMSD and
the per-atom deviation over the residues that are not padding. Nothing is aligned first: these
are two predictions of the same structure in the same frame, so a Kabsch fit would hide exactly
the rigid drift the eight frame updates are capable of producing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

BC2 = os.environ.get("BCX_BC2", "/home/ttuser/bcx_e2e/bc2")
if BC2 not in sys.path:
    sys.path.insert(0, BC2)


def atoms_from(traj_last, unnormalized, aatype, jnp, geometry, all_atom_multimer):
    """`traj_last` is `[n, 3, 4]` with the translation already in Angstrom, `unnormalized` is
    `[n, 14]`, which is `MultiRigidSidechain`'s unnormalised `[n, 7, 2]` flattened."""
    rigid = geometry.Rigid3Array.from_array(jnp.asarray(traj_last))
    unnorm = jnp.asarray(np.asarray(unnormalized).reshape(-1, 7, 2))
    angles = unnorm / jnp.sqrt(jnp.maximum(jnp.sum(unnorm ** 2, axis=-1, keepdims=True), 1e-12))
    frames = all_atom_multimer.torsion_angles_to_frames(jnp.asarray(aatype), rigid, angles)
    pos = all_atom_multimer.frames_and_literature_positions_to_atom14_pos(
        jnp.asarray(aatype), frames)
    atom14 = pos.to_array()
    return np.asarray(atom14), np.asarray(
        all_atom_multimer.atom14_to_atom37(atom14, jnp.asarray(aatype)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="the float64 reference dump")
    ap.add_argument("--dump", required=True, help="a device_arm.py --dump npz")
    ap.add_argument("--label", default="device")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ["JAX_ENABLE_X64"] = "1"
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from bindcraft.af.alphafold.model import all_atom_multimer
    from bindcraft.af.alphafold.model import geometry
    from bindcraft.af.alphafold.model.geometry import (rigid_matrix_vector, rotation_matrix,
                                                       vector)
    for cls in (vector.Vec3Array, rotation_matrix.Rot3Array, rigid_matrix_vector.Rigid3Array):
        cls.__post_init__ = lambda self: None

    ref = np.load(args.ref)
    dev = np.load(args.dump)
    aatype = np.asarray(ref["aatype"])
    live = np.asarray(ref["seq_mask"]) > 0

    # The reference's own atoms, rebuilt through the same tail rather than read off the dump,
    # so the tail itself cannot be the difference. It is checked against the dump below.
    ref14, ref37 = atoms_from(np.asarray(ref["traj"])[-1],
                              np.asarray(ref["sidechains_unnormalized"])[-1].reshape(-1, 14),
                              aatype, jnp, geometry, all_atom_multimer)
    dev14, dev37 = atoms_from(np.asarray(dev["traj"])[-1],
                              np.asarray(dev["unnormalized_angles"])[-1],
                              aatype, jnp, geometry, all_atom_multimer)

    tail_check = float(np.max(np.abs(
        ref14 - np.asarray(ref["final_atom14_positions"]))))

    mask14 = np.asarray(all_atom_multimer.get_atom14_mask(jnp.asarray(aatype))) > 0
    mask14 = mask14 & live[:, None]
    d = np.linalg.norm(dev14 - ref14, axis=-1)[mask14]

    # The backbone alone, which is what a fold is judged on before sidechains: atom14 slots
    # 0-3 are N, CA, C, CB in every residue type.
    bb = np.zeros_like(mask14)
    bb[:, :4] = True
    dbb = np.linalg.norm(dev14 - ref14, axis=-1)[mask14 & bb]

    report = {
        "label": args.label, "ref": args.ref, "dump": args.dump,
        "n": int(aatype.shape[0]), "n_live": int(live.sum()),
        "tail_selfcheck_max_abs_A": tail_check,
        "atoms_graded": int(mask14.sum()),
        "all_atom_rmsd_A": float(np.sqrt((d ** 2).mean())),
        "all_atom_median_A": float(np.median(d)),
        "all_atom_p95_A": float(np.percentile(d, 95)),
        "all_atom_max_A": float(d.max()),
        "backbone_atoms": int((mask14 & bb).sum()),
        "backbone_rmsd_A": float(np.sqrt((dbb ** 2).mean())),
        "backbone_max_A": float(dbb.max()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
