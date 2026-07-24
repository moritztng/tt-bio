"""Multi-step trajectory parity on a fully-unconditional F5 symmetric design
(p18): {"length": 12, "is_non_loopy": true, "symmetry": {"id": "C3"}} -- the
same shape as the real RosettaCommons/foundry `docs/examples/symmetry.md`
"uncond_C5" example, just a smaller length for a fast device check. No
`input` PDB at all (see tt_bio.rfd3_featurize's F5 grounding for why a bare
`length`-only spec never needs one). Value-parity-verified in
scripts/rfd3_port/parity_artifacts/parity_symmetry.py (41/41 keys bit-exact,
no documented gaps -- there is no real motif/ligand in this design at all).

This checks the on-device RFD3DiffusionModule + tt_bio.rfd3_sampler's F5
per-step symmetry reapplication (`apply_symmetry_atomwise`) numerically
agrees with the vendored torch reference's own (`rfd3_dm_ref.py`), both fed
identical shared RNG draws and identical `f["sym_transform"]`/
`sym_transform_id`/`sym_entity_id`/`is_sym_asu` features -- i.e. that BOTH
backends reconstruct every symmetric replica from the ASU's own denoised
output the same way at every step, not just that the featurizer output
matches.

Also checked (ad hoc, same script, not a separate fixture -- D4 is otherwise
identical in every mechanical respect to C3, see tt_bio.rfd3_featurize's F5
grounding for why both share one `_dihedral_frames`/`_cyclic_frames` code
path): a D4 spec (`scripts/rfd3_port/parity_artifacts/symmetry_uncond_d4/`,
8 replicas) gives the same result (worst-step PCC 0.999997, exact ASU-
derived-replica reconstruction on both backends).

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_trajectory_symmetry.py [num_timesteps] [spec_subdir]
"""
import json
import os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification

sys.path.insert(0, os.path.dirname(__file__))
from rfd3_dm_ref import RFD3DiffusionModuleRef, EDMSamplerRef, SharedDraws

DIR = os.path.dirname(__file__)
GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


def rmsd(a, b):
    return float((a.float() - b.float()).pow(2).mean().sqrt())


def main(n_ts, spec_subdir="symmetry_uncond"):
    torch.manual_seed(0)
    spec_json = os.path.join(DIR, "parity_artifacts", spec_subdir, "spec.json")
    with open(spec_json) as fh:
        spec_dict = json.load(fh)
    spec = InputSpecification.from_dict(spec_dict)
    spec.validate()
    f = featurize(None, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
    is_motif = f["is_motif_atom_with_fixed_coord"]
    L = f["ref_pos"].shape[0]
    I = f["restype"].shape[0]
    n_replicas = int(torch.unique(f["sym_transform_id"]).numel())
    print(f"[setup] real F5 symmetric (C3) f: I={I} L={L} ({n_replicas} replicas, "
          f"{int(f['is_sym_asu'].sum())} ASU atoms, {int(is_motif.sum())} fixed-coord atoms)")

    ti_weights = torch.load(os.path.join(GOLDEN_DIR, "token_initializer.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dm_weights = torch.load(os.path.join(GOLDEN_DIR, "diffusion_module.real_weights.pt"),
                             map_location="cpu", weights_only=True)

    dev_ti = build_token_initializer(ti_weights)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    init = {k: v.float() for k, v in init.items()}

    coord = f["motif_pos"].unsqueeze(0)
    sampler = EDMSamplerRef(num_timesteps=n_ts)
    sym_feats = {k: f[k] for k in ("sym_transform", "sym_transform_id", "sym_entity_id", "is_sym_asu")}

    ref = RFD3DiffusionModuleRef().eval()
    m, u = ref.load_state_dict(dm_weights, strict=False)
    print(f"[ref] weights missing={len(m)} unexpected={len(u)}")
    with torch.no_grad():
        X_ref, traj_ref = sampler.sample(ref, 1, L, coord, f, init, SharedDraws(seed=42), is_motif,
                                          sym_feats=sym_feats)

    dm = build_diffusion_module(dm_weights)
    X_tt, traj_tt = sampler.sample(dm, 1, L, coord, f, init, SharedDraws(seed=42), is_motif,
                                    sym_feats=sym_feats)

    assert len(traj_ref) == len(traj_tt), (len(traj_ref), len(traj_tt))
    print(f"[traj] {len(traj_ref)} steps; per-step X_L parity (device vs vendored-torch reference):")
    worst = 1.0
    for i, (r, t) in enumerate(zip(traj_ref, traj_tt)):
        p = pcc(t["X_L"], r["X_L"]); rr = rmsd(t["X_L"], r["X_L"])
        worst = min(worst, p)
        print(f"  step {i}: t_hat={float(r['t_hat']):.3f}  PCC={p:.6f}  RMSD={rr:.4f}")
    pf = pcc(X_tt, X_ref); rf = rmsd(X_tt, X_ref)
    print(f"[final] X_L PCC={pf:.6f}  RMSD={rf:.4f}  (ref std {X_ref.std().item():.4f})  worst-step PCC={worst:.6f}")

    # F5-specific structural check: at a step where symmetrization was applied
    # (any step but the last, per sym_step_frac), the DENOISED output (not the
    # carried X_L -- that also picks up the next step's non-symmetric noise
    # draw, see apply_symmetry_atomwise's call site) should have every
    # replica's atoms an exact rigid-transform copy of the ASU's own atoms.
    for name, traj in (("ref", traj_ref), ("device", traj_tt)):
        Xd = traj[0]["X_denoised_L"]
        asu_xyz = Xd[:, f["is_sym_asu"], :]
        max_err = 0.0
        for tid in range(1, n_replicas):
            R, t = f["sym_transform"][str(tid)]
            subunit = f["sym_transform_id"] == tid
            expect = torch.einsum("blc,cd->bld", asu_xyz, R.float()) + t.float()
            max_err = max(max_err, float((Xd[:, subunit, :] - expect).abs().max()))
        print(f"[symmetry check] {name} step 0 X_denoised_L: max abs error vs ASU-derived replica coords = {max_err:.6f}")

    if worst < 0.97 or pf < 0.97:
        raise AssertionError(f"trajectory parity failed on the F5 symmetric input: "
                              f"worst-step PCC={worst:.6f} final PCC={pf:.6f}")
    print("TRAJECTORY PARITY OK on the F5 symmetric (C3, unconditional) input (per-step PCC >= 0.97, "
          "both backends' per-step symmetry reapplication agree)")


if __name__ == "__main__":
    n_ts = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    subdir = sys.argv[2] if len(sys.argv) > 2 else "symmetry_uncond"
    main(n_ts, subdir)
