"""Multi-step trajectory parity on F5 symmetry COMBINED with a real motif
(p19) -- the two real, value-parity-verified targets from
`scripts/rfd3_port/parity_artifacts/parity_symmetry_motif.py`:

1. `unsym_C3_6t8h` (verbatim real example): unconditional C3 oligomer around
   a real DNA helix, DNA excluded from symmetrization via `is_unsym_motif`
   (mechanisms (a) Kabsch frames + (b) is_unsym_motif exclusion).
2. `unindexed_C2_1j79` minus `ligand` (deterministic variant, `ligand`+
   `symmetry` out of scope this pass -- see tt_bio.rfd3_featurize's module
   docstring): C2 design with an unindexed catalytic residue "within a
   subunit" (mechanisms (a) again, different real frame + (c) unindexed-
   motif replication-then-forced-fixed).

Checks the on-device RFD3DiffusionModule + tt_bio.rfd3_sampler's F5 per-step
symmetry reapplication (`apply_symmetry_atomwise`) numerically agrees with
the vendored torch reference's own, AND that the reference's generic
"never add noise to a fixed-coord atom" mechanism keeps the real motif atom
(DNA / the unindexed catalytic residue) pinned at its seeded `motif_pos`,
exactly as it does with no symmetry involved at all.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_trajectory_symmetry_motif.py [num_timesteps] [case]
  case: "6t8h" (default) or "1j79_nolig"
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

CASES = {
    "6t8h": ("symmetry_motif_6t8h", "6t8h_C3.pdb", "spec.json"),
    "1j79_nolig": ("symmetry_motif_1j79_nolig", "1j79_C2.pdb", "spec.json"),
    "6t8h_small": ("symmetry_motif_6t8h", "6t8h_C3.pdb", "spec_small.json"),
    "1j79_nolig_small": ("symmetry_motif_1j79_nolig", "1j79_C2.pdb", "spec_small.json"),
}


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


def rmsd(a, b):
    return float((a.float() - b.float()).pow(2).mean().sqrt())


def main(n_ts, case):
    subdir, pdb_name, spec_name = CASES[case]
    PDB = os.path.join(DIR, "parity_artifacts", subdir, pdb_name)
    SPEC_JSON = os.path.join(DIR, "parity_artifacts", subdir, spec_name)

    torch.manual_seed(0)
    with open(SPEC_JSON) as fh:
        spec_dict = json.load(fh)
    spec_dict["input"] = PDB
    spec = InputSpecification.from_dict(spec_dict)
    spec.validate()
    f = featurize(PDB, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
    is_motif = f["is_motif_atom_with_fixed_coord"]
    L = f["ref_pos"].shape[0]
    I = f["restype"].shape[0]
    n_replicas = int(torch.unique(f["sym_transform_id"][f["sym_transform_id"] >= 0]).numel())
    n_fixed_sym = int((f["sym_entity_id"] == -1).sum())
    print(f"[setup] real F5+motif ({case}) f: I={I} L={L} ({n_replicas} replicas, "
          f"{int(f['is_sym_asu'].sum())} ASU atoms, {int(is_motif.sum())} fixed-coord atoms, "
          f"{n_fixed_sym} sym-fixed (never-resymmetrized) atoms)")

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

    if int(is_motif.sum()) > 0:
        motif_seed_ok_ref = bool(torch.equal(X_ref[0, is_motif], coord[0, is_motif]))
        motif_seed_ok_tt = bool(torch.equal(X_tt[0, is_motif], coord[0, is_motif]))
        print(f"[motif] final fixed-atom coords == seeded motif_pos: ref={motif_seed_ok_ref} device={motif_seed_ok_tt}")
        if not (motif_seed_ok_ref and motif_seed_ok_tt):
            raise AssertionError("fixed atoms moved away from their seeded ground-truth position")

    # F5-specific structural check (same as verify_trajectory_symmetry.py):
    # at step 0, every REPLICATED (non-fixed) atom's denoised position should
    # be an exact rigid-transform copy of the ASU's own.
    for name, traj in (("ref", traj_ref), ("device", traj_tt)):
        Xd = traj[0]["X_denoised_L"]
        asu_xyz = Xd[:, f["is_sym_asu"], :]
        max_err = 0.0
        for tid in range(1, n_replicas):
            R, t = f["sym_transform"][str(tid)]
            subunit = (f["sym_transform_id"] == tid)
            if int(subunit.sum()) == 0:
                continue
            expect = torch.einsum("blc,cd->bld", asu_xyz, R.float()) + t.float()
            max_err = max(max_err, float((Xd[:, subunit, :] - expect).abs().max()))
        print(f"[symmetry check] {name} step 0 X_denoised_L: max abs error vs ASU-derived replica coords = {max_err:.6f}")

    if worst < 0.97 or pf < 0.97:
        raise AssertionError(f"trajectory parity failed on the F5+motif ({case}) input: "
                              f"worst-step PCC={worst:.6f} final PCC={pf:.6f}")
    print(f"TRAJECTORY PARITY OK on the F5+motif ({case}) input (per-step PCC >= 0.97, both "
          "backends' per-step symmetry reapplication agree, fixed motif atoms held exact)")


if __name__ == "__main__":
    n_ts = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    case = sys.argv[2] if len(sys.argv) > 2 else "6t8h"
    main(n_ts, case)
