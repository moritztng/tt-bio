"""Multi-step trajectory parity for the p21 INDEXED-motif atom-subsetting case:
same real IAI_protein.pdb + contig "A1-10,20,A31-40" as
verify_trajectory_from_pdb.py, but with `select_fixed_atoms: {"A5":
"CB,CG,CD"}` -- ARG5 stays a real motif residue (all 11 atoms present, known
sequence) but only CB/CG/CD are individually fixed-coord; its other 8 atoms
(including its own backbone N/CA/C/O) are diffused like a designed atom.

This is the real regression check for `_indexed_fixed_atom_names`: the
sampler must seed+lock ONLY the atoms `is_motif_atom_with_fixed_coord` marks
true (now a genuine per-atom, not per-token, mask) -- verified below by
checking ARG5's own backbone stays UNLOCKED (moves under diffusion) while
CB/CG/CD stay locked at the seeded `motif_pos`, on BOTH backends.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_trajectory_indexed_atomsubset.py [num_timesteps]
"""
import os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification

sys.path.insert(0, os.path.dirname(__file__))
from rfd3_dm_ref import RFD3DiffusionModuleRef, EDMSamplerRef, SharedDraws

PDB = os.path.join(os.path.dirname(__file__), "parity_artifacts", "iai_protein", "IAI_protein.pdb")
CONTIG = "A1-10,20,A31-40"
SELECT_FIXED_ATOMS = {"A5": "CB,CG,CD"}
GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


def rmsd(a, b):
    return float((a.float() - b.float()).pow(2).mean().sqrt())


def main(n_ts):
    torch.manual_seed(0)
    spec = InputSpecification.from_dict({"input": PDB, "contig": CONTIG,
                                          "select_fixed_atoms": SELECT_FIXED_ATOMS})
    spec.validate()
    f = featurize(PDB, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
    is_motif = f["is_motif_atom_with_fixed_coord"]
    L = f["ref_pos"].shape[0]
    print(f"[setup] real IAI-protein f: I={f['restype'].shape[0]} L={L} "
          f"({int(is_motif.sum())} individually-fixed atoms, ARG5 partially subsetted)")

    # ARG5's own token span: verify the partial split is real BEFORE running any
    # trajectory (a real regression check on `_indexed_fixed_atom_names`/
    # `_motif_atom_layout`'s `fixed` mask): some but not all of its 11 atoms fixed.
    tok_fixed = f["is_motif_token_with_fully_fixed_coord"]
    assert not bool(tok_fixed[4]), "ARG5 (token 4) must NOT be fully-fixed (partial select_fixed_atoms)"
    assert bool(tok_fixed[0]) and bool(tok_fixed[3]) and bool(tok_fixed[5]), \
        "neighboring real motif residues (no select_fixed_atoms override) must stay fully-fixed"
    atom_to_token = f["atom_to_token_map"]
    arg5_atoms = is_motif[atom_to_token == 4]
    print(f"[setup] ARG5: {int(arg5_atoms.sum())}/{arg5_atoms.numel()} atoms individually fixed-coord "
          f"(expect 3/11: CB,CG,CD)")
    assert int(arg5_atoms.sum()) == 3 and arg5_atoms.numel() == 11

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

    ref = RFD3DiffusionModuleRef().eval()
    m, u = ref.load_state_dict(dm_weights, strict=False)
    print(f"[ref] weights missing={len(m)} unexpected={len(u)}")
    with torch.no_grad():
        X_ref, traj_ref = sampler.sample(ref, 1, L, coord, f, init, SharedDraws(seed=42), is_motif)

    dm = build_diffusion_module(dm_weights)
    X_tt, traj_tt = sampler.sample(dm, 1, L, coord, f, init, SharedDraws(seed=42), is_motif)

    assert len(traj_ref) == len(traj_tt), (len(traj_ref), len(traj_tt))
    print(f"[traj] {len(traj_ref)} steps; per-step X_L parity (device vs vendored-torch reference):")
    worst = 1.0
    for i, (r, t) in enumerate(zip(traj_ref, traj_tt)):
        p = pcc(t["X_L"], r["X_L"]); rr = rmsd(t["X_L"], r["X_L"])
        worst = min(worst, p)
        print(f"  step {i}: t_hat={float(r['t_hat']):.3f}  PCC={p:.6f}  RMSD={rr:.4f}")
    pf = pcc(X_tt, X_ref); rf = rmsd(X_tt, X_ref)
    motif_seed_ok_ref = bool(torch.equal(X_ref[0, is_motif], coord[0, is_motif]))
    motif_seed_ok_tt = bool(torch.equal(X_tt[0, is_motif], coord[0, is_motif]))
    # The regression check this fixture exists for: ARG5's own NON-fixed atoms
    # (backbone N/CA/C/O + NE/CZ/NH1/NH2) must have MOVED under diffusion, on
    # both backends -- if a bug silently kept them locked (treating the whole
    # token as motif), this would be a false pass.
    arg5_unfixed = (atom_to_token == 4) & (~is_motif)
    arg5_moved_ref = not bool(torch.equal(X_ref[0, arg5_unfixed], coord[0, arg5_unfixed]))
    arg5_moved_tt = not bool(torch.equal(X_tt[0, arg5_unfixed], coord[0, arg5_unfixed]))
    print(f"[final] X_L PCC={pf:.6f}  RMSD={rf:.4f}  (ref std {X_ref.std().item():.4f})  worst-step PCC={worst:.6f}")
    print(f"[motif] final fixed-atom coords == seeded motif_pos: ref={motif_seed_ok_ref} device={motif_seed_ok_tt}")
    print(f"[arg5]  ARG5's non-selected atoms moved away from their seed: ref={arg5_moved_ref} device={arg5_moved_tt}")
    if worst < 0.97 or pf < 0.97:
        raise AssertionError(f"trajectory parity failed on the indexed-atomsubset input: "
                              f"worst-step PCC={worst:.6f} final PCC={pf:.6f}")
    if not (motif_seed_ok_ref and motif_seed_ok_tt):
        raise AssertionError("fixed atoms moved away from their seeded ground-truth position")
    if not (arg5_moved_ref and arg5_moved_tt):
        raise AssertionError("ARG5's non-selected atoms stayed locked -- partial fixing not applied")
    print("TRAJECTORY PARITY OK on the indexed-motif atom-subsetting input (per-step PCC >= 0.97, "
          "fixed atoms verified locked, partially-fixed residue's other atoms verified diffused, "
          "on both backends)")


if __name__ == "__main__":
    n_ts = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(n_ts)
