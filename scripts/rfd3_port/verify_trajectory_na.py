"""Multi-step trajectory parity on a REAL from-PDB F2/F8 nucleic-acid-binder
input (p15): 1bna.pdb (real B-DNA dodecamer duplex, chains A+B, fixed target)
+ a designed protein binder chain, contig "A1-10,/0,B15-24,/0,20" — value-
parity-verified 42/43 keys bit-exact vs a real reference capture in
scripts/rfd3_port/parity_artifacts/parity_dna.py (the lone gap, `ref_pos`'s
real reference-conformer geometry, is left at 0 on both the featurizer output
AND is irrelevant to the trajectory: DNA atoms are motif/fixed, so their true
geometry flows through `motif_pos`, not `ref_pos`).

This checks the on-device RFD3DiffusionModule numerically agrees with the
vendored torch reference on a real NA-binder input, not just the featurizer
output.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_trajectory_na.py [num_timesteps]
"""
import os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification

sys.path.insert(0, os.path.dirname(__file__))
from rfd3_dm_ref import RFD3DiffusionModuleRef, EDMSamplerRef, SharedDraws

PDB = os.path.join(os.path.dirname(__file__), "parity_artifacts", "dsdna_basic", "1bna.pdb")
CONTIG = "A1-10,/0,B15-24,/0,20"
GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


def rmsd(a, b):
    return float((a.float() - b.float()).pow(2).mean().sqrt())


def main(n_ts):
    torch.manual_seed(0)
    spec = InputSpecification.from_dict({"input": PDB, "contig": CONTIG})
    spec.validate()
    f = featurize(PDB, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
    is_motif = f["is_motif_atom_with_fixed_coord"]
    L = f["ref_pos"].shape[0]
    I = f["restype"].shape[0]
    n_dna = int(f["is_dna"].sum())
    print(f"[setup] real dsDNA-binder f: I={I} L={L} ({n_dna} DNA tokens, "
          f"{int(is_motif.sum())} motif atoms)")

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
    print(f"[final] X_L PCC={pf:.6f}  RMSD={rf:.4f}  (ref std {X_ref.std().item():.4f})  worst-step PCC={worst:.6f}")
    print(f"[motif] final DNA-atom coords == seeded motif_pos: ref={motif_seed_ok_ref} device={motif_seed_ok_tt}")
    if worst < 0.97 or pf < 0.97:
        raise AssertionError(f"trajectory parity failed on the NA-binder input: "
                              f"worst-step PCC={worst:.6f} final PCC={pf:.6f}")
    if not (motif_seed_ok_ref and motif_seed_ok_tt):
        raise AssertionError("DNA motif atoms moved away from their seeded ground-truth position")
    print("TRAJECTORY PARITY OK on the NA-binder input (per-step PCC >= 0.97, "
          "DNA motif atoms verified fixed at their true position on both backends)")


if __name__ == "__main__":
    n_ts = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(n_ts)
