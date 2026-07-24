"""Multi-step trajectory parity on a REAL from-PDB input (not the golden bridge):
runs the EDM sampler loop with the ttnn RFD3DiffusionModule vs the vendored torch
reference, using IDENTICAL shared random draws (SharedDraws seed=42), starting
from the p12 parity-verified featurizer's `f` (real IAI protein + contig) and the
p12-fixed motif_pos coordinate seed (not the old all-zero seed).

This closes the p12 §2l.6 item-3 gap for the F1/F6 case: p12 proved --from_pdb
produces a geometrically sane CIF (real motif coordinates, real denoised design
coordinates) but never checked that the on-device DiffusionModule numerically
agrees with the reference module ON THIS REAL INPUT + THE NEW motif-seeded
coord — the only trajectory parity numbers on record (p5/p6) used the golden
dsDNA_basic bridge with a zero coord seed on both sides.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_trajectory_from_pdb.py [num_timesteps]
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
    print(f"[setup] real IAI-protein f: I={f['restype'].shape[0]} L={L} "
          f"(p12 value-parity-verified featurizer, {int(is_motif.sum())} motif atoms)")

    ti_weights = torch.load(os.path.join(GOLDEN_DIR, "token_initializer.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dm_weights = torch.load(os.path.join(GOLDEN_DIR, "diffusion_module.real_weights.pt"),
                             map_location="cpu", weights_only=True)

    # TokenInitializer: run once on device, reuse the SAME init for both trajectories
    # (init is deterministic given f + weights; only the diffusion module differs).
    dev_ti = build_token_initializer(ti_weights)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    init = {k: v.float() for k, v in init.items()}

    # p12 fix: seed the trajectory at the real (centered) motif position, not zero.
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
    # motif atoms must be bit-identical to their seed on BOTH backends (the sampler
    # never touches them) — a real regression check for the p12 motif-seed fix.
    motif_seed_ok_ref = bool(torch.equal(X_ref[0, is_motif], coord[0, is_motif]))
    motif_seed_ok_tt = bool(torch.equal(X_tt[0, is_motif], coord[0, is_motif]))
    print(f"[final] X_L PCC={pf:.6f}  RMSD={rf:.4f}  (ref std {X_ref.std().item():.4f})  worst-step PCC={worst:.6f}")
    print(f"[motif] final motif-atom coords == seeded motif_pos: ref={motif_seed_ok_ref} device={motif_seed_ok_tt}")
    if worst < 0.97 or pf < 0.97:
        raise AssertionError(f"trajectory parity failed on the real from-PDB input: "
                              f"worst-step PCC={worst:.6f} final PCC={pf:.6f}")
    if not (motif_seed_ok_ref and motif_seed_ok_tt):
        raise AssertionError("motif atoms moved away from their seeded ground-truth position")
    print("TRAJECTORY PARITY OK on the real from-PDB input (per-step PCC >= 0.97, "
          "motif atoms verified fixed at their true position on both backends)")


if __name__ == "__main__":
    n_ts = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(n_ts)
