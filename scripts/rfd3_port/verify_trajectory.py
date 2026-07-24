"""Multi-step trajectory parity: run the EDM sampler loop with the ttnn RFD3DiffusionModule
vs the vendored torch reference, using IDENTICAL shared random draws (SharedDraws seed=42)
so both backends see the same noise stream. Reports per-step X_L PCC + RMSD and the final
structure RMSD. num_timesteps>=2 (default 4 -> 3 trajectory steps).

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_trajectory.py [capture_dir] [num_timesteps]
"""
import os, sys, glob, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module

sys.path.insert(0, os.path.dirname(__file__))
from rfd3_dm_ref import RFD3DiffusionModuleRef, EDMSamplerRef, SharedDraws


def load(cap, n):
    return torch.load(os.path.join(cap, n + ".pt"), map_location="cpu", weights_only=True)


def reconstruct_f(cap):
    keys = [os.path.basename(p)[len("token_initializer.in_f_"):-3]
            for p in glob.glob(os.path.join(cap, "token_initializer.in_f_*.pt"))]
    return {k: load(cap, "token_initializer.in_f_" + k) for k in keys}


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


def rmsd(a, b):
    return float((a.float() - b.float()).pow(2).mean().sqrt())


def main(cap, n_ts):
    torch.manual_seed(0)
    f = reconstruct_f(cap)
    weights = load(cap, "diffusion_module.real_weights")
    Q_L_init = load(cap, "token_initializer.out_Q_L_init").float()
    C_L = load(cap, "token_initializer.out_C_L").float()
    P_LL = load(cap, "token_initializer.out_P_LL").float()
    S_I = load(cap, "token_initializer.out_S_I").float()
    Z_II = load(cap, "token_initializer.out_Z_II").float()
    L = Q_L_init.shape[0]
    is_motif = f["is_motif_atom_with_fixed_coord"]
    coord = torch.zeros(1, L, 3)
    init = dict(Q_L_init=Q_L_init, C_L=C_L, P_LL=P_LL, S_I=S_I, Z_II=Z_II)

    sampler = EDMSamplerRef(num_timesteps=n_ts)

    # --- reference trajectory (real weights, shared draws seed=42) ---
    ref = RFD3DiffusionModuleRef().eval()
    m, u = ref.load_state_dict(weights, strict=False)
    print(f"[ref] weights missing={len(m)} unexpected={len(u)}")
    with torch.no_grad():
        X_ref, traj_ref = sampler.sample(ref, 1, L, coord, f, init, SharedDraws(seed=42), is_motif)

    # --- ttnn trajectory (real weights, device, identical shared draws seed=42) ---
    dm = build_diffusion_module(weights)
    X_tt, traj_tt = sampler.sample(dm, 1, L, coord, f, init, SharedDraws(seed=42), is_motif)

    assert len(traj_ref) == len(traj_tt), (len(traj_ref), len(traj_tt))
    print(f"[traj] {len(traj_ref)} steps; per-step X_L parity (device vs ref):")
    worst = 1.0
    for i, (r, t) in enumerate(zip(traj_ref, traj_tt)):
        p = pcc(t["X_L"], r["X_L"]); rr = rmsd(t["X_L"], r["X_L"])
        worst = min(worst, p)
        print(f"  step {i}: t_hat={float(r['t_hat']):.3f}  PCC={p:.6f}  RMSD={rr:.4f}")
    pf = pcc(X_tt, X_ref); rf = rmsd(X_tt, X_ref)
    print(f"[final] X_L PCC={pf:.6f}  RMSD={rf:.4f}  (ref std {X_ref.std().item():.4f})  worst-step PCC={worst:.6f}")
    if worst < 0.97 or pf < 0.97:
        raise AssertionError(f"trajectory parity failed: worst-step PCC={worst:.6f} final PCC={pf:.6f}")
    print("TRAJECTORY PARITY OK (per-step PCC >= 0.97 vs vendored torch reference, shared draws)")


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(__file__), "..", "..", ".scratch", "rfd3-ref", "goldens", "capture")
    cap = sys.argv[1] if len(sys.argv) > 1 else default
    n_ts = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    main(cap, n_ts)
