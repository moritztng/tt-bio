"""Verify the full RFD3 denoise step (the assembled DiffusionModule) on device vs the
vendored torch reference. Real checkpoint weights loaded into BOTH the ttnn port
(build_diffusion_module) and the vendored RFD3DiffusionModuleRef. Shared random
X_noisy_L + t (seed=42) + the captured TokenInitializer outputs (Q_L_init, C_L, P_LL,
S_I, Z_II) + the reconstructed f (43 keys) are fed to both. This is the full-assembly
shared-inputs wiring gate (device-vs-host-reference); a passing PCC proves the assembled
denoise step (encoder + DiffusionTokenEncoder + DiT + decoder + all glue +
forward_with_recycle n_recycle=2) is wired correctly end-to-end.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_assembly.py [capture_dir]
"""
import os, sys, glob, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module

sys.path.insert(0, os.path.dirname(__file__))
from rfd3_dm_ref import RFD3DiffusionModuleRef, SharedDraws


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


def main(cap):
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

    # shared inputs
    draws = SharedDraws(seed=42)
    X_noisy_L = draws.initial(torch.tensor(80.0), 1, L, torch.zeros(1, L, 3), is_motif)  # [1,L,3]
    t = torch.tensor([80.0])

    # --- vendored torch reference (real weights) ---
    ref = RFD3DiffusionModuleRef().eval()
    missing, unexpected = ref.load_state_dict(weights, strict=False)
    print(f"[ref] loaded real weights: missing={len(missing)} unexpected={len(unexpected)}")
    with torch.no_grad():
        ref_out = ref(X_noisy_L=X_noisy_L, t=t, f=f, Q_L_init=Q_L_init, C_L=C_L, P_LL=P_LL,
                        S_I=S_I, Z_II=Z_II, n_recycle=2)
    X_ref = ref_out["X_L"]
    print(f"[ref] X_L {tuple(X_ref.shape)} finite={torch.isfinite(X_ref).all().item()} "
          f"std={X_ref.std().item():.4f}")

    # --- ttnn port (real weights, device) ---
    dm = build_diffusion_module(weights)
    tt_out = dm(X_noisy_L=X_noisy_L, t=t, f=f, Q_L_init=Q_L_init, C_L=C_L, P_LL=P_LL,
                 S_I=S_I, Z_II=Z_II, n_recycle=2)
    X_tt = tt_out["X_L"]
    print(f"[ttnn] X_L {tuple(X_tt.shape)} finite={torch.isfinite(X_tt).all().item()} "
          f"std={X_tt.std().item():.4f}")

    v = pcc(X_tt, X_ref)
    r = rmsd(X_tt, X_ref)
    print(f"[parity] full-denoise-step PCC = {v:.6f}  RMSD = {r:.4f}  (ref std {X_ref.std().item():.4f})")
    if v < 0.99:
        raise AssertionError(f"full-denoise-step PCC {v:.6f} < 0.99")
    print("FULL ASSEMBLY PARITY OK (denoise step PCC >= 0.99 vs vendored torch reference)")


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(__file__), "..", "..", ".scratch", "rfd3-ref", "goldens", "capture")
    cap = sys.argv[1] if len(sys.argv) > 1 else default
    main(cap)
