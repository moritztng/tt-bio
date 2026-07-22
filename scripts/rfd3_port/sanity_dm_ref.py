"""CPU sanity check for the vendored RFD3DiffusionModuleRef: reconstruct f from the
captured token_initializer goldens, load captured TI outputs, run a single denoise step
with random weights + shared X_noisy/t, assert finite output. (Real weights loaded later
for the PCC bridge.)"""
import os, sys, glob, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from rfd3_dm_ref import RFD3DiffusionModuleRef, SharedDraws

CAP = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")


def load(n):
    return torch.load(os.path.join(CAP, n + ".pt"), map_location="cpu", weights_only=True)


def reconstruct_f():
    keys = [os.path.basename(p)[len("token_initializer.in_f_"):-3]
            for p in glob.glob(os.path.join(CAP, "token_initializer.in_f_*.pt"))]
    f = {}
    for k in keys:
        f[k] = load("token_initializer.in_f_" + k)
    return f


def main():
    torch.manual_seed(0)
    f = reconstruct_f()
    print(f"[f] {len(f)} keys; unindexing_pair_mask {tuple(f['unindexing_pair_mask'].shape)}")
    Q_L_init = load("token_initializer.out_Q_L_init").float()
    C_L = load("token_initializer.out_C_L").float()
    P_LL = load("token_initializer.out_P_LL").float()
    S_I = load("token_initializer.out_S_I").float()
    Z_II = load("token_initializer.out_Z_II").float()
    L = Q_L_init.shape[0]; I = S_I.shape[0]
    print(f"[shapes] L={L} I={I} Q_L_init={tuple(Q_L_init.shape)} C_L={tuple(C_L.shape)} "
          f"P_LL={tuple(P_LL.shape)} S_I={tuple(S_I.shape)} Z_II={tuple(Z_II.shape)}")

    dm = RFD3DiffusionModuleRef()
    dm.eval()
    n_params = sum(p.numel() for p in dm.parameters())
    print(f"[dm] {n_params/1e6:.1f}M params, {len(list(dm.parameters()))} tensors")

    D = 1
    is_motif_fixed = f["is_motif_atom_with_fixed_coord"]
    coord = torch.zeros(D, L, 3)
    draws = SharedDraws(seed=42)
    X_noisy_L = draws.initial(torch.tensor(80.0), D, L, coord, is_motif_fixed)
    t = torch.tensor([80.0])
    print(f"[inputs] X_noisy_L {tuple(X_noisy_L.shape)} t={t.item()} finite={torch.isfinite(X_noisy_L).all().item()}")

    with torch.no_grad():
        outs = dm(X_noisy_L=X_noisy_L, t=t, f=f, Q_L_init=Q_L_init, C_L=C_L, P_LL=P_LL,
                  S_I=S_I, Z_II=Z_II, n_recycle=2)
    X_out = outs["X_L"]
    print(f"[out] X_L {tuple(X_out.shape)} finite={torch.isfinite(X_out).all().item()} "
          f"mean={X_out.mean().item():.4f} std={X_out.std().item():.4f}")
    seq = outs["sequence_logits_I"]
    print(f"[out] seq_logits {tuple(seq.shape)} finite={torch.isfinite(seq).all().item()}")
    assert torch.isfinite(X_out).all(), "NON-FINITE X_out"
    assert torch.isfinite(seq).all(), "NON-FINITE seq"
    print("SANITY OK: vendored DiffusionModule reference runs + finite output (random weights)")


if __name__ == "__main__":
    main()
