"""Sampler-mode parity: run the shipped RFD3Sampler (default / partial / CFG) on the
ttnn RFD3DiffusionModule vs the vendored torch reference, with IDENTICAL shared
random draws (same-seed torch.Generator on both backends) so the comparison isolates
the device forward under each sampler mode.

  default  -> regression (must reproduce the p5 ~0.9999 trajectory parity)
  partial  -> F7 partial diffusion (subset the schedule; start from a real structure)
  cfg      -> classifier-free guidance (unconditional ref pass with cfg_features zeroed)

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_sampler.py [capture_dir] [num_timesteps]
"""
import os, sys, glob, copy, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
from tt_bio.rfd3_sampler import RFD3Sampler, strip_f

sys.path.insert(0, os.path.dirname(__file__))
from rfd3_dm_ref import RFD3DiffusionModuleRef
import rfd3_ref as R


def load(cap, n):
    return torch.load(os.path.join(cap, n + ".pt"), map_location="cpu", weights_only=True)


def reconstruct_f(cap):
    keys = [os.path.basename(p)[len("token_initializer.in_f_"):-3]
            for p in glob.glob(os.path.join(cap, "token_initializer.in_f_*.pt"))]
    f = {}
    for k in keys:
        t = load(cap, "token_initializer.in_f_" + k)
        # Golden f was captured under bf16 AMP; the torch ref runs in fp32. Cast
        # fp tensors to float32; leave bool/int index/mask tensors untouched.
        if t.is_floating_point() and t.dtype != torch.float32:
            t = t.float()
        f[k] = t
    return f


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


def rmsd(a, b):
    return float((a.float() - b.float()).pow(2).mean().sqrt())


def main(cap, n_ts):
    torch.manual_seed(0)
    f = reconstruct_f(cap)
    f = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()}
    dm_weights = load(cap, "diffusion_module.real_weights")
    ti_weights = load(cap, "token_initializer.real_weights")
    Q_L_init = load(cap, "token_initializer.out_Q_L_init").float()
    C_L = load(cap, "token_initializer.out_C_L").float()
    P_LL = load(cap, "token_initializer.out_P_LL").float()
    S_I = load(cap, "token_initializer.out_S_I").float()
    Z_II = load(cap, "token_initializer.out_Z_II").float()
    L = Q_L_init.shape[0]
    is_motif = f["is_motif_atom_with_fixed_coord"]
    init = dict(Q_L_init=Q_L_init, C_L=C_L, P_LL=P_LL, S_I=S_I, Z_II=Z_II)
    coord_zero = torch.zeros(1, L, 3)
    coord_real = f["ref_pos"].float().unsqueeze(0).clone()
    cfg_features = ["ref_atomwise_rasa", "active_donor", "active_acceptor"]

    # reference DM + reference TokenInitializer (real weights)
    ref_dm = RFD3DiffusionModuleRef().eval()
    m, u = ref_dm.load_state_dict(dm_weights, strict=False)
    print(f"[ref-dm] missing={len(m)} unexpected={len(u)}")
    ref_ti = R.build_token_initializer()
    mt, ut = ref_ti.load_state_dict(ti_weights, strict=False)
    print(f"[ref-ti] missing={len(mt)} unexpected={len(ut)}")
    ref_ti.eval()
    # ttnn device DM + device TokenInitializer (real weights)
    dev_dm = build_diffusion_module(dm_weights)
    dev_ti = build_token_initializer(ti_weights)

    sampler = RFD3Sampler(num_timesteps=n_ts)
    seed = 42
    results = {}

    # 1) default (regression): captured init for BOTH backends -> clean DM-forward parity
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed)
        Xr, _ = sampler.sample(ref_dm, 1, L, coord_zero, f, init, is_motif, generator=g)
        g = torch.Generator().manual_seed(seed)
        Xd, _ = sampler.sample(dev_dm, 1, L, coord_zero, f, init, is_motif, generator=g)
    p = pcc(Xd, Xr); results["default"] = p
    print(f"[default ] final PCC={p:.6f} RMSD={rmsd(Xd, Xr):.4f} (ref std {Xr.std().item():.4f})")

    # 2) partial (F7): subset schedule, start from the real input structure.
    #    TI is step-invariant and partial_t does not touch TI, so captured init is valid for both.
    sched = sampler.noise_schedule(coord_zero.device)
    partial_t = float(sched[1])  # keep ~half the schedule -> a multi-step partial run
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed)
        Xr, tr = sampler.sample(ref_dm, 1, L, coord_real, f, init, is_motif,
                                  generator=g, partial_t=partial_t)
        g = torch.Generator().manual_seed(seed)
        Xd, trd = sampler.sample(dev_dm, 1, L, coord_real, f, init, is_motif,
                                  generator=g, partial_t=partial_t)
    p = pcc(Xd, Xr); results["partial"] = p
    worst = min(min(pcc(a["X_L"], b["X_L"]) for a, b in zip(tr, trd)), p)
    print(f"[partial ] final PCC={p:.6f} RMSD={rmsd(Xd, Xr):.4f} (partial_t={partial_t:.3f}, "
          f"{len(tr)} steps, worst-step={worst:.6f})")

    # 3) CFG: conditional + unconditional (cfg_features zeroed) ref pass, combined.
    #    The captured dsDNA_basic design has all cfg_features already zero, so CFG would
    #    be a no-op on it. To exercise the CFG mechanism (second forward + combine) we
    #    synthesize non-zero cfg_features on a copy of f; both backends see the SAME
    #    synthetic conditioning, so parity still isolates the device forward under f and
    #    f_ref. (A real non-zero-cfg design capture is queued for p7 to validate CFG on a
    #    physically-conditioned target.)
    g = torch.Generator().manual_seed(123)
    f_cfg = copy.deepcopy(f)
    f_cfg["ref_atomwise_rasa"] = torch.randint(0, 3, f["ref_atomwise_rasa"].shape, generator=g)
    f_cfg["active_donor"] = torch.randint(0, 2, f["active_donor"].shape, generator=g)
    f_cfg["active_acceptor"] = torch.randint(0, 2, f["active_acceptor"].shape, generator=g)
    f_cfg_ref = strip_f(f_cfg, cfg_features)
    # confirm the cfg features actually differ (so CFG is non-trivial)
    assert float((f_cfg["ref_atomwise_rasa"] - f_cfg_ref["ref_atomwise_rasa"]).abs().sum()) > 0
    # ref TI on f_cfg and f_cfg_ref (real weights) -> init for both backends' two passes
    with torch.no_grad():
        init_cfg = ref_ti(copy.deepcopy(f_cfg))
        ref_init_cfg_ref = ref_ti(copy.deepcopy(f_cfg_ref))
    dev_init_cfg = dev_ti(copy.deepcopy(f_cfg))
    dev_ref_init_cfg_ref = dev_ti(copy.deepcopy(f_cfg_ref))
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed)
        Xr, _ = sampler.sample(ref_dm, 1, L, coord_zero, f_cfg, init_cfg, is_motif, generator=g,
                                  cfg=True, cfg_scale=1.5, cfg_features=cfg_features,
                                  ref_initializer_outputs=ref_init_cfg_ref, f_ref=f_cfg_ref)
        g = torch.Generator().manual_seed(seed)
        Xd, _ = sampler.sample(dev_dm, 1, L, coord_zero, f_cfg, dev_init_cfg, is_motif, generator=g,
                                  cfg=True, cfg_scale=1.5, cfg_features=cfg_features,
                                  ref_initializer_outputs=dev_ref_init_cfg_ref, f_ref=f_cfg_ref)
    p = pcc(Xd, Xr); results["cfg"] = p
    print(f"[cfg     ] final PCC={p:.6f} RMSD={rmsd(Xd, Xr):.4f} (cfg_scale=1.5 [edm.yaml default], "
          f"synthetic non-zero cfg_features; device uses dev_ti(f_cfg_ref), ref uses "
          f"ref_ti(f_cfg_ref) for the unconditional pass)")

    ok = all(v >= 0.97 for v in results.values())
    print(f"\nSAMPLER PARITY: default={results['default']:.6f} "
          f"partial={results['partial']:.6f} cfg={results['cfg']:.6f}  -> {'OK' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(f"sampler parity failed: {results}")
    print("SAMPLER PARITY OK (device vs vendored torch reference, shared draws)")


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(__file__), "..", "..", ".scratch", "rfd3-ref", "goldens", "capture")
    cap = sys.argv[1] if len(sys.argv) > 1 else default
    n_ts = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    main(cap, n_ts)
