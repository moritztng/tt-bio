#!/usr/bin/env python3
"""Screen the unbuilt rollout invariants named in state/openfold3-512aa-deep-perf.md section 5.

Times the EXACT op sequences that would be hoisted, at the real 512 aa shapes and the real
production dtype (OF3_DIFFUSION_FP32_DEVICE=1 -> fp32 inside the diffusion module), so the
prediction is a screen of the code that gets hoisted rather than of a proxy.

Groups, and how many times each runs per rollout today:
  npe_single_bcast    200x  (OF3NoisyPositionEmbedder: cl  = cl0 + bcast(linear_s(LN(si_trunk))))
  npe_pair_bcast      200x  (OF3NoisyPositionEmbedder: plm = plm0 + to_blocks(linear_z(LN(zij))))
  enc_pair_update     200x  (OF3DiffusionModule: cl_pad/cl_l/cl_m -> linear_l+linear_m -> pair_mlp)
  at_zbias            400x  (OF3AtomTransformer: layer_norm_z + 3x linear_z + permute; enc AND dec)
  at_sq_sk            400x  (OF3AtomTransformer: s query blocks + s key gather)
  at_ada_cg           400x  (OF3AtomTransformer: 3x linear_ada_out(s) + 3x linear_g(s))

A hoisted group runs once per rollout instead of N times, so the predicted saving is
(N - 1) / N of its measured total.
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T

    # Real 512 aa geometry, from the decomposition run.
    n_token, n_tok_pad = 512, 512
    n_atom, NP, nb = 4116, 4128, 129
    NQ, NK, C, CZ, CS, CT = 32, 128, 128, 16, 384, 768

    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    DT = ttnn.float32  # OF3_DIFFUSION_FP32_DEVICE=1 production default

    def dv(shape, dtype=DT, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(torch.randn(*shape), layout=layout, device=dev, dtype=dtype)

    def lin(x, w, activation=None):
        return ttnn.linear(x, w, activation=activation, compute_kernel_config=ckc,
                           core_grid=T.CORE_GRID_MAIN)

    # ---- tensors ----
    si_trunk = dv((1, n_tok_pad, CS))
    zij      = dv((1, n_tok_pad, n_tok_pad, C))
    cl0      = dv((1, NP, C))
    plm0     = dv((1, nb, NQ, NK, CZ))
    atom_mask_col = dv((1, NP, 1))
    zij_mask = dv((1, nb, NQ, NK, 1))
    valid_mask = dv((1, nb, NK, 1))
    pair_mask = dv((1, nb, NQ, NK, 1))
    cl_dev   = dv((1, NP, C))
    plm_dev  = dv((1, nb, NQ, NK, CZ))

    idx_atom = ttnn.from_torch(torch.randint(0, n_token, (1, NP)).int(),
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    idx_zij = ttnn.from_torch(torch.randint(0, n_tok_pad * n_tok_pad, (1, nb * NQ * NK)).int(),
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    idx_key = ttnn.from_torch(torch.randint(0, NP, (1, nb * NK)).int(),
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)

    # ---- weights ----
    w_ls_npe = dv((CS, C)); ln_s_w = dv((CS,))
    w_lz_npe = dv((C, CZ)); ln_z_w = dv((C,))
    w_ll = dv((C, CZ)); w_lm = dv((C, CZ))
    w_pm = [dv((CZ, CZ)) for _ in range(3)]
    ln_zat_w = dv((CZ,))
    w_lz_at = [dv((CZ, 4)) for _ in range(3)]          # linear_z -> H=4 heads
    w_ada = [dv((C, C)) for _ in range(3)]
    b_ada = [dv((C,)) for _ in range(3)]
    w_cg = [dv((C, C)) for _ in range(3)]
    b_cg = [dv((C,)) for _ in range(3)]

    def gather(x2d, idx, shape):
        g = ttnn.embedding(idx, x2d, layout=ttnn.ROW_MAJOR_LAYOUT,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.to_layout(ttnn.reshape(g, shape), ttnn.TILE_LAYOUT)

    # ---------------- the six groups ----------------
    def npe_single_bcast():
        si_ln = ttnn.layer_norm(si_trunk, weight=ln_s_w, epsilon=1e-5, compute_kernel_config=ckc)
        si_tok = lin(si_ln, w_ls_npe)
        bf = ttnn.typecast(si_tok, ttnn.bfloat16)
        t2d = ttnn.reshape(ttnn.to_layout(bf, ttnn.ROW_MAJOR_LAYOUT), (n_tok_pad, C))
        at = gather(t2d, idx_atom, (1, NP, C))
        at = ttnn.typecast(at, DT)
        at = ttnn.multiply(at, atom_mask_col)
        out = ttnn.add(cl0, at)
        for t in (si_ln, si_tok, bf, at):
            ttnn.deallocate(t)
        return out

    def npe_pair_bcast():
        z_ln = ttnn.layer_norm(zij, weight=ln_z_w, epsilon=1e-5, compute_kernel_config=ckc)
        z_tok = lin(z_ln, w_lz_npe)
        bf = ttnn.typecast(z_tok, ttnn.bfloat16)
        t2d = ttnn.reshape(ttnn.to_layout(bf, ttnn.ROW_MAJOR_LAYOUT), (n_tok_pad * n_tok_pad, CZ))
        blk = gather(t2d, idx_zij, (1, nb, NQ, NK, CZ))
        blk = ttnn.typecast(blk, DT)
        blk = ttnn.multiply(blk, zij_mask)
        out = ttnn.add(plm0, blk)
        for t in (z_ln, z_tok, bf, blk):
            ttnn.deallocate(t)
        return out

    def enc_pair_update():
        cl_l = ttnn.to_layout(cl_dev, ttnn.ROW_MAJOR_LAYOUT)
        cl_l = ttnn.to_layout(ttnn.reshape(cl_l, (1, nb, NQ, C)), ttnn.TILE_LAYOUT)
        bf = ttnn.typecast(cl_dev, ttnn.bfloat16)
        cl2d = ttnn.reshape(ttnn.to_layout(bf, ttnn.ROW_MAJOR_LAYOUT), (NP, C))
        cl_m = gather(cl2d, idx_key, (1, nb, NK, C))
        cl_m = ttnn.typecast(cl_m, DT)
        cl_m = ttnn.multiply(cl_m, valid_mask)
        ll = ttnn.unsqueeze(lin(ttnn.relu(cl_l), w_ll), -2)
        lm = ttnn.unsqueeze(lin(ttnn.relu(cl_m), w_lm), -3)
        cl_lm = ttnn.multiply(ttnn.add(ll, lm), pair_mask)
        plm = ttnn.add(plm_dev, cl_lm)
        pm = plm
        for w in w_pm:
            pm = lin(ttnn.relu(pm), w)
        plm = ttnn.multiply(ttnn.add(plm, pm), pair_mask)
        for t in (cl_l, bf, cl_m, ll, lm, cl_lm, pm):
            ttnn.deallocate(t)
        return plm

    def at_zbias():
        z_ln = ttnn.layer_norm(plm_dev, weight=ln_zat_w, epsilon=1e-5, compute_kernel_config=ckc)
        outs = []
        for b in range(3):
            zb = lin(z_ln, w_lz_at[b])
            zb = ttnn.to_layout(zb, ttnn.ROW_MAJOR_LAYOUT)
            zb = ttnn.permute(ttnn.reshape(zb, (1, nb, NQ, NK, 4)), (0, 1, 4, 2, 3))
            outs.append(ttnn.to_layout(zb, ttnn.TILE_LAYOUT))
        ttnn.deallocate(z_ln)
        return outs

    def at_sq_sk():
        s_q = ttnn.to_layout(cl_dev, ttnn.ROW_MAJOR_LAYOUT)
        s_q = ttnn.to_layout(ttnn.reshape(s_q, (1, nb, NQ, C)), ttnn.TILE_LAYOUT)
        bf = ttnn.typecast(cl_dev, ttnn.bfloat16)
        s2d = ttnn.reshape(ttnn.to_layout(bf, ttnn.ROW_MAJOR_LAYOUT), (NP, C))
        s_k = gather(s2d, idx_key, (1, nb, NK, C))
        s_k = ttnn.typecast(s_k, DT)
        s_k = ttnn.multiply(s_k, valid_mask)
        ttnn.deallocate(bf)
        return s_q, s_k

    def at_ada_cg():
        outs = []
        for b in range(3):
            outs.append(ttnn.linear(cl_dev, w_ada[b], bias=b_ada[b],
                                    compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN))
            outs.append(ttnn.linear(cl_dev, w_cg[b], bias=b_cg[b],
                                    compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN))
        return outs

    GROUPS = [
        ("npe_single_bcast", npe_single_bcast, 200),
        ("npe_pair_bcast", npe_pair_bcast, 200),
        ("enc_pair_update", enc_pair_update, 200),
        ("at_zbias", at_zbias, 400),
        ("at_sq_sk", at_sq_sk, 400),
        ("at_ada_cg", at_ada_cg, 400),
    ]

    def free(o):
        if isinstance(o, (list, tuple)):
            for x in o:
                free(x)
        elif isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)

    res = {}
    for name, fn, ncalls in GROUPS:
        for _ in range(a.warmup):
            free(fn())
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(a.reps):
            free(fn())
        ttnn.synchronize_device(dev)
        per_call = (time.perf_counter() - t0) / a.reps
        saving = per_call * (ncalls - 1)
        res[name] = {"ms_per_call": round(per_call * 1e3, 4),
                     "calls_today": ncalls,
                     "predicted_saving_s": round(saving, 4)}
        print(f"{name:20s} {per_call*1e3:8.3f} ms x {ncalls:4d}  -> saves {saving:7.3f} s", flush=True)

    total = sum(v["predicted_saving_s"] for v in res.values())
    print(f"\nTOTAL predicted saving: {total:.3f} s")
    out = {"host": os.uname().nodename, "ttnn": getattr(ttnn, "__version__", "0.68.0"),
           "dtype": str(DT), "reps": a.reps,
           "geometry": {"n_token": n_token, "n_atom": n_atom, "NP": NP, "nb": nb},
           "groups": res, "total_predicted_saving_s": round(total, 3)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
