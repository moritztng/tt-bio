#!/usr/bin/env python3
"""Token-DiT AttentionPairBias chain, arm by arm, at production shapes.

Shapes come from W10's diffusion ledger (state/perfwar/diffusion-ledger-298aa.md §3/§6):

  protenix-v2  NT=320, H=16, head_dim 48 padded to 64, FLOAT32,  bias [1,16,320,320] fp32
  opendde      NT=608, H=16, head_dim 48 padded to 64, BFLOAT16, bias [1,16,608,608] bf16

opendde reuses Protenix-v2's diffusion stack verbatim on the structural-token axis, so both models
run the same code; they take different branches of ``AttentionPairBias.__call__`` only because the
DiT dtype differs (``self.dtype == ttnn.float32 and self.fp32_raw_matmul_attention``).

No model and no checkpoint: q/k/v/bias are random at the exact production shape, dtype and memory
config, so an arm's number here is an op-chain time, not a fold time. Every timed region
synchronises on both sides. The pair bias is precomputed once per fold in production
(``protenix.py:_dit_block_biases``) and replayed for all 200 steps, so it is built outside the
timed region here too.

    TT_MESH_GRAPH_DESC_PATH=<ttnn>/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:perfwar-dit-attention-fusion PYTHONPATH=$PWD \
      python3 perf/dit_attn/chain_ab.py --model protenix-v2 --out perf/dit_attn/chain_protenix-v2_c0.json
"""
import argparse, json, statistics as st, time

import torch
import ttnn

from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

SHAPES = {
    "protenix-v2": dict(n=320, dtype=ttnn.float32, tdtype=torch.float32),
    "opendde": dict(n=608, dtype=ttnn.bfloat16, tdtype=torch.bfloat16),
}
H, HEAD_DIM, PAD_HEAD_DIM = 16, 48, 64
SCALE = HEAD_DIM ** -0.5

dev = get_device()
CKC = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True,
)


def timed(fn, warm=4, pipe=5, reps=7):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
    return st.median(out) * 1e6


def compare(a, b):
    a64, b64 = a.double().flatten(), b.double().flatten()
    eq = bool(torch.equal(a.float().flatten(), b.float().flatten()))
    d = a64 - b64
    rmsd, std = float(torch.sqrt((d * d).mean())), float(a64.std())
    ac, bc = a64 - a64.mean(), b64 - b64.mean()
    return eq, rmsd / std if std else float("nan"), float((ac * bc).sum() / (ac.norm() * bc.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SHAPES))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = SHAPES[a.model]
    N, DT, TDT = cfg["n"], cfg["dtype"], cfg["tdtype"]
    fp32 = DT == ttnn.float32
    torch.manual_seed(0)

    def up(t, dtype=DT):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)

    q_t, k_t, v_t = (torch.randn(1, H, N, PAD_HEAD_DIM) * 0.5 for _ in range(3))
    for t in (q_t, k_t, v_t):
        t[..., HEAD_DIM:] = 0.0        # the qkv weight is zero-padded 48 -> 64 in production
    z_t = torch.randn(1, H, N, N) * (HEAD_DIM ** 0.5) * 0.3   # sqrt(head_dim)-baked, as z_weight is

    q, k, v = up(q_t.to(TDT)), up(k_t.to(TDT)), up(v_t.to(TDT))
    z = up(z_t.to(TDT))
    z_pre = ttnn.multiply(z, SCALE)                 # bias with the sqrt(head_dim) bake undone
    z_pre_rs = ttnn.reshape(z_pre, (H, 1, N, N))    # head axis -> batch axis, for scale_mask_softmax
    kT = ttnn.permute(k, (0, 1, 3, 2))              # what nlp_create_qkv_heads(transpose_k_heads=True) hands back
    q_scaled = ttnn.multiply(q, SCALE)              # what folding head_dim**-0.5 into qkv_weight gives
    print(f"model={a.model} N={N} dtype={DT}", flush=True)

    arms = {}

    def arm(name):
        def deco(f):
            arms[name] = f
            return f
        return deco

    def tail(sc, scale=True):
        if scale:
            sc = ttnn.multiply(sc, SCALE)
        sc = ttnn.add(sc, z_pre)
        p = ttnn.softmax(sc, dim=-1)
        ttnn.deallocate(sc)
        return p

    def pv(p, **kw):
        o = ttnn.matmul(p, v, compute_kernel_config=CKC, **kw)
        ttnn.deallocate(p)
        return o

    GRID = dict(core_grid=CORE_GRID_MAIN)

    if fp32:
        @arm("prod")                                   # tenstorrent.py:1645-1657
        def _():
            zz = ttnn.multiply(z, SCALE)
            kt = ttnn.permute(k, (0, 1, 3, 2))
            sc = ttnn.matmul(q, kt, compute_kernel_config=CKC)
            ttnn.deallocate(kt)
            sc = ttnn.multiply(sc, SCALE)
            sc = ttnn.add(sc, zz)
            ttnn.deallocate(zz)
            p = ttnn.softmax(sc, dim=-1)
            ttnn.deallocate(sc)
            return pv(p)
    else:
        @arm("prod")                                   # tenstorrent.py:1669-1680
        def _():
            kt = ttnn.transpose(k, -2, -1)
            lg = ttnn.matmul(q, kt, compute_kernel_config=CKC)
            ttnn.deallocate(kt)
            lg = ttnn.add(lg, z)
            lg = ttnn.multiply(lg, SCALE)
            p = ttnn.softmax(lg, dim=-1, compute_kernel_config=CKC)
            ttnn.deallocate(lg)
            return pv(p)

    @arm("A_prescale_kt")        # hoist the per-call bias scale + take kT from the head split
    def _():
        return pv(tail(ttnn.matmul(q, kT, compute_kernel_config=CKC)))

    @arm("B_A_pvgrid")           # + core_grid=CORE_GRID_MAIN on probs@v
    def _():
        return pv(tail(ttnn.matmul(q, kT, compute_kernel_config=CKC)), **GRID)

    @arm("C_B_qkgrid")           # + core_grid on q@kT too
    def _():
        return pv(tail(ttnn.matmul(q, kT, compute_kernel_config=CKC, **GRID)), **GRID)

    @arm("D_B_qfold")            # + head_dim**-0.5 folded into q (foldable into qkv_weight: 0 ops)
    def _():
        return pv(tail(ttnn.matmul(q_scaled, kT, compute_kernel_config=CKC), scale=False), **GRID)

    @arm("E_D_qkgrid")
    def _():
        return pv(tail(ttnn.matmul(q_scaled, kT, compute_kernel_config=CKC, **GRID), scale=False),
                  **GRID)

    @arm("K_sms_reshape")        # KILLED: fused scale+mask+softmax via the head->batch reshape
    def _():
        sc = ttnn.matmul(q, kT, compute_kernel_config=CKC)
        sc_rs = ttnn.reshape(sc, (H, 1, N, N))
        p_rs = ttnn.scale_mask_softmax_in_place(sc_rs, SCALE, z_pre_rs)
        return pv(ttnn.reshape(p_rs, (1, H, N, N)), **GRID)

    @arm("K_logits_l1")          # KILLED: logits held in L1
    def _():
        sc = ttnn.matmul(q, kT, compute_kernel_config=CKC, memory_config=ttnn.L1_MEMORY_CONFIG)
        return pv(tail(sc))

    @arm("K_pv_batch_reshape")   # probs@v with the head axis moved to batch, no core_grid
    def _():
        p = tail(ttnn.matmul(q, kT, compute_kernel_config=CKC))
        p_rs = ttnn.reshape(p, (H, 1, N, N))
        o = ttnn.matmul(p_rs, ttnn.reshape(v, (H, 1, N, PAD_HEAD_DIM)), compute_kernel_config=CKC)
        ttnn.deallocate(p_rs)
        ttnn.deallocate(p)
        return ttnn.reshape(o, (1, H, N, PAD_HEAD_DIM))

    @arm("sdpa")                 # release-gated: ttnn SDPA, the arithmetic ba6ede96 removed
    def _():
        if fp32:
            qb, kb, vb, zb = (ttnn.typecast(t, ttnn.bfloat16, memory_config=t.memory_config())
                              for t in (q, k, v, z))
        else:
            qb, kb, vb, zb = q, k, v, z
        ob = ttnn.transformer.scaled_dot_product_attention(
            qb, kb, vb, attn_mask=zb, is_causal=False, scale=SCALE)
        if fp32:
            for t in (qb, kb, vb, zb):
                ttnn.deallocate(t)
            o = ttnn.typecast(ob, ttnn.float32, memory_config=ob.memory_config())
            ttnn.deallocate(ob)
            return o
        return ob

    if fp32:
        # Numerics control for the SDPA arm: the SAME unfused chain, bf16 operands. Whatever
        # error this leaves is the fp32->bf16 crossing; SDPA's excess over it is SDPA's own.
        qb16 = ttnn.typecast(q, ttnn.bfloat16)
        kTb16 = ttnn.typecast(kT, ttnn.bfloat16)
        vb16 = ttnn.typecast(v, ttnn.bfloat16)
        zb16 = ttnn.typecast(z_pre, ttnn.bfloat16)

        @arm("ctrl_unfused_bf16")
        def _():
            sc = ttnn.matmul(qb16, kTb16, compute_kernel_config=CKC)
            sc = ttnn.multiply(sc, SCALE)
            sc = ttnn.add(sc, zb16)
            p = ttnn.softmax(sc, dim=-1)
            ttnn.deallocate(sc)
            o = ttnn.matmul(p, vb16, compute_kernel_config=CKC, **GRID)
            ttnn.deallocate(p)
            return ttnn.typecast(o, ttnn.float32)

    res, outs = {}, {}
    for name, fn in arms.items():
        try:
            o = fn()
            outs[name] = ttnn.to_torch(o).float()
            ttnn.deallocate(o)
            res[name] = timed(fn)
            print(f"{name:20s} {res[name]:9.2f} us", flush=True)
        except Exception as e:
            res[name] = None
            print(f"{name:20s} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)

    base, ref = res["prod"], outs["prod"]
    print("\narm                     us   x(prod)  equal      rmsd/std       pcc", flush=True)
    table = {}
    for name in arms:
        if res.get(name) is None:
            continue
        eq, r, p = compare(ref, outs[name])
        table[name] = dict(us=res[name], ratio=base / res[name], equal=eq, rmsd_over_std=r, pcc=p)
        print(f"{name:20s} {res[name]:8.2f} {base/res[name]:8.3f}  {str(eq):6s} {r:12.6f} {p:11.7f}",
              flush=True)
    if a.out:
        json.dump(dict(model=a.model, n=N, n_heads=H, head_dim=HEAD_DIM,
                       pad_head_dim=PAD_HEAD_DIM, dtype=str(DT), calls_per_fold=4800, arms=table),
                  open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
