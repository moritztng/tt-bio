"""SaProt's qkv projection written straight into per-head q, k, v ([B, H, L, 64]) by the minimal_matmul writer
(kernels/mm_gate, MM_HEADS_ST), against minimal_matmul + nlp_create_qkv_heads. The page shape, the shipped WH qkv
block config, with the qkv bias; the output should be bit-identical and the split op's time gone. The rope arm
(tt_bio/mm_rope: bias, NeoX RoPE and head split in the epilogue, permuted weight) is checked against today's path,
minimal_matmul + qk_rope_heads.rope_heads_packed; under QKRH_SFPU and HiFi4 it should be bit-identical too.
f64: every RoPE arm (chain = nlp_create_qkv_heads + esmc._rope, packed with FPU / SFPU sums, mm_rope) against float64 math
on the same bf16 x and the weight as the device holds it: rel_l2 per tensor, per-head worst q / k, and the signed bias
mean((got - ref) * sign(ref)) / mean(|ref|) (negative: |q|, |k| come out small). --fid picks the matmul fidelity.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/mm_rope_check.py --model saprot-650m [--fast] --out check.jsonl
"""
import argparse
import json
import math
import time

import torch
import ttnn

MODELS = {"saprot-650m": (1280, 20), "saprot-1.3b": (1280, 20)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m", choices=sorted(MODELS))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--cfg", default=None, help="qkv minimal_matmul blocks; default saprot._PROGS' for this arch")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--fid", choices=("HiFi2", "HiFi3", "HiFi4"), default=None, help="default HiFi3 --fast, HiFi4 normal")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import tenstorrent as T, mm_generic as MG, mm_gate, mm_rope as MR
    from tt_bio import qk_rope_heads as QK, esmc
    T.set_fast_mode(a.fast)
    dev = T.get_device()
    d, H = MODELS[a.model]
    B, L = 7, 544
    from tt_bio import saprot
    cfg = (tuple(int(v) for v in a.cfg.split(",")) if a.cfg else saprot._PROGS[(T.is_wormhole(), d, 3 * d)][1])
    grid = tuple(T.COMPUTE_GRID_MAIN)
    gen = torch.Generator().manual_seed(0)
    x = ttnn.from_torch(torch.randn(B, L, d, generator=gen).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    wt = (torch.randn(d, 3 * d, generator=gen) / math.sqrt(d)).to(torch.bfloat16)
    bt = (torch.randn(1, 3 * d, generator=gen) * 0.1).to(torch.bfloat16)
    wdt = ttnn.bfloat8_b if a.fast else ttnn.bfloat16
    w = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt)
    b = ttnn.from_torch(bt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    nb = cfg[2]
    nc = MR.cols(3 * d // 32, grid[0], nb)   # Blackhole's 11 / 13 columns split no whole heads: it runs 10
    wp = ttnn.from_torch(MR.permute(wt, nc, nb), layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt)
    bp = ttnn.from_torch(MR.permute(bt, nc, nb), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
    fr = torch.outer(torch.arange(L).float(), inv)
    emb = torch.cat([fr, fr], -1)[None, None]
    cos = ttnn.from_torch(emb.cos().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    sin = ttnn.from_torch(emb.sin().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    fid = getattr(ttnn.MathFidelity, a.fid or ("HiFi3" if a.fast else "HiFi4"))
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True,
                                          packer_l1_acc=True)
    mcfg = ttnn.MinimalMatmulConfig(M_block_size=cfg[0], K_block_size=cfg[1], N_block_size=cfg[2], subblock_h=cfg[3],
                                    subblock_w=cfg[4], compute_with_storage_grid_size=ttnn.CoreCoord(*grid))

    def ref():
        qkv = ttnn.experimental.minimal_matmul(x, w, bias_tensor=b, compute_kernel_config=ck, config=mcfg,
                                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(ttnn.unsqueeze(qkv, 1), num_heads=H, num_kv_heads=H,
                                                          transpose_k_heads=False,
                                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qkv)
        return q, k, v

    def heads():
        outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([B, H, L, 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                               ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
        return MG.generic_minimal_matmul(dev, x, w, outs, (cfg, grid), MG.ckc_args(ck),
                                         defines={"MM_HEADS_ST": L // 32}, kernel_dir=mm_gate.KERNEL_DIR, bias=b)

    def plain():
        return MG.generic_minimal_matmul(dev, x, w, [ttnn.allocate_tensor_on_device(
            ttnn.Shape([B, L, 3 * d]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)],
            (cfg, grid), MG.ckc_args(ck), kernel_dir=mm_gate.KERNEL_DIR, bias=b)

    def ref_rope():   # today's SaProt path: minimal_matmul with bias, then qk_rope_heads' packed RoPE + head split
        qkv = ttnn.experimental.minimal_matmul(x, w, bias_tensor=b, compute_kernel_config=ck, config=mcfg,
                                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = QK.rope_heads_packed(qkv, cos, sin)
        ttnn.deallocate(qkv)
        return o

    def chain():   # Blackhole normal's path today: minimal_matmul with bias, nlp_create_qkv_heads, two rotary_embedding
        qkv = ttnn.experimental.minimal_matmul(x, w, bias_tensor=b, compute_kernel_config=ck, config=mcfg,
                                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(ttnn.unsqueeze(qkv, 1), num_heads=H, num_kv_heads=H,
                                                          transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qkv)
        qr, kr = esmc._rope(q, k, cos, sin, l1=False)
        ttnn.deallocate(q); ttnn.deallocate(k)
        return qr, kr, v

    def rope():
        return MR.qkv_rope_heads(x, wp, bp, cos, sin, H, cfg, (nc, grid[1]), ck)

    out = open(a.out, "a")
    r = [ttnn.to_torch(t) for t in ref()]
    h = [ttnn.to_torch(t) for t in heads()]
    p = ttnn.to_torch(plain())
    rq = torch.cat([t.permute(0, 2, 1, 3).reshape(B, L, d) for t in r], -1)
    rec = dict(model=a.model, fast=a.fast, cfg=list(cfg), arch="wh" if T.is_wormhole() else "bh", heads_equal=[bool(torch.equal(u, v)) for u, v in zip(r, h)],
               heads_maxdiff=[float((u.float() - v.float()).abs().max()) for u, v in zip(r, h)],
               plain_vs_ref_equal=bool(torch.equal(p, rq)), qkrh_sfpu=QK._sfpu())
    rr = [ttnn.to_torch(t) for t in ref_rope()]
    ro = rope()
    if ro is None:
        rec["rope_declined"] = MR.plan(x, wp, cfg, grid, L)
    else:
        ro = [ttnn.to_torch(t) for t in ro]
        rec.update(rope_equal=[bool(torch.equal(u, v)) for u, v in zip(rr, ro)],
                   rope_maxdiff=[float((u.float() - v.float()).abs().max()) for u, v in zip(rr, ro)],
                   rope_ndiff=[int((u != v).sum()) for u, v in zip(rr, ro)])
    f64_arms = {"chain": (chain, None), "packed_fpu": (ref_rope, False), "packed_sfpu": (ref_rope, True)}
    if ro is not None:
        f64_arms["mm_rope"] = (rope, None)
    rec.update(fid=fid.name, qkrh_sfpu_default=QK._sfpu(), f64=f64_errors(x, w, bt, d, H, L, f64_arms, QK))
    arms = (("ref", ref), ("heads", heads), ("plain", plain), ("ref_rope", ref_rope)) + (
        (("rope", rope),) if ro is not None else ())
    for name, fn in arms:
        for _ in range(2):
            o = fn()
            for t in (o if isinstance(o, (list, tuple)) else [o]):
                ttnn.deallocate(t)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(a.calls):
            o = fn()
            for t in (o if isinstance(o, (list, tuple)) else [o]):
                ttnn.deallocate(t)
        ttnn.synchronize_device(dev)
        rec[f"{name}_ms"] = round((time.perf_counter() - t0) / a.calls * 1e3, 4)
    print(json.dumps(rec), flush=True)
    out.write(json.dumps(rec) + "\n")
    return 0


def f64_errors(x, w, bt, d, H, L, arms, QK) -> dict:
    """Each RoPE arm's q, k, v ({name: (fn, qk_rope_heads.SFPU or None to keep)}) against float64 RoPE of float64
    (x W + b) on the device's own x and W values."""
    x64, w64 = ttnn.to_torch(x).double(), ttnn.to_torch(w).double()
    qkv = x64 @ w64 + bt.double()
    B, hd = x64.shape[0], d // H
    inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    f = torch.outer(torch.arange(L, dtype=torch.float64), inv)
    c, s = f.cos(), f.sin()

    def heads(t, rot):
        t = t.reshape(B, L, H, hd).transpose(1, 2)
        if not rot:
            return t
        t1, t2 = t[..., : hd // 2], t[..., hd // 2:]
        return torch.cat([t1 * c - t2 * s, t2 * c + t1 * s], -1)
    want = [heads(qkv[..., :d], True), heads(qkv[..., d:2 * d], True), heads(qkv[..., 2 * d:], False)]

    def grade(o):
        got = [ttnn.to_torch(t).double().reshape(B, H, L, hd) for t in o]
        for t in o:
            ttnn.deallocate(t)
        r = {}
        for n, g, ref in zip("qkv", got, want):
            dlt = g - ref
            r[n] = {"rel_l2": float(dlt.norm() / ref.norm()),
                    "bias": float((dlt * ref.sign()).mean() / ref.abs().mean())}
            if n != "v":
                ph = (dlt.pow(2).sum((0, 2, 3)) / ref.pow(2).sum((0, 2, 3))).sqrt()
                r[n]["head_max"] = float(ph.max())
        return r

    keep, out = QK.SFPU, {}
    for name, (fn, sfpu) in arms.items():
        QK.SFPU = keep if sfpu is None else sfpu
        out[name] = grade(fn())
    QK.SFPU = keep
    return out


if __name__ == "__main__":
    raise SystemExit(main())
