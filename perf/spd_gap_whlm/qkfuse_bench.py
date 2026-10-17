"""ESMC's q/k path after the qkv projection, today's chain against one fused RMSNorm + head split + RoPE op, vs float64.

Today (Attention._qk_heads + _rope): layer_norm(q), layer_norm(k), concat with v, nlp_create_qkv_heads, two
rotary_embedding calls. On the WH page cell that chain is ~35 ms of a ~128 ms 300M batch (cen5: LN, rope, heads, concat).

The fused arm rests on two exact rewrites of the weights:
* q_ln / k_ln have no bias, and LayerNorm(x W) = RMSNorm(x W_c) * g with W_c = W minus its mean over the output
  features (the row mean of x W_c is 0 for every x), so the norm needs only sum(q**2).
* RoPE pairs (i, i + hd/2) of a head; permuting each head's features so that pair sits at (2i, 2i+1) turns it into
  the adjacent-pair (llama) RoPE the fused op takes, and q.k is unchanged when q and k share the permutation.
Then wan_fused_rmsnorm_pre_allgather (sum q**2 per row) + wan_fused_rmsnorm_post_allgather (norm, gamma, head split,
RoPE) replace the chain. v keeps its own head split. The op's output is [1, H, S, hd], so a batch tries [B, 1, S, d].

    python3 perf/spd_gap_whlm/qkfuse_bench.py --out qkfuse.jsonl
"""
import argparse
import json
import math
import time

import torch
import ttnn

SHAPES = {"300m": (7, 960, 15), "600m": (7, 1152, 18), "6b": (1, 2560, 40)}   # batch, d_model, heads
S = 544


def rope64(S, hd):
    inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    f = torch.outer(torch.arange(S, dtype=torch.float64), inv)
    return f.cos(), f.sin()   # [S, hd/2]


def ref_heads(q, g, H, cos, sin):
    """float64 LayerNorm(q) * g (no bias), split into heads, rotate-half RoPE: [B, H, S, hd]."""
    q = q - q.mean(-1, keepdim=True)
    q = q / torch.sqrt((q * q).mean(-1, keepdim=True) + 1e-5) * g
    B, S_, d = q.shape
    q = q.view(B, S_, H, d // H).transpose(1, 2)
    h = d // H // 2
    x1, x2 = q[..., :h], q[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def err(got, ref):
    d = (got.double() - ref).flatten()
    return {"rel_l2": float(d.norm() / ref.flatten().norm()), "max_abs": float(d.abs().max())}


def timed(fn, dev, repeat, inner):
    fn()
    ts = []
    for _ in range(repeat):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / inner * 1e3)
    return round(sorted(ts)[len(ts) // 2], 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio.esmc import _rope, rope_tables
    dev = T.get_device()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t.contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    trans = torch.zeros(1, 1, 32, 32)
    trans[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
    trans[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1
    trans_tt = up(trans)
    out = open(args.out, "w")
    gen = torch.Generator().manual_seed(0)
    for name in args.shapes.split(","):
        B, d, H = SHAPES[name]
        hd = d // H
        x = torch.randn(B, S, d, generator=gen).to(torch.bfloat16)
        w = (torch.randn(d, d, generator=gen) / math.sqrt(d) + 0.02 * torch.randn(d, 1, generator=gen)).to(torch.bfloat16)
        g = (1 + 0.1 * torch.randn(d, generator=gen)).to(torch.bfloat16)
        q64 = x.double() @ w.double()
        c64, s64 = rope64(S, hd)
        ref = ref_heads(q64, g.double(), H, c64, s64)
        # per head: new 2i <- old i, new 2i+1 <- old i + hd/2; the inverse maps the fused output back
        ph = torch.stack([torch.arange(hd // 2), torch.arange(hd // 2) + hd // 2], 1).flatten()
        perm = (torch.arange(H)[:, None] * hd + ph[None]).flatten()
        inv = torch.argsort(ph)
        wc = w.double() - w.double().mean(1, keepdim=True)
        rec = {"shape": name, "bsd_h": [B, S, d, H]}
        x_tt = up(x)
        cos, sin = rope_tables(S, hd, device=dev)

        # today's chain, q and k from the same weights (so the timing covers both norms and both ropes)
        q = ttnn.linear(x_tt, up(w), compute_kernel_config=ck, dtype=ttnn.bfloat16)
        v = ttnn.clone(q)
        g_tt = up(g.view(1, d))

        def chain():
            qn = ttnn.layer_norm(q, weight=g_tt, epsilon=1e-5, compute_kernel_config=ck)
            kn = ttnn.layer_norm(q, weight=g_tt, epsilon=1e-5, compute_kernel_config=ck)
            qkv = ttnn.unsqueeze(ttnn.concat([qn, kn, v], dim=-1), 1)
            qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
                                                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for t in (qn, kn, qkv):
                ttnn.deallocate(t)
            qr, kr = _rope(qh, kh, cos, sin, l1=False)
            for t in (qh, kh):
                ttnn.deallocate(t)
            return qr, kr, vh

        qr, kr, vh = chain()
        rec["chain"] = err(ttnn.to_torch(qr).reshape(B, H, S, hd), ref)
        for t in (qr, kr, vh):
            ttnn.deallocate(t)
        rec["chain_ms"] = timed(lambda: [ttnn.deallocate(t) for t in chain()], dev, args.repeat, args.inner)

        # fused: centred, permuted weights; v keeps its plain head split
        qf = ttnn.linear(x_tt, up(wc[:, perm].to(torch.bfloat16)), compute_kernel_config=ck, dtype=ttnn.bfloat16)
        gp_tt = up(g[perm].view(1, d))
        cos_i = up(torch.repeat_interleave(c64, 2, -1).float().view(1, 1, S, hd))
        sin_i = up(torch.repeat_interleave(s64, 2, -1).float().view(1, 1, S, hd))
        for layout in ("b1sd", "1bsd"):
            qin = ttnn.reshape(qf, (B, 1, S, d) if layout == "b1sd" else (1, 1, B * S, d))

            def fused():
                outs = []
                for _ in range(2):   # q and k
                    st = ttnn.experimental.wan_fused_rmsnorm_pre_allgather(qin, compute_kernel_config=ck, dtype=ttnn.float32)
                    outs.append(ttnn.experimental.wan_fused_rmsnorm_post_allgather(
                        qin, st, epsilon=1e-5, num_heads_per_device=H, weight=gp_tt, transformation_mat=trans_tt,
                        rope_cos=cos_i, rope_sin=sin_i, compute_kernel_config=ck))
                    ttnn.deallocate(st)
                vh = ttnn.experimental.nlp_create_qkv_heads(ttnn.unsqueeze(ttnn.concat([v, v, v], dim=-1), 1), num_heads=H,
                                                            num_kv_heads=H, transpose_k_heads=False,
                                                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
                return outs + list(vh)

            try:
                o = fused()
                shp = list(o[0].shape)
                got = ttnn.to_torch(o[0]).double()
                got = got.reshape(B, H, S, hd) if layout == "b1sd" else got.reshape(H, B, S, hd).transpose(0, 1)
                rec[f"fused_{layout}"] = {"out_shape": shp, **err(got[..., inv], ref)}
                for t in o:
                    ttnn.deallocate(t)
                rec[f"fused_{layout}_ms"] = timed(lambda: [ttnn.deallocate(t) for t in fused()], dev, args.repeat, args.inner)
            except Exception as exc:   # an unsupported shape is a result here, not a crash
                rec[f"fused_{layout}"] = {"error": str(exc).splitlines()[0][:300]}
            if B == 1:
                break
        # v alone to [B, H, S, hd]: what the fused arm needs beside q and k (above it pays a 3x concat as an upper bound)
        vsplits = {
            "concat3_nlp": lambda: ttnn.experimental.nlp_create_qkv_heads(
                ttnn.unsqueeze(ttnn.concat([v, v, v], dim=-1), 1), num_heads=H, num_kv_heads=H, transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)[2],
            "reshape_permute": lambda: ttnn.permute(ttnn.reshape(v, (B, S, H, hd)), (0, 2, 1, 3)),
            "sep_tensors": lambda: ttnn.experimental.create_qkv_heads_from_separate_tensors(
                v, ttnn.concat([v, v], dim=-1), num_heads=H, num_kv_heads=H, transpose_k_heads=False)[1],
        }
        vref = ttnn.to_torch(v).reshape(B, S, H, hd).transpose(1, 2)
        for vn, fn in vsplits.items():
            try:
                ok = bool(torch.equal(ttnn.to_torch(fn()).reshape(B, H, S, hd), vref))
                rec[f"v_{vn}"] = {"exact": ok, "ms": timed(lambda fn=fn: ttnn.deallocate(fn()), dev, args.repeat, args.inner)}
            except Exception as exc:
                rec[f"v_{vn}"] = {"error": str(exc).splitlines()[0][:200]}
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
