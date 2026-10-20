"""SaProt's RoPE + head split: today's nlp_create_qkv_heads + rotary_embedding against qk_rope_heads.rope_heads_packed,
both vs float64 on the same bf16 packed qkv (packed-sfpu: the RoPE sums on the SFPU, qk_rope_heads.SFPU). Reports rel_l2 / max_abs of q and k heads, whether v is exact, median ms,
and the signed bias mean((got - ref) * sign(ref)) / mean(|ref|) (negative: the arm shrinks |q|, |k|, e.g. by truncating),
against float64 math on the float64 tables and on the bf16 tables the device reads (the kernel's own arithmetic).

    python3 perf/spd_gap_whlm/rope_packed_bench.py --out rope_packed.jsonl
"""
import argparse
import json

import torch
import ttnn

from qkfuse_bench import S, err, rope64, timed

SHAPES = {"saprot650m": (7, 1280, 20)}   # batch, d_model, heads (1.3B has the same width)


def ref_rope(x, H, cos, sin):
    B, S_, d = x.shape
    x = x.view(B, S_, H, d // H).transpose(1, 2)
    h = d // H // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def bias(got, ref):
    return float(((got - ref) * ref.sign()).mean() / ref.abs().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio.esmc import _rope, rope_tables
    from tt_bio import qk_rope_heads as qkrh
    from tt_bio.qk_rope_heads import rope_heads_packed
    dev = T.get_device()
    out = open(args.out, "w")
    gen = torch.Generator().manual_seed(0)
    for name, (B, d, H) in SHAPES.items():
        hd = d // H
        qkv = (2 * torch.randn(B, S, 3 * d, generator=gen)).to(torch.bfloat16)
        qkv_tt = ttnn.from_torch(qkv, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        x64 = qkv.double()
        c64, s64 = rope64(S, hd)
        ref = [ref_rope(x64[..., :d], H, c64, s64), ref_rope(x64[..., d:2 * d], H, c64, s64),
               x64[..., 2 * d:].view(B, S, H, hd).transpose(1, 2)]
        cos, sin = rope_tables(S, hd, device=dev)
        cb, sb = (ttnn.to_torch(t).double().reshape(S, hd)[:, : hd // 2] for t in (cos, sin))   # halves are equal
        refb = [ref_rope(x64[..., :d], H, cb, sb), ref_rope(x64[..., d:2 * d], H, cb, sb)]

        def chain():
            t = ttnn.unsqueeze(qkv_tt, 1)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(t, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
                                                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
            qr, kr = _rope(q, k, cos, sin, l1=False)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            return qr, kr, v

        rec = {"shape": name, "bsd_h": [B, S, d, H]}
        def packed(sfpu):
            qkrh.SFPU = sfpu
            return rope_heads_packed(qkv_tt, cos, sin)

        for arm, fn in (("chain", chain), ("packed", lambda: packed(False)), ("packed-sfpu", lambda: packed(True))):
            o = fn()
            got = [ttnn.to_torch(t).double().reshape(B, H, S, hd) for t in o]
            rec[arm] = {"q": err(got[0], ref[0]), "k": err(got[1], ref[1]), "v_exact": bool(torch.equal(got[2], ref[2])),
                        "bias": [bias(got[i], ref[i]) for i in (0, 1)],
                        "q_bf16tab": err(got[0], refb[0]), "bias_bf16tab": [bias(got[i], refb[i]) for i in (0, 1)]}
            for t in o:
                ttnn.deallocate(t)
            rec[arm]["ms"] = timed(lambda fn=fn: [ttnn.deallocate(t) for t in fn()], dev, args.repeat, args.inner)
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
