"""ESMC's RoPE at the page shape, per kernel, against float64.

ESMC rotates half-split pairs (i, i + d/2). The llama kernel rotates interleaved pairs (2i, 2i + 1); permuting each
head's q and k columns the same way turns one into the other and leaves q.k unchanged, so on the model side the
permutation would live in Wq/Wk and the q/k LayerNorm weights at load. Here the input is permuted on the host and the
output permuted back before the comparison.

Variants: rot (ttnn.experimental.rotary_embedding, shipped) in DRAM and L1; llama (rotary_embedding_llama, prefill) on
[B, H, L, d] and on [1, B*H, L, d]; mm (x*cos + (x @ R)*sin, R the 64 x 64 rotate-half matrix).

    python3 perf/spd_gap_whlm/rope_bench.py --out rope.jsonl
"""
import argparse
import json
import time

import torch
import ttnn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="7x15,7x18,1x40,7x20", help="BxH pairs (300M, 600M, 6B, SaProt 650M)")
    ap.add_argument("--seq", type=int, default=544)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    dev = T.get_device()
    L, D, h = args.seq, args.d, args.d // 2
    inv = 1.0 / (10000 ** (torch.arange(0, h, dtype=torch.float64) / h))
    ang = torch.arange(L, dtype=torch.float64)[:, None] * inv[None]  # [L, h]
    cos_h, sin_h = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)  # half-split layout
    cos_i, sin_i = ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1)  # interleaved layout
    perm = torch.stack([torch.arange(h), torch.arange(h) + h], -1).flatten()  # interleaved slot j <- half-split col perm[j]
    unperm = torch.argsort(perm)
    rot_half = torch.zeros(D, D, dtype=torch.float64)  # x @ R = cat(-x2, x1)
    rot_half[h:, :h] = -torch.eye(h, dtype=torch.float64)
    rot_half[:h, h:] = torch.eye(h, dtype=torch.float64)
    tm = torch.zeros(32, 32)
    tm[torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
    tm[torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1

    up = lambda t, mc=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                                                               dtype=ttnn.bfloat16, memory_config=mc)
    c4 = lambda t: t.reshape(1, 1, L, D)
    ch, sh, ci, si = up(c4(cos_h)), up(c4(sin_h)), up(c4(cos_i)), up(c4(sin_i))
    tmd, rd = up(tm.reshape(1, 1, 32, 32)), up(rot_half.reshape(1, 1, D, D))

    def bench(fn):
        ts = []
        for i in range(args.repeat + 2):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            if i >= 2:
                ts.append((time.perf_counter() - t0) * 1000)
            ttnn.deallocate(o)
        return round(sorted(ts)[len(ts) // 2], 4)

    with open(args.out, "w") as f:
        for shp in args.shapes.split(","):
            B, H = (int(x) for x in shp.split("x"))
            g = torch.Generator().manual_seed(0)
            x = torch.randn(B, H, L, D, generator=g).to(torch.bfloat16).double()
            ref = x * cos_h + (x @ rot_half) * sin_h
            xd, xp = up(x), up(x[..., perm])
            xpf = ttnn.reshape(xp, (1, B * H, L, D))

            def mm():
                r = ttnn.matmul(xd, rd)
                a = ttnn.multiply(xd, ch)
                b = ttnn.multiply(r, sh)
                ttnn.deallocate(r)
                o = ttnn.add(a, b)
                ttnn.deallocate(a); ttnn.deallocate(b)
                return o

            variants = {
                "rot_dram": (lambda: ttnn.experimental.rotary_embedding(xd, ch, sh), False),
                "rot_l1": (lambda: ttnn.experimental.rotary_embedding(xd, ch, sh, memory_config=ttnn.L1_MEMORY_CONFIG), False),
                "llama_bh": (lambda: ttnn.experimental.rotary_embedding_llama(xp, ci, si, tmd, is_decode_mode=False), True),
                "llama_flat": (lambda: ttnn.experimental.rotary_embedding_llama(xpf, ci, si, tmd, is_decode_mode=False), True),
                "llama_flat_l1": (lambda: ttnn.experimental.rotary_embedding_llama(
                    xpf, ci, si, tmd, is_decode_mode=False, memory_config=ttnn.L1_MEMORY_CONFIG), True),
                "mm": (mm, False),
            }
            for name, (fn, permuted) in variants.items():
                row = dict(shape=shp, seq=L, variant=name)
                try:
                    o = ttnn.to_torch(fn()).double().reshape(B, H, -1, D)[:, :, :L]
                    if permuted:
                        o = o[..., unperm]
                    err = o - ref
                    row.update(finite=bool(torch.isfinite(o).all()), max_abs=float(err.abs().max()),
                               rel_l2=float(err.norm() / ref.norm()), ms=bench(fn))
                except Exception as exc:  # a variant the op refuses is a result, not a crash
                    row["error"] = repr(exc)[:300]
                print(row, flush=True)
                f.write(json.dumps(row) + "\n")
            for t in (xd, xp):
                ttnn.deallocate(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
