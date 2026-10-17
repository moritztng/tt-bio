"""ESMC's LM SDPA (esmc.lm_sdpa, the shipped program) with the additive key-padding mask in different formats.

At the 300M page batch (7 x 15 heads x 544 tokens, d 64) the [B, 1, L, L] bf16 mask is ~62 MB a layer read through
DRAM against ~29 MB for q, k, v and the output. Its values are 0 or -inf, which block-float formats hold exactly as
long as a shared exponent never has to cover both (0 has a zero mantissa at any exponent). Arms per shape:
bf16 (shipped), bfp8 and bfp4 at -inf, the same at a finite -1e4, a [B, 1, 32, L] mask (rows broadcast, if the op
takes it) and the all-zero [1, 1, L, L] mask (no padding masked: a speed floor, wrong output by design).
Per arm: device ms (median of --repeat, synchronized) and error over the valid rows vs float64.

    python3 perf/spd_gap_whlm/sdpa_mask_bench.py --out mask.jsonl
"""
import argparse
import json
import time

import torch
import ttnn

SHAPES = {"300m": (7, 15), "600m": (7, 18), "6b": (1, 40)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=544)
    ap.add_argument("--valid", type=int, default=514)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio import esmc
    dev = T.get_device()
    L, V, D = args.seq, args.valid, 64
    scale = D ** -0.5
    with open(args.out, "w") as f:
        for name, (B, H) in SHAPES.items():
            g = torch.Generator().manual_seed(0)
            q, k, v = (torch.randn(B, H, L, D, generator=g).to(torch.bfloat16) for _ in range(3))
            s = (q.double() @ k.double().transpose(-1, -2)) * scale
            s[..., V:] = float("-inf")
            ref = torch.softmax(s, -1) @ v.double()
            up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
            qd, kd, vd = up(q), up(k), up(v)

            def mask(neg, rows=L):
                m = torch.zeros(B, 1, rows, L)
                m[..., V:] = neg
                return m.to(torch.bfloat16)

            arms = {
                "bf16": lambda: up(mask(float("-inf"))),
                "bfp8": lambda: up(mask(float("-inf")), ttnn.bfloat8_b),
                "bfp4": lambda: up(mask(float("-inf")), ttnn.bfloat4_b),
                "bfp8_fin": lambda: up(mask(-1e4), ttnn.bfloat8_b),
                "bfp4_fin": lambda: up(mask(-1e4), ttnn.bfloat4_b),
                "rows32": lambda: up(mask(float("-inf"), 32)),
                "zero_floor": lambda: None,
            }
            for arm, mk in arms.items():
                row = {"shape": name, "bhld": [B, H, L, D], "arm": arm}
                try:
                    md = mk()
                    run = lambda: esmc.lm_sdpa(qd, kd, vd, md, scale)
                    o = ttnn.to_torch(run()).double()[:, :, :V]
                    err = o - ref[:, :, :V]
                    a, b = o.flatten(), ref[:, :, :V].flatten()
                    ts = []
                    for i in range(args.repeat + 2):
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        out = run()
                        ttnn.synchronize_device(dev)
                        if i >= 2:
                            ts.append((time.perf_counter() - t0) * 1000)
                        ttnn.deallocate(out)
                    row.update(ms=round(sorted(ts)[len(ts) // 2], 4), finite=bool(torch.isfinite(o).all()),
                               max_abs=float(err.abs().max()), rel_l2=float(err.norm() / ref[:, :, :V].norm()),
                               pcc=float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
                    if md is not None:
                        ttnn.deallocate(md)
                except Exception as exc:  # a format the op refuses is a result, not a crash
                    row["error"] = repr(exc)[:300]
                print(row, flush=True)
                f.write(json.dumps(row) + "\n")
                f.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
