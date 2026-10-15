"""ESMC's attention SDPA at the perf-page shape, swept over (q_chunk, k_chunk), against a float64 reference.

Shape: batch 7 x 18 heads x 544 tokens x d 64 (600M at prot512 x7, rows 514 long, padded keys masked with -inf and
zeroed through key_valid as esmc.Attention does). Per config: device ms (median of --repeat, synchronized), max abs
error and PCC vs float64 torch, and whether dropping the key_valid multiplies changes a single output byte.

    python3 perf/wh-embed/sdpa_sweep.py --out sdpa.jsonl
"""
import argparse
import json
import time

import torch
import ttnn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=7)
    ap.add_argument("--heads", type=int, default=18)
    ap.add_argument("--seq", type=int, default=544)
    ap.add_argument("--valid", type=int, default=514)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--configs", default="", help="q:k pairs, comma separated (default: the 5 x 5 power-of-two grid)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    dev = T.get_device()
    B, H, L, D, V = args.batch, args.heads, args.seq, args.d, args.valid
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(B, H, L, D, generator=g) for _ in range(3))
    q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
    mask = torch.zeros(B, 1, L, L)
    mask[:, :, :, V:] = float("-inf")
    kv = torch.ones(B, 1, L, 1)
    kv[:, :, V:] = 0
    scale = D ** -0.5
    s = (q.double() @ k.double().transpose(-1, -2)) * scale + mask.double()
    ref = torch.softmax(s, -1) @ v.double()

    up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    qd, kd, vd, md, kvd = up(q), up(k), up(v), up(mask), up(kv)
    shipped = T._sdpa_program_config_for_lengths(L, L, B * H, site="esmc", d=D)
    print("shipped", shipped.q_chunk_size, shipped.k_chunk_size, "grid", T.COMPUTE_GRID_MAIN, flush=True)

    def run(cfg, masked_kv):
        kk, vv = (ttnn.multiply(kd, kvd), ttnn.multiply(vd, kvd)) if masked_kv else (kd, vd)
        o = T.fused_sdpa(q=qd, k=kk, v=vv, attn_mask=md, scale=scale, program_config=cfg)
        if masked_kv:
            ttnn.deallocate(kk); ttnn.deallocate(vv)
        return o

    def bench(cfg, masked_kv):
        ts = []
        for i in range(args.repeat + 2):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = run(cfg, masked_kv)
            ttnn.synchronize_device(dev)
            if i >= 2:
                ts.append((time.perf_counter() - t0) * 1000)
            ttnn.deallocate(o)
        return sorted(ts)[len(ts) // 2]

    grid = ([tuple(int(x) for x in c.split(":")) for c in args.configs.split(",")] if args.configs else
            [(a, b) for a in (32, 64, 128, 256, 512) for b in (32, 64, 128, 256, 512)])
    configs = [(shipped.q_chunk_size, shipped.k_chunk_size)] + grid
    seen = set()
    with open(args.out, "w") as f:
        for qc, kc in configs:
            if (qc, kc) in seen:
                continue
            seen.add((qc, kc))
            cfg = T._sdpa_program_config(qc, kc)
            row = dict(q_chunk=qc, k_chunk=kc, shipped=(qc, kc) == (shipped.q_chunk_size, shipped.k_chunk_size))
            try:
                o1 = ttnn.to_torch(run(cfg, True)).double()
                o0 = ttnn.to_torch(run(cfg, False)).double()
                err = (o1[:, :, :L] - ref)[:, :, :V]
                a, b = o1[:, :, :V].flatten(), ref[:, :, :V].flatten()
                row.update(
                    finite=bool(torch.isfinite(o1).all()),
                    max_abs=float(err.abs().max()),
                    pcc=float(torch.corrcoef(torch.stack([a, b]))[0, 1]),
                    no_kv_mul_identical=bool(torch.equal(o0, o1)),
                    no_kv_mul_max_abs=float((o0 - o1)[:, :, :V].abs().max()),
                    ms_kv_mul=round(bench(cfg, True), 3),
                    ms_no_kv_mul=round(bench(cfg, False), 3),
                )
            except Exception as exc:  # a config the op refuses is a result, not a crash
                row["error"] = repr(exc)[:300]
            print(row, flush=True)
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
