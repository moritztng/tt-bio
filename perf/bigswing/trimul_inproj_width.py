#!/usr/bin/env python3
"""The trimul in-projection re-reads its 134 MB activation once per weight chunk. Measure the price.

Census (`perf/bigswing/mmcfg/census512_qb2c0.json`) says the largest matmul class in a 512 aa
protenix-v2 fold is `tenstorrent.py:1700`, `[1,512,512,256] @ [256,128]`, **8384 calls, 6.199 s**,
144.0 TFLOP/fold at 23.2 TFLOP/s. The 128 is `4 * TRIANGLE_MULT_CHUNK_SIZE`: `_trimul_chunk_size`
returns the narrowest chunk (32) as soon as `seq_len > _trimul_l1_max_seq()`, and `_gp_in_chunks`
then splits one weight into `n_pairs = 8` pieces. So per trimul the same 134.2 MB activation is
streamed from DRAM eight times to produce eight 67.1 MB outputs.

The chunk width is a partition of an independent-channel sum (`_trimul_chunk_size`'s own docstring:
"bit-exact at every width"), so the matmul's N can be widened without touching the downstream
per-chunk loop: one matmul of N = 128*G, then a 4G-way split feeding the same body.

This measures the in-projection of ONE trimul -- all `n_pairs` matmuls, amortized inside a single
synchronise..synchronise region, which is W4's per-region-overhead correction -- at G in {1,2,4,8},
arms alternating, and checks `torch.equal` of every widened output against the G=1 columns it
replaces. Nothing here is a fold gain: it is a per-call screen whose job is to size and de-risk the
fold A/B, and the fold A/B is the only thing that may be quoted as a result.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--cz", type=int, default=256, help="K, the pair channel width")
    ap.add_argument("--c", type=int, default=32, help="TRIANGLE_MULT_CHUNK_SIZE")
    ap.add_argument("--pairs", type=int, default=8, help="n_pairs at this size")
    ap.add_argument("--groups", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    import ttnn
    from tt_bio.tenstorrent import get_device

    torch.manual_seed(0)
    dev = get_device()
    S, K, C, P = a.seq, a.cz, a.c, a.pairs
    NW = 4 * C * P                       # the full fused [g_a|g_b|p_a|p_b] width, all pairs

    at = torch.randn(1, S, S, K, dtype=torch.bfloat16)
    wt = torch.randn(K, NW, dtype=torch.bfloat16)
    A = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)

    groups = [int(g) for g in a.groups.split(",")]
    W = {}
    for g in groups:
        if P % g:
            continue
        # G pair-chunks side by side: [ ...pair i... | ...pair i+1... ], each 4*C wide.
        W[g] = [ttnn.from_torch(wt[:, i * 4 * C * g:(i + 1) * 4 * C * g].contiguous(),
                                layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
                for i in range(P // g)]

    def one_inproj(g):
        """All of one trimul's in-projection matmuls at grouping g. Returns the outputs."""
        return [ttnn.experimental.minimal_matmul(
            input_tensor=A, weight_tensor=w, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16, compute_kernel_config=ckc) for w in W[g]]

    res = {"host": "qb2", "chip": 0, "seq": S, "K": K, "C": C, "n_pairs": P,
           "shape": f"[1,{S},{S},{K}] @ [{K},{4*C}g]", "reps": a.reps, "arms": []}
    import importlib.metadata as im
    res["ttnn"] = im.version("ttnn")

    # ---- warm every shape's kernel first; JIT is a compile-time cost, not a rate ----
    for g in groups:
        if g in W:
            for t in one_inproj(g):
                ttnn.deallocate(t)
    ttnn.synchronize_device(dev)

    # ---- bit-exactness: every widened output against the G=1 columns it replaces ----
    base = [ttnn.to_torch(t) for t in one_inproj(1)]
    exact = {}
    for g in groups:
        if g == 1 or g not in W:
            continue
        outs = [ttnn.to_torch(t) for t in one_inproj(g)]
        ok, maxabs = True, 0.0
        for i, o in enumerate(outs):
            ref = torch.cat(base[i * g:(i + 1) * g], dim=-1)
            ok &= torch.equal(o.float(), ref.float())
            maxabs = max(maxabs, (o.float() - ref.float()).abs().max().item())
        exact[str(g)] = {"torch_equal": bool(ok), "max_abs": maxabs}
        print(f"G={g}: torch.equal={ok} max_abs={maxabs}", flush=True)
    res["bit_exact"] = exact
    del base

    # ---- timing, arms alternating over G, median of reps ----
    order = [g for _ in range(a.reps) for g in groups if g in W]
    times = {g: [] for g in groups if g in W}
    for g in order:
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        outs = one_inproj(g)
        ttnn.synchronize_device(dev)
        times[g].append((time.perf_counter() - t0) * 1000.0)
        for t in outs:
            ttnn.deallocate(t)

    flop_per_trimul = 2 * (S * S) * K * NW
    for g in sorted(times):
        ms = sorted(times[g])
        med = st.median(ms)
        rd_mb = (S * S * K * 2 / 2**20) * (P // g)
        wr_mb = S * S * NW * 2 / 2**20
        res["arms"].append({
            "G": g, "n_calls_per_trimul": P // g, "N": 4 * C * g,
            "ms_median": round(med, 4), "ms_min": round(ms[0], 4), "ms_max": round(ms[-1], 4),
            "spread_ms": round(ms[-1] - ms[0], 4),
            "tflops": round(flop_per_trimul / (med / 1000) / 1e12, 2),
            "read_MB": round(rd_mb, 1), "write_MB": round(wr_mb, 1),
            "agg_GBps": round((rd_mb + wr_mb) / 2**10 / (med / 1000), 1),
            "torch_equal_vs_G1": exact.get(str(g), {}).get("torch_equal", True),
        })
        print(f"G={g:2d} N={4*C*g:5d} {P//g} calls  {med:8.3f} ms  "
              f"{res['arms'][-1]['tflops']:6.2f} TFLOP/s  read {rd_mb:7.1f} MB", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
