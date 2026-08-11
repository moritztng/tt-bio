#!/usr/bin/env python3
"""triatt-absolute-optimal, planning pass: the three unmeasured quantities the design turns on.

Card 2 on qb2, P300c, 11x10 = 110 cores, ttnn 0.68.0, branch wk/triatt-absolute-optimal.

PREDICTIONS, WRITTEN BEFORE THE RUN (state/triatt-absolute-optimal.md 0):

P1  Card-2 roofs land within 5 % of the card-1 figures the sister leg measured: DRAM read
    352-356 GB/s, DRAM write 249-264, DRAM->DRAM copy 375-387, L1->L1 836-893.
P2  qkv at a 64-row chunk (M=32768, K=256, N=768) DRAM->DRAM costs 8 x ~0.263 = ~2.10 ms,
    i.e. chunking alone is free (+-5 %) because the op is traffic-bound, not launch-bound.
P3  qkv L1->L1 is 1.30-1.60x the DRAM->DRAM time. It removes 512 MiB of a 512 MiB traffic
    bill, so if the op were purely write-bound it would be much more; I predict it stops at
    a compute rate near 55-70 TF/s. THIS IS THE MEASUREMENT THAT DECIDES THE L1-RESIDENT
    ROW-CHUNK DESIGN: below 1.15x that design is dead, above 1.35x it is the biggest cheap
    lever in the sub-block.
P4  gate / out (K=256, N=256) L1->L1 is 1.25-1.60x their DRAM->DRAM time.
P5  the fused projection (one matmul, N=768+256+32 -> 1056) beats the three separate
    matmuls by 1.35-1.60x, and every output column is bit-exact against the separate ops
    (same K, same accumulation order, weights only concatenated along N).
P6  SDPA at batch 64 x 8 chunks is 1.00-1.12x the batch-512 time: the work unit is
    (batch, head, q_chunk), so 64x8 = 512 units over 110 cores wastes at most one wave
    (512/(5*110) = 93 % occupancy) and the bias re-read per batch element is unchanged.
P7  SDPA with q/k/v in L1 and the bias still in DRAM is 1.00-1.10x the all-DRAM time: the
    bias is 84 % of the read, so removing the q/k/v read cannot pay much.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio.tenstorrent import get_device

MiB = 2 ** 20
RES = {"predictions": __doc__}


def timed(fn, warm=3, reps=5):
    dev = T.get_device()
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
    return st.median(ts)


def roofs(dev, ckc):
    """Named per row, same method as the sister leg: ttnn.clone between buffer types."""
    out = []
    for mb in (52, 105, 128):
        n = mb * MiB // 2
        rows = n // 1024
        t = torch.randn(rows, 1024, dtype=torch.float32).to(torch.bfloat16)
        for src, dst, label, counted in (
                (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, "dram_read", 1),
                (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, "dram_write", 1),
                (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, "dram_copy", 2),
                (ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, "l1_copy", 2)):
            try:
                a = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,
                                    dtype=ttnn.bfloat16, memory_config=src)
            except Exception as e:
                out.append({"mb": mb, "roof": label, "error": repr(e)[:120]})
                continue
            try:
                dt = timed(lambda: ttnn.clone(a, memory_config=dst))
                out.append({"mb": mb, "roof": label, "ms": dt * 1e3,
                            "gbps": counted * mb * MiB / dt / 1e9})
            except Exception as e:
                out.append({"mb": mb, "roof": label, "error": repr(e)[:120]})
            ttnn.deallocate(a)
    return out


def mm_case(dev, ckc, M, K, N, src, dst, cfg_nt24=True):
    """One minimal_matmul at (M,K,N) with the operand in `src` and the output in `dst`."""
    x = ttnn.from_torch(torch.randn(M, K).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16, memory_config=src)
    w = ttnn.from_torch(torch.randn(K, N).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16)
    nt, mt, kt = -(-N // 32), -(-M // 32), -(-K // 32)
    cfg = None
    if cfg_nt24 and mt % 4 == 0 and kt % 8 == 0:
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=4, K_block_size=8, N_block_size=1, subblock_h=4, subblock_w=1,
            compute_with_storage_grid_size=ttnn.CoreCoord(*T.COMPUTE_GRID_MAIN))
    kw = {"input_tensor": x, "weight_tensor": w, "compute_kernel_config": ckc,
          "dtype": ttnn.bfloat16, "config": cfg}
    if dst.buffer_type == ttnn.BufferType.L1:
        kw["memory_config"] = dst
    try:
        dt = timed(lambda: ttnn.experimental.minimal_matmul(**kw))
    except TypeError:
        kw.pop("memory_config", None)
        dt = timed(lambda: ttnn.experimental.minimal_matmul(**kw))
    flop = 2 * M * K * N
    rd = M * K * 2 / MiB
    wr = M * N * 2 / MiB
    ttnn.deallocate(x)
    ttnn.deallocate(w)
    return {"M": M, "K": K, "N": N, "src": str(src.buffer_type).split(".")[-1],
            "dst": str(dst.buffer_type).split(".")[-1], "cfg": cfg is not None,
            "ms": dt * 1e3, "tflops": flop / dt / 1e12,
            "read_MiB": rd, "write_MiB": wr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="roofs,mm,fused,sdpa")
    args = ap.parse_args()
    only = set(args.only.split(","))

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    RES["loadavg"] = open("/proc/loadavg").read().split()[:3]
    RES["grid"] = list(T.COMPUTE_GRID_MAIN)
    print(f"grid={RES['grid']} load={RES['loadavg']}", flush=True)

    if "roofs" in only:
        RES["roofs"] = roofs(dev, ckc)
        for r in RES["roofs"]:
            print("ROOF", r, flush=True)

    # ---- L1 residency of the three projections, at the row-chunk sizes the design would use ----
    if "mm" in only:
        rows = []
        D, L = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
        for R in (32, 64):
            M = R * 512
            for N in (768, 256):
                for src, dst in ((D, D), (L, L), (D, L), (L, D)):
                    try:
                        r = mm_case(dev, ckc, M, 256, N, src, dst)
                        r["row_chunk"] = R
                        r["chunks"] = 512 // R
                        r["ms_full"] = r["ms"] * (512 // R)
                        rows.append(r)
                        print("MM", {k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in r.items()}, flush=True)
                    except Exception as e:
                        rows.append({"M": M, "N": N, "src": str(src.buffer_type),
                                     "dst": str(dst.buffer_type), "error": repr(e)[:200]})
                        print("MM FAIL", M, N, repr(e)[:160], flush=True)
        # the unchunked production reference
        for N in (768, 256):
            try:
                r = mm_case(dev, ckc, 512 * 512, 256, N, D, D)
                r["row_chunk"] = 512
                r["chunks"] = 1
                r["ms_full"] = r["ms"]
                rows.append(r)
                print("MM", {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in r.items()}, flush=True)
            except Exception as e:
                print("MM FAIL full", N, repr(e)[:160], flush=True)
        RES["mm"] = rows

    # ---- one fused projection instead of three ----------------------------------------------
    if "fused" in only:
        rows = []
        M = 512 * 512
        x = ttnn.from_torch(torch.randn(M, 256).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16)
        wq = torch.randn(256, 768).to(torch.bfloat16)
        wg = torch.randn(256, 256).to(torch.bfloat16)
        wb = torch.zeros(256, 32).to(torch.bfloat16)
        wb[:, :8] = torch.randn(256, 8).to(torch.bfloat16)
        parts = {"qkv768": wq, "gate256": wg, "bias32": wb}
        combos = {"qkv768": ["qkv768"], "gate256": ["gate256"], "bias32": ["bias32"],
                  "qkv+bias_800": ["qkv768", "bias32"],
                  "qkv+gate_1024": ["qkv768", "gate256"],
                  "all_1056": ["qkv768", "gate256", "bias32"]}
        outs = {}
        for name, keys in combos.items():
            w_t = torch.cat([parts[k] for k in keys], dim=1)
            w = ttnn.from_torch(w_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            nt, kt = -(-w_t.shape[1] // 32), 8
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=4, K_block_size=8, N_block_size=1, subblock_h=4, subblock_w=1,
                compute_with_storage_grid_size=ttnn.CoreCoord(*T.COMPUTE_GRID_MAIN))
            try:
                dt = timed(lambda: ttnn.experimental.minimal_matmul(
                    input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
                    dtype=ttnn.bfloat16, config=cfg))
                o = ttnn.experimental.minimal_matmul(
                    input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
                    dtype=ttnn.bfloat16, config=cfg)
                outs[name] = ttnn.to_torch(o)
                ttnn.deallocate(o)
                row = {"combo": name, "N": int(w_t.shape[1]), "ms": dt * 1e3,
                       "write_MiB": M * w_t.shape[1] * 2 / MiB}
            except Exception as e:
                row = {"combo": name, "N": int(w_t.shape[1]), "error": repr(e)[:200]}
            ttnn.deallocate(w)
            rows.append(row)
            print("FUSED", {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in row.items()}, flush=True)
        # bit-exactness of the fused columns against the separate ops
        eq = {}
        if "all_1056" in outs:
            a = outs["all_1056"]
            for name, sl in (("qkv768", slice(0, 768)), ("gate256", slice(768, 1024)),
                             ("bias32", slice(1024, 1056))):
                if name in outs:
                    eq[name] = bool(torch.equal(a[..., sl], outs[name]))
        if "qkv+gate_1024" in outs:
            a = outs["qkv+gate_1024"]
            eq["qkv+gate:qkv"] = bool(torch.equal(a[..., :768], outs["qkv768"]))
            eq["qkv+gate:gate"] = bool(torch.equal(a[..., 768:1024], outs["gate256"]))
        RES["fused"] = rows
        RES["fused_bitexact"] = eq
        print("FUSED bit-exact:", eq, flush=True)
        ttnn.deallocate(x)

    # ---- SDPA: batch chunking, and q/k/v in L1 ------------------------------------------------
    if "sdpa" in only:
        rows = []
        S, H, Dh = 512, 8, 32
        bias = ttnn.from_torch(torch.randn(1, H, S, S).to(torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        for B in (512, 64, 32):
            for mc in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
                try:
                    q, k, v = (ttnn.from_torch(torch.randn(B, H, S, Dh).to(torch.bfloat16),
                                               layout=ttnn.TILE_LAYOUT, device=dev,
                                               dtype=ttnn.bfloat16, memory_config=mc)
                               for _ in range(3))
                except Exception as e:
                    rows.append({"B": B, "qkv_mem": str(mc.buffer_type), "error": repr(e)[:160]})
                    print("SDPA alloc fail", B, mc.buffer_type, repr(e)[:120], flush=True)
                    continue
                try:
                    dt = timed(lambda: T._tri_att_sdpa(q, k, v, bias, Dh ** -0.5), warm=2, reps=3)
                    row = {"B": B, "qkv_mem": str(mc.buffer_type).split(".")[-1],
                           "ms": dt * 1e3, "chunks": 512 // B, "ms_full": dt * 1e3 * (512 // B)}
                except Exception as e:
                    row = {"B": B, "qkv_mem": str(mc.buffer_type).split(".")[-1],
                           "error": repr(e)[:200]}
                for t in (q, k, v):
                    ttnn.deallocate(t)
                rows.append(row)
                print("SDPA", {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                               for kk, vv in row.items()}, flush=True)
        # the bias-off ceiling on THIS card, with the wide-q pick in place
        for B in (512,):
            q, k, v = (ttnn.from_torch(torch.randn(B, H, S, Dh).to(torch.bfloat16),
                                       layout=ttnn.TILE_LAYOUT, device=dev,
                                       dtype=ttnn.bfloat16) for _ in range(3))
            try:
                dt = timed(lambda: T._tri_att_sdpa(q, k, v, None, Dh ** -0.5), warm=2, reps=3)
                rows.append({"B": B, "qkv_mem": "DRAM", "bias": "off", "ms": dt * 1e3})
                print("SDPA bias-off", round(dt * 1e3, 4), flush=True)
            except Exception as e:
                print("SDPA bias-off FAIL", repr(e)[:200], flush=True)
            for t in (q, k, v):
                ttnn.deallocate(t)
        RES["sdpa"] = rows

    json.dump(RES, open(args.out, "w"), indent=1)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
