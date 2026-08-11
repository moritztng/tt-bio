#!/usr/bin/env python3
"""Roofs on THIS card, then the three trimul matmul classes at 512 aa with the production configs.

One process, one device context. Everything is a device time: every timed region has a
`ttnn.synchronize_device` on both sides, and the region holds `--reps` back-to-back calls so host
dispatch is amortised rather than measured (the unsynced-`to_torch` trap and the per-region-overhead
correction both apply -- see the campaign record).

Roof method, each roof measured with an op whose OTHER side is much faster, so the roof it reports is
the slow side's:
  dram_read   ttnn.clone DRAM -> L1     bytes = read bytes ; the L1 write is ~4x faster
  dram_write  ttnn.clone L1  -> DRAM    bytes = write bytes
  dram_copy   ttnn.clone DRAM -> DRAM   bytes each way; the read-once-write-once shape most ops have
  l1_copy     ttnn.clone L1   -> L1     bytes each way
  compute     ttnn.matmul square, K large, no program config, DRAM in/out
Reported with its contamination named, never as a datasheet number.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[2] / "tt_bio").is_dir() else Path("/home/ttuser/.coworker/wt/trimul-bottleneck-rootcause")
sys.path.insert(0, str(ROOT))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio.tenstorrent import get_device


def timed(dev, fn, reps, warm=2):
    for _ in range(warm):
        r = fn()
        del r
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    keep = [fn() for _ in range(reps)]
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / reps
    del keep
    return dt


def med(dev, fn, reps, n=5):
    return st.median([timed(dev, fn, reps) for _ in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--cz", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = get_device()
    R = {"grid": list(T.COMPUTE_GRID_MAIN), "seq": a.seq, "cz": a.cz, "chunk": a.chunk,
         "l1_bank_bytes": int(T._l1_bank_bytes()), "reps": a.reps, "roofs": {}, "classes": {}}
    S, K, C = a.seq, a.cz, a.chunk
    P = K // C                                          # n_pairs = 8 at 512 aa

    # production compute kernel config, read off the model rather than re-specified
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    ckc_np = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=True)

    DR, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG

    # ---------------- roofs ----------------
    # copy roofs on a 16.78 MB tensor -- one trimul chunk [1,32,512,512] bf16, the shape that
    # dominates the channel loop, so the roof is measured on the bytes it will be applied to.
    ct = torch.randn(1, C, S, S, dtype=torch.bfloat16)
    nb = C * S * S * 2
    x_dr = ttnn.from_torch(ct, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DR)
    x_l1 = ttnn.from_torch(ct, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=L1)
    for name, src, dst in (("dram_read", x_dr, L1), ("dram_write", x_l1, DR),
                           ("dram_copy", x_dr, DR), ("l1_copy", x_l1, L1)):
        t = med(dev, lambda s=src, d=dst: ttnn.clone(s, memory_config=d), a.reps)
        R["roofs"][name] = {"bytes_each_way": nb, "s": t, "gbs_each_way": nb / t / 1e9,
                            "gbs_aggregate": 2 * nb / t / 1e9, "shape": [1, C, S, S]}
    ttnn.deallocate(x_l1)

    # compute roof: square matmul, K large, no program config (ttnn picks), DRAM in/out
    for M in (2048, 4096):
        at = torch.randn(M, M, dtype=torch.bfloat16)
        A = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DR)
        B = ttnn.from_torch(at.t().contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DR)
        fl = 2.0 * M * M * M
        t = med(dev, lambda: ttnn.matmul(A, B, compute_kernel_config=ckc, memory_config=DR, dtype=ttnn.bfloat16), 1, n=3)
        R["roofs"][f"compute_square_{M}"] = {"flop": fl, "s": t, "tflops": fl / t / 1e12}
        ttnn.deallocate(A); ttnn.deallocate(B)

    # ---------------- the three trimul matmul classes, production shapes + configs ----------------
    xt = torch.randn(1, S, S, K, dtype=torch.bfloat16)
    X = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DR)
    x_bytes = S * S * K * 2

    # in_proj: minimal_matmul [1,S,S,K] @ [K, 4*C*G], 8/G calls per trimul
    for G in (1, 4, 8):
        if P % G:
            continue
        NW = 4 * C * G
        W = ttnn.from_torch(torch.randn(K, NW, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16, memory_config=DR)
        calls = P // G
        def one(W=W, calls=calls):
            outs = [ttnn.experimental.minimal_matmul(X, W, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                     dtype=ttnn.bfloat16, compute_kernel_config=ckc_np)
                    for _ in range(calls)]
            return outs
        t = med(dev, one, 1, n=5)
        fl = 2.0 * S * S * K * NW * calls
        rd = (x_bytes + K * NW * 2) * calls
        wr = S * S * NW * 2 * calls
        R["classes"][f"in_proj_G{G}"] = {
            "shape": f"[1,{S},{S},{K}]@[{K},{NW}]", "calls_per_trimul": calls, "s_per_trimul": t,
            "flop_per_trimul": fl, "tflops": fl / t / 1e12, "read_MB": rd / 1e6, "write_MB": wr / 1e6,
            "gbs_aggregate": (rd + wr) / t / 1e9, "ai_flop_per_byte": fl / (rd + wr)}
        ttnn.deallocate(W)

    # the mandatory unit: in-projection matmul + the split the channel loop actually consumes.
    # pair-major (today) splits 4*G ways -> 1-tile pieces at G=1; role-major splits 4 ways -> G-tile
    # pieces. Same bytes, same arithmetic, different piece width.
    for G, mode in ((1, "pair"), (2, "role"), (4, "role"), (8, "role"), (4, "pair"), (8, "pair")):
        if P % G:
            continue
        NW = 4 * C * G
        W = ttnn.from_torch(torch.randn(K, NW, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16, memory_config=DR)
        calls, nsp = P // G, (4 * G if mode == "pair" else 4)
        def unit(W=W, calls=calls, nsp=nsp):
            keep = []
            for _ in range(calls):
                o = ttnn.experimental.minimal_matmul(X, W, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                    dtype=ttnn.bfloat16, compute_kernel_config=ckc_np)
                keep.append(ttnn.chunk(o, chunks=nsp, dim=-1))
                ttnn.deallocate(o)
            return keep
        t = med(dev, unit, 1, n=5)
        R["classes"][f"unit_G{G}_{mode}"] = {
            "calls": calls, "chunks_per_call": nsp, "piece_tiles": (4 * C * G // nsp) // 32,
            "s_per_trimul": t, "ms_per_trimul": t * 1e3}
        ttnn.deallocate(W)

    # tri_matmul: [1,C,S,S] @ [1,C,S,S], 8 calls per trimul, with and without the program config
    A = ttnn.from_torch(ct, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DR)
    B = ttnn.from_torch(torch.randn(1, C, S, S, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16, memory_config=DR)
    St = (S + 31) // 32
    pc = T._triangle_mul_program_config(St)
    gx, gy = T.COMPUTE_GRID_MAIN
    cores = -(-St // pc.per_core_M) * -(-St // pc.per_core_N)
    fl8 = 2.0 * C * S * S * S * P
    rd8 = 2 * C * S * S * 2 * P
    wr8 = C * S * S * 2 * P
    for tag, kw in (("prodcfg", {"program_config": pc}), ("nocfg", {})):
        def eight(kw=kw):
            return [ttnn.matmul(A, B, compute_kernel_config=ckc, memory_config=DR,
                                dtype=ttnn.bfloat16, **kw) for _ in range(P)]
        t = med(dev, eight, 1, n=5)
        R["classes"][f"tri_matmul_{tag}"] = {
            "shape": f"[1,{C},{S},{S}]@[1,{C},{S},{S}]", "calls_per_trimul": P, "s_per_trimul": t,
            "flop_per_trimul": fl8, "tflops": fl8 / t / 1e12, "read_MB": rd8 / 1e6,
            "write_MB": wr8 / 1e6, "gbs_aggregate": (rd8 + wr8) / t / 1e9,
            "ai_flop_per_byte": fl8 / (rd8 + wr8),
            "per_core_M": pc.per_core_M, "per_core_N": pc.per_core_N,
            "in0_block_w": pc.in0_block_w, "cores_engaged": cores, "grid_cores": gx * gy}
    # occupancy: every legal (per_core_M, per_core_N) at St=16 on this grid
    occ = []
    for pm in range(1, St + 1):
        for pn in range(1, St + 1):
            rows, cols = -(-St // pm), -(-St // pn)
            if rows <= gy and cols <= gx:
                occ.append({"per_core_M": pm, "per_core_N": pn, "cores": rows * cols})
    R["classes"]["tri_matmul_legal_occupancy"] = sorted(occ, key=lambda d: -d["cores"])[:6]
    ttnn.deallocate(A); ttnn.deallocate(B)

    # out_proj: minimal_matmul [1,S,S,K] @ [K,K], 2 calls per trimul (p_out, g_out)
    W2 = ttnn.from_torch(torch.randn(K, K, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                         device=dev, dtype=ttnn.bfloat16, memory_config=DR)
    def two():
        return [ttnn.experimental.minimal_matmul(X, W2, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                 dtype=ttnn.bfloat16, compute_kernel_config=ckc_np)
                for _ in range(2)]
    t = med(dev, two, 1, n=5)
    fl2, rd2, wr2 = 2.0 * S * S * K * K * 2, (x_bytes + K * K * 2) * 2, S * S * K * 2 * 2
    R["classes"]["out_proj"] = {"shape": f"[1,{S},{S},{K}]@[{K},{K}]", "calls_per_trimul": 2,
                                "s_per_trimul": t, "flop_per_trimul": fl2, "tflops": fl2 / t / 1e12,
                                "read_MB": rd2 / 1e6, "write_MB": wr2 / 1e6,
                                "gbs_aggregate": (rd2 + wr2) / t / 1e9,
                                "ai_flop_per_byte": fl2 / (rd2 + wr2)}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
