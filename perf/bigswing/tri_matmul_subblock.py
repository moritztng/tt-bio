#!/usr/bin/env python3
"""`trimul.tri_matmul` runs at 17.41 TFLOP/s, the worst rate in the fold. Its dest register holds one tile.

§12's amortized-region table prices eight matmul classes in a 512 aa protenix-v2 fold. Seven sit at
23-49 TFLOP/s; `trimul.tri_matmul` sits at **17.41** over **74.8 TFLOP/fold = 4.30 s**, and the note
against it says only "at 17.41 TFLOP/s it is bound by something else entirely". Nobody named the
something else.

`_triangle_mul_program_config` (tenstorrent.py:620) emits `out_subblock_h=1, out_subblock_w=1`
unconditionally. The model runs `fp32_dest_acc_en=True`, which halves the dest register file to **4
tiles** -- this codebase's own rule, written at tenstorrent.py:424 for the batched-matmul helper and
not applied here. At 512 aa the config's `per_core_M` and `per_core_N` are both 2, so `2x2` (area 4,
exactly the dest limit) divides the block and is legal. A 1x1 subblock re-reads both operand tiles
from the CB for every single output tile; a 2x2 subblock amortizes each read over two.

The op is not traffic-bound: `[1,32,512,512] @ [1,32,512,512]` moves 48 MB to do 8.59 GFLOP, an
arithmetic intensity of 179 FLOP/byte. So unlike the in-projection lever (§55-56, which removed
reads), any gain here has to come from the pipeline, which is exactly why it is measured and not
argued.

Arms are program config only. The K accumulation order into the fp32 dest register does not depend on
the subblock shape, so every arm should be `torch.equal` to the shipped one -- checked, not assumed,
because that is what decides whether a parity gate is owed.

This is a per-call screen. Its seconds size and de-risk a fold A/B; they are not a fold gain.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=32, help="TRIANGLE_MULT_CHUNK_SIZE, the batch dim here")
    ap.add_argument("--pairs", type=int, default=8, help="tri_matmuls per trimul at this size")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    import ttnn
    import tt_bio.tenstorrent as T

    torch.manual_seed(0)
    dev = T.get_device()
    S, C, P = a.seq, a.chunk, a.pairs
    St = (S + 31) // 32
    gx, gy = T.COMPUTE_GRID_MAIN

    # The model's own kernel config, not a stand-in: fp32_dest_acc_en=True is what caps the subblock.
    ckc = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    # a_chunk/b_chunk as the loop hands them over: [batch=1, chunk, S, S], DRAM, bf16.
    A = ttnn.from_torch(torch.randn(1, C, S, S, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    B = ttnn.from_torch(torch.randn(1, C, S, S, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    shipped = T._triangle_mul_program_config(St)
    per_core_M, per_core_N = shipped.per_core_M, shipped.per_core_N
    bw = shipped.in0_block_w
    print(f"shipped: grid={gx}x{gy} St={St} per_core_M={per_core_M} per_core_N={per_core_N} "
          f"in0_block_w={bw} subblock={shipped.out_subblock_h}x{shipped.out_subblock_w} "
          f"-> {-(-St//per_core_M)}x{-(-St//per_core_N)} = "
          f"{(-(-St//per_core_M))*(-(-St//per_core_N))} of {gx*gy} cores", flush=True)

    def cfg(sh, sw, block_w=None, pcm=None, pcn=None):
        pcm = per_core_M if pcm is None else pcm
        pcn = per_core_N if pcn is None else pcn
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=bw if block_w is None else block_w,
            out_subblock_h=sh, out_subblock_w=sw,
            out_block_h=pcm, out_block_w=pcn,
            per_core_M=pcm, per_core_N=pcn,
            transpose_mcast=False, fused_activation=None, fuse_batch=False)

    arms = {"shipped_1x1": shipped}
    for sh, sw in ((1, 2), (2, 1), (2, 2)):
        if per_core_M % sh == 0 and per_core_N % sw == 0 and sh * sw <= 4:
            arms[f"sub{sh}x{sw}"] = cfg(sh, sw)
    if per_core_M % 2 == 0 and per_core_N % 2 == 0 and St % 16 == 0:
        arms["sub2x2_bw16"] = cfg(2, 2, block_w=16)
    if per_core_M % 2 == 0 and per_core_N % 2 == 0 and St % 4 == 0:
        arms["sub2x2_bw4"] = cfg(2, 2, block_w=4)
    arms["default_none"] = None

    def one_trimul(pc):
        """All P tri_matmuls of one trimul, amortized in one region (W4's overhead correction)."""
        kw = {} if pc is None else {"program_config": pc}
        return [ttnn.matmul(A, B, compute_kernel_config=ckc,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, **kw)
                for _ in range(P)]

    res = {"host": "qb2", "chip": 0, "seq": S, "chunk": C, "pairs": P, "reps": a.reps,
           "grid": [gx, gy], "seq_len_tiles": St, "per_core_M": per_core_M,
           "per_core_N": per_core_N, "in0_block_w": bw,
           "cores_used": (-(-St // per_core_M)) * (-(-St // per_core_N)), "cores_avail": gx * gy,
           "shape": f"[1,{C},{S},{S}] @ [1,{C},{S},{S}]", "arms": []}
    import importlib.metadata as im
    res["ttnn"] = im.version("ttnn")

    # ---- warm, and drop arms the op refuses, before anything is timed --------------------------
    live = {}
    for name, pc in arms.items():
        try:
            for t in one_trimul(pc):
                ttnn.deallocate(t)
            ttnn.synchronize_device(dev)
            live[name] = pc
        except Exception as e:                                                   # noqa: BLE001
            res["arms"].append({"arm": name, "error": f"{type(e).__name__}: {str(e)[:220]}"})
            print(f"{name}: REFUSED {type(e).__name__}: {str(e)[:180]}", flush=True)

    # ---- bit-exactness against the shipped config ----------------------------------------------
    ref = ttnn.to_torch(one_trimul(live["shipped_1x1"])[0]).float()
    exact = {}
    for name, pc in live.items():
        if name == "shipped_1x1":
            continue
        got = ttnn.to_torch(one_trimul(pc)[0]).float()
        exact[name] = {"torch_equal": bool(torch.equal(got, ref)),
                       "max_abs": (got - ref).abs().max().item()}
        print(f"{name}: torch.equal={exact[name]['torch_equal']} "
              f"max_abs={exact[name]['max_abs']}", flush=True)
    del ref
    res["bit_exact"] = exact

    # ---- timing, arms alternating, median of reps ----------------------------------------------
    names = list(live)
    times = {n: [] for n in names}
    for _ in range(a.reps):
        for n in names:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            outs = one_trimul(live[n])
            ttnn.synchronize_device(dev)
            times[n].append((time.perf_counter() - t0) * 1000.0)
            for t in outs:
                ttnn.deallocate(t)

    flop = 2 * C * S * S * S * P
    traffic_mb = 3 * C * S * S * 2 / 2**20 * P
    base = st.median(times["shipped_1x1"])
    for n in names:
        ms = sorted(times[n])
        med = st.median(ms)
        pc = live[n]
        res["arms"].append({
            "arm": n, "calls_per_trimul": P,
            "subblock": None if pc is None else [pc.out_subblock_h, pc.out_subblock_w],
            "in0_block_w": None if pc is None else pc.in0_block_w,
            "ms_median": round(med, 4), "ms_min": round(ms[0], 4), "ms_max": round(ms[-1], 4),
            "spread_ms": round(ms[-1] - ms[0], 4),
            "tflops": round(flop / (med / 1000) / 1e12, 2),
            "traffic_MB": round(traffic_mb, 1),
            "agg_GBps": round(traffic_mb / 2**10 / (med / 1000), 1),
            "speedup_vs_shipped": round(base / med, 4),
            "torch_equal_vs_shipped": exact.get(n, {}).get("torch_equal", True),
        })
        print(f"{n:16s} {med:9.3f} ms  {res['arms'][-1]['tflops']:6.2f} TFLOP/s  "
              f"{base/med:6.4f}x", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
