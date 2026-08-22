#!/usr/bin/env python3
"""The floor for one OF3/OpenBind Pairformer block, at the roofs of THIS card.

Nothing in this campaign may carry a "% of peak" sentence until this file has run, and it
measures both roofs rather than inheriting them (`roofline-roof-must-be-measured-not-asserted`:
this lineage has published 668 GB/s on a ~400 GB/s card by inheriting one).

Two floors, per pairformer block execution:

  compute   every multiply-accumulate the block's algorithm requires, against the best
            matmul rate this card actually reaches (a large square bf16 HiFi4 matmul with
            fp32 dest accumulation, the trunk's compute kernel config).
  bandwidth the pair-shaped tensors the DATA FLOW forces through DRAM, against the measured
            DRAM copy rate (which counts a read and a write, so the accounting below counts
            reads and writes too).

The bandwidth floor is deliberately the ALGORITHM's minimum, not a model of today's op list. A
tensor counts only when the next stage cannot start without the whole of it AND it cannot live in
L1: at 1024 tokens a [N,N,128] bf16 pair tensor is 268.4 MB against 160.8 MB of L1 on this grid,
so every pair-shaped intermediate is a DRAM round trip and no fusion can remove it. That is what
makes the floor a floor.

    python3 perf/openbind/trunk_floor.py --tokens 512 1024 --out perf/openbind/tt_results/floor.json
"""
from __future__ import annotations

import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402

# OF3 trunk pairformer, from tt_bio/openfold3_trunk.py: 48 blocks, 4 recycle cycles,
# _PF_DIMS = (tri_att_head_dim 32, tri_att_n_heads 4, att_head_dim 24, att_n_heads 16).
N_BLOCKS, N_CYCLES = 48, 4
C_Z, C_S = 128, 384
TRI_ATT_HD, TRI_ATT_H = 32, 4
APB_HD, APB_H = 24, 16
C_HIDDEN = 128          # trimul hidden width (g_in/p_in are c_z -> 2*128 each)
TRANSITION_MULT = 4     # SwiGLU: two up-projections of width 4*c_z plus one down


def bench(dev, fn, n=7, warm=2):
    for _ in range(warm):
        r = fn()
        ttnn.synchronize_device(dev)
        ttnn.deallocate(r)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    return st.median(ts)


def rand(dev, shape, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(*shape), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def flops_per_block(n: int) -> dict:
    """Multiply-accumulates the algorithm requires, x2 for FLOP, per pairformer block."""
    n2, n3 = n * n, n * n * n
    f = {}
    # Two triangle multiplications. Each: in-projection z -> (g_a,g_b,p_a,p_b) = 4 chunks of
    # c_hidden, the per-channel triangle product (the N^3 term), the output gate and projection.
    f["trimul_in_proj"] = 2 * (n2 * C_Z * 4 * C_HIDDEN) * 2
    f["trimul_triangle"] = 2 * (n3 * C_HIDDEN) * 2
    f["trimul_out"] = 2 * (n2 * C_HIDDEN * C_Z + n2 * C_Z * C_Z) * 2
    # Two triangle attentions. q,k,v projections (h*d wide), the bias projection (h wide),
    # scores q@k^T and attn@v (both N^3 over h*d), the output gate and projection.
    hd = TRI_ATT_H * TRI_ATT_HD
    f["triatt_qkv_proj"] = 2 * (n2 * C_Z * 3 * hd) * 2
    f["triatt_scores"] = 2 * (n3 * hd) * 2
    f["triatt_av"] = 2 * (n3 * hd) * 2
    f["triatt_out"] = 2 * (n2 * hd * C_Z + n2 * C_Z * C_Z) * 2
    # Pair transition, SwiGLU: two up-projections and one down.
    f["transition_z"] = 3 * (n2 * C_Z * TRANSITION_MULT * C_Z) * 2
    # The single track. Attention-pair-bias reads the pair tensor for its bias (N^2 * h) and
    # attends over N tokens; the s transition is N-shaped. Small, counted for completeness.
    ahd = APB_H * APB_HD
    f["apb"] = (n * C_S * 3 * ahd + n * n * APB_H + n * n * ahd + n * n * ahd
                + n * ahd * C_S) * 2
    f["transition_s"] = 3 * (n * C_S * TRANSITION_MULT * C_S) * 2
    f["_total"] = sum(v for k, v in f.items() if not k.startswith("_"))
    return f


def bytes_per_block(n: int, elem: int = 2) -> dict:
    """Pair-shaped bytes the data flow forces through DRAM per block, reads + writes.

    Counted per stage, and only for tensors the next stage needs whole. Every pair-shaped
    tensor here is larger than this card's whole L1 at 1024 tokens, so none of these round
    trips can be fused away; below ~560 tokens some can, which is exactly the L1-residency
    lever the census found going dark.
    """
    z = n * n * C_Z * elem                     # the pair tensor
    h = n * n * C_HIDDEN * elem                # a trimul hidden tensor, same width here
    hd = n * n * TRI_ATT_H * TRI_ATT_HD * elem  # packed q / k / v
    b = {}
    # Two trimuls. Read z once for the in-projection; write a and b; the triangle product reads
    # both in channel-major order and writes its result; the output stage reads it and writes the
    # update; the residual add reads z and the update and writes z.
    b["trimul"] = 2 * (z + 2 * h                # read z, write a,b
                       + 2 * h + h              # read a,b, write the product
                       + h + z                  # read the product, write the update
                       + 2 * z + z)             # residual add: read z + update, write z
    # Two triangle attentions. Read z for q,k,v and for the bias; write q,k,v; the scores are
    # N^2 per row and never materialise pair-wide beyond one row block, so only the attention
    # output is counted; then the same output stage and residual add.
    b["triatt"] = 2 * (z + 3 * hd
                       + 3 * hd + hd
                       + hd + z
                       + 2 * z + z)
    # The transition reads z, writes the two up-projections (4x wide, but streamed row-block by
    # row-block, so counted once each way), reads them back for the down-projection, writes the
    # update, then the residual add.
    b["transition"] = (z + 4 * z + 4 * z + z + 2 * z + z)
    b["_total"] = sum(v for k, v in b.items() if not k.startswith("_"))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    g = dev.compute_with_storage_grid_size()
    out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "arch": T.arch_name(), "grid": [g.x, g.y],
           "compute_grid_main": list(T.COMPUTE_GRID_MAIN),
           "l1_per_bank": int(T._l1_bank_bytes()),
           "l1_total_MB": round(T._l1_bank_bytes() * T.COMPUTE_GRID_MAIN[0]
                                * T.COMPUTE_GRID_MAIN[1] / 1e6, 1),
           "roofs": {}, "cells": {}}

    # --- roof 1: DRAM copy, read + write, at the pair shapes these blocks actually move.
    for n in a.tokens:
        for c in (C_Z,):
            t = rand(dev, (n, n, c))
            ms = bench(dev, lambda t=t: ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)) * 1e3
            gb = 2 * n * n * c * 2 / 1e9
            out["roofs"][f"copy_{n}x{n}x{c}"] = {"ms": round(ms, 4),
                                                 "GBps": round(gb / (ms / 1e3), 1)}
            ttnn.deallocate(t)
            print(f"copy {n}x{n}x{c}: {ms:.3f} ms = {gb / (ms / 1e3):.1f} GB/s", flush=True)

    # --- roof 2: the best matmul rate this card reaches, bf16 HiFi4 + fp32 dest acc, which is
    # the trunk's own compute kernel config. A large square matmul, so the rate is the card's
    # and not a shape artefact.
    for m in (4096,):
        x = rand(dev, (1, 1, m, m))
        w = rand(dev, (m, m))
        ms = bench(dev, lambda x=x, w=w: ttnn.linear(
            x, w, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)) * 1e3
        tf = 2 * m ** 3 / (ms / 1e3) / 1e12
        out["roofs"][f"matmul_{m}"] = {"ms": round(ms, 4), "TFLOPs": round(tf, 2)}
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        print(f"matmul {m}^3: {ms:.3f} ms = {tf:.2f} TFLOP/s", flush=True)

    # --- the trimul triangle product at the real shape and the shipped program config, which is
    # the single biggest matmul in the block. Its rate is not the roof; the distance between the
    # two is the lever.
    for n in a.tokens:
        st_ = (n + 31) // 32
        cs = T._trimul_chunk_size(n, C_HIDDEN, 1)
        pc = T._triangle_mul_program_config(st_)
        mc = T._triangle_mul_memory_config(n)
        aa = rand(dev, (1, cs, n, n))
        bb = rand(dev, (1, cs, n, n))
        try:
            ms = bench(dev, lambda aa=aa, bb=bb: ttnn.matmul(
                aa, bb, compute_kernel_config=ckc, memory_config=mc,
                program_config=pc, dtype=ttnn.bfloat16), n=5) * 1e3
            tf = 2 * cs * n ** 3 / (ms / 1e3) / 1e12
            out["roofs"][f"trimul_mm_{n}_c{cs}"] = {
                "ms": round(ms, 4), "TFLOPs": round(tf, 2),
                "chunk": cs, "chunks_per_trimul": C_HIDDEN // cs,
                "l1_dest": mc.buffer_type == ttnn.BufferType.L1}
            print(f"trimul mm {n} chunk {cs}: {ms:.3f} ms = {tf:.2f} TFLOP/s", flush=True)
        except Exception as e:                                                # noqa: BLE001
            out["roofs"][f"trimul_mm_{n}_c{cs}"] = {"err": str(e)[:200]}
            print(f"trimul mm {n} chunk {cs}: {e}"[:200], flush=True)
        ttnn.deallocate(aa)
        ttnn.deallocate(bb)

    # --- the two floors
    for n in a.tokens:
        f = flops_per_block(n)
        b = bytes_per_block(n)
        roof_tf = out["roofs"]["matmul_4096"]["TFLOPs"]
        roof_gb = out["roofs"][f"copy_{n}x{n}x{C_Z}"]["GBps"]
        comp_ms = f["_total"] / (roof_tf * 1e12) * 1e3
        bw_ms = b["_total"] / (roof_gb * 1e9) * 1e3
        execs = N_BLOCKS * N_CYCLES
        out["cells"][str(n)] = {
            "flops_per_block": {k: float(v) for k, v in f.items()},
            "bytes_per_block": {k: float(v) for k, v in b.items()},
            "roof_TFLOPs": roof_tf, "roof_GBps": roof_gb,
            "compute_floor_ms_per_block": round(comp_ms, 3),
            "bandwidth_floor_ms_per_block": round(bw_ms, 3),
            "binds": "compute" if comp_ms > bw_ms else "bandwidth",
            "floor_ms_per_block": round(max(comp_ms, bw_ms), 3),
            "block_execs": execs,
            "stack_floor_s": round(max(comp_ms, bw_ms) * execs / 1e3, 3),
            "arithmetic_intensity_FLOP_per_byte": round(f["_total"] / b["_total"], 2),
        }
        print(f"\n=== {n} tokens: compute floor {comp_ms:.2f} ms/block, bandwidth floor "
              f"{bw_ms:.2f} ms/block -> {'COMPUTE' if comp_ms > bw_ms else 'BANDWIDTH'} binds; "
              f"{max(comp_ms, bw_ms) * execs / 1e3:.2f} s for {execs} block executions",
              flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
