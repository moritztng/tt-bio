#!/usr/bin/env python3
"""W12 step 2: isolate the pair-tensor transpose memory config INSIDE the real block.

W6 measured the Pairformer block at 45.26 -> 39.67 ms with four changes bundled (the SDPA
program config, _transpose_memory_config, minimal_matmul, the silu-on-multiply). W12
measured the permute alone at 1.4624 ms -> 0.5976 ms in a microbenchmark, 2.45x. Those two
numbers do not compose: a microbenchmark times an op with nothing else in flight.

This runs W6's tenstorrent.py unmodified and flips ONE thing between processes:

  --force-dram   T._transpose_memory_config is replaced by a lambda returning DRAM
  (default)      W6's function, which picks L1 at this size

Everything else W6 changed is identical in both arms, so the delta is the transpose
destination and nothing else. One arm per process: an L1 overflow fragments the allocator
and poisons every later measurement in the same process.

  TT_VISIBLE_DEVICES=0 python3 perf/permute_l1/insitu_transpose.py --n 320 --out X.json
"""

import argparse
import json
import time

import torch

import ttnn
from tt_bio import protenix_weights as PW
import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import PairformerLayer, get_device

CKPT = "/home/ttuser/.boltz/protenix-v2.pt"
TRI_HEAD_DIM = 32
DEV = None


def timeit(fn, warm, iters):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(DEV)
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2], ts


def build_layer(ckc):
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    blk = {k[len("pairformer_stack.blocks.0."):]: v
           for k, v in sd.items() if k.startswith("pairformer_stack.blocks.0.")}
    remapped = PW.remap_pairformer_block(blk)
    c_z = remapped["tri_mul_out.p_in.weight"].shape[1]
    return PairformerLayer(TRI_HEAD_DIM, c_z // TRI_HEAD_DIM, 384 // 16, 16, True,
                           remapped, ckc), c_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--force-dram", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--dump-z", type=str, default=None)
    args = ap.parse_args()

    global DEV
    DEV = get_device()

    arm = "dram" if args.force_dram else "l1"
    if args.force_dram:
        T._transpose_memory_config = lambda t: ttnn.DRAM_MEMORY_CONFIG

    ckc = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build_layer(ckc)
    N = args.n
    torch.manual_seed(0)
    s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=DEV,
                        dtype=ttnn.bfloat16)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z) * 0.1, layout=ttnn.TILE_LAYOUT, device=DEV,
                        dtype=ttnn.bfloat16)

    # What does the config actually decide for the tensor the transpose sees? The permute is
    # applied to the 3-D view [N,N,c_z] inside TriangleAttention, so probe that shape.
    probe = ttnn.from_torch(torch.zeros(N, N, c_z), layout=ttnn.TILE_LAYOUT, device=DEV,
                            dtype=ttnn.bfloat16)
    mc = T._transpose_memory_config(probe)
    decided = "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"
    ttnn.deallocate(probe)
    print("arm=%s  N=%d c_z=%d  transpose destination = %s" % (arm, N, c_z, decided), flush=True)

    # Parity dump first, on clones: the block updates s and z in place, so a dump taken
    # after the timing loops would be the ninth block applied to itself.
    if args.dump_z:
        sc, zc = ttnn.clone(s), ttnn.clone(z)
        so, zo = layer(sc, zc)
        torch.save({"z": ttnn.to_torch(zo), "s": ttnn.to_torch(so)}, args.dump_z)
        print("  dumped one-block output -> %s" % args.dump_z, flush=True)

    res = {}
    holder = {}

    def sub(name, fn):
        med, ts = timeit(fn, args.warm, args.iters)
        res[name] = {"median_ms": med, "series": [round(t, 3) for t in ts]}
        print("  %-18s %8.3f ms   %s" % (name, med, [round(t, 2) for t in ts]), flush=True)

    def timed_free(name, fn):
        def g():
            fn()
            ttnn.deallocate(holder["u"])
        sub(name, g)

    # tri_att_end is the only site that runs the dim0/dim1 transpose (twice: input and
    # output). tri_att_start is the control: same op, same shapes, no transpose.
    timed_free("tri_att_end",
               lambda: holder.__setitem__("u", layer.triangle_attention_end(z, None)))
    timed_free("tri_att_start",
               lambda: holder.__setitem__("u", layer.triangle_attention_start(z, None)))
    timed_free("transition_z",
               lambda: holder.__setitem__("u", layer.transition_z(z)))

    # No deallocate: the block returns its inputs updated in place, so freeing the result
    # frees s and z themselves.
    sub("FULL_BLOCK", lambda: holder.__setitem__("blk", layer(s, z)))

    out = {"arm": arm, "n": N, "c_z": c_z, "transpose_destination": decided,
           "grid": list(T.COMPUTE_GRID_MAIN), "results": res}
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
