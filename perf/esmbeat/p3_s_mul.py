#!/usr/bin/env python3
"""S-G: price the SwiGLU multiply, the single largest non-matmul term in the pair FFN.

p3_s_fork puts it at 3.588 ms/call at 512 aa = 1.93 s of fold, on operands that never touch DRAM.
805 MB of L1 traffic in 3.588 ms is 224 GB/s -- DRAM-roof class for an on-chip op, so the binding
term is either the SFPU (the SiLU pass) or the NoC (L1 INTERLEAVED is not core-local). This tells
the fusion fork which one to attack: an SFPU-bound multiply must be absorbed into a kernel with
spare SFPU, a NoC-bound one is fixed by sharding and needs no fused kernel at all.

Method is the one that killed protenix-v2's L2 without building anything (memory
fusion-into-compute-bound-kernel-unhides-arithmetic): toggle exactly one SFPU pass and read the
wall-clock delta.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio import esmc as EC


def timed(fn, dev, reps=4, batches=5, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / reps)
    return st.median(out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    L, C_Z, D_FF, ROWS = a.size, 256, 1024, EC._PAIR_FFN_ROW_BLOCK
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    ck = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    nblk = -(-L // ROWS)
    L1 = ttnn.L1_MEMORY_CONFIG
    DR = ttnn.DRAM_MEMORY_CONFIG
    mk = lambda mc: ttnn.from_torch(torch.randn(1, ROWS, L, D_FF), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "size": L, "blocks": nblk, "ms": {}, "raw": {}}
    h1l, h2l = mk(L1), mk(L1)
    h1d, h2d = mk(DR), mk(DR)
    SIL = [ttnn.UnaryOpType.SILU]

    arms = {
        # L1 operands, L1 out -- the shipped shape
        "l1_silu": lambda: ttnn.deallocate(
            ttnn.multiply(h1l, h2l, input_tensor_a_activations=SIL, memory_config=L1)),
        # same, SiLU removed: the one-SFPU-pass toggle
        "l1_plain": lambda: ttnn.deallocate(ttnn.multiply(h1l, h2l, memory_config=L1)),
        # DRAM operands, DRAM out -- the pre-L1 shipped shape, for the traffic reference
        "dram_silu": lambda: ttnn.deallocate(
            ttnn.multiply(h1d, h2d, input_tensor_a_activations=SIL, memory_config=DR)),
        "dram_plain": lambda: ttnn.deallocate(ttnn.multiply(h1d, h2d, memory_config=DR)),
        # a bare copy of one operand: pure movement at this shape and memory class
        "l1_copy": lambda: ttnn.deallocate(ttnn.clone(h1l, memory_config=L1)),
        "dram_copy": lambda: ttnn.deallocate(ttnn.clone(h1d, memory_config=DR)),
    }
    for name, fn in arms.items():
        m, raw = timed(fn, dev)
        res["ms"][name], res["raw"][name] = round(m, 4), [round(v, 4) for v in raw]
        print("%-11s %8.4f ms/block  x%d = %8.4f ms/call" % (name, m, nblk, m * nblk), flush=True)
    ms = res["ms"]
    res["per_call_ms"] = {k: round(v * nblk, 4) for k, v in ms.items()}
    res["silu_pass_ms_per_call"] = round((ms["l1_silu"] - ms["l1_plain"]) * nblk, 4)
    res["silu_pass_s_per_fold"] = round((ms["l1_silu"] - ms["l1_plain"]) * nblk * 538 / 1e3, 3)
    res["l1_vs_dram_ms_per_call"] = round((ms["l1_silu"] - ms["dram_silu"]) * nblk, 4)
    # bytes the multiply moves per block: read 2 operands, write 1
    b = 3 * ROWS * L * D_FF * 2
    res["GBps"] = {k: round(b / (v / 1e3) / 1e9, 1) for k, v in ms.items()
                   if not k.endswith("copy")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "raw"}, indent=1))


if __name__ == "__main__":
    main()
