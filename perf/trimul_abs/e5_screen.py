#!/usr/bin/env python3
"""E5's screen (P4): can a per-core contraction with NO multicast beat the 2D-mcast matmul's 40.5 TF/s?

The fused-kernel case in state/trimul-absolute-optimal.md §6 rests on one claim: for a single channel
h the operands A_h and B_h are 512 KB each and the output 512 KB, so a core can own a whole channel
if the output is produced in halves, and 256 half-channels over 110 cores is 93 % occupancy with zero
multicast against the 2D-mcast factory's structurally capped 58 %. P4 says that measures >= 60 TFLOP/s
aggregate; under 45 kills the fused kernel and E1-E4 are the answer.

This screens it with the wheel's own no-multicast batched matmul rather than a hand-rolled kernel,
which is the cheap version of the question: `MatmulMultiCoreReuseProgramConfig` gives every batch
element its own `per_core_M x per_core_N` output on ONE core and multicasts nothing. If the wheel's
own per-core path cannot beat 40.5 on this shape, a hand-written one will not either -- the previous
hand-rolled contraction in this lineage measured 1.3 TFLOP/s.

Baseline is the PRODUCTION path: `_triangle_mul_program_config`, the config the fold actually calls.
"""
import json, sys, time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, get_device  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
C = 256
dev = get_device()
gx, gy = COMPUTE_GRID_MAIN
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
Nt = N // 32
FLOP = 2 * C * N * N * N
torch.manual_seed(0)
a = ttnn.from_torch(torch.randn(1, C, N, N).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)
b = ttnn.from_torch(torch.randn(1, C, N, N).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)

OUT = {"n": N, "c": C, "grid": [gx, gy], "gflop": FLOP / 1e9, "rows": []}


def timed(pc, label, n=4):
    try:
        for _ in range(2):
            ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                        program_config=pc, dtype=ttnn.bfloat16))
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        outs = [ttnn.matmul(a, b, compute_kernel_config=ckc,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            program_config=pc, dtype=ttnn.bfloat16) for _ in range(n)]
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) * 1e3 / n
        for o in outs:
            ttnn.deallocate(o)
        row = {"label": label, "ms": ms, "tflops": FLOP / (ms * 1e-3) / 1e12}
    except Exception as e:                                                  # noqa: BLE001
        row = {"label": label, "err": str(e)[:200]}
    OUT["rows"].append(row)
    print(f"{label:52s} {row.get('ms', 0):8.3f} ms  {row.get('tflops', 0):6.2f} TF/s "
          f"{row.get('err', '')}", flush=True)
    return row


print(f"contraction [1,{C},{N},{N}] @ same, {FLOP/1e9:.2f} GFLOP, grid {gx}x{gy}\n")
prod = timed(T._triangle_mul_program_config(Nt), "PRODUCTION 2D-mcast (_triangle_mul_program_config)")

# The no-multicast path: one batch element per core, whole per-core output, K blocked so the CBs fit
# the 1.43 MB bank. out_subblock_h*out_subblock_w <= 4 with fp32_dest_acc_en.
best = None
for ibw in (1, 2, 4, 8, 16):
    if Nt % ibw:
        continue
    for osh, osw in ((1, 1), (2, 2), (1, 4), (4, 1), (1, 2), (2, 1)):
        if Nt % osh or Nt % osw or osh * osw > 4:
            continue
        pc = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=ibw, out_subblock_h=osh, out_subblock_w=osw,
            per_core_M=Nt, per_core_N=Nt,
        )
        r = timed(pc, f"no-mcast batch-split ibw={ibw} osub={osh}x{osw}")
        if "tflops" in r and (best is None or r["tflops"] > best["tflops"]):
            best = r

OUT["production"] = prod
OUT["best_no_mcast"] = best
cores = gx * gy
per_core = -(-C // cores)
OUT["occupancy"] = C / (cores * per_core)
print(f"\nproduction {prod.get('tflops', 0):.2f} TF/s | best no-mcast "
      f"{(best or {}).get('tflops', 0):.2f} TF/s | batch {C} on {cores} cores = "
      f"{per_core} each, occupancy {OUT['occupancy']*100:.1f} %")
if best and "tflops" in best:
    OUT["balanced_projection"] = best["tflops"] / OUT["occupancy"]
    print(f"at perfect balance that projects to {OUT['balanced_projection']:.2f} TF/s")
    print(f"P4 verdict: {'PASS (>=60)' if OUT['balanced_projection'] >= 60 else ('AMBIGUOUS (45-60)' if OUT['balanced_projection'] >= 45 else 'FAIL (<45) -- the fused kernel is dead')}")
Path(sys.argv[2] if len(sys.argv) > 2 else "e5_screen.json").write_text(json.dumps(OUT, indent=1))
