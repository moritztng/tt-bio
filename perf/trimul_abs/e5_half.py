#!/usr/bin/env python3
"""E5's screen, the half-output form -- the only one that fits a core.

The whole-channel-per-core form refuses: `MatmulMultiCoreReuseProgramConfig` with per_core_N = Nt
needs a 512 KB bf16 output CB plus a 1 MB fp32 accumulator (HiFi4 + fp32_dest_acc_en) on a 1.43 MB
bank, and all 30 blockings throw `Statically allocated circular buffers ... beyond max L1 size`.
P4's own premise was that a core owns a channel "if the output is produced in halves", so this
measures exactly that: two matmuls over the halves of the output's N, per_core_N = Nt/2, no multicast.

The FLOP count is the full contraction's, so the rate is comparable to the production 2D-mcast number
measured in the same session. Reading A twice is priced in, deliberately: any real halved schedule
pays it unless it keeps A resident, which is the fused kernel's own claim and not this screen's.
"""
import json, sys, time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, get_device  # noqa: E402

N, C = 512, 256
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

prod = T._triangle_mul_program_config(Nt)


def run(fn, label, n=4):
    try:
        for _ in range(2):
            for o in fn():
                ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        got = [fn() for _ in range(n)]
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) * 1e3 / n
        for g in got:
            for o in g:
                ttnn.deallocate(o)
        row = {"label": label, "ms": ms, "tflops": FLOP / (ms * 1e-3) / 1e12}
    except Exception as e:                                                  # noqa: BLE001
        row = {"label": label, "err": str(e)[:220]}
    OUT["rows"].append(row)
    print(f"{label:56s} {row.get('ms', 0):8.3f} ms  {row.get('tflops', 0):6.2f} TF/s "
          f"{row.get('err', '')}", flush=True)
    return row


run(lambda: [ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                         program_config=prod, dtype=ttnn.bfloat16)],
    "PRODUCTION 2D-mcast, whole output")

halves = [b[:, :, :, : N // 2], b[:, :, :, N // 2:]]
best = None
for frac, pcn in ((2, Nt // 2), (4, Nt // 4)):
    parts = [b[:, :, :, i * (N // frac): (i + 1) * (N // frac)] for i in range(frac)]
    for ibw in (1, 2, 4, 8):
        if Nt % ibw:
            continue
        for osh, osw in ((1, 1), (2, 2), (1, 4), (4, 1)):
            if pcn % osw or Nt % osh or osh * osw > 4:
                continue
            pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=ibw, out_subblock_h=osh, out_subblock_w=osw,
                per_core_M=Nt, per_core_N=pcn,
            )
            r = run(lambda pc=pc, parts=parts: [
                ttnn.matmul(a, p, compute_kernel_config=ckc,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            program_config=pc, dtype=ttnn.bfloat16) for p in parts],
                f"no-mcast 1/{frac} output, per_core_N={pcn} ibw={ibw} osub={osh}x{osw}")
            if "tflops" in r and (best is None or r["tflops"] > best["tflops"]):
                best = r

cores, per_core = gx * gy, -(-C // (gx * gy))
OUT["occupancy"] = C / (cores * per_core)
OUT["production"] = OUT["rows"][0]
OUT["best_no_mcast"] = best
print(f"\nproduction {OUT['rows'][0].get('tflops', 0):.2f} TF/s | best no-mcast "
      f"{(best or {}).get('tflops', 0):.2f} TF/s | batch {C} on {cores} cores = {per_core} each, "
      f"occupancy {OUT['occupancy']*100:.1f} %")
if best and "tflops" in best:
    proj = best["tflops"] / OUT["occupancy"]
    OUT["balanced_projection"] = proj
    v = "PASS (>=60)" if proj >= 60 else ("AMBIGUOUS (45-60)" if proj >= 45 else "FAIL (<45)")
    print(f"at perfect occupancy that projects to {proj:.2f} TF/s -- P4 verdict: {v}")
Path(sys.argv[1] if len(sys.argv) > 1 else "e5_half.json").write_text(json.dumps(OUT, indent=1))
