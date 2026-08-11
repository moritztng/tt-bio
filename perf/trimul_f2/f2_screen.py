#!/usr/bin/env python3
"""Phase-1 screen for the F2 fused kernel: does fusion actually remove the module's residuals?

The predecessor's screen (state/trimul-absolute-optimal.md §13) timed the CONTRACTION, which is
8.9 % of the module, and read 1.135x. A screen on a class that does not dominate cannot predict a
fused result. F2's case is the byte argument: 6307.8 MB -> 1073.7 MB by keeping the intermediates
in L1. So the screen that predicts F2 asks, for each class F2 claims to fix, whether the class is
DRAM-bound (fusion removes it) or transaction/issue-bound (fusion does not).

S1  the forward channel move, L1 -> L1 vs DRAM -> DRAM at the SAME shape. This is the decisive one:
    the move is 3.049 ms, the largest residual, and §5 already measured it at 86.7 GB/s to L1 vs
    88.8 to DRAM with a DRAM source. If L1 -> L1 is also ~88, the cost is the gather's transaction
    count and F2 inherits every millisecond of it.
S2  the contraction at C=64, which is what F2 issues per channel pass, against C=256, which is what
    §13 measured. 2C output blocks on 110 cores: 512 at C=256 (93 % occupancy) but 128 at C=64
    (58 %). §13's 1.135x may not survive the schedule that uses it.
S3  the in-projection at N=256 (F2's per-pass width) against N=1024 (today's G=8 width), DRAM and
    L1 output. F2 cannot issue the wide one: its per-pass output is 4 roles x 64 channels.

Every rate is bytes / time and is checked against the 375 GB/s combined roof measured on this card.
"""
import json
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tt_bio.tenstorrent as T                                              # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, get_device                # noqa: E402
from tt_bio import reblock_permute as RP                                    # noqa: E402

ROOF = 375.0                                                                # GB/s, measured, §3
OUT = {"roof_gbs": ROOF, "s1": [], "s2": [], "s3": []}
dev = get_device()
gx, gy = COMPUTE_GRID_MAIN
OUT["grid"] = [gx, gy]
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
torch.manual_seed(0)


def timed(fn, n=5, warm=2):
    """Best-of style: warm, then n runs behind one sync. Returns ms/call and the last outputs."""
    outs = []
    for _ in range(warm):
        for o in fn():
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        outs.append(fn())
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) * 1e3 / n
    for g in outs[:-1]:
        for o in g:
            ttnn.deallocate(o)
    return ms, outs[-1]


# ---------------------------------------------------------------- S1: the forward channel move
print("=== S1  forward channel move, source x destination ===", flush=True)
for N, C in ((512, 256), (512, 64), (512, 32), (352, 64)):
    ref = None
    host = torch.randn(1, N, N, C).bfloat16()
    payload = N * N * C * 2 / 1e6                                           # MB, one direction
    for src_name, src_mc in (("dram", DRAM), ("l1", L1)):
        try:
            x = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=src_mc)
        except Exception as e:                                              # noqa: BLE001
            OUT["s1"].append({"n": N, "c": C, "src": src_name, "err": str(e)[:180]})
            print(f"  N={N} C={C} src={src_name}: alloc refused: {str(e)[:110]}", flush=True)
            continue
        for dst_name, dst_mc in (("dram", DRAM), ("l1", L1)):
            row = {"n": N, "c": C, "src": src_name, "dst": dst_name, "payload_mb": payload}
            try:
                ms, outs = timed(lambda: [RP.reblock_permute(x, memory_config=dst_mc, device=dev)])
                got = ttnn.to_torch(outs[0])
                if ref is None:
                    ref = torch.permute(host, (0, 3, 1, 2)).contiguous()
                row["equal"] = bool(torch.equal(got.float(), ref.float()))
                row["ms"] = ms
                row["gbs_each_way"] = payload / 1e3 / (ms * 1e-3)
                row["pct_roof"] = 100.0 * (2 * payload / 1e3 / (ms * 1e-3)) / ROOF
                for o in outs:
                    ttnn.deallocate(o)
            except Exception as e:                                          # noqa: BLE001
                row["err"] = str(e)[:180]
            OUT["s1"].append(row)
            print(f"  N={N:3d} C={C:3d}  {src_name:4s} -> {dst_name:4s}  "
                  f"{row.get('ms', 0):7.3f} ms  {row.get('gbs_each_way', 0):6.1f} GB/s each way  "
                  f"{row.get('pct_roof', 0):5.1f} % of combined roof  equal={row.get('equal')}  "
                  f"{row.get('err', '')}", flush=True)
        ttnn.deallocate(x)

# ---------------------------------------------------------------- S2: contraction occupancy
print("\n=== S2  contraction at F2's per-pass channel count ===", flush=True)
N = 512
Nt = N // 32
prod = T._triangle_mul_program_config(Nt)
best_half = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=(gx, gy),
    in0_block_w=4, out_subblock_h=4, out_subblock_w=1, per_core_M=Nt, per_core_N=Nt // 2)
for C in (256, 128, 64, 32):
    a = ttnn.from_torch(torch.randn(1, C, N, N).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch.randn(1, C, N, N).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    flop = 2 * C * N * N * N
    halves = [b[:, :, :, : N // 2], b[:, :, :, N // 2:]]
    for label, fn in (
        ("production 2D-mcast", lambda: [ttnn.matmul(
            a, b, compute_kernel_config=ckc, memory_config=DRAM, program_config=prod,
            dtype=ttnn.bfloat16)]),
        ("no-mcast half-output", lambda: [ttnn.matmul(
            a, p, compute_kernel_config=ckc, memory_config=DRAM, program_config=best_half,
            dtype=ttnn.bfloat16) for p in halves]),
    ):
        row = {"c": C, "label": label, "blocks": 2 * C, "cores": gx * gy}
        try:
            ms, outs = timed(fn, n=4)
            for o in outs:
                ttnn.deallocate(o)
            row["ms"] = ms
            row["tflops"] = flop / (ms * 1e-3) / 1e12
            row["ms_per_channel"] = ms / C
        except Exception as e:                                              # noqa: BLE001
            row["err"] = str(e)[:180]
        OUT["s2"].append(row)
        print(f"  C={C:3d}  {label:22s} {row.get('ms', 0):7.3f} ms  "
              f"{row.get('tflops', 0):6.2f} TF/s  {row.get('ms_per_channel', 0)*1e3:6.2f} us/channel"
              f"  {row.get('err', '')}", flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)

# ---------------------------------------------------------------- S3: the in-projection width
print("\n=== S3  in-projection width, F2 per-pass (N=256) vs today's G=8 (N=1024) ===", flush=True)
M = 512 * 512
K = 256
x = ttnn.from_torch(torch.randn(1, 1, M, K).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)
for Nw in (256, 512, 1024):
    w = ttnn.from_torch(torch.randn(K, Nw).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    flop = 2 * M * K * Nw
    bytes_rw = (M * K + M * Nw) * 2 / 1e6
    for l1_out in (False, True):
        row = {"n_out": Nw, "l1_out": l1_out, "gflop": flop / 1e9, "mb_rw": bytes_rw}
        try:
            ms, outs = timed(lambda: [T._pair_proj_linear(x, w, ckc, ttnn.bfloat16, l1_out=l1_out)],
                             n=4)
            row["out_mc"] = str(outs[0].memory_config().buffer_type)
            for o in outs:
                ttnn.deallocate(o)
            row["ms"] = ms
            row["tflops"] = flop / (ms * 1e-3) / 1e12
            row["gbs"] = bytes_rw / 1e3 / (ms * 1e-3)
        except Exception as e:                                              # noqa: BLE001
            row["err"] = str(e)[:180]
        OUT["s3"].append(row)
        print(f"  N={Nw:4d} l1_out={int(l1_out)}  {row.get('ms', 0):7.3f} ms  "
              f"{row.get('tflops', 0):6.2f} TF/s  {row.get('gbs', 0):6.1f} GB/s  "
              f"out={row.get('out_mc', '')}  {row.get('err', '')}", flush=True)
    ttnn.deallocate(w)

Path(sys.argv[1] if len(sys.argv) > 1 else "f2_screen.json").write_text(json.dumps(OUT, indent=1))
print("\nwrote", sys.argv[1] if len(sys.argv) > 1 else "f2_screen.json", flush=True)
