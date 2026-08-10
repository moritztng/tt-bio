#!/usr/bin/env python3
"""The compute-side ceiling for a K=256 contraction on THIS card, with the output kept in L1 so
DRAM is not what is being measured. Charter §4.6 says K=256 tops out well under the square roof;
this measures that number rather than inheriting it."""
import json, statistics as st, sys, time
from pathlib import Path
import torch, ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device  # noqa: E402

DEV = get_device()
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
ckc = ttnn.init_device_compute_kernel_config(
    DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(fn, warm=3, pipe=3, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(DEV)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(DEV)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


out = {}
for lbl, (m, k, n), mc in (("K256_M4096_N4096_oL1", (4096, 256, 4096), L1),
                           ("K256_M8192_N2048_oL1", (8192, 256, 2048), L1),
                           ("K256_M2048_N2048_oL1", (2048, 256, 2048), L1),
                           ("K1024_M4096_N4096_oL1", (4096, 1024, 4096), L1),
                           ("K4096_M4096_N4096_oL1", (4096, 4096, 4096), L1),
                           ("K256_M4096_N4096_oDRAM", (4096, 256, 4096), DRAM)):
    try:
        a = ttnn.from_torch(torch.randn(m, k), layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
        b = ttnn.from_torch(torch.randn(k, n), layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
        s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=mc)))
        out[lbl] = {"ms": round(s * 1e3, 4), "tflops": round(2 * m * k * n / s / 1e12, 2)}
        ttnn.deallocate(a); ttnn.deallocate(b)
    except Exception as e:                                          # noqa: BLE001
        out[lbl] = {"err": str(e)[:120]}
    print(f"  {lbl}: {json.dumps(out[lbl])}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
