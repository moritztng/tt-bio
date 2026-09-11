#!/usr/bin/env python3
"""The two roofs the BioIR roofline comparison is checked against, measured on THIS p150a.

A vendor peak is not a roof (memory `roofline-roof-must-be-measured-not-asserted`), so every
denominator in FINDINGS.md comes from this file: a DRAM->DRAM copy roof and a DRAM read roof at
the tensor shapes Boltz-2 actually moves at 512 tokens, and a dense bf16 matmul rate at each
fidelity.  Median of 9 synced calls after 2 warm; every timed region syncs on both sides.
"""
import json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch, ttnn                                                            # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "arch": str(dev.arch()), "grid": [g.x, g.y],
       "compute_grid_main": list(T.COMPUTE_GRID_MAIN), "copy": [], "read": [], "matmul": []}


def bench(fn, n=9, warm=2):
    for _ in range(warm):
        r = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(r)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    return st.median(ts)


def mk(shape):
    return ttnn.from_torch(torch.zeros(*shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                           dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# --- copy roof: read N bytes + write N bytes -------------------------------------------
for shape in ([512, 512, 128], [512, 512, 384], [1024, 1024, 128], [8192, 8192]):
    nbytes = 2
    for d in shape:
        nbytes *= d
    t = mk(shape)
    ms = bench(lambda: ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)) * 1e3
    row = {"shape": shape, "MiB": round(nbytes / 2**20, 1), "clone_ms": round(ms, 4),
           "copy_roof_GBps": round(2 * nbytes / (ms * 1e-3) / 1e9, 1)}
    out["copy"].append(row); print("COPY " + json.dumps(row), flush=True)
    ttnn.deallocate(t)

# --- read-heavy roof: binary add, 2 reads + 1 write ------------------------------------
for shape in ([512, 512, 128], [8192, 8192]):
    nbytes = 2
    for d in shape:
        nbytes *= d
    a, b = mk(shape), mk(shape)
    ms = bench(lambda: ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)) * 1e3
    row = {"shape": shape, "MiB": round(nbytes / 2**20, 1), "add_ms": round(ms, 4),
           "rw_roof_GBps": round(3 * nbytes / (ms * 1e-3) / 1e9, 1)}
    out["read"].append(row); print("READ " + json.dumps(row), flush=True)
    ttnn.deallocate(a); ttnn.deallocate(b)

# --- dense bf16 matmul rate -------------------------------------------------------------
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
for n in (2048, 4096, 8192):
    a, b = mk([n, n]), mk([n, n])
    fl = 2.0 * n ** 3
    for fname, fid in FID.items():
        cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, fp32_dest_acc_en=False,
                                               packer_l1_acc=False)
        try:
            ms = bench(lambda: ttnn.matmul(a, b, compute_kernel_config=cfg,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG), n=5, warm=2) * 1e3
        except Exception as e:  # noqa: BLE001
            print("MM FAIL %d %s %s" % (n, fname, str(e)[:120]), flush=True); continue
        row = {"N": n, "fidelity": fname, "GFLOP": round(fl / 1e9, 2), "ms": round(ms, 4),
               "TFLOPs": round(fl / (ms * 1e-3) / 1e12, 2)}
        out["matmul"].append(row); print("MM " + json.dumps(row), flush=True)
    ttnn.deallocate(a); ttnn.deallocate(b)

Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("WROTE " + sys.argv[1])
T.cleanup()
