#!/usr/bin/env python3
"""The two roofs every byte model in this doc is checked against, measured on THIS card.

This lineage has published 668 GB/s on a ~400 GB/s card and separately made an op at 96 % of the read
roof look like 13 %, both by inheriting a roof. So both roofs are measured on card 1 here, at
OpenDDE's own pair-tensor shapes, and nothing in the doc mixes them with the card-2 fold wall or the
card-3 roofs the predecessor measured.

  copy roof     `ttnn.clone` DRAM->DRAM, read+write bytes / time
  compute roof  HiFi4 bf16 square matmul with the production compute kernel config, DRAM result
"""
import json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch, ttnn                                                            # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor  # noqa: E402

if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
ckc = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "grid": [g.x, g.y], "loadavg": open("/proc/loadavg").read().split()[:3],
       "copy": [], "compute": []}


def med(fn, warm=2, reps=9):
    for _ in range(warm):
        ttnn.deallocate(fn())
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
    return st.median(ts)


for shape in ((512, 512, 384), (512, 512, 512), (512, 512, 1152)):
    t = ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    gb = 2 * torch.tensor(shape).prod().item() / 1e9
    s = med(lambda: ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    out["copy"].append({"shape": list(shape), "gb": round(gb, 4), "ms": round(s * 1e3, 4),
                        "read_plus_write_gbs": round(2 * gb / s, 1)})
    print(" copy", out["copy"][-1], flush=True)
    ttnn.deallocate(t)

for n in (2048, 3072, 4096):
    x = ttnn.from_torch(torch.randn(n, n), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    w = ttnn.from_torch(torch.randn(n, n), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    gflop = 2 * n ** 3 / 1e9
    s = med(lambda: ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                core_grid=T.CORE_GRID_MAIN))
    out["compute"].append({"n": n, "gflop": round(gflop, 1), "ms": round(s * 1e3, 4),
                           "tflops": round(gflop / s / 1e3, 2)})
    print(" compute", out["compute"][-1], flush=True)
    ttnn.deallocate(x)
    ttnn.deallocate(w)

out["copy_roof_gbs"] = max(r["read_plus_write_gbs"] for r in out["copy"])
out["compute_roof_tflops"] = max(r["tflops"] for r in out["compute"])
out["machine_balance_flop_per_byte"] = round(
    out["compute_roof_tflops"] * 1e12 / (out["copy_roof_gbs"] * 1e9), 1)
p = ROOT / "perf" / "odde4x" / "roofs_card1.json"
p.write_text(json.dumps(out, indent=1))
print("copy roof", out["copy_roof_gbs"], "GB/s   compute roof", out["compute_roof_tflops"],
      "TFLOP/s   balance", out["machine_balance_flop_per_byte"], "FLOP/byte", flush=True)
print("wrote", p, flush=True)
T.cleanup()
