#!/usr/bin/env python3
"""Turn a roofs_card.py measurement into the schema mm_replay.py reads, and add the one roof
roofs_card.py does not measure: fp32 compute.

roofs_card.py reports a bf16 HiFi4 compute peak and directional DRAM read/write peaks. Three of the
batched sites in this audit run on fp32 operands (openfold3's `fp32_softmax` attention pair and
ESMFold2's `_attn_fp32`), and sizing those against the bf16 roof would overstate their headroom by
whatever the fp32 throughput ratio turns out to be. So the fp32 square-matmul roof is measured here
with the same method as the bf16 one -- same sizes, same HiFi4 + fp32_dest_acc + packer_l1_acc
kernel config, same synchronise-both-sides timing -- so the two are comparable.

    TT_VISIBLE_DEVICES=0 python3 perf/attn_sites/roofs_adapt.py raw.json out.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
import ttnn

from tt_bio.tenstorrent import get_device

raw = json.loads(Path(sys.argv[1]).read_text())
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(fn, warm=4, pipe=4, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


print("=== fp32 compute roof: square matmul, HiFi4, DRAM out ===", flush=True)
fp32 = {}
for n in (2048, 4096):
    a = ttnn.from_torch(torch.randn(1, 1, n, n), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(torch.randn(1, 1, n, n), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    try:
        s = timed(lambda: ttnn.deallocate(ttnn.matmul(
            a, b, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)))
        tf = 2 * n ** 3 / s / 1e12
        fp32[str(n)] = {"ms": round(s * 1e3, 4), "tflops": round(tf, 2)}
        print(f"  N={n:<5} {s*1e3:9.4f} ms {tf:8.2f} TFLOP/s", flush=True)
    except Exception as e:                                                    # noqa: BLE001
        print(f"  N={n} ERR {str(e)[:120]}", flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    del a, b

peak_fp32 = max((v["tflops"] for v in fp32.values()), default=0.0)
out = {
    "host": "pc", "card": 0, "ttnn": "0.68.0",
    "compute_bf16_TFLOPs": raw["compute_roof"]["peak_TFLOPs"],
    "compute_fp32_TFLOPs": peak_fp32,
    "dram_read_GBs": raw["dram_roofs"]["read_peak_GBs"],
    "dram_write_GBs": raw["dram_roofs"]["write_peak_GBs"],
    "machine_balance_FLOP_per_byte_read": raw["machine_balance_FLOP_per_byte_read"],
    "fp32_runs": fp32,
    "raw": raw,
}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(f"\nbf16 compute roof {out['compute_bf16_TFLOPs']} TFLOP/s   "
      f"fp32 {peak_fp32} TFLOP/s   read {out['dram_read_GBs']} GB/s   "
      f"write {out['dram_write_GBs']} GB/s   balance "
      f"{out['machine_balance_FLOP_per_byte_read']} FLOP/byte", flush=True)
print("wrote", sys.argv[2], flush=True)
