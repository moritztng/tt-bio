#!/usr/bin/env python3
"""Does moving dispatch off the Tensix column give Boltz-2 the 12th column back?

Blackhole is 14x10 Tensix. qb2 chip 3 harvests 2 columns (UMD mask tensix 0x140), leaving 120.
`blackhole_140_arch.yaml` spends the last remaining column (10 cores) on dispatch, so the model is
handed 11x10 = 110 -- which is the number both campaign roofs were priced against.
`blackhole_140_arch_eth_dispatch.yaml` puts dispatch on the idle ethernet cores instead and hands
back 12x10 = 120. On a p300c pinned to one chip the ethernet links carry no chip-to-chip traffic,
so they are free. ttnn only defaults to ETH dispatch on N300/T3K/N300_2x2; a Blackhole p300c always
gets WORKER.

One arm per process: the dispatch core config is baked into MetalContext at first device open.
"""
import json, os, sys, time

ARM = sys.argv[1]
import ttnn

cfg = None
if ARM == "eth":
    cfg = ttnn.DispatchCoreConfig(ttnn.device.DispatchCoreType.ETH)
elif ARM == "worker":
    cfg = ttnn.DispatchCoreConfig(ttnn.device.DispatchCoreType.WORKER)

kw = {"device_id": 0}
if cfg is not None:
    kw["dispatch_core_config"] = cfg

dev = ttnn.open_device(**kw)
out = {"arm": ARM}
try:
    cg = dev.compute_with_storage_grid_size()
    out["compute_grid"] = [cg.x, cg.y]
    out["compute_cores"] = cg.x * cg.y
    out["dispatch_core_type"] = str(cfg.type) if cfg is not None else "default"
    out["num_hw_cqs"] = getattr(dev, "num_hw_cqs", lambda: None)()
    out["l1_size_per_core"] = dev.l1_size_per_core()
    out["num_dram_channels"] = dev.num_dram_channels()
    out["dram_size_per_channel"] = dev.dram_size_per_channel()
    out["arch"] = str(dev.arch())
    out["ttnn"] = getattr(ttnn, "__version__", None)

    # prove the extra column is usable: an op that spans the whole compute grid
    import torch
    t = torch.randn(1, 1, 8192, 8192, dtype=torch.bfloat16)
    a = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ttnn.synchronize_device(dev)
    for _ in range(3):
        b = ttnn.clone(a)
        ttnn.deallocate(b)
    ttnn.synchronize_device(dev)
    ms = []
    for _ in range(15):
        t0 = time.perf_counter()
        b = ttnn.clone(a)
        ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(b)
    ms.sort()
    med = ms[len(ms) // 2]
    out["clone_8192_ms_median"] = round(med, 4)
    out["clone_8192_GBps"] = round(2 * 8192 * 8192 * 2 / (med / 1e3) / 1e9, 1)
    ttnn.deallocate(a)
finally:
    ttnn.close_device(dev)

print("RESULT " + json.dumps(out))
