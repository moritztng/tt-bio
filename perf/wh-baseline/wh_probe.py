import os, time, json, statistics
import ttnn

DEV_ID = int(os.environ.get("PROBE_DEV", "26"))
out = {"probe_dev_umd_id": DEV_ID}
out["arch"] = ttnn.get_arch_name()

dev = ttnn.open_device(device_id=DEV_ID)
try:
    g = dev.compute_with_storage_grid_size()
    out["compute_with_storage_grid"] = [int(g.x), int(g.y)]
    out["compute_cores"] = int(g.x) * int(g.y)
    for name, fn in [
        ("worker_grid_size", lambda: dev.worker_grid_size()),
        ("dram_grid_size", lambda: dev.dram_grid_size()),
        ("num_dram_channels", lambda: dev.num_dram_channels()),
        ("dram_size_per_channel", lambda: dev.dram_size_per_channel()),
        ("l1_size_per_core", lambda: dev.l1_size_per_core()),
        ("num_program_cache_entries", lambda: dev.num_program_cache_entries()),
    ]:
        try:
            v = fn()
            out[name] = [int(v.x), int(v.y)] if hasattr(v, "x") else int(v)
        except Exception as e:
            out[name] = "ERR: %s" % e
    try:
        out["max_worker_l1_unreserved"] = int(ttnn.get_max_worker_l1_unreserved_size())
    except Exception as e:
        out["max_worker_l1_unreserved"] = "ERR: %s" % e

    # --- DRAM roofs. Batched iterations, ONE synchronize, so per-op sync does not
    # inflate the cost (memory: isolated per-op timing oversyncs ~2x).
    def bench(fn, iters=20, warm=3):
        for _ in range(warm):
            fn()
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) / iters)
        return min(ts), statistics.median(ts)

    import torch
    N = 4096
    x = ttnn.from_torch(torch.randn(1, 1, N, N, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    nbytes = N * N * 2

    # copy: reads nbytes, writes nbytes
    best, med = bench(lambda: ttnn.clone(x))
    out["dram_copy_MB"] = nbytes / 1e6
    out["dram_copy_best_ms"] = best * 1e3
    out["dram_copy_rw_GBs_best"] = 2 * nbytes / best / 1e9
    out["dram_copy_rw_GBs_med"] = 2 * nbytes / med / 1e9

    # unary read+write (same traffic shape, different kernel) as a cross-check
    best2, med2 = bench(lambda: ttnn.neg(x))
    out["dram_neg_rw_GBs_best"] = 2 * nbytes / best2 / 1e9

    # binary: reads 2*nbytes, writes nbytes
    y = ttnn.clone(x)
    best3, med3 = bench(lambda: ttnn.add(x, y))
    out["dram_add_rw_GBs_best"] = 3 * nbytes / best3 / 1e9

    # L1-resident unary: same op, operands in L1 -> upper bound not bound by DRAM
    try:
        xl = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        best4, _ = bench(lambda: ttnn.neg(xl))
        out["l1_neg_rw_GBs_best"] = 2 * nbytes / best4 / 1e9
        out["l1_resident_tensor_MB"] = nbytes / 1e6
    except Exception as e:
        out["l1_neg_rw_GBs_best"] = "ERR: %s" % e
finally:
    ttnn.close_device(dev)

print("PROBE_JSON " + json.dumps(out, indent=2))
