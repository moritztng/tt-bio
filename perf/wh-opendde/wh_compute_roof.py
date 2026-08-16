"""The Wormhole compute roof at OpenDDE's own matmul shapes, plus the shipped _MM_BLOCK entries.

The audit measured Wormhole's DRAM roofs and left the compute roof unmeasured, so no floor can be
derived for a module whose binding limit is the matmul rate (OpenDDE's TriangleAttention is
half traffic, half matmul rate -- state/opendde-512aa-deep-perf.md section 2.2). This measures:

  1. the same three matmuls the Blackhole decomposition timed, at OpenDDE's widths, so the
     architecture ratio is per-op and measured rather than modelled;
  2. a square 2048 matmul at HiFi4/fp32-acc and at LoFi, as the machine's own peak;
  3. the shipped _MM_BLOCK entries for OpenDDE ((12,36) and (12,12)) against the unconfigured op
     on the 8x9 grid -- they were swept on 110 Blackhole cores and carry no grid term in the
     block sizes, only in compute_with_storage_grid_size.

Batched iterations with one synchronize per block (memory: isolated per-op timing oversyncs ~2x).
"""
import os, json, time, statistics
import torch
import ttnn

DEV = int(os.environ.get("PROBE_DEV", "26"))
out = {"probe_dev_umd_id": DEV, "arch": ttnn.get_arch_name()}
dev = ttnn.open_device(device_id=0)
try:
    g = dev.compute_with_storage_grid_size()
    GX, GY = int(g.x), int(g.y)
    out["grid"] = [GX, GY]
    out["cores"] = GX * GY

    ckc = ttnn.types.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    ckc_lofi = ttnn.types.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    def bench(fn, iters=10, warm=2, blocks=5):
        for _ in range(warm):
            r = fn(); ttnn.deallocate(r)
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(blocks):
            t0 = time.perf_counter()
            for _ in range(iters):
                r = fn(); ttnn.deallocate(r)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) / iters)
        return min(ts), statistics.median(ts)

    def mk(shape):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    res = {}

    # --- 1/3. OpenDDE's own shapes, unconfigured and with the shipped entry -------------------
    #   qkv   [1,S,S,384] @ [384,1152]   key (kt=12, nt=36) -> (4,12,1,2,1)
    #   gate  [1,S,S,384] @ [384, 384]   key (kt=12, nt=12) -> (8,12,1,2,1)
    #   inproj[1,S,S,384] @ [384,1536]   no entry, unconfigured on both machines
    SHIPPED = {1152: (4, 12, 1, 2, 1), 384: (8, 12, 1, 2, 1)}
    for S in (512, 320):
        x = mk((1, S, S, 384))
        for N in (1152, 384, 1536):
            w = mk((384, N))
            flop = 2 * S * S * 384 * N
            key = "S%d_384x%d" % (S, N)
            base_fn = lambda w=w, x=x: ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
                dtype=ttnn.bfloat16, config=None)
            b, m = bench(base_fn)
            res[key] = {"flop": flop, "base_ms": m * 1e3, "base_best_ms": b * 1e3,
                        "base_TFLOPs": flop / m / 1e12}
            if N in SHIPPED:
                M, K, Nb, sh, sw = SHIPPED[N]
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=M, K_block_size=K, N_block_size=Nb, subblock_h=sh, subblock_w=sw,
                    compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY))
                cfg_fn = lambda w=w, x=x, cfg=cfg: ttnn.experimental.minimal_matmul(
                    input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
                    dtype=ttnn.bfloat16, config=cfg)
                try:
                    b2, m2 = bench(cfg_fn)
                    r0 = ttnn.to_torch(base_fn()); r1 = ttnn.to_torch(cfg_fn())
                    res[key].update({
                        "shipped_entry": list(SHIPPED[N]),
                        "cfg_ms": m2 * 1e3, "cfg_TFLOPs": flop / m2 / 1e12,
                        "ratio_base_over_cfg": m / m2,
                        "torch_equal": bool(torch.equal(r0, r1)),
                        "max_abs": float((r0.float() - r1.float()).abs().max()),
                    })
                except Exception as e:
                    res[key]["cfg_err"] = "%s" % e
            ttnn.deallocate(w)
        ttnn.deallocate(x)

    # --- 2. the machine's own square peak ------------------------------------------------------
    for n in (2048,):
        a = mk((1, 1, n, n)); b_ = mk((1, 1, n, n))
        flop = 2 * n ** 3
        for tag, kc in (("hifi4", ckc), ("lofi", ckc_lofi)):
            bb, mm = bench(lambda a=a, b_=b_, kc=kc: ttnn.matmul(
                a, b_, compute_kernel_config=kc, dtype=ttnn.bfloat16))
            res["square%d_%s" % (n, tag)] = {"flop": flop, "ms": mm * 1e3,
                                             "TFLOPs": flop / mm / 1e12,
                                             "TFLOPs_best": flop / bb / 1e12}
        ttnn.deallocate(a); ttnn.deallocate(b_)

    out["results"] = res
finally:
    ttnn.close_device(dev)
print("ROOF_JSON " + json.dumps(out, indent=2))
