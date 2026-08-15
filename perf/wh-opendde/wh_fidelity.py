"""What a math-fidelity pass and fp32 dest accumulation actually cost on Wormhole, at OpenDDE's shapes.

tenstorrent.py:238-252 prices HiFi3 on Blackhole: 48 against 64 cycles per tile, 1.071x time-weighted
at 512 aa, 0.08 % of relative RMS where the bf16 output already costs 1.8 %. It ships behind
TT_BIO_TRUNK_MATH_FIDELITY and defaults to hifi4. That arithmetic is a Blackhole arithmetic. This
measures the same ladder on the 8x9 Galaxy grid at the three matmuls the OpenDDE trunk is made of,
so the Wormhole value of an already-built, zero-build-cost lever is a number rather than a transfer.

fp32_dest_acc_en is measured beside it because Wormhole halves the DEST register when it is on and
Blackhole does not, so the two parts can disagree about what that flag costs.
"""
import os, json, time, statistics
import torch
import ttnn

out = {"arch": ttnn.get_arch_name()}
dev = ttnn.open_device(device_id=0)
try:
    g = dev.compute_with_storage_grid_size()
    GX, GY = int(g.x), int(g.y)
    out["grid"] = [GX, GY]

    def ckc(fid, fp32=True):
        return ttnn.types.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
            fp32_dest_acc_en=fp32, packer_l1_acc=True)

    def bench(fn, iters=10, warm=2, blocks=5):
        for _ in range(warm):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(blocks):
            t0 = time.perf_counter()
            for _ in range(iters):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) / iters)
        return statistics.median(ts)

    def mk(shape):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch.manual_seed(0)
    S = 512
    xt = torch.randn(1, S, S, 384, dtype=torch.bfloat16)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    res = {}
    for N in (1152, 384, 1536):
        wt = torch.randn(384, N, dtype=torch.bfloat16)
        w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        flop = 2 * S * S * 384 * N
        ref = (xt.float().reshape(-1, 384) @ wt.float()).reshape(1, S, S, N)
        refn = ref.pow(2).mean().sqrt()
        cell = {}
        base_t = None
        for tag, fid, fp32 in (("hifi4_fp32acc", "HiFi4", True), ("hifi3_fp32acc", "HiFi3", True),
                               ("hifi2_fp32acc", "HiFi2", True), ("lofi_fp32acc", "LoFi", True),
                               ("hifi4_nofp32acc", "HiFi4", False), ("hifi3_nofp32acc", "HiFi3", False)):
            k = ckc(fid, fp32)
            fn = lambda k=k, w=w: ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w, compute_kernel_config=k,
                dtype=ttnn.bfloat16, config=None)
            try:
                t = bench(fn)
            except Exception as e:
                cell[tag] = {"err": "%s" % e}
                continue
            r = ttnn.to_torch(fn()).float()
            if base_t is None:
                base_t = t
                base_r = r
            cell[tag] = {
                "ms": t * 1e3, "TFLOPs": flop / t / 1e12, "speedup_vs_hifi4_fp32acc": base_t / t,
                "relRMSD_vs_fp32": float((r - ref).pow(2).mean().sqrt() / refn),
                "torch_equal_to_hifi4_fp32acc": bool(torch.equal(r, base_r)),
            }
        res["S%d_384x%d" % (S, N)] = cell
        ttnn.deallocate(w)
    out["results"] = res
finally:
    ttnn.close_device(dev)
print("FID_JSON " + json.dumps(out, indent=2))
