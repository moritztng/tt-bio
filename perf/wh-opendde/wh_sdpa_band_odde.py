"""Does `_sdpa_chunks_shipped`'s 64/64 band still win on a 72-core Wormhole grid?

The band (`256 < q_len <= 384 and 256 < k_len <= 384 -> (64, 64)`) has no grid term, and its
2.45x justification was measured by microbench M7/M7b/M7c on a 13x10 Blackhole. A 298-residue
protein pads to 320 and lands in it, so this is a size JapanFold serves every day.

Shapes are M7's: the tri-att SDPA, batch=seq, h=8, d=32, bf16. Each N is run under both chunk
configs -- the shipped band pick and what `_capped_sdpa_chunk_size` would have returned without
the band -- and the ratio is band/capped, so >1 means the band is winning.

Run with TT_VISIBLE_DEVICES=<free umd id>; the visible chip re-indexes to device 0.
"""
import json, os, statistics, sys, time

import ttnn

DEV = int(os.environ.get("PROBE_DEV", "0"))
WARMUP = int(os.environ.get("PROBE_WARMUP", "2"))
ITERS = int(os.environ.get("PROBE_ITERS", "3"))
BLOCKS = int(os.environ.get("PROBE_BLOCKS", "5"))
H = int(os.environ.get("PROBE_H", "8"))
D = int(os.environ.get("PROBE_D", "32"))

# N, and the two chunk picks to compare. 256/512 are the controls outside the band.
CASES = [
    (256, (256, 256), (256, 256)),   # control: band does not apply, both arms identical
    (288, (64, 64), (256, 256)),
    (320, (64, 64), (256, 256)),     # 298 aa pads here
    (352, (64, 64), (256, 256)),
    (384, (64, 64), (256, 256)),
    (512, (256, 256), (256, 256)),   # control
]


def pcfg(grid, q, k):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, exp_approx_mode=False,
        q_chunk_size=q, k_chunk_size=k,
    )


def time_sdpa(device, q, k, v, cfg, kcfg):
    for _ in range(WARMUP):
        ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, program_config=cfg, compute_kernel_config=kcfg)
    ttnn.synchronize_device(device)
    ms = []
    for _ in range(BLOCKS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, program_config=cfg, compute_kernel_config=kcfg)
        ttnn.synchronize_device(device)
        ms.append((time.perf_counter() - t0) * 1e3 / ITERS)
        o.deallocate()
    return {"best": min(ms), "median": statistics.median(ms), "all": [round(x, 3) for x in ms]}


def main():
    device = ttnn.open_device(device_id=DEV)
    try:
        g = device.compute_with_storage_grid_size()
        grid = ttnn.CoreCoord(g.x, g.y)
        kcfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        out = {"grid": [g.x, g.y], "cores": g.x * g.y, "h": H, "d": D,
               "warmup": WARMUP, "iters": ITERS, "blocks": BLOCKS, "cases": []}
        import torch
        for N, band, capped in CASES:
            shp = (N, H, N, D)
            t = torch.randn(shp, dtype=torch.float32).to(torch.bfloat16)
            q = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            k = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            v = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            rec = {"N": N, "band_chunks": list(band), "capped_chunks": list(capped)}
            try:
                rec["band"] = time_sdpa(device, q, k, v, pcfg(grid, *band), kcfg)
                rec["capped"] = time_sdpa(device, q, k, v, pcfg(grid, *capped), kcfg)
                rec["ratio_capped_over_band"] = round(
                    rec["capped"]["median"] / rec["band"]["median"], 4)
            except Exception as e:  # a config the wheel refuses is itself a result
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            for x in (q, k, v):
                x.deallocate()
            out["cases"].append(rec)
            print(json.dumps(rec), flush=True)
        print("RESULT " + json.dumps(out))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
