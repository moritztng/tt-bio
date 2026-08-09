"""Dump the matmul program config ttnn actually chooses, per shape, with no timing.

ttnn resolves the config inside the device op and logs it at debug level
(`matmul_program_config.cpp: "Auto generated program config"`), so run this with
TT_METAL_LOGGER_LEVEL=Debug and read the line that follows each SHAPE marker.
`create_matmul_attributes` does NOT resolve it -- it returns program_config=None.
"""
import os, sys, ttnn, torch

CG = None  # set in main

SHAPES = [
    ("pv2 atom-attn AV    protenix.py:414", (75, 4, 32, 32), (75, 4, 32, 128), ttnn.float32, False),
    ("pv2 atom-attn QK^T  protenix.py:417", (75, 4, 32, 128), (75, 4, 128, 32), ttnn.float32, False),
    ("pv2 DiT AV         tenstorrent:1656", (1, 16, 320, 320), (1, 16, 320, 64), ttnn.float32, False),
    ("pv2 DiT QK^T       tenstorrent:1650", (1, 16, 320, 64), (1, 16, 64, 320), ttnn.float32, False),
    ("odde DiT AV        tenstorrent:1678", (1, 16, 608, 608), (1, 16, 608, 64), ttnn.bfloat16, False),
    ("odde DiT QK^T      tenstorrent:1670", (1, 16, 608, 64), (1, 16, 64, 608), ttnn.bfloat16, False),
    ("pair proj 298aa kt=8  no core_grid", (1, 102400, 256), (1, 256, 256), ttnn.bfloat16, False),
    ("pair proj 298aa kt=8  core_grid", (1, 102400, 256), (1, 256, 256), ttnn.bfloat16, True),
    ("single 298aa kt=24    core_grid", (1, 320, 768), (1, 768, 768), ttnn.bfloat16, True),
]


def main():
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    print(f"SHAPEMARK grid x={g.x} y={g.y} cores={g.x * g.y}", flush=True)
    cg = ttnn.CoreGrid(y=g.y, x=g.x)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    for label, sa, sb, dt, use_cg in SHAPES:
        a = ttnn.from_torch(torch.zeros(sa), dtype=dt, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.from_torch(torch.zeros(sb), dtype=dt, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"SHAPEMARK {label} kt={sa[-1] // 32} core_grid={use_cg}", flush=True)
        kw = {"core_grid": cg} if use_cg else {}
        try:
            o = ttnn.matmul(a, b, compute_kernel_config=ckc, **kw)
            ttnn.synchronize_device(dev)
            ttnn.deallocate(o)
        except Exception as e:
            print(f"SHAPEMARK   FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    ttnn.close_device(dev)


main()
