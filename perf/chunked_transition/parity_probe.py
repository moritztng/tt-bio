"""E6: what the fitted Transition config does to the numbers, on the shapes the fold runs.

Both arms see the same live operands and the same L1 residency; the reference arm is byte-for-byte
the `ttnn.linear(core_grid=...)` call this branch replaces.
"""
import sys

import torch
import ttnn

sys.path.insert(0, "/home/ttuser/.coworker/wt/perfwar-chunked-transition-cb")
from tt_bio import tenstorrent as T  # noqa: E402


def main():
    dev = ttnn.open_device(device_id=0)
    T._configure_active_compute_grid(dev)
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    print("grid", T.COMPUTE_GRID_MAIN)

    def mk(shape, l1=False):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16), device=dev,
                               layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                               memory_config=ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG)

    CASES = [
        ("protenix-v2 fc1 silu", 30, 320, 256, 1024, "up", "silu"),
        ("protenix-v2 fc2     ", 30, 320, 256, 1024, "up", None),
        ("protenix-v2 fc3     ", 30, 320, 256, 1024, "down", None),
        ("protenix-v2 fc1 h28 ", 28, 320, 256, 1024, "up", "silu"),
        ("protenix-v2 fc3 h28 ", 28, 320, 256, 1024, "down", None),
        ("opendde     fc1 silu", 30, 320, 384, 1536, "up", "silu"),
        ("opendde     fc3     ", 30, 320, 384, 1536, "down", None),
    ]
    for name, h, W, c, hid, leg, act in CASES:
        out_l1 = leg == "up"
        mem = ttnn.L1_MEMORY_CONFIG if out_l1 else ttnn.DRAM_MEMORY_CONFIG
        res = ([mk((1, h, W, c), l1=True), mk((1, h, W, hid), l1=True)] if out_l1
               else [mk((1, h, W, hid), l1=True)])
        x = mk((1, h, W, c) if out_l1 else (1, h, W, hid))
        w = mk((c, hid) if out_l1 else (hid, c))
        ref = ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                          memory_config=mem, core_grid=T.CORE_GRID_MAIN, activation=act)
        got = T._transition_linear(x, w, ckc, ttnn.bfloat16, mem, activation=act)
        a, b = ttnn.to_torch(ref).float(), ttnn.to_torch(got).float()
        eq = torch.equal(ttnn.to_torch(ref), ttnn.to_torch(got))
        d = (a - b).abs()
        cfg = T._transition_linear.__wrapped__ if hasattr(T._transition_linear, "__wrapped__") else None
        print(f"{name}  torch.equal={eq!s:<5}  max|d|={d.max():.3e}  rel_max={d.max() / a.abs().max():.3e}  "
              f"rmsd/rms={(d.pow(2).mean().sqrt() / a.pow(2).mean().sqrt()):.3e}  fired={cfg is None}")
        for t in (ref, got, x, w, *res):
            ttnn.deallocate(t)
    print("\nconfigs chosen:")
    for k, v in T._transition_program_config.cache_info()._asdict().items():
        print(" ", k, v)
    ttnn.close_device(dev)


if __name__ == "__main__":
    sys.exit(main())
