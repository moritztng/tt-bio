"""Is the tape's compute-kernel config actually taking effect on Blackhole?

`autograd.precise_config()` builds a WormholeComputeKernelConfig and this is a Blackhole
card. Production builds its config through `ttnn.init_device_compute_kernel_config(arch,
...)` instead. If the Wormhole struct is silently ignored or partially applied here, every
backward in this row has been running at a lower fidelity than it claims, which is exactly
the size of effect the atom-block check is sitting on (1.6e-2 where torch fp32 reads
7e-7).

One matmul, one variable: the config. Reference is float64 numpy on the same inputs.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ttnn
from tt_bio.tenstorrent import get_device


def rel(a, b):
    return float(np.linalg.norm(np.float64(a) - np.float64(b)) / np.linalg.norm(np.float64(b)))


def main():
    dev = get_device()
    print("arch:", dev.arch())
    rng = np.random.default_rng(3)
    M = K = N = 512
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    b = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    ref = np.float64(a) @ np.float64(b)
    import torch
    ta = ttnn.from_torch(torch.from_numpy(a), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(torch.from_numpy(b), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    cfgs = {}
    cfgs["none"] = None
    cfgs["wormhole HiFi4 fp32acc"] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    try:
        cfgs["arch HiFi4 fp32acc"] = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
    except Exception as e:
        print("init_device_compute_kernel_config failed:", e)
    try:
        cfgs["arch HiFi4 no-fp32acc"] = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=False, packer_l1_acc=True)
    except Exception:
        pass
    try:
        cfgs["arch LoFi"] = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
    except Exception:
        pass

    print(f"{'config':<26} {'type':<34} {'rel L2 vs float64':>18}")
    for name, cfg in cfgs.items():
        out = ttnn.matmul(ta, tb, compute_kernel_config=cfg)
        got = ttnn.to_torch(out).float().numpy()
        print(f"{name:<26} {type(cfg).__name__:<34} {rel(got, ref):>18.3e}")


main()
