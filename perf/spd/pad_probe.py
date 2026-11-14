"""pad_dim's device route against its host route, repeated, at OpenFold3 c730's diffusion shapes.

ab20 (t20, 60bfff52f) hung the device in the second diffusion step after the first went through,
so each case runs REPS times through the same program cache and is checked bit for bit against
the host route every time. Prints one line per case: max |diff|, ms per call on device and host.

usage (one chip, tt-bio's device path): TT_VISIBLE_DEVICES=N timeout 600 python perf/spd/pad_probe.py [REPS]
"""
import sys
import time

import torch
import ttnn

import tt_bio.tenstorrent as T

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
dev = T.get_device()
F32, BF16 = ttnn.float32, ttnn.bfloat16
L1 = ttnn.L1_MEMORY_CONFIG


def host(x, dtype, n, n_pad):
    th = torch.nn.functional.pad(ttnn.to_torch(x).float(), (0, 0, 0, n_pad - n))
    return ttnn.from_torch(th, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)


# (shape, n_pad, input dtype, output dtype, memory): atoms ql at 5917 -> 5920, tokens 730 -> 736
CASES = [((5, 5917, 128), 5920, F32, F32, None), ((5, 5917, 128), 5920, BF16, F32, None),
         ((1, 730, 768), 736, F32, F32, None), ((1, 730, 384), 736, BF16, F32, None),
         ((5, 730, 768), 736, F32, F32, None), ((5, 5917, 128), 5920, BF16, BF16, None),
         ((1, 730, 768), 736, F32, F32, L1)]
torch.manual_seed(0)
for shape, n_pad, din, dout, mem in CASES:
    n = shape[-2]
    x = ttnn.from_torch(torch.randn(shape), layout=ttnn.TILE_LAYOUT, device=dev, dtype=din,
                        memory_config=mem or ttnn.DRAM_MEMORY_CONFIG)
    ref = ttnn.to_torch(host(x, dout, n, n_pad))
    worst, t_dev = 0.0, []
    for _ in range(REPS):
        t0 = time.perf_counter()
        y = T.pad_dim(x, dout, n, n_pad)
        ttnn.synchronize_device(dev)
        t_dev.append(time.perf_counter() - t0)
        out = ttnn.to_torch(y)
        worst = max(worst, (out - ref).abs().max().item() if out.shape == ref.shape else float("inf"))
        ttnn.deallocate(y)
    t0 = time.perf_counter()
    for _ in range(3):
        ttnn.deallocate(host(x, dout, n, n_pad))
    ttnn.synchronize_device(dev)
    t_host = (time.perf_counter() - t0) / 3
    print(f"{shape} {din}->{dout} {'L1' if mem else 'DRAM'}: max|diff| {worst:.3g}  "
          f"device {1e3 * min(t_dev[1:] or t_dev):.2f} ms  host {1e3 * t_host:.2f} ms", flush=True)
print("PROBE-DONE", flush=True)
