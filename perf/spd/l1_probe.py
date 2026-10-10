"""DRAM vs L1-interleaved for the OpenFold3 DiT's fp32 stream ops at c730 (5 samples, 768 padded tokens, 768 channels).

cen25 puts ~12 s of the WH fold in the DiT's elementwise ops, each a 4800-call fp32 op at 0.6-0.9 of the DRAM roof.
This times each op with every operand in DRAM and with every operand in L1 (interleaved), plus the whole adaLN
(layer_norm, multiply by sigmoid(scale), add bias) and the to_memory_config that would move the stream in and out.
Checks that both placements return the same bits.

usage: TT_VISIBLE_DEVICES=N timeout 600 python perf/spd/l1_probe.py
"""
import glob
import statistics
import threading
import time

import torch
import ttnn

import tt_bio.tenstorrent as T

dev = T.get_device()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
S, N, C = 5, 768, 768
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
torch.manual_seed(0)
A = torch.randn(S, N, C)
SC, BI = torch.randn(1, N, C), torch.randn(1, N, C)


def up(x, mc, dt=ttnn.float32):
    return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def timed(f, n=50):
    f()
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(n):
        r = f()
        ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / n * 1e6


def adaln(a, sc, bi, mc):
    x = ttnn.layer_norm(a, epsilon=1e-5, compute_kernel_config=ckc, memory_config=mc)
    x = ttnn.multiply_(x, sc, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    return ttnn.add_(x, bi)


CLK = []


def sample_clock():
    # Every node's AICLK every 0.5 s while the probe runs: the min shows whether ANY chip on the box throttled.
    while True:
        v = [int(open(f).read().split()[0]) for f in glob.glob("/sys/class/tenstorrent/*/tt_aiclk")]
        CLK.extend(x for x in v if 0 < x < 5000)
        time.sleep(0.5)


threading.Thread(target=sample_clock, daemon=True).start()
res = {}
for name, mc in (("dram", DRAM), ("l1", L1)):
    a, sc, bi, a2 = up(A, mc), up(SC, mc), up(BI, mc), up(A * 0.5, mc)
    ops = {
        "layer_norm": lambda: ttnn.layer_norm(a, epsilon=1e-5, compute_kernel_config=ckc, memory_config=mc),
        "mul_sig_b": lambda: ttnn.multiply(a, sc, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
                                           memory_config=mc),
        "add_b": lambda: ttnn.add(a, bi, memory_config=mc),
        "add_full": lambda: ttnn.add(a, a2, memory_config=mc),
        "adaln": lambda: adaln(a, sc, bi, mc),
    }
    res[name] = {}
    for k, f in ops.items():
        us = timed(f)
        out = f()
        res[name][k] = (us, ttnn.to_torch(out).float())
        ttnn.deallocate(out)
    for t in (a, sc, bi, a2):
        ttnn.deallocate(t)

for k in res["dram"]:
    (d, od), (l, ol) = res["dram"][k], res["l1"][k]
    print(f"{k:11s} dram {d:8.1f} us  l1 {l:8.1f} us  ratio {d / l:5.2f}  bitexact {torch.equal(od, ol)}", flush=True)

a = up(A, DRAM)
print(f"to_l1      {timed(lambda: ttnn.to_memory_config(a, L1)):8.1f} us", flush=True)
a1 = up(A, L1)
print(f"to_dram    {timed(lambda: ttnn.to_memory_config(a1, DRAM)):8.1f} us", flush=True)
print(f"aiclk all nodes during the probe: median {statistics.median(CLK) if CLK else 'n/a'} min {min(CLK, default='n/a')} n {len(CLK)}",
      flush=True)
