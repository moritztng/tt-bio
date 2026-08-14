#!/usr/bin/env python3
"""S6 -- layer_norm at the trimul's production shape, against its own DRAM roof.

state/esmfold2-to-3p4x.md §10 left this unscreened: 2 per trimul call plus 1 per pair transition,
never measured above 66 % of any roof in this repo. At 1084 trimul calls a 0.3 ms/call saving is
0.6 s of fold, larger than the 0.440 s between 32.258 s and the target, so it is worth the sweep.

[1, 512, 512, 256] bf16 is 134.2 MB in and 134.2 MB out = 268.4 MB, which at the MEASURED
422.9 GB/s roof is 0.635 ms. Anything the sweep finds is bounded by that.

Every candidate torch.equal against the shipped call: a layer_norm whose reduction order changes
is a parity change, not a speedup.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, ttnn
import tt_bio.tenstorrent as T

ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--L', type=int, default=512)
ap.add_argument('--n', type=int, default=7)
a = ap.parse_args()
from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
if _detect_p300_devices() and not os.environ.get('TT_MESH_GRAPH_DESC_PATH'):
    m = _find_ttnn_mesh_graph_descriptor('p150_mesh_graph_descriptor.textproto')
    if m:
        os.environ['TT_MESH_GRAPH_DESC_PATH'] = m
dev = T.get_device()
g = dev.compute_with_storage_grid_size()
L, CZ = a.L, 256
CK = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
R = {'host': os.uname().nodename, 'card': os.environ.get('TT_VISIBLE_DEVICES'),
     'grid': [g.x, g.y], 'L': L, 'n': a.n, 'arms': {},
     'bytes_mb': round(2 * L * L * CZ * 2 / 1e6, 1),
     'roof_gbs': 422.9, 'loadavg': open('/proc/loadavg').read().split()[0]}
R['roof_ms'] = round(R['bytes_mb'] / 1e3 / R['roof_gbs'] * 1e3, 4)


def bench(fn, warm=2):
    for _ in range(warm):
        o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts = []
    for _ in range(a.n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
    return st.median(ts)


f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
torch.manual_seed(0)
z = f(torch.randn(1, L, L, CZ))
w, b = f(torch.randn(CZ)), f(torch.randn(CZ))

shipped = lambda: ttnn.layer_norm(z, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=CK)
ref = ttnn.to_torch(shipped())
ms = bench(shipped)
R['arms']['shipped'] = {'ms': round(ms, 4), 'equal': True,
                        'pct_of_roof': round(R['roof_ms'] / ms * 100, 1)}
print(f"  shipped            {ms:8.4f} ms   {R['arms']['shipped']['pct_of_roof']:5.1f} % of the "
      f"{R['roof_ms']:.4f} ms DRAM roof", flush=True)

CANDS = {
    'lofi': dict(compute_kernel_config=ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=True)),
    'no_fp32_acc': dict(compute_kernel_config=ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=True)),
    'approx': dict(compute_kernel_config=ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
        fp32_dest_acc_en=True, packer_l1_acc=True)),
    'default_ck': dict(),
}
for tag, kw in CANDS.items():
    try:
        call = lambda kw=kw: ttnn.layer_norm(z, weight=w, bias=b, epsilon=1e-5, **kw)
        o = call()
        eq = bool(torch.equal(ref, ttnn.to_torch(o)))
        ttnn.deallocate(o)
        ms = bench(call)
        R['arms'][tag] = {'ms': round(ms, 4), 'equal': eq,
                          'pct_of_roof': round(R['roof_ms'] / ms * 100, 1),
                          'x_vs_shipped': round(R['arms']['shipped']['ms'] / ms, 4)}
        print(f"  {tag:18s} {ms:8.4f} ms   {R['arms'][tag]['pct_of_roof']:5.1f} % of roof  "
              f"equal={eq}  {R['arms'][tag]['x_vs_shipped']:.3f}x", flush=True)
    except Exception as e:
        R['arms'][tag] = {'refused': str(e)[:120]}
        print(f'  {tag:18s} REFUSED {str(e)[:70]}', flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(R, indent=1))
print('wrote', a.out)
