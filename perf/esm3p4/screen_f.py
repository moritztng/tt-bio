#!/usr/bin/env python3
"""S5 -- what the row-blocked FFN's `chunk` and `concat` actually cost, and whether a
preallocated destination can replace them.

The blocked pair FFN is `chunk(x, 16, dim=1)` -> 16 x _ffn -> `concat(..., dim=1)`. Both copy the
whole [1,512,512,256] tensor, 134 MB each way. At the MEASURED 422.9 GB/s roof that is ~1.27 ms
per call and 0.68 s of a 512 aa fold -- larger than the 0.440 s still between the fold and the
target, which is why it is worth a screen rather than an assumption.

Measured here:
  A  the two ops alone, at the production shape
  B  chunk against a per-block `ttnn.slice`, which does not materialise all 16 parts at once
  C  whether `ttnn.linear` will write into a caller-supplied output tensor at all -- if it will
     not, the concat cannot be replaced by a preallocated destination without a kernel.
"""
import argparse, inspect, json, os, statistics as st, sys, time
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
L, CZ, FF, ROWS = a.L, 256, 1024, 32
R = {'host': os.uname().nodename, 'card': os.environ.get('TT_VISIBLE_DEVICES'),
     'L': L, 'n': a.n, 'arms': {}, 'loadavg': open('/proc/loadavg').read().split()[0]}


def bench(fn, warm=2):
    for _ in range(warm):
        o = fn(); ttnn.synchronize_device(dev)
        del o
    ts = []
    for _ in range(a.n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        del o
    return st.median(ts)


f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
torch.manual_seed(0)
z = f(torch.randn(1, L, L, CZ))
nblk = L // ROWS


def chunk_only():
    return ttnn.chunk(z, nblk, dim=1)


def slice_only():
    return [ttnn.slice(z, [0, r, 0, 0], [1, r + ROWS, L, CZ]) for r in range(0, L, ROWS)]


parts = ttnn.chunk(z, nblk, dim=1)
R['arms']['chunk_x16'] = round(bench(chunk_only), 4)
R['arms']['slice_x16'] = round(bench(slice_only), 4)
R['arms']['concat_x16'] = round(bench(lambda: ttnn.concat(list(parts), dim=1)), 4)
for k in ('chunk_x16', 'slice_x16', 'concat_x16'):
    print(f'  {k:14s} {R["arms"][k]:8.4f} ms', flush=True)
R['chunk_plus_concat_ms'] = round(R['arms']['chunk_x16'] + R['arms']['concat_x16'], 4)
print(f"  chunk+concat  {R['chunk_plus_concat_ms']:8.4f} ms of the 14.657 ms chain", flush=True)

# C -- can ttnn.linear take a caller-supplied destination?
sig = None
try:
    sig = str(inspect.signature(ttnn.linear))
except Exception as e:
    sig = f'unavailable: {e}'
R['linear_signature'] = sig
R['linear_has_output_tensor'] = any(k in sig for k in ('output_tensor', 'optional_output'))
print('  ttnn.linear signature:', sig[:220], flush=True)
print('  accepts a caller-supplied output tensor:', R['linear_has_output_tensor'], flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(R, indent=1))
print('wrote', a.out)
