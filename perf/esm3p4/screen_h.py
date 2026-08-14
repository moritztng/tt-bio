#!/usr/bin/env python3
"""S7 -- where the FFN's remaining 12.2 ms goes, and whether fc2 has a bit-exact config.

The chain is norm 0.9575 + chunk 0.7386 + 16 x [fc1 x2, multiply, fc2] + concat 0.7265 = 14.657 ms.
fc1 was swept (L2) and is 0.160 ms per block. fc2's operand class, [1,32,512,1024] x [1024,256],
has NEVER been swept: k_tiles 32, n_tiles 8, so it is neither the fc1 class nor the narrow class
`_narrow_proj_linear` handles (n_tiles <= 2). This screens it the same way S2 screened fc1.

Also settles, with evidence rather than reasoning, whether `ttnn.slice` can ever be a VIEW over the
parent buffer. If it can, the chunk and the concat are both free and the FFN's 1.4651 ms of
assembly disappears without a kernel; if it cannot, then a copy-into-offset kernel IS the concat
and only a fused FFN can delete them.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, ttnn
import tt_bio.tenstorrent as T

ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--n', type=int, default=7)
a = ap.parse_args()
from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
if _detect_p300_devices() and not os.environ.get('TT_MESH_GRAPH_DESC_PATH'):
    m = _find_ttnn_mesh_graph_descriptor('p150_mesh_graph_descriptor.textproto')
    if m:
        os.environ['TT_MESH_GRAPH_DESC_PATH'] = m
dev = T.get_device()
gx, gy = T.COMPUTE_GRID_MAIN
L, CZ, FF, ROWS = 512, 256, 1024, 32
CK = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
R = {'host': os.uname().nodename, 'card': os.environ.get('TT_VISIBLE_DEVICES'),
     'grid': [gx, gy], 'n': a.n, 'fc2': {}, 'parts': {}}


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

# --- is a slice ever a view? ------------------------------------------------------------------
big = f(torch.randn(1, L, L, CZ))
sl = ttnn.slice(big, [0, 64, 0, 0], [1, 96, L, CZ])
R['slice_is_view'] = bool(sl.buffer_address() == big.buffer_address())
print(f"  slice buffer == parent buffer: {R['slice_is_view']}", flush=True)
ttnn.deallocate(sl); ttnn.deallocate(big)

# --- the per-block pieces ---------------------------------------------------------------------
gated = f(torch.randn(1, ROWS, L, FF))
w2 = f((torch.randn(CZ, FF) * 0.02).t())
h1, h2 = f(torch.randn(1, ROWS, L, FF)), f(torch.randn(1, ROWS, L, FF))
SILU = [ttnn.UnaryOpType.SILU]

shipped = lambda: ttnn.linear(gated, w2, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                              core_grid=T.CORE_GRID_MAIN)
ref = ttnn.to_torch(shipped())
ms_ref = bench(shipped)
R['fc2']['shipped_core_grid'] = {'ms': round(ms_ref, 4), 'equal': True}
R['parts']['multiply_silu'] = round(bench(
    lambda: ttnn.multiply(h1, h2, input_tensor_a_activations=SILU,
                          memory_config=ttnn.L1_MEMORY_CONFIG)), 4)
print(f"  fc2 shipped        {ms_ref:8.4f} ms   x16 = {ms_ref*16:6.3f} ms/call", flush=True)
print(f"  multiply+silu      {R['parts']['multiply_silu']:8.4f} ms   "
      f"x16 = {R['parts']['multiply_silu']*16:6.3f} ms/call", flush=True)

m_tiles = ROWS * (L // 32)
n_tiles = CZ // 32
best = None
for bw in (1, 2, 4, 8, 16, 32):
    if (FF // 32) % bw:
        continue
    for obh in (5, 4, 2, 1):
        per_core_M = -(-(-(-m_tiles // (gx * gy))) // obh) * obh
        if per_core_M > m_tiles or -(-m_tiles // per_core_M) > gx * gy:
            continue
        for obw in (8, 4, 2, 1):
            if n_tiles % obw:
                continue
            sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
            sw = max(w for w in range(min(4 // sh, obw), 0, -1) if obw % w == 0)
            c = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
                out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=obw,
                per_core_M=per_core_M, per_core_N=n_tiles, fuse_batch=True,
                fused_activation=None, mcast_in0=False)
            tag = f'bw{bw}_obh{obh}_obw{obw}'
            try:
                call = lambda c=c: ttnn.linear(gated, w2, compute_kernel_config=CK,
                                               dtype=ttnn.bfloat16,
                                               memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                               program_config=c)
                o = call()
                eq = bool(torch.equal(ref, ttnn.to_torch(o)))
                ttnn.deallocate(o)
                ms = bench(call)
                R['fc2'][tag] = {'ms': round(ms, 4), 'equal': eq}
                if eq and (best is None or ms < best[1]):
                    best = (tag, ms)
            except Exception as e:
                R['fc2'][tag] = {'refused': str(e)[:90]}
R['fc2_best_bit_exact'] = None if best is None else {
    'tag': best[0], 'ms': round(best[1], 4), 'x': round(ms_ref / best[1], 4),
    'fold_s_if_landed': round((ms_ref - best[1]) * 16 * 538 / 1.0315 / 1000, 3)}
print('  fc2 best bit-exact:', R['fc2_best_bit_exact'], flush=True)
print(f"  ({len([1 for v in R['fc2'].values() if isinstance(v, dict) and 'ms' in v])} candidates, "
      f"{len([1 for v in R['fc2'].values() if isinstance(v, dict) and v.get('equal')])} bit-exact)",
      flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(R, indent=1))
print('wrote', a.out)
