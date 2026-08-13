#!/usr/bin/env python3
"""S2b -- the whole row-blocked pair FFN with an L1-resident fc1, at the shapes it ships at.

screen_a proved a bit-exact L1 destination for fc1 exists (0.312 -> 0.164 ms at rows=32) and that
the widest config clashes once a half is already resident. This screens the CHAIN, which is the
number the landing is predicted from, over the configs whose circular-buffer footprint leaves room
for what is live, and over both residency plans:

    both  -- h1, h2 AND the gated product in L1   (100.7 MB, the largest)
    fc1   -- h1, h2 in L1, the gated product to DRAM (67.1 MB)

Every arm torch.equal against the unblocked reference. Nothing is built until this says so.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, ttnn
import tt_bio.tenstorrent as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--L', type=int, default=512)
    ap.add_argument('--n', type=int, default=5)
    a = ap.parse_args()
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get('TT_MESH_GRAPH_DESC_PATH'):
        mgd = _find_ttnn_mesh_graph_descriptor('p150_mesh_graph_descriptor.textproto')
        if mgd:
            os.environ['TT_MESH_GRAPH_DESC_PATH'] = mgd
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    L, CZ, FF = a.L, 256, 1024
    CK = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    R = {'host': os.uname().nodename, 'card': os.environ.get('TT_VISIBLE_DEVICES'),
         'grid': [g.x, g.y], 'L': L, 'n': a.n, 'arms': {},
         'l1_bank_bytes': T._l1_bank_bytes(),
         'loadavg': open('/proc/loadavg').read().split()[0]}

    def bench(fn, n=None, warm=2):
        n = n or a.n
        for _ in range(warm):
            o = fn(); ttnn.synchronize_device(dev)
            if isinstance(o, ttnn.Tensor): ttnn.deallocate(o)
        ts = []
        for _ in range(n):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
            if isinstance(o, ttnn.Tensor): ttnn.deallocate(o)
        return st.median(ts)

    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    torch.manual_seed(0)
    nw, nb = f(torch.randn(CZ)), f(torch.randn(CZ))
    w1a = f((torch.randn(FF, CZ) * 0.02).t())
    w1b = f((torch.randn(FF, CZ) * 0.02).t())
    w2 = f((torch.randn(CZ, FF) * 0.02).t())
    z = f(torch.randn(1, L, L, CZ))
    SILU = [ttnn.UnaryOpType.SILU]

    def cfg(m_tiles, bw, obh, obw, n_tiles=FF // 32):
        gx, gy = T.COMPUTE_GRID_MAIN
        per_core_M = -(-(-(-m_tiles // (gx * gy))) // obh) * obh
        sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
        sw = max(w for w in range(min(4 // sh, obw), 0, -1) if obw % w == 0)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
            out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=obw,
            per_core_M=per_core_M, per_core_N=n_tiles, fuse_batch=True,
            fused_activation=None, mcast_in0=False)

    def chain(rows, c, gated_l1):
        xn = ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=CK)
        parts = ttnn.chunk(xn, -(-L // rows), dim=1) if rows else [xn]
        if rows: ttnn.deallocate(xn)
        outs = []
        for p in parts:
            if c is None:
                kw = dict(compute_kernel_config=CK, dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)
            else:
                kw = dict(compute_kernel_config=CK, dtype=ttnn.bfloat16,
                          memory_config=ttnn.L1_MEMORY_CONFIG, program_config=c)
            h1, h2 = ttnn.linear(p, w1a, **kw), ttnn.linear(p, w1b, **kw)
            ttnn.deallocate(p)
            gt = ttnn.multiply(h1, h2, input_tensor_a_activations=SILU,
                               **({'memory_config': ttnn.L1_MEMORY_CONFIG} if gated_l1 else {}))
            ttnn.deallocate(h1); ttnn.deallocate(h2)
            outs.append(ttnn.linear(gt, w2, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                                    core_grid=T.CORE_GRID_MAIN))
            ttnn.deallocate(gt)
        if len(outs) == 1: return outs[0]
        o = ttnn.concat(outs, dim=1)
        for x in outs: ttnn.deallocate(x)
        return o

    ref = ttnn.to_torch(chain(0, None, False))
    R['arms']['unblocked'] = {'ms': round(bench(lambda: chain(0, None, False)), 4), 'equal': True}
    R['arms']['shipped_rows32'] = {
        'ms': round(bench(lambda: chain(32, None, True)), 4),
        'equal': bool(torch.equal(ref, ttnn.to_torch(chain(32, None, True))))}
    for k in ('unblocked', 'shipped_rows32'):
        print(f"  {k:24s} {R['arms'][k]['ms']:8.3f} ms equal={R['arms'][k]['equal']}", flush=True)

    mt = 32 * (L // 32)
    for obw in (32, 16, 8, 4):
        for gl1 in (True, False):
            tag = f'rows32_bw1_obh5_obw{obw}_gated{"L1" if gl1 else "DRAM"}'
            try:
                c = cfg(mt, 1, 5, obw)
                out = ttnn.to_torch(chain(32, c, gl1))
                eq = bool(torch.equal(ref, out))
                ms = bench(lambda: chain(32, c, gl1))
                R['arms'][tag] = {'ms': round(ms, 4), 'equal': eq}
                print(f'  {tag:36s} {ms:8.3f} ms equal={eq}', flush=True)
            except Exception as e:
                R['arms'][tag] = {'refused': str(e)[:160]}
                print(f'  {tag:36s} REFUSED {str(e)[:90]}', flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1))
    print('wrote', a.out)


main()
