#!/usr/bin/env python3
"""S3b -- the pair FFN chain at a 64-row block (§7 step 4.1, part B).

screen_c part A settled the bare op: a 64-row block DOES admit a bit-exact L1 destination, best
`bw=1 obh=2 obw=32` at 0.3009 ms against 0.7193 ms for the DRAM `core_grid` call, 2.390x. The
predecessor's "R=64 is refused" came from `out_block_w` hardcoded to n_tiles, not from capacity.

The chain is a different question, as screen_b already showed once: at rows=32 the bare-op optimum
(obw=32) clashes once one d_ff-wide half is resident and obw=16 wins instead. At rows=64 the two
halves are 67 MB each, so the gated product cannot also be L1-resident (201 MB against 110 x
1.46 MB), and it goes to DRAM -- worth 0.018 ms at rows=32, so not the term that matters.

Every candidate output is deallocated. screen_c part B died on exactly that leak.
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
         'loadavg': open('/proc/loadavg').read().split()[0]}

    def bench(fn, n=None, warm=2):
        n = n or a.n
        for _ in range(warm):
            o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
        ts = []
        for _ in range(n):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o)
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
        if per_core_M > m_tiles or -(-m_tiles // per_core_M) > gx * gy:
            return None
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
        if rows:
            ttnn.deallocate(xn)
        outs = []
        for p in parts:
            kw = (dict(compute_kernel_config=CK, dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)
                  if c is None else
                  dict(compute_kernel_config=CK, dtype=ttnn.bfloat16,
                       memory_config=ttnn.L1_MEMORY_CONFIG, program_config=c))
            h1, h2 = ttnn.linear(p, w1a, **kw), ttnn.linear(p, w1b, **kw)
            ttnn.deallocate(p)
            gt = ttnn.multiply(h1, h2, input_tensor_a_activations=SILU,
                               **({'memory_config': ttnn.L1_MEMORY_CONFIG} if gated_l1 else {}))
            ttnn.deallocate(h1); ttnn.deallocate(h2)
            outs.append(ttnn.linear(gt, w2, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                                    core_grid=T.CORE_GRID_MAIN))
            ttnn.deallocate(gt)
        if len(outs) == 1:
            return outs[0]
        o = ttnn.concat(outs, dim=1)
        for t in outs:
            ttnn.deallocate(t)
        return o

    def equal_to(refT, rows, c, gl1):
        o = chain(rows, c, gl1)
        eq = bool(torch.equal(refT, ttnn.to_torch(o)))
        ttnn.deallocate(o)
        return eq

    o = chain(0, None, False)
    ref = ttnn.to_torch(o)
    ttnn.deallocate(o)
    R['arms']['unblocked'] = {'ms': round(bench(lambda: chain(0, None, False)), 4), 'equal': True}
    R['arms']['shipped_l2_rows32_obw16'] = {
        'ms': round(bench(lambda: chain(32, cfg(512, 1, 5, 16), True)), 4),
        'equal': equal_to(ref, 32, cfg(512, 1, 5, 16), True)}
    for k, v in R['arms'].items():
        print(f"  {k:28s} {v['ms']:8.3f} ms equal={v['equal']}", flush=True)

    for rows, gl1 in ((64, False), (64, True), (128, False)):
        mt = rows * (L // 32)
        for obh in (5, 4, 2, 1):
            for obw in (32, 16, 8):
                c = cfg(mt, 1, obh, obw)
                if c is None:
                    continue
                tag = f'rows{rows}_bw1_obh{obh}_obw{obw}_gated{"L1" if gl1 else "DRAM"}'
                try:
                    eq = equal_to(ref, rows, c, gl1)
                    ms = bench(lambda: chain(rows, c, gl1))
                    R['arms'][tag] = {'ms': round(ms, 4), 'equal': eq}
                    print(f'  {tag:40s} {ms:8.3f} ms equal={eq}', flush=True)
                except Exception as e:
                    R['arms'][tag] = {'refused': str(e)[:110]}
                    print(f'  {tag:40s} REFUSED {str(e)[:60]}', flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1))
    print('wrote', a.out)


main()
