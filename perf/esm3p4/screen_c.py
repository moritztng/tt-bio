#!/usr/bin/env python3
"""S3 -- does a 64-row block admit a bit-exact L1 destination? (§7 step 4.1)

The trimul in-projection and the pair FFN's fc1 are the SAME operand class,
[1, R, 512, 256] x [256, 1024] (k_tiles 8, n_tiles 32), which is why one sweep decides both. L2
settled R=32 at `bw=1 obh=5 obw=16`. R=64 was reported REFUSED by the predecessor, but that
refusal came from a config whose circular buffers were too wide (`out_block_w` hardcoded to
n_tiles), not from capacity: at obw=16 the static budget is 1,363,968 B against a 1,461,760 B
bank, so it is expressible and the only question left is what the allocator and the clock say.

A -- the bare op at R in {32, 64, 128}, L1 destination, bw=1 x obh x obw, torch.equal per
     candidate against the `core_grid` DRAM call it would replace.
B -- the whole FFN chain at rows in {32, 64}, best bit-exact config per row height, torch.equal
     against the unblocked reference. A 64-row block halves the launches; the bare-op optimum was
     already shown not to be the chain optimum (screen_b), so the chain is what decides.
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
         'grid': [g.x, g.y], 'L': L, 'n': a.n, 'l1_bank_bytes': T._l1_bank_bytes(),
         'A_bare': {}, 'B_chain': {}, 'loadavg': open('/proc/loadavg').read().split()[0]}

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
    w1a = f((torch.randn(FF, CZ) * 0.02).t())
    w1b = f((torch.randn(FF, CZ) * 0.02).t())
    w2 = f((torch.randn(CZ, FF) * 0.02).t())
    nw, nb = f(torch.randn(CZ)), f(torch.randn(CZ))
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

    # ---- A: the bare op ---------------------------------------------------------------
    best = {}
    for rows in (32, 64, 128):
        x = f(torch.randn(1, rows, L, CZ))
        ref_t = ttnn.to_torch(ttnn.linear(x, w1a, compute_kernel_config=CK,
                                          dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN))
        ms_ref = bench(lambda: ttnn.linear(x, w1a, compute_kernel_config=CK,
                                           dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN))
        R['A_bare'][f'rows{rows}'] = {'dram_core_grid_ms': round(ms_ref, 4), 'cands': {}}
        print(f'rows={rows} DRAM core_grid reference {ms_ref:.4f} ms', flush=True)
        mt = rows * (L // 32)
        for obh in (5, 4, 2, 1):
            for obw in (2, 4, 8, 16, 32):
                c = cfg(mt, 1, obh, obw)
                if c is None:
                    continue
                tag = f'bw1_obh{obh}_obw{obw}'
                try:
                    call = lambda: ttnn.linear(x, w1a, compute_kernel_config=CK,
                                               dtype=ttnn.bfloat16,
                                               memory_config=ttnn.L1_MEMORY_CONFIG,
                                               program_config=c)
                    o = call()
                    eq = bool(torch.equal(ref_t, ttnn.to_torch(o)))
                    ttnn.deallocate(o)
                    ms = bench(call)
                    R['A_bare'][f'rows{rows}']['cands'][tag] = {'ms': round(ms, 4), 'equal': eq}
                    if eq and (tag not in best.get(rows, {}) or True):
                        cur = best.get(rows)
                        if cur is None or ms < cur[1]:
                            best[rows] = (tag, ms, obh, obw)
                except Exception as e:
                    R['A_bare'][f'rows{rows}']['cands'][tag] = {'refused': str(e)[:120]}
        b = best.get(rows)
        R['A_bare'][f'rows{rows}']['best_bit_exact'] = None if b is None else {
            'tag': b[0], 'ms': round(b[1], 4), 'speedup_vs_dram': round(ms_ref / b[1], 4)}
        print(f'  best bit-exact L1: {R["A_bare"][f"rows{rows}"]["best_bit_exact"]}', flush=True)
        ttnn.deallocate(x)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(R, indent=1))

    # ---- B: the chain lives in perf/esm3p4/screen_d.py, which deallocates every candidate
    # and sweeps the row heights that matter. Part A is the kill gate and it stands alone.
    print(wrote, a.out)
    return
    z = f(torch.randn(1, L, L, CZ))

    def chain(rows, c, gated_l1=True):
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

    ref = ttnn.to_torch(chain(0, None))
    R['B_chain']['unblocked'] = {'ms': round(bench(lambda: chain(0, None)), 4), 'equal': True}
    print(f"chain unblocked {R['B_chain']['unblocked']['ms']:.3f} ms", flush=True)
    for rows in (32, 64):
        mt = rows * (L // 32)
        for obh in (5, 4, 2, 1):
            for obw in (16, 8, 32, 4):
                c = cfg(mt, 1, obh, obw)
                if c is None:
                    continue
                tag = f'rows{rows}_bw1_obh{obh}_obw{obw}'
                try:
                    eq = bool(torch.equal(ref, ttnn.to_torch(chain(rows, c))))
                    ms = bench(lambda: chain(rows, c))
                    R['B_chain'][tag] = {'ms': round(ms, 4), 'equal': eq}
                    print(f'  {tag:28s} {ms:8.3f} ms equal={eq}', flush=True)
                except Exception as e:
                    R['B_chain'][tag] = {'refused': str(e)[:120]}
                    print(f'  {tag:28s} REFUSED {str(e)[:70]}', flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1))
    print('wrote', a.out)


main()
