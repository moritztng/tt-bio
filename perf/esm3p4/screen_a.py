#!/usr/bin/env python3
"""Two screens the plan for 3.40x turns on, in one session.

S2 -- the pair FFN's fc1 halves have NEVER been L1-resident. probe_legs_c0.json shows
`_pair_proj_config` returns None for this operand class at every row height and every block
width, so the landed arm C reaches `ttnn.linear(core_grid=...)` with a DRAM output and its
1.708 s comes from the L1 multiply alone. `out_block_w` is hardcoded to n_tiles, which is what
makes the CB budget overflow. Screen: does a config with a smaller out_block_w take an L1
output, is it bit-exact, and what does the whole FFN cost then.

S1 -- the gated move reads 256 MiB and writes 128 MiB. The -3.42 s in state/esmfold2-to-4x.md
assumes deleting the read scales with bytes, but that pass also established the move is
transaction-bound, in which case the read is not what costs. Screen: the same move with the
source in L1 instead of DRAM, at the largest N whose 4-way fused input fits.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.reblock_permute as RP

MB = 2 ** 20


def ckc():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--L', type=int, default=512)
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--warm', type=int, default=2)
    a = ap.parse_args()
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get('TT_MESH_GRAPH_DESC_PATH'):
        mgd = _find_ttnn_mesh_graph_descriptor('p150_mesh_graph_descriptor.textproto')
        if mgd:
            os.environ['TT_MESH_GRAPH_DESC_PATH'] = mgd
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    L, CZ, FF = a.L, 256, 1024
    CK = ckc()
    R = {'host': os.uname().nodename, 'card': os.environ.get('TT_VISIBLE_DEVICES'),
         'grid': [g.x, g.y], 'L': L, 'n': a.n, 'S2': {}, 'S1': {}, 'loadavg': open('/proc/loadavg').read().split()[0]}

    def bench(fn, n=None, warm=None):
        n, warm = n or a.n, a.warm if warm is None else warm
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

    f = lambda t, mc=None: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,
                                           dtype=ttnn.bfloat16,
                                           **({'memory_config': mc} if mc is not None else {}))
    torch.manual_seed(0)

    # ---------------- S2: an L1-resident fc1 ----------------
    nw, nb = f(torch.randn(CZ)), f(torch.randn(CZ))
    w1a = f((torch.randn(FF, CZ) * 0.02).t())
    w1b = f((torch.randn(FF, CZ) * 0.02).t())
    w2 = f((torch.randn(CZ, FF) * 0.02).t())
    z = f(torch.randn(1, L, L, CZ))
    SILU = [ttnn.UnaryOpType.SILU]

    def cfg(m_tiles, bw, obh, obw, n_tiles=FF // 32):
        gx, gy = T.COMPUTE_GRID_MAIN
        nc = gx * gy
        per_core_M = -(-(-(-m_tiles // nc)) // obh) * obh
        sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
        sw = max(w for w in range(min(4 // sh, obw), 0, -1) if obw % w == 0)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
            out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=obw,
            per_core_M=per_core_M, per_core_N=n_tiles, fuse_batch=True,
            fused_activation=None, mcast_in0=False)

    def ffn(rows, fc1_l1_cfg=None, mul_l1=True):
        xn = ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=CK)
        parts = ttnn.chunk(xn, -(-L // rows), dim=1) if rows else [xn]
        if rows: ttnn.deallocate(xn)
        outs = []
        for p in parts:
            if fc1_l1_cfg is not None:
                kw = dict(compute_kernel_config=CK, dtype=ttnn.bfloat16,
                          memory_config=ttnn.L1_MEMORY_CONFIG, program_config=fc1_l1_cfg)
                h1, h2 = ttnn.linear(p, w1a, **kw), ttnn.linear(p, w1b, **kw)
            else:
                kw = dict(compute_kernel_config=CK, dtype=ttnn.bfloat16,
                          core_grid=T.CORE_GRID_MAIN)
                h1, h2 = ttnn.linear(p, w1a, **kw), ttnn.linear(p, w1b, **kw)
            ttnn.deallocate(p)
            gt = ttnn.multiply(h1, h2, input_tensor_a_activations=SILU,
                               **({'memory_config': ttnn.L1_MEMORY_CONFIG} if mul_l1 else {}))
            ttnn.deallocate(h1); ttnn.deallocate(h2)
            outs.append(ttnn.linear(gt, w2, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                                    core_grid=T.CORE_GRID_MAIN))
            ttnn.deallocate(gt)
        if len(outs) == 1: return outs[0]
        o = ttnn.concat(outs, dim=1)
        for x in outs: ttnn.deallocate(x)
        return o

    ref = ttnn.to_torch(ffn(0, None, mul_l1=False))
    R['S2']['unblocked_ref_ms'] = round(bench(lambda: ffn(0, None, mul_l1=False)), 4)
    R['S2']['shipped_rows32_ms'] = round(bench(lambda: ffn(32, None, True)), 4)
    print(f"  unblocked ref        {R['S2']['unblocked_ref_ms']:8.3f} ms", flush=True)
    print(f"  shipped rows=32      {R['S2']['shipped_rows32_ms']:8.3f} ms", flush=True)

    # bare fc1 at one row block: does an L1 output take, and is it bit-exact?
    p0 = ttnn.chunk(ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5,
                                    compute_kernel_config=CK), L // 32, dim=1)[0]
    mt = 32 * (L // 32)
    fc1_ref = ttnn.to_torch(ttnn.linear(p0, w1a, compute_kernel_config=CK,
                                        dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN))
    R['S2']['fc1_ref_ms'] = round(bench(lambda: ttnn.linear(
        p0, w1a, compute_kernel_config=CK, dtype=ttnn.bfloat16,
        core_grid=T.CORE_GRID_MAIN)), 4)
    print(f"  fc1 half, DRAM ref   {R['S2']['fc1_ref_ms']:8.3f} ms", flush=True)
    best = None
    for bw in (1, 2, 4, 8):
        for obh in (5, 4, 2, 1):
            for obw in (2, 4, 8, 16, 32):
                tag = f'bw{bw}_obh{obh}_obw{obw}'
                try:
                    c = cfg(mt, bw, obh, obw)
                    o = ttnn.linear(p0, w1a, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                                    memory_config=ttnn.L1_MEMORY_CONFIG, program_config=c)
                    ok = bool(torch.equal(fc1_ref, ttnn.to_torch(o)))
                    ttnn.deallocate(o)
                    ms = bench(lambda: ttnn.linear(
                        p0, w1a, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                        memory_config=ttnn.L1_MEMORY_CONFIG, program_config=c))
                    R['S2'][f'fc1_l1_{tag}'] = {'ms': round(ms, 4), 'equal': ok}
                    print(f'   fc1 L1 {tag:18s} {ms:8.3f} ms equal={ok}', flush=True)
                    if ok and (best is None or ms < best[1]):
                        best = (tag, ms, bw, obh, obw)
                except Exception as e:
                    R['S2'][f'fc1_l1_{tag}'] = {'refused': str(e)[:120]}
    R['S2']['fc1_best_bitexact'] = best[:2] if best else None
    print('  fc1 best bit-exact L1:', best, flush=True)

    if best is not None:
        c = cfg(mt, best[2], best[3], best[4])
        for rows in (32,):
            try:
                out = ttnn.to_torch(ffn(rows, c, True))
                eq = bool(torch.equal(ref, out))
                ms = bench(lambda: ffn(rows, c, True))
                R['S2'][f'chain_rows{rows}_l1fc1'] = {'ms': round(ms, 4), 'equal': eq}
                print(f'  CHAIN rows={rows} L1 fc1  {ms:8.3f} ms equal={eq}', flush=True)
            except Exception as e:
                R['S2'][f'chain_rows{rows}_l1fc1'] = {'refused': str(e)[:200]}
                print('  CHAIN refused:', str(e)[:200], flush=True)
    ttnn.deallocate(z)

    # ---------------- S1: does an L1 source speed the gated move? ----------------
    for N in (128, 192, 256):
        try:
            t = torch.randn(1, N, N, 4 * CZ)
            xd = f(t)
            xl = f(t, ttnn.L1_MEMORY_CONFIG)
            mv = lambda x: RP.reblock_permute_gated(x, 0, 2 * CZ, CZ,
                                                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                    device=dev)
            od, ol = ttnn.to_torch(mv(xd)), ttnn.to_torch(mv(xl))
            mbs = (2 * N * N * CZ * 2 + N * N * CZ * 2) / MB
            ms_d, ms_l = bench(lambda: mv(xd)), bench(lambda: mv(xl))
            R['S1'][f'N{N}'] = {
                'dram_src_ms': round(ms_d, 4), 'l1_src_ms': round(ms_l, 4),
                'ratio': round(ms_d / ms_l, 4), 'MB': round(mbs, 1),
                'dram_GBps': round(mbs * MB / (ms_d * 1e-3) / 1e9, 1),
                'equal': bool(torch.equal(od, ol))}
            print(f"  S1 N={N:4d} DRAMsrc {ms_d:7.3f}  L1src {ms_l:7.3f}  "
                  f"ratio {ms_d/ms_l:5.3f}  equal={R['S1'][f'N{N}']['equal']}", flush=True)
            ttnn.deallocate(xd); ttnn.deallocate(xl)
        except Exception as e:
            R['S1'][f'N{N}'] = {'refused': str(e)[:200]}
            print(f'  S1 N={N} refused:', str(e)[:160], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1))
    print('wrote', a.out)


main()
