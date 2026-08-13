#!/usr/bin/env python3
"""Acceptance for L2 -- the BUILT L1-resident fc1, through the real `SwiGLUFFN`.

screen_b measured the chain out of hand-built ttnn calls. This runs the shipped module, so a
config the gate still refuses, or a call site that does not reach the new leg, fails here rather
than in a fold. Three arms off one baseline, every arm `torch.equal` against the unblocked path:

    unblocked      set_pair_l1_rows(0)   -- what ships by default today
    shipped_rows32 set_pair_l1_rows(32) with block_w unset -- the predecessor's arm C
    l2_rows32      set_pair_l1_rows(32) with the named fc1 config
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.esmc as E


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

    # the config the gate used to refuse, at the production fc1 operand class
    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    torch.manual_seed(0)
    xb = f(torch.randn(1, 32, L, CZ))
    wb = f(torch.randn(CZ, FF) * 0.02)
    old = T._pair_proj_config(xb, wb, bw_cap=T._PAIR_PROJ_L1_BW, out_l1=True)
    new = T._pair_proj_config(xb, wb, bw_cap=T._PAIR_FFN_FC1_BW, out_l1=True,
                              block_w=T._PAIR_FFN_FC1_BLOCK_W)
    R['fc1_config'] = {
        'default_gate': None if old is None else str(old),
        'named': None if new is None else
        {'in0_block_w': new.in0_block_w, 'out_block_h': new.out_block_h,
         'out_block_w': new.out_block_w, 'per_core_M': new.per_core_M,
         'per_core_N': new.per_core_N, 'out_subblock_h': new.out_subblock_h,
         'out_subblock_w': new.out_subblock_w}}
    ttnn.deallocate(xb); ttnn.deallocate(wb)
    print('default gate for fc1:', R['fc1_config']['default_gate'])
    print('named config        :', R['fc1_config']['named'], flush=True)
    assert new is not None, 'the named fc1 config is still refused by the gate'

    sd = {'0.weight': torch.randn(CZ), '0.bias': torch.randn(CZ),
          '1.weight': torch.randn(2 * FF, CZ) * 0.02,
          '3.weight': torch.randn(CZ, FF) * 0.02}
    ffn = E.SwiGLUFFN(sd, CK, fuse_swiglu=True)
    assert ffn.split_swiglu and not ffn.fuse_swiglu, 'not the split-swiglu path this task targets'
    z = f(torch.randn(1, L, L, CZ))

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

    def arm(rows, block_w):
        prev = E.set_pair_l1_rows(rows)
        pbw, pbl = E._PAIR_FFN_FC1_BW, E._PAIR_FFN_FC1_BLOCK_W
        E._PAIR_FFN_FC1_BLOCK_W = block_w
        try:
            out = ttnn.to_torch(ffn(z))
            ms = bench(lambda: ffn(z))
        finally:
            E.set_pair_l1_rows(prev)
            E._PAIR_FFN_FC1_BW, E._PAIR_FFN_FC1_BLOCK_W = pbw, pbl
        return ms, out

    ms, ref = arm(0, None)
    R['arms']['unblocked'] = {'ms': round(ms, 4), 'equal': True}
    for tag, rows, bw in (('shipped_rows32', 32, None),
                          ('l2_rows32', 32, T._PAIR_FFN_FC1_BLOCK_W)):
        ms, out = arm(rows, bw)
        R['arms'][tag] = {'ms': round(ms, 4), 'equal': bool(torch.equal(ref, out))}
    for k, v in R['arms'].items():
        print(f"  {k:18s} {v['ms']:8.3f} ms equal={v['equal']}", flush=True)
    u = R['arms']['unblocked']['ms']
    for k, v in R['arms'].items():
        v['speedup_vs_unblocked'] = round(u / v['ms'], 4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1))
    print('wrote', a.out)


main()
