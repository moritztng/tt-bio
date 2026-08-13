#!/usr/bin/env python3
"""Which matmul leg the row-blocked pair FFN actually takes, and why rows=16 is not bit-exact.

Not a timing. It answers three questions the plan needs before anything is built:
  1. what `_l1_bank_bytes()` really is on this card,
  2. what `_pair_proj_program_config` returns for fc1 [1,rows,512,256]x[256,1024] at every
     row height and both destinations, so the L1 leg's eligibility is read off the gate and not
     guessed,
  3. which branch of `_pair_proj_linear` each fc1 call lands on at rows 8/16/32, counted by a
     wrapper, plus torch.equal of the whole FFN against the unblocked reference.
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, ttnn
import tt_bio.tenstorrent as T


def ckc():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--L', type=int, default=512)
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
         'grid': [g.x, g.y], 'L': L,
         'l1_bank_bytes': T._l1_bank_bytes(),
         'PAIR_PROJ_BW': T._PAIR_PROJ_BW, 'PAIR_PROJ_L1_BW': T._PAIR_PROJ_L1_BW,
         'PAIR_PROJ_L1_OUT': T._PAIR_PROJ_L1_OUT,
         'cfg': {}, 'legs': {}, 'exact': {}}
    print('l1_bank_bytes', R['l1_bank_bytes'], 'grid', R['grid'], flush=True)

    # 2. the gate, read directly
    for rows in (8, 16, 32, 64, 512):
        mt = rows * (L // 32)
        for bw in (8, 4, 2, 1):
            for l1 in (True, False):
                c = T._pair_proj_program_config(mt, 8, FF // 32, bw, 2, l1)
                R['cfg'][f'rows{rows}_bw{bw}_l1{int(l1)}'] = None if c is None else {
                    'in0_block_w': c.in0_block_w, 'out_block_h': c.out_block_h,
                    'out_block_w': c.out_block_w, 'per_core_M': c.per_core_M,
                    'out_subblock_h': c.out_subblock_h, 'out_subblock_w': c.out_subblock_w}
    for k, v in R['cfg'].items():
        if v is not None:
            print(' CFG', k, v, flush=True)

    # 3. which leg each fc1 call takes, and the FFN's own bit-exactness
    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    torch.manual_seed(0)
    nw, nb = f(torch.randn(CZ)), f(torch.randn(CZ))
    w1a = f((torch.randn(FF, CZ) * 0.02).t())
    w1b = f((torch.randn(FF, CZ) * 0.02).t())
    w2 = f((torch.randn(CZ, FF) * 0.02).t())
    z = f(torch.randn(1, L, L, CZ))
    SILU = [ttnn.UnaryOpType.SILU]
    legs = []

    orig = T._pair_proj_linear

    def traced(x, w, ckc_, dtype, l1_out=False):
        tag = None
        if l1_out and T._PAIR_PROJ_L1_OUT:
            key = (tuple(x.padded_shape), tuple(w.shape), str(dtype))
            if key not in T._L1_OUT_REFUSED:
                c = T._pair_proj_config(x, w, bw_cap=T._PAIR_PROJ_L1_BW, out_l1=True)
                if c is not None:
                    tag = f'L1_linear(bw={c.in0_block_w},obh={c.out_block_h})'
        if tag is None:
            if T._pair_proj_minimal_matmul(x, w, ckc_, dtype) is not None:
                tag = 'minimal_matmul'
            else:
                c = T._pair_proj_config(x, w)
                tag = (f'DRAM_linear_cfg(bw={c.in0_block_w},obh={c.out_block_h})'
                       if c is not None else 'DRAM_linear_core_grid')
        legs.append(tag)
        return orig(x, w, ckc_, dtype, l1_out=l1_out)

    def ffn(rows, l1_out):
        xn = ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=CK)
        if rows:
            parts = ttnn.chunk(xn, -(-L // rows), dim=1)
            ttnn.deallocate(xn)
        else:
            parts = [xn]
        outs = []
        for p in parts:
            if l1_out:
                h1 = traced(p, w1a, CK, ttnn.bfloat16, l1_out=True)
                h2 = traced(p, w1b, CK, ttnn.bfloat16, l1_out=True)
            else:
                h1 = ttnn.linear(p, w1a, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                                 core_grid=T.CORE_GRID_MAIN)
                h2 = ttnn.linear(p, w1b, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                                 core_grid=T.CORE_GRID_MAIN)
            ttnn.deallocate(p)
            gt = ttnn.multiply(h1, h2, input_tensor_a_activations=SILU,
                               **({'memory_config': ttnn.L1_MEMORY_CONFIG} if l1_out else {}))
            ttnn.deallocate(h1); ttnn.deallocate(h2)
            outs.append(ttnn.linear(gt, w2, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                                    core_grid=T.CORE_GRID_MAIN))
            ttnn.deallocate(gt)
        if len(outs) == 1:
            return outs[0]
        o = ttnn.concat(outs, dim=1)
        for x in outs:
            ttnn.deallocate(x)
        return o

    ref = ttnn.to_torch(ffn(0, False))
    for rows in (32, 16, 8):
        legs.clear()
        out = ttnn.to_torch(ffn(rows, True))
        R['legs'][f'rows{rows}'] = sorted(set(legs))
        R['exact'][f'rows{rows}'] = {
            'equal': bool(torch.equal(ref, out)),
            'max_abs': float((ref.float() - out.float()).abs().max()),
            'n_diff': int((ref != out).sum()),
            'n_elem': int(ref.numel())}
        print(' rows', rows, R['legs'][f'rows{rows}'], R['exact'][f'rows{rows}'], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1))
    print('wrote', a.out)


main()
