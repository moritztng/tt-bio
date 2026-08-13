#!/usr/bin/env python3
"""S4 -- the row-blocked gated move at the production N=512 shape (§7 step 4.2).

Kill gate, pre-committed by the plan: a row-blocked move that cannot beat 1.11 ms -- half the
MEASURED 2.2182 ms the whole-tensor move costs -- makes L1' worth under 1 s and kills it.

Arms, all producing the same [1, 256, 512, 512] destination:

  whole_dram    the shipped move, one call, source in DRAM              (the reference)
  blocked_dram  8 or 16 row blocks, source still in DRAM                (the split penalty alone)
  blocked_l1    the same blocks with the source in L1                   (the arm L1' needs)

`blocked_l1` re-uses ONE pre-staged L1 block across all the row offsets. The values are then
wrong, which is deliberate: this measures the move off an L1 source and nothing else, and
bit-exactness is established separately by perf/esm3p4/rowblock_parity.py. Staging the block is
not counted because in L1' the in-projection writes it there directly.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.reblock_permute as RP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--N', type=int, default=512)
    ap.add_argument('--n', type=int, default=7)
    a = ap.parse_args()
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get('TT_MESH_GRAPH_DESC_PATH'):
        m = _find_ttnn_mesh_graph_descriptor('p150_mesh_graph_descriptor.textproto')
        if m:
            os.environ['TT_MESH_GRAPH_DESC_PATH'] = m
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    N, C, CW = a.N, 256, 1024
    R = {'host': os.uname().nodename, 'card': os.environ.get('TT_VISIBLE_DEVICES'),
         'grid': [g.x, g.y], 'N': N, 'n': a.n, 'arms': {},
         'loadavg': open('/proc/loadavg').read().split()[0]}

    def bench(fn, warm=2):
        for _ in range(warm):
            fn(); ttnn.synchronize_device(dev)
        ts = []
        for _ in range(a.n):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        return st.median(ts)

    torch.manual_seed(0)
    xw = ttnn.from_torch(torch.randn(1, N, N, CW), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, C, N, N]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)

    ms = bench(lambda: RP.reblock_permute_gated(xw, 2 * C, 0, C, out=out, row_off=0))
    R['arms']['whole_dram'] = {'ms': round(ms, 4), 'blocks': 1}
    print(f"  whole_dram          {ms:8.4f} ms", flush=True)

    for Rb in (64, 32):
        blocks = [ttnn.slice(xw, [0, r, 0, 0], [1, r + Rb, N, CW]) for r in range(0, N, Rb)]
        offs = list(range(0, N, Rb))

        def run_dram():
            for b, r in zip(blocks, offs):
                RP.reblock_permute_gated(b, 2 * C, 0, C, out=out, row_off=r)
        ms = bench(run_dram)
        R['arms'][f'blocked{Rb}_dram'] = {'ms': round(ms, 4), 'blocks': len(blocks)}
        print(f"  blocked{Rb}_dram      {ms:8.4f} ms  ({len(blocks)} blocks)", flush=True)
        for b in blocks:
            ttnn.deallocate(b)

        try:
            l1blk = ttnn.to_memory_config(
                ttnn.slice(xw, [0, 0, 0, 0], [1, Rb, N, CW]), ttnn.L1_MEMORY_CONFIG)

            def run_l1():
                for r in offs:
                    RP.reblock_permute_gated(l1blk, 2 * C, 0, C, out=out, row_off=r)
            ms = bench(run_l1)
            R['arms'][f'blocked{Rb}_l1'] = {'ms': round(ms, 4), 'blocks': len(offs)}
            print(f"  blocked{Rb}_l1        {ms:8.4f} ms  ({len(offs)} blocks)", flush=True)
            ttnn.deallocate(l1blk)
        except Exception as e:
            R['arms'][f'blocked{Rb}_l1'] = {'refused': str(e)[:140]}
            print(f"  blocked{Rb}_l1        REFUSED {str(e)[:80]}", flush=True)

    base = R['arms']['whole_dram']['ms']
    for k, v in R['arms'].items():
        if 'ms' in v:
            v['x_vs_whole_dram'] = round(base / v['ms'], 4)
    R['kill_gate_ms'] = 1.11
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1))
    print('wrote', a.out)


main()
