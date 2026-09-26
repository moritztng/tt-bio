#!/usr/bin/env python3
"""What does `ttnn.generic_op` actually charge for: the program, the cores, or the arg words?

`decomp.py` put 50 % of a dispatch inside the `generic_op` call and 0.4 % in the two
`buffer_address()` reads the brief blamed, so the fix has to be aimed at whatever inside the
dispatch scales. This drives a kernel that touches nothing -- no CB, no NOC -- so the only thing
being timed is the dispatch, and sweeps the two axes a descriptor has: how many cores carry
per-core runtime args, and how many words each of them carries.

A core placed with NO per-core runtime args reads stale L1 and would send a NOC write into the
bottom of DRAM in a real kernel, which is why the zero-word point is measured HERE, on a kernel
with no NOC transaction in it, and not by emptying a shipped kernel's args.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'perf' / 'bcx_stack'))
from perf.bcx_stack import stack as S            # noqa: E402
from perf.bcx_p10_devmap import devmap as D      # noqa: E402
from perf.bcx_p10_genq import bench as B         # noqa: E402

OUT = ROOT / 'perf' / 'bcx_p10_genq' / 'out'
KD = pathlib.Path(__file__).resolve().parent / 'kernels' / 'nullk'
SRC, SRC_C = str(KD / 'nullk.cpp'), str(KD / 'nullk_compute.cpp')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=300)
    ap.add_argument('--out', default='nullk.json')
    ap.add_argument('--params', default=D.A.DEFAULT_PARAMS)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    args.card = int(os.environ.get('TT_VISIBLE_DEVICES', '0'))
    torch.set_num_threads(args.threads)

    import ttnn
    _lv, _dev, _ref = S.open_all(args)
    dev = _dev.device
    clock = S.Clock()
    g = dev.compute_with_storage_grid_size()
    # generic_op requires at least one input and one output tensor; the null kernel touches
    # neither, so these two exist only to satisfy that check.
    mk = lambda: ttnn.from_torch(torch.zeros(1, 32, 32), layout=ttnn.TILE_LAYOUT, device=dev,
                                 dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    scratch, scratch2 = mk(), mk()
    spans, res = [], {}

    def build(ncores, nwords, nkernels, ncommon=1):
        """`ncores` cores row-major, `nwords` per-core words, `nkernels` kernels, `ncommon`
        words in the shared table every core indexes by its own logical coordinates."""
        cores = [(i % g.x, i // g.x) for i in range(ncores)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
                                 for x, y in cores])
        # One kernel per processor class: a program refuses two of the same class on one core.
        slots = [(SRC, ttnn.ReaderConfigDescriptor()), (SRC, ttnn.WriterConfigDescriptor()),
                 (SRC_C, ttnn.ComputeConfigDescriptor())]
        ks = []
        for src, cfg in slots[:nkernels]:
            rt = ttnn.RuntimeArgs()
            for x, y in cores:
                rt[x][y] = [7] * nwords
            ks.append(ttnn.KernelDescriptor(
                kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs, compile_time_args=[nwords, ncommon], runtime_args=rt,
                common_runtime_args=[0] * ncommon, config=cfg))
        return ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=[])

    plan = []
    for nc in (1, 8, 32, 64, 110, 130):
        plan.append((f'cores{nc}_w2_k3', nc, 2, 3))
    for nw in (0, 2, 8, 32):
        plan.append((f'cores130_w{nw}_k3', 130, nw, 3))
    for nk in (1, 2, 3):
        plan.append((f'cores130_w2_k{nk}', 130, 2, nk))
    plan.append(('cores130_w0_k1', 130, 0, 1))
    # The axis the fix rides on: with no per-core args at all, what does a LONG shared table cost?
    # 341 words is the hard ceiling: tt-metal refuses a kernel whose unique+common runtime args
    # exceed it on any core (kernel.cpp:453), which is what bounds how big a shared table can get.
    for ncom in (1, 130, 260, 341):
        plan.append((f'cores130_w0_k3_common{ncom}', 130, 0, 3, ncom))

    for tag, nc, nw, nk, *rest in plan:
        if nc > g.x * g.y:
            continue
        pd = build(nc, nw, nk, rest[0] if rest else 1)
        ttnn.generic_op([scratch, scratch2], pd)       # JIT-build once, outside the clock
        ttnn.synchronize_device(dev)
        r = B.timed(tag, lambda pd=pd: ttnn.generic_op([scratch, scratch2], pd), dev, ttnn, args.reps, spans)
        r.update(cores=nc, words=nw, kernels=nk, common=rest[0] if rest else 1)
        res[tag] = r

    blob = {'stamp': S.stamp(args, clock), 'grid': [g.x, g.y], 'reps': args.reps,
            'aiclk': clock.window(spans), 'points': res}
    clock.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print('wrote ' + str(OUT / args.out), flush=True)


if __name__ == '__main__':
    main()
