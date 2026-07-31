#!/usr/bin/env python3
"""Emit a controlled number of ttnn ops so the device profiler's scale limits can be measured.

Purpose: find the op count at which device profiling stops being usable, and how much
the instrument perturbs the thing it measures.  Run it twice -- once bare, once under
`python -m tracy` -- and compare wall clock and artifact size.

Relevant hard limits, read out of tt-metal source (tt_metal/impl/profiler/
profiler_state_manager.cpp and tt_metal/hostdevcommon/api/hostdevcommon/profiler_common.h):
  * DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT = 1000 dispatched programs per dump window.
  * 48 bytes of DRAM per program per RISC (2 program-id words + 4 guaranteed markers).
  * PROFILER_L1_BUFFER_SIZE = 2048 bytes per RISC (250 optional + 4 guaranteed + 2 id
    markers x 2 uint32).
Past the program-support count the device first silently drops OPTIONAL (custom) zones and
keeps only the 4 guaranteed FW/KERNEL markers, then hard-fails with a host-side TT_FATAL
about missing data.  Raise it with TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=<n>, or dump
periodically with `python -m tracy --dump-device-data-mid-run`.
"""
import argparse
import time

import ttnn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--ops", type=int, default=1000, help="number of dispatched ttnn ops")
    p.add_argument("--tile", type=int, default=32, help="square tile side per operand")
    p.add_argument("--distinct_shapes", type=int, default=1,
                   help="cycle over this many shapes; >1 forces kernel recompiles")
    args = p.parse_args()

    device = ttnn.open_device(device_id=args.device_id)
    try:
        shapes = [(1, 1, args.tile * (i + 1), args.tile) for i in range(args.distinct_shapes)]
        operands = [
            (ttnn.ones(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
             ttnn.ones(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))
            for s in shapes
        ]
        # warm every shape so the timed region pays no JIT
        for a, b in operands:
            ttnn.add(a, b)
        ttnn.synchronize_device(device)

        t0 = time.perf_counter()
        for i in range(args.ops):
            a, b = operands[i % len(operands)]
            ttnn.add(a, b)
        ttnn.synchronize_device(device)
        dt = time.perf_counter() - t0
        print(f"OPS {args.ops} SHAPES {len(shapes)} WALL_S {dt:.4f} US_PER_OP {dt/args.ops*1e6:.2f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
