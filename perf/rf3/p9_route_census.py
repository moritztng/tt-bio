#!/usr/bin/env python3
"""Which route each pair-tensor transpose takes on THIS board, before any fold is timed.

Lever 8 (pass 7) only pays where the shipped `_transpose_memory_config` has already fallen to
DRAM. On qb2 (p300c, 11x10, 110 cores) that is 768 aa and only 768 aa. qb1 is a p150a with a
13x10 grid, 130 cores, so its L1 budget is 130/110 = 1.18x larger and the fall to DRAM may
happen at a different rung, or at 768 aa not at all, which would make lever 8 DARK here.

That is a route question, not a timing question, so it is answered by reading the decision
rather than by timing a fold: co-tenant load cannot move it. Both the arithmetic predicate and
the allocation itself are reported, because a static budget can be refused by the live
allocator.
"""
from __future__ import annotations

import argparse
import enum
import json
import sys
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="128,256,512,768,1024")
    ap.add_argument("--c", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T

    device = T.get_device()
    per_core = int(ttnn.get_max_worker_l1_unreserved_size())
    grid = T.COMPUTE_GRID_MAIN
    cores = grid[0] * grid[1]
    budget = per_core * cores
    reserve = T._TRANSPOSE_L1_RESERVE_PER_CORE
    head = T._TRANSPOSE_L1_HEADROOM
    print("grid {} = {} cores, {} B/core unreserved, L1 budget {:.2f} MB, "
          "headroom {}, reserve {} B/core".format(
              grid, cores, per_core, budget / 2 ** 20, head, reserve), flush=True)

    def bt(mc):
        return "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"

    rows = []
    for S in [int(s) for s in args.sizes.split(",")]:
        nbytes = S * S * args.c * 2
        xt = torch.zeros(S, S, args.c, dtype=torch.bfloat16)
        x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=device,
                            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # The shipped decision: both transposes pre-lever-8, and the INPUT transpose still.
        shipped = T._transpose_memory_config(x)
        # Lever 8, OUTPUT transpose: per-core reserve instead of multiplicative headroom.
        l8_out = T._transpose_memory_config(x, reserve)
        # Lever 8, INPUT transpose: L1 staging, then a copy to DRAM.
        l8_stage = T._l1_memory_config_if_it_fits(x, 1.0, reserve_per_core=reserve)
        # The allocation itself is the real test for whichever route asks for L1.
        allocated = {}
        for name, mc in (("shipped", shipped), ("l8_out", l8_out), ("l8_stage", l8_stage)):
            if mc.buffer_type != ttnn.BufferType.L1:
                allocated[name] = "n/a"
                continue
            try:
                o = T._pair_transpose_impl(x, ttnn.L1_MEMORY_CONFIG)
                ok = torch.equal(ttnn.to_torch(o), xt.permute(1, 0, 2))
                ttnn.deallocate(o)
                allocated[name] = "ok-bitexact" if ok else "OK-BUT-DIFFERS"
            except Exception as e:                                      # noqa: BLE001
                allocated[name] = "REFUSED: " + str(e).splitlines()[0][:90]
        row = dict(S=S, mb=nbytes / 2 ** 20, pct_budget=100.0 * nbytes / budget,
                   shipped=bt(shipped), l8_out=bt(l8_out), l8_stage=bt(l8_stage),
                   alloc=allocated)
        rows.append(row)
        print("{:5d} {:7.1f} MB  {:5.1f} % of budget  shipped={:4s} l8_out={:4s} "
              "l8_stage={:4s}  {}".format(
                  S, row["mb"], row["pct_budget"], row["shipped"], row["l8_out"],
                  row["l8_stage"], allocated), flush=True)
        ttnn.deallocate(x)

    out = dict(grid=list(grid), cores=cores, per_core=per_core, budget=budget,
               headroom=head, reserve=reserve, c_z=args.c, rows=rows)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote " + args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
