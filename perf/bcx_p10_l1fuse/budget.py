#!/usr/bin/env python3
"""bcx-p10-l1fuse leg 2: the target card's OWN L1 budget, read off the allocator.

`bcx-p10-tril1`'s budget table was measured on pc card 0 at a 13x10 grid. This row runs on qb2,
which is 11x10, so every aggregate number in that table is 15 % smaller here and a chain sized
against the remembered one would be sized against a card that is not under it. The allocator is
asked directly, the same way `tt_bio.tenstorrent._l1_bank_bytes` asks it, and the free number is
read beside the idle capacity because a chain has to fit next to whatever the block is already
holding.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MIB = 1024.0 ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "perf/bcx_p10_l1fuse/out/budget_qb2.json"))
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tn

    dev = tn.get_device()
    gx, gy = tn.COMPUTE_GRID_MAIN
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
    per_bank = int(mv.total_bytes_per_bank)
    banks = int(mv.num_banks)
    free_per_bank = int(mv.total_bytes_free_per_bank)
    dram = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)

    out = {
        "host": os.uname().nodename,
        "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "compute_grid_main": [gx, gy],
        "compute_grid_cores": gx * gy,
        "l1_total_bytes_per_bank": per_bank,
        "l1_num_banks": banks,
        "l1_free_bytes_per_bank_idle": free_per_bank,
        "l1_bank_bytes_helper": int(tn._l1_bank_bytes()),
        "l1_max_worker_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
        "l1_aggregate_grid_mib": per_bank * gx * gy / MIB,
        "l1_aggregate_banks_mib": per_bank * banks / MIB,
        "dram_total_bytes_per_bank": int(dram.total_bytes_per_bank),
        "dram_num_banks": int(dram.num_banks),
        "dram_total_gib": int(dram.total_bytes_per_bank) * int(dram.num_banks) / 1024.0 ** 3,
        "trimul_tail_l1_share": float(tn._TRIMUL_TAIL_L1_SHARE),
        "trimul_l1_max_seq": int(tn.TRIANGLE_MULT_L1_MAX_SEQ),
    }

    #: What one pair-shaped tensor costs, per bank and as a share of the grid's aggregate L1.
    #: 288x288 pads to 288 tiles-wise already (288 = 9 tiles of 32), so the padded axis is 288.
    shapes = {
        "pair 1x288x288x128 bf16": (1, 288, 288, 128, 2),
        "pair 1x288x288x128 f32": (1, 288, 288, 128, 4),
        "pair 1x288x288x512 bf16": (1, 288, 288, 512, 2),
        "pair 1x288x288x384 bf16": (1, 288, 288, 384, 2),
        "pair 1x288x288x32 bf16": (1, 288, 288, 32, 2),
        "trimul chunk 128x288x288 bf16": (1, 128, 288, 288, 2),
        "msa 2x288x256 bf16": (1, 2, 288, 256, 2),
    }
    agg = per_bank * gx * gy
    out["tensors"] = {}
    for name, (b, c, h, w, e) in shapes.items():
        nbytes = b * c * h * w * e
        out["tensors"][name] = {
            "mib": nbytes / MIB,
            "bytes_per_bank_if_spread": nbytes / (gx * gy),
            "share_of_grid_l1": nbytes / agg,
            "fits_grid_l1": nbytes <= agg,
        }

    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
