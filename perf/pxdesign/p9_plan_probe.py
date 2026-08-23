"""What the fp32-softmax L1 plan answers at AF2-IG's token counts, logical against padded.

Pure arithmetic over `_fp32_softmax_l1_rows` / `_fp32_softmax_l1_free_rows` /
`_fp32_softmax_l1_plan` / `_fp32_softmax_shard` at the trunk's four heads. No fold, but it does
open the device: `COMPUTE_GRID_MAIN` and `_fp32_softmax_core_budget` are read off the live chip,
and `import tt_bio.tenstorrent` opens one at import -- so it must be pinned like any device leg,
or UMD brings up every visible chip.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:pxdesign-perf-p9 \\
        PYTHONPATH=. python3 perf/pxdesign/p9_plan_probe.py --out perf/pxdesign/tt_pxd_p9_plan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#: AF2-IG's trunk pair heads. `AF2PairBlock` builds both triangle attentions at 4 x 32.
N_HEADS = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="208,336,592,848,1024")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio import tenstorrent as TT

    rows = []
    for token in [int(t) for t in args.tokens.split(",")]:
        for label, length in (("logical", token), ("padded", -(-token // 32) * 32)):
            hpr = N_HEADS * length
            per_row = hpr * length * 4
            tuned = TT._fp32_softmax_l1_rows(per_row, hpr)
            plan = TT._fp32_softmax_l1_plan(per_row, hpr, length, None, None, None)
            free = TT._fp32_softmax_l1_free_rows(per_row, hpr, None, None)
            shard = None
            if plan[0]:
                shard = TT._fp32_softmax_shard(plan[0], hpr, length, plan[1]) is not None
            cap = max(32, int(TT._FP32_SOFTMAX_BLOCK_BYTES // per_row) // 32 * 32)
            block = plan[0] or min(cap, token)
            rows.append({"tokens": token, "extent": label, "length": length,
                         "per_row_bytes": per_row, "height_per_row": hpr,
                         "tuned_rows": tuned, "plan": list(plan), "free_rows": list(free),
                         "block_cap": cap, "block_rows": block,
                         "blocks_per_call": -(-token // block), "shard_built": shard})

    out = {"mode": "af2ig_fp32_softmax_l1_plan_probe", "n_heads": N_HEADS,
           "compute_grid_main": list(TT.COMPUTE_GRID_MAIN),
           "core_budget": TT._fp32_softmax_core_budget(),
           "tuned_grid": list(TT._FP32_SOFTMAX_L1_GRID),
           "bytes_per_core": TT._FP32_SOFTMAX_L1_BYTES_PER_CORE,
           "any_cores": TT._FP32_SOFTMAX_L1_ANY_CORES,
           "float_cores": TT._FP32_SOFTMAX_L1_FLOAT_CORES,
           "rows": rows}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
