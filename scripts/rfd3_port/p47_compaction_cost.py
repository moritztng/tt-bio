"""Does the compaction pay for the chain it speeds up?

p46 showed the atom attention chain costs 15.06 % of its dense time at n_key=1024, the width S1's
0.1673 tile density implies. That is only collectable if K and V can be presented at that width,
and every occupied-tile list is per-32-row-block, so the shipped dense ops need a per-row-block
gather of K and V first.

Gathers on this card are element-rate limited (~10-14 cycles/element), which is exactly what made
lever A's one-hot route expensive. So this prices the gather at the real shape and puts it against
the 27.2 ms/call the chain saves.

Two forms, because they cost very differently:

  row gather   -- ttnn.embedding over a [n_key, head_dim] table, 190 blocks x 1024 rows. What a
                  naive compaction costs.
  one block    -- the same gather for a SINGLE row block, to expose whether the cost is per-element
                  (scales with blocks) or dominated by dispatch (does not).

The verdict this feeds: if the gather costs more than ~15 ms/call the no-custom-kernel route is dead
and only a kernel that reads K/V in place by tile index can collect S1.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

B, H, L, HD = 2, 4, 6051, 32
FULL = 6080
NBLK = -(-L // 32)      # 190 row blocks
WIDTH = 1024            # compacted key width implied by the 0.1673 tile density


def timeit(dev, fn, reps=3):
    r = fn()
    ttnn.synchronize_device(dev)
    ttnn.deallocate(r)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    return statistics.median(ts) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    dev = ttnn.open_device(device_id=0)
    g = torch.Generator().manual_seed(7)
    w = a.width

    # K laid out as a gatherable table: one row per key position, per (B,H) plane folded into it.
    tbl = ttnn.from_torch(torch.randn(B * H * FULL, HD, generator=g) * 0.1,
                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    # K and V differ per head AND per design, so every one of the B*H planes needs its own
    # compacted copy. An earlier revision gathered a single plane and undercounted by 8x.
    idx_all = torch.randint(0, B * H * FULL, (1, B * H * NBLK * w), generator=g).to(torch.int32)
    idx_one = idx_all[:, :w].contiguous()

    res = {}
    for tag, idx, n in (("all planes", idx_all, B * H * NBLK), ("one block", idx_one, 1)):
        i_dev = ttnn.from_torch(idx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
        ms = timeit(dev, lambda: ttnn.embedding(i_dev, tbl, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                memory_config=ttnn.DRAM_MEMORY_CONFIG))
        el = n * w * HD
        res[tag] = {"ms": ms, "elements": el, "ns_per_elem": ms * 1e6 / el}
        print(f"{tag:12s} {n:4d} blocks x {w} keys x {HD}  =  {el / 1e6:7.2f} M elements   "
              f"{ms:8.3f} ms   {ms * 1e6 / el:6.3f} ns/element", flush=True)
        ttnn.deallocate(i_dev)

    # K and V both, which is what a real compaction needs, and it is the number that matters.
    kv = 2 * res["all planes"]["ms"]
    print(f"\nK and V together, all {B * H} planes x {NBLK} blocks: {kv:8.3f} ms/call")
    print(f"the chain saves 32.05 - 4.83 = 27.22 ms/call at this width (perf/p42/r4_keywidth.json)")
    print(f"net per call: {27.22 - kv:+8.3f} ms   -> "
          f"{'GO' if kv < 15.0 else 'NO-GO for the gather route'}")

    ttnn.close_device(dev)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"width": w, "blocks": NBLK, "rows": res,
                                     "kv_ms": kv, "chain_saving_ms": 27.22}, indent=2))
        print(f"[done] {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
