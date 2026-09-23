#!/usr/bin/env python3
"""Is this card actually able to dispatch? Bounded, so a wedge costs a minute and not a night.

`tenstorrent._assert_local_dispatch` already probes a freshly-opened chip with one trivial add,
and its docstring says a mis-initialised worker "fails HERE, at startup". It does not always
fail: on 2026-09-22 two arms sat inside that probe for 115 minutes each, at 100 % CPU, with
their cards clocked at 1350 MHz and drawing 9 W over idle. Every cheap liveness signal said
"computing". Nothing had been computed, and nothing would be -- the probe has no timeout, so a
wedged chip hangs in it and is indistinguishable from a long-running job.

A healthy open plus dispatch on qb1 is 1.6 seconds. Run this before an arm and a wedge is a
60-second failure with a message, instead of a run script that waits forever.

    python3 cardcheck.py <card>     # exit 0 healthy, 1 wedged/failed
"""
from __future__ import annotations

import os
import sys
import time


def main() -> int:
    card = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TT_VISIBLE_DEVICES", "0")
    t0 = time.time()
    import torch
    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device()
    open_s = time.time() - t0
    x = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn.add(x, x)
    ttnn.synchronize_device(dev)
    total = time.time() - t0
    T.cleanup()
    print(f"CARDCHECK card={card} open={open_s:.1f}s dispatch_total={total:.1f}s HEALTHY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
