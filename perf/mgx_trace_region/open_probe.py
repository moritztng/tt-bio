"""One pinned open at a given trace region, step by step, so a stall names its step.

    TT_VISIBLE_DEVICES=<c> TT_BIO_LEASE_CARDS=<c> TT_BIO_LEASE_HOLDER=worker:mgx-trace-region \
        timeout -s INT 60 python perf/mgx_trace_region/open_probe.py <region_bytes>

Prints one JSON line per step (flushed), so the last line of a killed run is the step that hung.
Uses tt_bio's own open (lease, dispatch-core choice, device-init lock), then the same 32x32 add +
synchronize `_assert_local_dispatch` runs, then one capture + replay of that add.
"""
import json
import sys
import time

import torch
import ttnn

from tt_bio import tenstorrent as T

T0 = time.time()


def say(step, **kw):
    print(json.dumps({"t": round(time.time() - T0, 2), "step": step, **kw}), flush=True)


def view(dev, kind):
    v = ttnn.get_memory_view(dev, kind)
    return {k: getattr(v, k) for k in ("num_banks", "total_bytes_per_bank",
                                       "total_bytes_allocated_per_bank",
                                       "total_bytes_free_per_bank",
                                       "largest_contiguous_bytes_free_per_bank")
            if hasattr(v, k)}


region = int(sys.argv[1])
say("start", region=region, arch=str(ttnn.get_arch_name()) if hasattr(ttnn, "get_arch_name") else None)
from tt_bio.device_lease import CardSetLease
lease = CardSetLease().acquire()
say("leased")
kwargs = {"trace_region_size": region} if region else {}
dev = T._open_device_locked(0, kwargs)
say("opened", dram=view(dev, ttnn.BufferType.DRAM), trace=view(dev, ttnn.BufferType.TRACE))
x = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
say("from_torch")
y = ttnn.add(x, x)
say("add_enqueued")
ttnn.synchronize_device(dev)
say("synchronized")
if region:
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    y2 = ttnn.add(x, y)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    say("captured", trace=view(dev, ttnn.BufferType.TRACE))
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    say("replayed")
    ttnn.release_trace(dev, tid)
T._close_device_locked(dev)
lease.release()
say("closed")
