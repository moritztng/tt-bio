#!/usr/bin/env python3
"""How much of the token DiT's wall time is dispatch, i.e. what would a trace of LocalTokenTransformer buy?

    gap_dit_trace.py OUT.json ARM [STEPS] [--fast=0|1]

cen3 (WH fast @1000): the DiT is ~1,500 ops a step at 62.4 ms/step of kernel time, and the step's plain wall sits
~54 ms above kernel time plus drained idle. Runs one design with ARM and, on DiT calls 3..3+N, times the eager call
(synchronized before and after) and a replay of a trace captured once from persistent copies of the same inputs
(copy-in + execute + synchronize), and counts unequal elements between the two outputs. The design always takes the
eager output, so its digest is the ARM's own. RFD3_ATTN_THREAD stays off (an upload from the index thread during
capture would throw), so pass an ARM without `at`.
"""
import json
import os
import pathlib
import sys
import time

OUT = pathlib.Path(sys.argv[1])
ARM = sys.argv[2]
STEPS = sys.argv[3] if len(sys.argv) > 3 else "4"
FAST = next((a for a in sys.argv[4:] if a.startswith("--fast")), "--fast=1")
assert "+at" not in ARM and not ARM.startswith("at+"), "drop `at`: index uploads would hit the open capture"
os.environ.setdefault("TT_BIO_TRACE_REGION_MB", "48")     # the rf3 entry: ~1,600 ops
sys.argv = [sys.argv[0], ARM, "/tmp/gap_dit_trace", STEPS, "60", FAST]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_knobs import apply                                               # noqa: E402
apply(ARM)
sys.path.insert(0, os.getcwd())
import torch                                                              # noqa: E402
import ttnn                                                               # noqa: E402
from tt_bio import tenstorrent as T                                       # noqa: E402
from tt_bio.rfd3 import model as M                                        # noqa: E402

N_MEASURE, FIRST = 12, 3
REC = {"arm": ARM, "card": os.environ.get("TT_VISIBLE_DEVICES"), "calls": []}
ST = {"n": 0}
_orig = M.LocalTokenTransformer.run_device


def _wall(fn):
    dev = T.get_device()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(dev)
    return out, 1e3 * (time.perf_counter() - t0)


def run_device(self, a, s, z, mask):
    ST["n"] += 1
    n = ST["n"]
    if n < FIRST or n >= FIRST + N_MEASURE or "fail" in ST:
        return _orig(self, a, s, z, mask)
    dev = T.get_device()
    ins = (a, s, z, mask)
    eager, ms_eager = _wall(lambda: _orig(self, *ins))
    e = ttnn.to_torch(eager).float()        # read before any replay: replay may write freed capture-time addresses
    try:
        if "tid" not in ST:
            ST["bufs"] = [ttnn.clone(t) for t in ins]
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            ST["tid"], ST["out"] = T.capture_trace(dev, lambda: _orig(self, *ST["bufs"]))
            REC["capture_ms"] = 1e3 * (time.perf_counter() - t0)

        def replay():
            for src, dst in zip(ins, ST["bufs"]):
                ttnn.copy(src, dst) if hasattr(ttnn, "copy") else ttnn.assign(src, dst)
            ttnn.execute_trace(dev, ST["tid"], cq_id=0, blocking=False)
            return ST["out"]
        out, ms_replay = _wall(replay)
        _, ms_exec = _wall(lambda: ttnn.execute_trace(dev, ST["tid"], cq_id=0, blocking=False))
        r = ttnn.to_torch(out).float()
        rec = {"call": n, "ms_eager": round(ms_eager, 3), "ms_replay_with_copy": round(ms_replay, 3),
               "ms_execute_only": round(ms_exec, 3), "neq": int((e != r).sum()),
               "maxabs": float((e - r).abs().max()), "shapes": [list(t.shape) for t in ins]}
    except Exception as exc:                                              # report, keep the design going
        ST["fail"] = True
        rec = {"call": n, "ms_eager": round(ms_eager, 3), "error": repr(exc)[:400]}
    print("[dittrace]", rec, flush=True)
    REC["calls"].append(rec)
    OUT.write_text(json.dumps(REC, indent=1))
    return eager


M.LocalTokenTransformer.run_device = run_device
T.get_device(trace="rf3")                 # reserve the trace region before gap_screen opens the chip
import runpy                                                              # noqa: E402
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gap_screen.py"), run_name="__main__")
