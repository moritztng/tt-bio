#!/usr/bin/env python3
"""Sample tt_aiclk for one node at ~1 kHz into a jsonl, until a deadline.

A separate file rather than inline code in the caller: an inline `python3 -c` string passed
through ssh had its quotes eaten and the sampler died silently, which cost a whole timing run.
The AICLK SETS the fold time on Blackhole, so a second taken without a clock sampled DURING the
interval is not a measurement.
"""
import json
import sys
import time
from pathlib import Path

out, node, secs = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
src = Path("/sys/class/tenstorrent/tenstorrent!%s/tt_aiclk" % node)
deadline = time.monotonic() + secs
with out.open("w") as f:
    while time.monotonic() < deadline:
        a = time.monotonic_ns()
        try:
            mhz = int(src.read_text())
            row = {"read_start_ns": a, "MHz": mhz, "read_end_ns": time.monotonic_ns()}
        except Exception as e:                                                 # noqa: BLE001
            row = {"read_start_ns": a, "error": repr(e), "read_end_ns": time.monotonic_ns()}
        f.write(json.dumps(row) + "\n")
        f.flush()
