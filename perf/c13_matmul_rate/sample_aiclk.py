#!/usr/bin/env python3
"""Sample Blackhole AICLK from sysfs only, for the duration of a measurement.

Deliberately holds NO fd on /dev/tenstorrent/N. perf/c12_genop_rate/clk.py both forces and
samples from one process, and on 2026-09-18 that process was signalled away the moment the
measuring process opened the chip: the force was released 21 s into a 4-minute session, every
arm after it ran at the governor's clock, and the record still said 99.8 percent of samples at
1350 MHz because the samples simply stopped. A sampler that touches the device can be taken
with the device. This one cannot, and the FORCE now lives inside the measuring process so it
cannot outlive or predecease the measurement.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path


def aiclk(node):
    try:
        return int(Path("/sys/class/tenstorrent/tenstorrent!%d/tt_aiclk" % node).read_text())
    except OSError as e:
        return "ERR:%s" % e.errno


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--period-ms", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=3600.0)
    a = ap.parse_args()
    nodes = [int(x) for x in a.nodes.split(",")]
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        f.write(json.dumps({"sampler": "sysfs-only", "nodes": nodes,
                            "period_ms": a.period_ms, "t_start": time.time()}) + "\n")
        t_end = time.time() + a.max_s
        while not stop["now"] and time.time() < t_end:
            f.write(json.dumps(dict({"t": time.time()},
                                    **{str(n): aiclk(n) for n in nodes})) + "\n")
            f.flush()
            time.sleep(a.period_ms / 1e3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
