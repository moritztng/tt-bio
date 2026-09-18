#!/usr/bin/env python3
"""Sample a Blackhole node's AICLK with monotonic read brackets until stdin says stop.

Brackets matter: a sample only counts for a timed region if the whole read happened inside it,
so `read_start_ns` and `read_end_ns` are both recorded. Same contract as
`perf/c10_bare_baseline/control.py`, reduced to the clock and board power.
"""
import json, select, sys, time
from pathlib import Path

def main():
    path, node = sys.argv[1], int(sys.argv[2])
    root = Path(f"/sys/class/tenstorrent/tenstorrent!{node}")
    clk = root / "tt_aiclk"
    try:
        hw = next(root.glob("device/hwmon/hwmon*")) / "power1_input"
    except StopIteration:
        hw = None
    with Path(path).open("w") as out:
        while not select.select([sys.stdin], [], [], 0.0015)[0]:
            row = {"read_start_ns": time.monotonic_ns()}
            try:
                row["MHz"] = int(clk.read_text())
                if hw is not None:
                    row["W"] = int(hw.read_text()) / 1e6
            except BaseException as e:
                row["error"] = repr(e)
            row["read_end_ns"] = time.monotonic_ns()
            out.write(json.dumps(row) + "\n")
        out.flush()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
