"""AICLK of the leased chip over a stage's own window, from aiclk.py's 1 Hz log.

A clock sampled before or after the work is not the clock the work ran at, so every figure here
is taken between two timestamps you pass in, which are the stage's own start and end.

The node is NOT the card number. On .107 card 30 is PCI 0000:c7:00.0 and sysfs node **6**; the
tt-smi/UMD id and the /dev/tenstorrent/N node disagree in general, so the mapping is given
explicitly rather than inferred from the lease.

Wormhole reads 1000 MHz at its ceiling, not 1350. The Blackhole rule "under ~1200 MHz means a
busy chip and a bad number" does NOT carry over: on this part 1000 is the architectural maximum
and a perfectly clean sitting reads 1000, so `--floor` defaults to 1000 and "samples under the
floor" counts only genuinely degraded ones.

    aiclk_window.py aiclk3.jsonl --node 6 --from <iso|epoch> --to <iso|epoch> [--floor 1000]
"""
import argparse
import json
import statistics
import sys
from datetime import datetime, timezone


def ts(s):
    try:
        return float(s)
    except ValueError:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(
            tzinfo=timezone.utc).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--node", type=int, default=6)
    ap.add_argument("--from", dest="t0", required=True)
    ap.add_argument("--to", dest="t1", required=True)
    ap.add_argument("--floor", type=float, default=1000.0)
    a = ap.parse_args()
    t0, t1 = ts(a.t0), ts(a.t1)

    vals = []
    for line in open(a.log):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line while the logger is still writing
        if t0 <= r["t"] <= t1:
            v = r.get("mhz", {}).get(str(a.node))
            if v is not None:
                vals.append(v)

    if not vals:
        print(f"no samples for node {a.node} in the window", file=sys.stderr)
        return 2
    under = sum(1 for v in vals if v < a.floor)
    print(f"node {a.node}  n={len(vals)} DURING the window  "
          f"median {statistics.median(vals):.0f}  min {min(vals):.0f}  max {max(vals):.0f}  "
          f"MHz;  {under} sample(s) under {a.floor:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
