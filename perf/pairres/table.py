"""One line per ladder row: verdict, wall, AICLK sampled during the fold, load, speed-bar status.

    python3 table.py ~/scratch/pairres/*.jsonl

AICLK comes from `aiclk.log` ("epoch card mhz", clk.sh) and load from `load.log` ("epoch load1
nproc", load.sh), both beside the jsonl files, restricted to the fold's own window
[ts - wall_s, ts] and to its card. A row whose median load exceeds 1.5x nproc is VOID under the
speed bar (1c87356b4), so its wall time is capacity evidence only.
"""
import json
import statistics
import sys
from pathlib import Path


def _samples(path, card=None):
    out = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) != 3:
            continue
        if card is None:
            out.append((int(f[0]), float(f[1]), int(f[2])))
        elif f[1] == str(card):
            out.append((int(f[0]), float(f[2])))
    return out


def _epoch(ts):
    from datetime import datetime, timezone
    return int(datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())


def main(paths):
    root = Path(paths[0]).parent
    load = _samples(root / "load.log")
    clk_cache = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            end = _epoch(r["ts"])
            start = end - int(r["wall_s"])
            card = r["device"]
            clk = clk_cache.setdefault(card, _samples(root / "aiclk.log", card))
            mhz = [v for t, v in clk if start <= t <= end]
            ld = [(v, n) for t, v, n in load if start <= t <= end]
            clk_s = (f"AICLK min/med {min(mhz):.0f}/{statistics.median(mhz):.0f} MHz n={len(mhz)}"
                     if mhz else "AICLK unsampled")
            if ld:
                med = statistics.median(v for v, _ in ld)
                bar = "VOID" if med > 1.5 * ld[0][1] else "scorable"
                ld_s = f"load {med:.0f}/{ld[0][1]} {bar}"
            else:
                ld_s = "load unsampled"
            print(f"{Path(p).stem:24s} {r['model']:14s} {r['rung']:22s} card {card:2d} "
                  f"{r['verdict']:9s} {r['wall_s']:7.1f} s  {clk_s}  {ld_s}")


if __name__ == "__main__":
    main(sys.argv[1:])
