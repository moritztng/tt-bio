#!/usr/bin/env python3
"""Which physical card is a Tenstorrent perf cell standing on?

Pass 6 measured the same RF3 fold 13 % apart on two boxes and traced it to the
compute-with-storage grid, 13x10 against 11x10, with the compute grid identical -- so
`TT_BIO_FORCE_GRID` does not reach it. This reads the cause straight off SMBus telemetry
without opening a device, so it costs nothing and does not contend for a card:
`ENABLED_TENSIX_COL` is a 14-bit mask, one bit per Blackhole tensix column, and the
storage grid is the enabled columns minus the one the dispatcher reserves.

    python3 perf/rf3/card_census.py [--out results/p7_card_census.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import subprocess
import tempfile

N_COLS = 14   # Blackhole tensix columns
N_ROWS = 10

SMI_CANDIDATES = (
    "/home/ttuser/tt-bio/env/bin/tt-smi",
    "/home/ttuser/.tenstorrent-venv/bin/tt-smi",
    "/home/ttuser/.uma_run/env/bin/tt-smi",
)


def snapshot() -> dict:
    """tt-smi -s writes the file on some builds and prints it on others. Take either."""
    smi = next((p for p in SMI_CANDIDATES if os.access(p, os.X_OK)), None)
    if smi is None:
        smi = "tt-smi"
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "snap.json")
        out = subprocess.run([smi, "-s", "-f", dst], capture_output=True, text=True).stdout
        if os.path.getsize(dst) if os.path.exists(dst) else 0:
            return json.load(open(dst))
        return json.loads(out[out.index("{"):])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cards = []
    for i, e in enumerate(snapshot()["device_info"]):
        b, t = e["board_info"], e["smbus_telem"]
        mask = int(t["ENABLED_TENSIX_COL"], 16)
        cols = bin(mask).count("1")
        cards.append({
            "device": i,
            "board_type": b["board_type"],
            "board_id": b["board_id"],
            "bus_id": b["bus_id"],
            "enabled_tensix_col": t["ENABLED_TENSIX_COL"],
            "cols_enabled": cols,
            "cols_harvested": N_COLS - cols,
            "tensix": cols * N_ROWS,
            # one column goes to the dispatcher, so the grid a tensor can be sharded over
            # is one narrower than the enabled grid
            "storage_grid": f"{cols - 1}x{N_ROWS}",
            "storage_cores": (cols - 1) * N_ROWS,
        })

    out = {"host": socket.gethostname(), "cards": cards}
    print(json.dumps(out, indent=2))
    if args.out:
        dst = args.out
        if not os.path.isabs(dst):
            dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(out, open(dst, "w"), indent=2)
        print("wrote", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
