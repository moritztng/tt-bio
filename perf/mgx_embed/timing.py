"""Warm per-sequence wall of one embedding model at a few lengths, for an A/B of two trees on one chip.

    python3 perf/mgx_embed/timing.py --model saprot-35m --pdb mtor.pdb --lengths 126,1534 --out t.jsonl

Each length is embedded alone --calls times after two untimed calls (compile, then trace capture on
the second sighting), and the row carries the median and the spread with AICLK sampled during the
timed calls. Run the same command from each tree; `tree` records which one it was.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "mgx_embed"))
from perf.clocksample import during  # noqa: E402
from accuracy import pdb_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--lengths", default="126,254,510,1534")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_grad_enabled(False)
    if a.model.startswith("saprot"):
        from tt_bio import saprot as mod
        model = mod.load_saprot(a.model)
    else:
        from tt_bio import esmc as mod
        model = mod.load_esmc(a.model)
    full = pdb_sequence(a.pdb)
    tree = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h"],
                          capture_output=True, text=True).stdout.strip() \
        or (ROOT / "REV").read_text().strip()  # a tree synced without .git names itself in REV
    with open(a.out, "a") as fh:
        for L in [int(x) for x in a.lengths.split(",")]:
            seqs = {"s": full[:L]}
            for _ in range(2):
                mod.embed_sequences(model, seqs, batch_size=1)
            walls = []
            with during() as clk:
                for _ in range(a.calls):
                    t = time.perf_counter()
                    mod.embed_sequences(model, seqs, batch_size=1)
                    walls.append(time.perf_counter() - t)
            row = {"model": a.model, "L": L, "tree": tree, "root": str(ROOT), "calls": a.calls,
                   "median_ms": round(1e3 * statistics.median(walls), 2),
                   "min_ms": round(1e3 * min(walls), 2), "max_ms": round(1e3 * max(walls), 2),
                   "aiclk": clk.summary(), "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES")}
            print(json.dumps(row), flush=True)
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
