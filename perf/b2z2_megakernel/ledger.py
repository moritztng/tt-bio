#!/usr/bin/env python3
"""Turn `census.py`'s graph captures into the block's tile-pass ledger, DRAM and L1 separately.

`census.py` counts only the DRAM side, and for this block that is the smaller half: the shipped
Transition already asks for `L1_MEMORY_CONFIG` on every intermediate, so its traffic never reaches
DRAM and a DRAM-only ledger reads it as nearly free. The term wave 1 measured -- the math thread
blocked on CB input tiles -- is an L1 term. So both are counted here and reported side by side.

Read the two columns as what they are:
  DRAM  tiles crossing the DRAM<->L1 boundary. Deleted by fusion, and also by any lever that keeps
        an intermediate in L1.
  L1    tiles a kernel's reader hands its compute thread from an L1 buffer, and the packer writes
        back to one. Deleted ONLY by fusion: there is no memory config that removes them.
Neither sees a matmul re-streaming an operand from L1 inside one op, so both are lower bounds on
what the unpacker actually moves.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "perf" / "b2x_difflayer"))
from itemize import itemize                                                  # noqa: E402

TILE = 2048                      # bf16 32x32; the Boltz-2 trunk is bf16 end to end
NO_TRAFFIC = {"ttnn.reshape", "ttnn.unsqueeze", "ttnn.squeeze", "ttnn.deallocate"}


def ledger(call, kind):
    ops, rows = itemize(call)
    bufs = [r for r in rows if r["kind"] == kind]
    alloc = defaultdict(int)
    for r in bufs:
        if r["alloc_op_i"] is not None:
            alloc[r["alloc_op_i"]] += r["size"]

    def moves(i):
        return ops[i]["name"] not in NO_TRAFFIC and (alloc[i] > 0 or ops[i]["name"].endswith("_"))

    w, rd = defaultdict(int), defaultdict(int)
    for r in bufs:
        readers = [i for i in r["consumers"] if moves(i)]
        if r["alloc_op_i"] is not None:
            w[r["alloc_op_i"]] += r["size"]
        for i in readers:
            rd[i] += r["size"]
            if ops[i]["name"].endswith("_") and alloc[i] == 0:
                w[i] += r["size"]
        if not readers:
            rd[r["alloc_op_i"] if r["alloc_op_i"] is not None else -1] += r["size"]
    per_op = {}
    for i, op in enumerate(ops):
        if w[i] + rd[i]:
            per_op[i] = (op["name"], rd[i] // TILE, w[i] // TILE)
    return ops, per_op, sum(rd.values()) // TILE, sum(w.values()) // TILE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    grand = {}
    for p in a.graphs:
        call = json.load(open(p))
        name = Path(p).stem
        ops, d_op, d_r, d_w = ledger(call, "DRAM")
        _, l_op, l_r, l_w = ledger(call, "L1")
        grand[name] = {"n_ops": len(ops), "dram_r": d_r, "dram_w": d_w,
                       "l1_r": l_r, "l1_w": l_w,
                       "tile_passes": d_r + d_w + l_r + l_w}
        print("### %s -- %d ttnn ops" % (name, len(ops)))
        print("%-46s %9s %9s %9s %9s %10s" % ("op", "DRAM r", "DRAM w", "L1 r", "L1 w", "passes"))
        rowsum = []
        for i in range(len(ops)):
            dr, dw = d_op.get(i, (None, 0, 0))[1:]
            lr, lw = l_op.get(i, (None, 0, 0))[1:]
            if dr + dw + lr + lw:
                rowsum.append((dr + dw + lr + lw, i, ops[i]["name"], dr, dw, lr, lw))
        for tot, i, nm, dr, dw, lr, lw in sorted(rowsum, reverse=True)[:a.top]:
            print("%-46s %9d %9d %9d %9d %10d" % (nm[:46], dr, dw, lr, lw, tot))
        print("%-46s %9d %9d %9d %9d %10d" % ("TOTAL", d_r, d_w, l_r, l_w,
                                              d_r + d_w + l_r + l_w))
        print()
    print("=== roll-up, tile passes ===")
    for k, v in sorted(grand.items(), key=lambda kv: -kv[1]["tile_passes"]):
        print("%-28s ops %4d  DRAM %9d  L1 %9d  TOTAL %10d"
              % (k, v["n_ops"], v["dram_r"] + v["dram_w"], v["l1_r"] + v["l1_w"],
                 v["tile_passes"]))
    (HERE / "out" / "ledger_rollup.json").write_text(json.dumps(grand, indent=1))


if __name__ == "__main__":
    main()
