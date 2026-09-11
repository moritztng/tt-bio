#!/usr/bin/env python3
"""Real DRAM traffic per captured call: every op charged for what it actually moves.

Three counts of the same capture, and the spread between them is the honest uncertainty:

* PUBLISHED  -- `perf/bioir_roofline/fold_bytes_512.py`: DRAM reads deduped by tensor id.
  Charges a full DRAM read for every metadata view, because ttnn gives a reshape or an
  unsqueeze a fresh tensor id over the same buffer. An over-count.
* FLOOR      -- every distinct DRAM buffer written once and read once. An under-count: a
  buffer read by four ops is charged one read.
* REAL       -- this file. Per buffer: one write if it was allocated in the call, one more
  write per in-place consumer, and one read per consuming op that actually touches DRAM.

An op is charged as touching DRAM unless it allocated nothing and is not in-place -- a
`reshape`, `unsqueeze`, `squeeze` or `deallocate` aliases or frees and moves nothing. An
in-place op (trailing underscore) reads and rewrites its destination. A buffer with no
recorded consumer is still charged one read: device-op scratch is written and consumed
inside one op and never becomes a ttnn-level tensor node, so it has no visible reader.

Still a lower bound on a fat matmul: a multi-pass matmul that re-streams an operand is not
modelled, and nothing here sees L1 spill.
"""
import argparse
import gzip
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itemize import itemize                                                   # noqa: E402

NO_TRAFFIC = {"ttnn.reshape", "ttnn.unsqueeze", "ttnn.squeeze", "ttnn.deallocate"}


def load(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.load(open(path))


def counts(call):
    ops, rows = itemize(call)
    dram = [r for r in rows if r["kind"] == "DRAM"]
    alloc_by_op = defaultdict(int)
    for r in dram:
        if r["alloc_op_i"] is not None:
            alloc_by_op[r["alloc_op_i"]] += r["size"]

    def moves_dram(i):
        name = ops[i]["name"]
        if name in NO_TRAFFIC:
            return False
        return alloc_by_op[i] > 0 or name.endswith("_")

    w = defaultdict(int)
    rd = defaultdict(int)
    for r in dram:
        readers = [i for i in r["consumers"] if moves_dram(i)]
        if r["alloc_op_i"] is not None:
            w[r["alloc_op_i"]] += r["size"]
        for i in readers:
            rd[i] += r["size"]
            if ops[i]["name"].endswith("_") and alloc_by_op[i] == 0:
                w[i] += r["size"]                      # in-place: rewrites what it read
        if not readers:
            rd[r["alloc_op_i"] if r["alloc_op_i"] is not None else -1] += r["size"]

    pre = sum(r["size"] for r in dram if r["alloc_op_i"] is None)
    al = sum(r["size"] for r in dram if r["alloc_op_i"] is not None)
    per_op = sorted(((w[i] + rd[i], ops[i]["name"], w[i], rd[i]) for i in range(len(ops))
                     if w[i] + rd[i]), reverse=True)
    return {"floor_MB": (pre + 2 * al) / 1e6, "once_MB": (pre + al) / 1e6,
            "real_MB": (sum(w.values()) + sum(rd.values())) / 1e6,
            "real_w_MB": sum(w.values()) / 1e6, "real_r_MB": sum(rd.values()) / 1e6,
            "n_ops": len(ops), "per_op": per_op}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    for path in a.jsons:
        print("### " + Path(path).name)
        for c in load(path)["calls"]:
            x = counts(c)
            print("%-34s once %9.3f  floor %9.3f  REAL %9.3f MB (w %8.3f r %8.3f)  real/floor %.2fx"
                  % (c["sig"], x["once_MB"], x["floor_MB"], x["real_MB"],
                     x["real_w_MB"], x["real_r_MB"], x["real_MB"] / x["floor_MB"]))
            for tot, name, ww, rr in x["per_op"][:a.top]:
                print("      %8.3f MB  %-40s w %7.3f r %7.3f" % (tot / 1e6, name, ww / 1e6, rr / 1e6))


if __name__ == "__main__":
    main()
