#!/usr/bin/env python3
"""The 512 aa fold's DRAM byte counter, with three charging defects fixed and each one switchable.

Same captures, same `itemize` buffer-address dedupe, same everything else as
`perf/b2x_difflayer/real_traffic.py`, with the same terminal-output correction (model version 2).
With all three corrections off its byte totals match that counter. Historical version-1
artifacts included an invented read of each unconsumed tensor output.

Terminal outputs carry a write only. Internal intermediates and opaque scratch with no
counted reader retain one ASSUMED read, itemised as assumed_read_MB (opaque buffers also
separately). Graphs do not expose all internal reads, rereads or spills, so these are estimates,
not exact physical traffic. floor_MB is the historical once-read/write estimate, now also
excluding terminal output reads; it is not a proven physical bound.

L1   `real_traffic.counts` decides whether an op touches DRAM from the DRAM rows alone
     (`alloc_by_op[i] > 0`), so an op whose only output landed in L1 is not charged as a reader of
     the DRAM it consumed. Here the test is "allocated anything, DRAM or L1". Found by
     `roof-redteam-2`; 110 of one capture's 410 ops are in the class.

PRE  A buffer from `ttnn.allocate_tensor_on_device` is passed to its op in a DESTINATION slot, so
     the op WRITES it. `real_traffic.counts` charges the allocation a write and then, because the
     consuming `generic_op` allocated nothing and so fails `moves_dram`, charges the allocation a
     second time as a phantom read. Here the allocation moves nothing (the capture shows
     `create_device_tensor` + `buffer_allocate` and no host copy) and the first consuming op is
     charged the write; later consumers are readers, which they were not before. The audit behind
     it is `audit_prealloc.py`: all 22 `allocate_tensor_on_device` call sites in `tt_bio`.

GATE `reblock_permute_gated` reads TWO channel-tile slices of the wide fused projection, not all of
     it: `reader_reblock_permute_gated.cpp` issues exactly two `noc_async_read_page` per
     (row-tile, col-tile, channel-tile) group, at `p_off + ct` and `g_off + ct`, with `ct < Ct`.
     At the trimul Ct = 4 and Ctw = 16, so the call reads half its input. The two calls that share
     one wide projection read disjoint halves, so the projection is read once between them and
     charging each of them all of it is a clean 2x. N = 512 is a multiple of the tile height, so
     the reader's padding-row path never fires and the fraction is exact.

The corrections do not push the same way: PRE and GATE delete bytes, L1 adds them.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b2x_difflayer"))
from itemize import itemize, top_level_spans, is_terminal_tensor_output         # noqa: E402

NO_TRAFFIC = {"ttnn.reshape", "ttnn.unsqueeze", "ttnn.squeeze", "ttnn.deallocate"}
ALLOC = "ttnn.allocate_tensor_on_device"
GENERIC = "ttnn.generic_op"
DEVOP = "GenericOpDeviceOperation"


def load_nodes(path):
    op = gzip.open if str(path).endswith(".gz") else open
    d = json.load(op(path, "rt"))
    return d if isinstance(d, list) else d["nodes"]


def _shape(p):
    return [int(x) for x in re.findall(r"-?\d+", str(p.get("shape", "")))]


def generic_op_args(nodes):
    """op index -> the tensor argument list `ttnn.generic_op` was handed, in order.

    Read off the `GenericOpDeviceOperation` node: every `tensor` node that connects to it is one
    positional argument, and the destination appears twice because it is also the return value.
    """
    ops, owner = top_level_spans(nodes)
    devop = {}
    for n in nodes:
        if n["node_type"] == "function_start" and (n.get("params") or {}).get("name") == DEVOP:
            i = owner.get(n["counter"])
            if i is not None and ops[i]["name"] == GENERIC:
                devop[n["counter"]] = i
    args = defaultdict(list)
    for n in nodes:
        if n["node_type"] != "tensor":
            continue
        for c in (n.get("connections") or []):
            if c in devop:
                args[devop[c]].append((_shape(n["params"]), int(n["params"]["size"])))
    return args


def gated_read_fraction(nodes):
    """op index -> {buffer size: fraction of it the kernel actually reads}, for the one op class
    in this fold that reads part of an operand. Matched on the signature `reblock_permute_gated`
    alone can have: `generic_op([xw, out])` with xw [1, N, N, Cw] and out [1, Cs, N, N], the
    destination repeated as the return value, and Cw a multiple of Cs greater than 2 Cs."""
    out = {}
    for i, args in generic_op_args(nodes).items():
        if len(args) != 3:
            continue
        (sa, za), (sb, zb), (sc, zc) = args
        if (sb, zb) != (sc, zc) or len(sa) != 4 or len(sb) != 4:
            continue
        N, Cw, Cs = sa[1], sa[3], sb[1]
        if not (sa[2] == N and sb[2] == sb[3] == N and Cs and Cw % Cs == 0 and Cw > 2 * Cs):
            continue
        out[i] = {za: 2.0 * Cs / Cw}
    return out


def counts(call, l1=True, pre=True, gate=True):
    nodes = call["nodes"]
    ops, rows = itemize(call)
    dram = [r for r in rows if r["kind"] == "DRAM"]
    frac = gated_read_fraction(nodes) if gate else {}

    alloc_by_op = defaultdict(int)
    for r in (rows if l1 else dram):
        if r["alloc_op_i"] is not None:
            alloc_by_op[r["alloc_op_i"]] += r["size"]

    writer_of, prealloc_writers = {}, set()
    if pre:
        for r in dram:
            i = r["alloc_op_i"]
            if i is None or ops[i]["name"] != ALLOC:
                continue
            cand = [c for c in r["consumers"] if ops[c]["name"] not in NO_TRAFFIC]
            writer_of[r["buffer"]] = cand[0] if cand else None
            if cand:
                prealloc_writers.add(cand[0])

    def moves_dram(i):
        name = ops[i]["name"]
        if name in NO_TRAFFIC:
            return False
        if pre and name == ALLOC:
            return False                      # a bare reservation moves nothing
        return alloc_by_op[i] > 0 or name.endswith("_") or i in prealloc_writers

    def read_bytes(i, size):
        return size * frac.get(i, {}).get(size, 1.0)

    w, rd, phantom, saved = defaultdict(float), defaultdict(float), defaultdict(float), 0.0
    terminal_read_removed = opaque_read = 0
    for r in dram:
        size, ai = r["size"], r["alloc_op_i"]
        wi = writer_of.get(r["buffer"], "none") if pre else "none"
        if wi != "none":                      # pre-allocated: charge the op that writes it
            w[wi if wi is not None else ai] += size
            for i in r["consumers"]:
                if i != wi and moves_dram(i):
                    rd[i] += read_bytes(i, size)
                    saved += size - read_bytes(i, size)
            continue
        readers = [i for i in r["consumers"] if moves_dram(i)]
        if ai is not None:
            w[ai] += size
        for i in readers:
            rd[i] += read_bytes(i, size)
            saved += size - read_bytes(i, size)
            if ops[i]["name"].endswith("_") and alloc_by_op[i] == 0:
                w[i] += size                  # in-place: rewrites what it read
        if is_terminal_tensor_output(r, readers):
            terminal_read_removed += size
        elif not readers:
            k = ai if ai is not None else -1
            rd[k] += size
            phantom[k] += size
            if not r["tensor_nodes"]:
                opaque_read += size

    pre_B = sum(r["size"] for r in dram if r["alloc_op_i"] is None)
    al = sum(r["size"] for r in dram if r["alloc_op_i"] is not None)
    per_op = sorted(((w[i] + rd[i], ops[i]["name"], w[i], rd[i]) for i in range(len(ops))
                     if w[i] + rd[i]), reverse=True)
    by_op = [w[i] + rd[i] for i in range(len(ops))]
    # Bytes with no owning op: a pre-existing buffer (a weight) whose only consumers all fail
    # `moves_dram`. Real traffic, so it must not vanish from a per-op sum; it carries no FLOPs.
    unattributed = w[-1] + rd[-1]
    return {"floor_MB": (pre_B + 2 * al - terminal_read_removed) / 1e6, "once_MB": (pre_B + al) / 1e6,
            "real_MB": (sum(w.values()) + sum(rd.values())) / 1e6,
            "real_w_MB": sum(w.values()) / 1e6, "real_r_MB": sum(rd.values()) / 1e6,
            "phantom_read_MB": sum(phantom.values()) / 1e6,
            "traffic_model_version": 2,
            "terminal_read_removed_MB": terminal_read_removed / 1e6,
            # Keep phantom_read_MB for old consumers; these retained reads are estimates,
            # not evidence that scratch traffic is absent or exactly one pass.
            "assumed_read_MB": sum(phantom.values()) / 1e6,
            "opaque_buffer_assumed_read_MB": opaque_read / 1e6,
            "gate_saved_MB": saved / 1e6,
            "prealloc_MB": sum(r["size"] for r in dram if r["alloc_op_i"] is not None
                               and ops[r["alloc_op_i"]]["name"] == ALLOC) / 1e6,
            "n_ops": len(ops), "per_op": per_op, "by_op": by_op,
            "unattributed_B": unattributed,
            "op_names": [o["name"] for o in ops]}


MODES = {"published": dict(l1=False, pre=False, gate=False),
         "+L1": dict(l1=True, pre=False, gate=False),
         "+PRE": dict(l1=False, pre=True, gate=False),
         "+GATE": dict(l1=False, pre=False, gate=True),
         "corrected": dict(l1=True, pre=True, gate=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    a = ap.parse_args()
    print("%-46s" % "capture" + "".join("%11s" % m for m in MODES))
    for p in a.captures:
        nodes = load_nodes(p)
        v = [counts({"nodes": nodes}, **kw)["real_MB"] for kw in MODES.values()]
        print("%-46s" % Path(p).name[:46] + "".join("%11.3f" % x for x in v))


if __name__ == "__main__":
    main()
