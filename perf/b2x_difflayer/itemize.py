#!/usr/bin/env python3
"""Itemise a captured DiffusionTransformerLayer call: per op, per buffer, read multiplicity.

Two counts come out of the same capture, and the difference between them is most of the
argument with BioIR:

* ONCE -- every distinct DRAM buffer charged once, full stop. This is the convention
  BioIR`s 54.8 MB is arithmetically in: 23.6 MB of weights + 8.39 MB of pair bias + 29
  token tensors at 0.786 MB each = 54.78 MB.
* ROUND-TRIP -- inputs charged one read, intermediates charged a write AND a read, which
  is what an intermediate physically costs. Stricter, and the honest compulsory floor.

Neither is what `perf/bioir_roofline/fold_bytes_512.py` reports. That instrument dedupes
DRAM reads by TENSOR ID, and ttnn gives a reshape / unsqueeze / slice a fresh tensor id
over the SAME buffer, so a metadata view is charged a full DRAM read it never performs:
102 tensor ids over 34 buffer addresses in one token-DiT call, and 214.45 MB where the
round-trip floor is 127.89 MB.

Consumers come out of the capture graph: a `buffer` node connects to its `tensor` nodes,
and each `tensor` node connects to the ops that took it as an argument. Both are mapped
back to the enclosing top-level `ttnn.*` call.
"""
import argparse
import json
from collections import defaultdict


def top_level_spans(nodes, rule="auto"):
    """counter -> (op_index, op_name) for every node inside a top-level ttnn.* call.

    Two ownership rules. STACK is the original: push on `function_start`, pop on `function_end`,
    a ttnn.* name at depth zero opens an op. It is exact when the capture is balanced and it fails
    silently when it is not -- a ttnn.* frame whose `function_end` never arrives leaves the stack
    permanently non-empty, every later call reads as nested, and the per-op table collapses. A
    512 aa trimul capture has 102 function_start against 93 function_end and the stack rule finds
    4 ops in it where the device runs 19.

    RANGE owns every node by the last ttnn.* `function_start` at or before it, which is immune to
    a missing end. It is exact here because ttnn.* names do not nest inside one another in these
    captures: a ttnn.linear's internals are `MatmulDeviceOperation` and `create_device_tensor`,
    not another ttnn.*.

    `rule="auto"` uses STACK when the capture is balanced and RANGE when it is not, so a balanced
    capture keeps byte-for-byte the numbers it reported before.
    """
    if rule in ("auto", "range"):
        n_start = sum(1 for n in nodes if n.get("node_type") == "function_start")
        n_end = sum(1 for n in nodes if n.get("node_type") == "function_end")
        if rule == "range" or n_start != n_end:
            return _spans_by_range(nodes)
    return _spans_by_stack(nodes)


def _spans_by_range(nodes):
    ops, owner, cur = [], {}, None
    for n in nodes:
        c = n["counter"]
        if n.get("node_type") == "function_start":
            name = str((n.get("params") or {}).get("name", ""))
            if name.startswith("ttnn."):
                cur = len(ops)
                ops.append({"name": name, "start": c, "end": None})
        if cur is not None:
            owner[c] = cur
            ops[cur]["end"] = c
    return ops, owner


def _spans_by_stack(nodes):
    owner = {}
    ops = []
    stack = []
    for n in nodes:
        t = n.get("node_type")
        c = n["counter"]
        if t == "function_start":
            name = str((n.get("params") or {}).get("name", ""))
            if not stack and name.startswith("ttnn."):
                stack.append(len(ops))
                ops.append({"name": name, "start": c, "end": None})
            elif stack:
                stack.append(None)
        elif t == "function_end":
            if stack:
                top = stack.pop()
                if top is not None:
                    ops[top]["end"] = c
        if stack and stack[0] is not None:
            owner[c] = stack[0]
    return ops, owner


def itemize(call):
    nodes = call["nodes"]
    by_counter = {n["counter"]: n for n in nodes}
    ops, owner = top_level_spans(nodes)

    # buffer node -> size, and the op that allocated it (None => pre-existing: weight or input)
    buf_size, buf_alloc_op, buf_kind = {}, {}, {}
    for n in nodes:
        p = n.get("params") or {}
        if n["node_type"] == "buffer":
            buf_size[n["counter"]] = int(p.get("size", 0) or 0)
            buf_kind[n["counter"]] = str(p.get("type", ""))
            buf_alloc_op[n["counter"]] = None
    for n in nodes:
        if n["node_type"] != "buffer_allocate":
            continue
        for c in n.get("connections") or []:
            if c in buf_size:
                buf_alloc_op[c] = owner.get(n["counter"])

    # buffer -> tensors -> consuming ops
    tensor_of_buffer = defaultdict(list)
    for b in buf_size:
        for c in (by_counter[b].get("connections") or []):
            if by_counter.get(c, {}).get("node_type") == "tensor":
                tensor_of_buffer[b].append(c)
    consumers = {}
    for b, tens in tensor_of_buffer.items():
        ops_seen = set()
        for t in tens:
            for c in (by_counter[t].get("connections") or []):
                o = owner.get(c)
                if o is not None:
                    ops_seen.add(o)
        # the allocating op is a writer, not a reader
        ops_seen.discard(buf_alloc_op.get(b))
        consumers[b] = sorted(ops_seen)

    rows = []
    for b, size in buf_size.items():
        rows.append({
            "buffer": b, "size": size, "kind": buf_kind[b],
            "alloc_op": (ops[buf_alloc_op[b]]["name"] if buf_alloc_op[b] is not None else None),
            "alloc_op_i": buf_alloc_op[b],
            # A buffer with no tensor node attached never entered `tensor_of_buffer`, so it has
            # no entry here. That is device-op scratch, not an error: it is written and consumed
            # inside one op and never becomes a ttnn-level tensor. `real_traffic` already charges
            # a consumer-less buffer one read; before this it crashed the itemisation instead.
            "n_consumers": len(consumers.get(b, ())),
            "consumers": list(consumers.get(b, ())),
            "consumer_names": [ops[i]["name"] for i in consumers.get(b, ())],
        })
    return ops, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--sig", default="512x768")
    a = ap.parse_args()
    d = json.load(open(a.json))
    call = [c for c in d["calls"] if a.sig in c["sig"]][0]
    ops, rows = itemize(call)
    dram = [r for r in rows if r["kind"] == "DRAM"]

    floor_w = sum(r["size"] for r in dram if r["alloc_op_i"] is not None)
    floor_r = sum(r["size"] for r in dram)
    real_r = sum(r["size"] * max(r["n_consumers"], 0) for r in dram)
    persistent = sum(r["size"] for r in dram if r["alloc_op_i"] is None)

    print("SIG %s  top-level ttnn ops=%d  DRAM buffers=%d" % (call["sig"], len(ops), len(dram)))
    print("ONCE        %8.3f MB   (pre-existing %.3f + intermediates %.3f)"
          % (floor_r / 1e6, persistent / 1e6, floor_w / 1e6))
    print("ROUND-TRIP  %8.3f MB   (pre-existing %.3f + 2 x intermediates %.3f)"
          % ((persistent + 2 * floor_w) / 1e6, persistent / 1e6, floor_w / 1e6))
    print("reads charged per consuming op (view ops included, so an over-count): %.3f MB"
          % (real_r / 1e6))
    print()
    print("--- buffers >= 0.5 MB, by line-by-line read traffic ---")
    print("%9s %4s %5s  %-34s %s" % ("size MB", "rds", "MB", "allocated by", "consumers"))
    for r in sorted(dram, key=lambda r: -r["size"] * max(r["n_consumers"], 1)):
        if r["size"] < 500_000:
            continue
        print("%9.3f %4d %5.1f  %-34s %s"
              % (r["size"] / 1e6, r["n_consumers"], r["size"] * r["n_consumers"] / 1e6,
                 r["alloc_op"] or "(pre-existing)",
                 ",".join(sorted(set(r["consumer_names"])))))
    print()
    print("--- per top-level op: bytes written, bytes read ---")
    w = defaultdict(float)
    rd = defaultdict(float)
    for r in dram:
        if r["alloc_op_i"] is not None:
            w[r["alloc_op_i"]] += r["size"]
        for i in r["consumers"]:
            rd[i] += r["size"]
    for i, op in enumerate(ops):
        print("%3d %-46s w %8.3f  r %8.3f  MB" % (i, op["name"], w[i] / 1e6, rd[i] / 1e6))
    print()
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    for i, op in enumerate(ops):
        agg[op["name"]][0] += 1
        agg[op["name"]][1] += w[i]
        agg[op["name"]][2] += rd[i]
    print("--- by op name ---")
    for name, (n, ww, rr) in sorted(agg.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
        print("%-46s n=%2d  w %8.3f  r %8.3f  tot %8.3f MB"
              % (name, n, ww / 1e6, rr / 1e6, (ww + rr) / 1e6))


if __name__ == "__main__":
    main()
