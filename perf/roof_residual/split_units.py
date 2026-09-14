#!/usr/bin/env python3
"""Split a roof-budget capture into its marked child units and the ops belonging to no child.

`baseline_attrib.py` pushes a `unit::<Class>` marker into the graph capture every time a bracketed
module runs inside a capture that is already open, so the capture of a parent already carries the
boundary between "inside a child unit" and "inside the parent and nothing else". That second set is
exactly what `roof-budget`'s module-level table cannot see. Nothing here is a new measurement: it
reads the 26 committed captures and re-uses `exec_flops.py` for FLOPs and `real_traffic.py`'s
charging rule for bytes, unchanged.

Two things the marker stream does not give you for free.

* **Ends go missing.** ttnn's graph tracker drops a `function_end` when the frame below it leaked,
  and several ops in this model leak frames: a `PairformerLayer` capture has 1347 `function_start`
  against 1265 `function_end`, and `unit::TriangleMultiplication` has 2 starts and 0 ends in it.
  `stacking_level` is no help, it drifts with the same leak (a token `AttentionPairBias` opens at
  level 5 and the next marker opens at level 9). So a child region is closed by ALIGNMENT: the
  child's own standalone capture is a list of top-level ttnn op names, and the region in the
  parent is that list followed by whatever the parent did next. Walk the two in lockstep, stop at
  the first mismatch, and the tail belongs to the parent.
* **The outermost frame never closes.** The capture's own `unit::X` end is emitted after
  `end_graph_capture`, so it is not in the node list. Expected, and the only open frame allowed.

The alignment is checked, not assumed: `align_report()` reports, per child region, whether the
whole of the standalone child matched. A region that does not match in full is reported rather
than silently truncated.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "roof_budget"))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

import exec_flops as EF                                                      # noqa: E402
from itemize import itemize, top_level_spans                                 # noqa: E402
from real_traffic import NO_TRAFFIC                                          # noqa: E402

UNIT = "unit::"
CAPDIR = ROOT / "perf" / "roof_budget" / "captures"


def cap_path(sig):
    return CAPDIR / ("cap_" + sig.replace("|", "__") + ".json.gz")


def op_names(nodes):
    ops, _ = top_level_spans(nodes)
    return [o["name"] for o in ops]


def _markers(nodes):
    """[(op_index_at_or_after, kind, unit_name)] in capture order, kind in {'start','end'}."""
    ops, _ = top_level_spans(nodes)
    starts = sorted((o["start"], i) for i, o in enumerate(ops))
    out = []
    j = 0
    for n in nodes:
        nm = str((n.get("params") or {}).get("name", ""))
        if not nm.startswith(UNIT):
            continue
        c = n["counter"]
        while j < len(starts) and starts[j][0] < c:
            j += 1
        idx = starts[j][1] if j < len(starts) else len(ops)
        out.append((idx, "start" if n["node_type"] == "function_start" else "end",
                    nm[len(UNIT):]))
    return ops, out


def per_op_bytes(nodes, ops):
    """real_traffic's charge, kept per top-level op instead of summed."""
    _ops, rows = itemize({"nodes": nodes})
    dram = [r for r in rows if r["kind"] == "DRAM"]
    alloc = defaultdict(int)
    for r in dram:
        if r["alloc_op_i"] is not None:
            alloc[r["alloc_op_i"]] += r["size"]

    def moves(i):
        nm = ops[i]["name"]
        return False if nm in NO_TRAFFIC else (alloc[i] > 0 or nm.endswith("_"))

    w, rd = defaultdict(int), defaultdict(int)
    for r in dram:
        readers = [i for i in r["consumers"] if moves(i)]
        if r["alloc_op_i"] is not None:
            w[r["alloc_op_i"]] += r["size"]
        for i in readers:
            rd[i] += r["size"]
            if ops[i]["name"].endswith("_") and alloc[i] == 0:
                w[i] += r["size"]
        if not readers:
            rd[r["alloc_op_i"] if r["alloc_op_i"] is not None else -1] += r["size"]
    return [w[i] + rd[i] for i in range(len(ops))], rd[-1]


# Bookkeeping ops: no FLOPs, and a view op moves nothing either. They are skipped when a child
# region is aligned against that child's own capture, because the two are not always recorded the
# same way: `top_level_spans` falls back to its RANGE rule on an unbalanced capture, and RANGE
# opens a separate top-level op for a ttnn.* call nested inside another one (a `ttnn.slice` inside
# a `ttnn.Tensor.__getitem__`) where STACK folds it into its parent.
SKIP_IN_ALIGN = set(EF.FREE) | {"ttnn.deallocate", "ttnn.allocate_tensor_on_device"}


def _subst(seq):
    return [x for x in seq if x not in SKIP_IN_ALIGN]


def split(sig, child_names=None, edges=frozenset()):
    """(ops, owner, bytes, flops, orphan, report) for the capture of `sig`.

    owner[i] is '' when top-level ttnn op i runs in the captured unit itself and in no marked
    child. A child frame opens on its `unit::` start and closes on whichever comes first:

      * its own `unit::` end, which is exact;
      * the point where its substantive ops stop matching the child's own standalone capture,
        counted inclusive of the child's own children, since the standalone capture contains
        them too. That is what recovers a region whose end marker ttnn dropped.

    A frame with no standalone capture to align against stays open until an enclosing frame
    closes it, and is reported as `unaligned` so the reader can see which rows rest on a marker
    and which on an alignment.
    """
    nodes = EF.nodes_of(cap_path(sig))
    ops, marks = _markers(nodes)
    by_bytes, orphan = per_op_bytes(nodes, ops)
    ins, outs = EF.operands(nodes)[1:]
    flops = [EF.op_flops(o["name"], ins[i], outs[i]) for i, o in enumerate(ops)]
    names = [o["name"] for o in ops]
    n = len(ops)

    ref = {}
    for _i, kind, u in marks:
        if kind == "start" and u not in ref:
            ref[u] = [_subst(op_names(EF.nodes_of(cap_path(s))))
                      for s in (child_names or [])
                      if s.split("|")[0] == u and cap_path(s).is_file()]

    by_pos = defaultdict(list)
    for idx, kind, u in marks[1:]:                      # marks[0] is the capture's own unit
        by_pos[idx].append((kind, u))

    owner = [""] * n
    report = []
    stack = []

    def close(frame, at, how):
        report.append((frame["u"], frame["start"], at - frame["start"], how))

    for i in range(n + 1):
        for kind, u in by_pos.get(i, []):
            if kind == "start":
                # a start marker also closes every frame that cannot be this unit's parent.
                # `edges` is the set of (parent class, child class) the bracket tree recorded in
                # the same fold, so this is the model's real nesting, not a guess.
                while stack and edges and (stack[-1]["u"], u) not in edges:
                    close(stack.pop(), i, "closed-by-" + u)
                cands = [(r, 0) for r in ref.get(u, [])]
                stack.append({"u": u, "start": i, "cands": cands,
                              "how": "aligned" if cands else "unaligned"})
            else:
                while stack:
                    f = stack.pop()
                    close(f, i, "marker-end" if f["u"] == u else "closed-by-" + u)
                    if f["u"] == u:
                        break
        if i == n:
            break
        if names[i] not in SKIP_IN_ALIGN:
            while stack:
                f = stack[-1]
                if not f["cands"]:
                    break                               # nothing to align against: stays open
                if any(k < len(r) and r[k] == names[i] for r, k in f["cands"]):
                    break
                if not any(k == len(r) for r, k in f["cands"]):
                    # the reference ran out of agreement before it ran out of ops, so it is the
                    # wrong reference for this call, not the end of the region. Stop aligning and
                    # let a marker or an edge close the frame.
                    f["cands"], f["how"] = [], "unaligned(ref-diverged)"
                    break
                close(stack.pop(), i, "aligned %d/%d" % (max(k for _r, k in f["cands"]),
                                                         len(f["cands"][0][0])))
            for f in stack:
                live = [(r, k + 1) for r, k in f["cands"] if k < len(r) and r[k] == names[i]]
                if live:
                    f["cands"] = live
        if stack:
            owner[i] = stack[-1]["u"]
    while stack:
        close(stack.pop(), n, "to-end")
    return ops, owner, by_bytes, flops, orphan, report


def residual(sig, child_names):
    """{op name: {n, B, F, F_log}} for the ops of `sig` that belong to no marked child."""
    ops, owner, by, fl, orphan, report = split(sig, child_names)
    agg = defaultdict(lambda: {"n": 0, "B": 0.0, "F": 0, "F_log": 0})
    kids = defaultdict(lambda: {"n": 0, "B": 0.0, "F": 0})
    for i, o in enumerate(ops):
        log, pad, kind = fl[i]
        tgt = agg[o["name"]] if owner[i] == "" else kids[owner[i]]
        tgt["n"] += 1
        tgt["B"] += by[i]
        tgt["F"] += pad
        if owner[i] == "":
            tgt["F_log"] += log
    return dict(agg), dict(kids), orphan, report
