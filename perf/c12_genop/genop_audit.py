#!/usr/bin/env python3
"""`ttnn.generic_op` -- the largest floor in the 512 aa fold, priced by no C10 row. Where its 3,920
calls are, and whether its 0.9127 TB is this op's and counted once.

Reads the three top-level captures `perf/roof_launch/op_census.py` walks (`true_floor.TOP`,
weighted by `roof_budget`'s own call counts) and the same
`perf/roof_arb/corrected_traffic.counts()`, so every number joins the census on (capture, op index)
and is not a second convention. No device, no new measurement.

WHY THE CENSUS CANNOT SEE THIS OP. `true_floor.LAUNCH_ARM` has no entry for `ttnn.generic_op`, so
`launch_arm` returns None, `launch_key` returns None, and `op_census.py`'s ladder loop drops the op
before the arm list exists. That is why it is in neither the 70 priced nor the 52 refused launch
keys of `perf/c10_fold_census/runs/sweep*/replay.json`: `fold_shapes.json` has 122 keys and 286,752
calls and no generic arm at all.

THREE THINGS THIS FILE CHECKS.

* dedupe. `itemize` keys a charge on the buffer NODE. A node is one allocation, not one address:
  this re-keys every charge on the address its tensors carry and separates address reuse after a
  deallocate (real, two passes) from a live-overlap collision (a double charge).
* the L1 pre-allocated destination. `counts(pre=True)` builds `prealloc_writers` from DRAM rows
  only, and `counts(l1=True)` fixes `alloc_by_op` for an op that ALLOCATES into L1. An op whose
  destination was RESERVED in L1 by `ttnn.allocate_tensor_on_device` is in neither set, fails
  `moves_dram`, and is charged nothing -- not even the DRAM it reads. Reported as a repair beside
  the published number, never folded into it.
* the band. `c12-orchestrator` (9d16d1455, `perf/c12_orchestrator/replay_bias/`) measured that the
  census's per-key prices carry two uncontrolled biases that push opposite ways: `replay.py`
  allocates every operand AND every output `DRAM_MEMORY_CONFIG` while the fold runs many of these
  ops L1-resident (byte ratio 2.23x on multiply_, 1.83x on linear, and the control classes the fold
  really does run through DRAM agree at 1.00-1.03x), and the published `linear`/`matmul` seconds are
  the `@grid110` arm while the fold passes no core_grid. So the 10.5368 s is not a point and the
  closure below is stated as a band with THREE unknowns, of which only F is measured. The pinned
  copy of that artifact is `replay_bias_9d16d1455.json` beside this file.
* the arithmetic term's basis. Every generic_op matmul row is covered by a measured class, and for
  three of the four the rate comes from a NATIVE qb2 card-3 arm on the shipped kernel at the fold's
  own FLOP count (`perf/roof_triatt_rate/rate_ab_512_qb2c3.json`: ship_sdpa, ship_in, ship_out).
  The trimul in-projection has no shipped arm and takes a pc-transferred rate for a variant the
  fold does not run. Printed per site so the two bases are never summed as one.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERF = HERE.parent
for d in ("roof_true", "roof_budget", "b2x_difflayer", "roof_residual", "roof_arb", "roof_shape"):
    sys.path.insert(0, str(PERF / d))
import true_floor as TF                                                       # noqa: E402
import corrected_traffic as CT                                                # noqa: E402
import exec_flops as EF                                                       # noqa: E402
from itemize import itemize, top_level_spans                                  # noqa: E402

GENERIC = "ttnn.generic_op"
# the census's own measured roofs, from perf/c10_fold_census/runs/sweep2 at a during-sampled 1350 MHz
DRAM_ROOF = 442.877e9
CUBE_MEAS = 122.29e12
MATMUL_CLASS_MEAS = 21.38e12
FOLD_S = 14.881
MHZ = 1350.0
# site -> (file:line, what it is). The graph gives the call count, the owner unit and the shapes;
# these lines are the only `ttnn.generic_op` in tt_bio whose operand list matches each signature.
SITE_SRC = {
    ("TriangleMultiplication", "in-proj"):
        ("tt_bio/mm_dualnoc.py:123 (in_proj), via tt_bio/tenstorrent.py:5665 "
         "(_in_proj_matmul) -> tt_bio/mm_generic.py:372",
         "x[1,N,N,128] @ w[128,640], split=[512,128] -> xw[1,N,N,512] + gate[1,N,N,128]"),
    ("TriangleMultiplication", "reblock-gated"):
        ("tt_bio/reblock_permute.py:823 (reblock_permute_gated)",
         "two disjoint channel halves of xw -> [1,128,N,N]"),
    ("TriangleMultiplication", "reblock-back"):
        ("tt_bio/reblock_permute.py:563 (reblock_permute_back)",
         "[1,128,N,N] -> [1,N,N,128]"),
    ("TriangleAttention", "in-proj"):
        ("tt_bio/triatt_qkv.py:420 (qkvgb_heads) -> tt_bio/mm_generic.py:372",
         "x[N,N,128] @ w[128,544], five N chunks -> q|k|v|g[N,4,N,32] + bias[N,N,4]"),
    ("TriangleAttention", "fused SDPA"):
        ("tt_bio/sdpa_generic.py:590 (sdpa)", "q,k,v,mask -> out[N,4,N,32]"),
    ("TriangleAttention", "out-proj"):
        ("tt_bio/triatt_qkv.py:321 (out_proj) -> tt_bio/mm_generic.py:372",
         "gated[N,4,N,32] @ w[128,128] -> [N,N,128]; memory_config picks L1 or DRAM"),
}


class A:
    perf = PERF
    roofs = "shape_roofs_qb2c3_shipped.json"
    eltwise_rate = None
    uncovered_lo = None
    uncovered_hi = None


def cap(sig):
    p = PERF / "roof_budget" / "captures" / ("cap_" + sig.replace("|", "__") + ".json.gz")
    return json.load(gzip.open(p, "rt"))


def buf_addr(nodes):
    """buffer node -> the address its tensor nodes report. A node is one allocation; an address is
    reused after a deallocate, so the address alone is not a key."""
    by = {n["counter"]: n for n in nodes}
    out = {}
    for n in nodes:
        if n["node_type"] != "buffer":
            continue
        a = set()
        for c in (n.get("connections") or []):
            t = by.get(c)
            if t and t["node_type"] == "tensor":
                ad = (t.get("params") or {}).get("address")
                if ad is not None:
                    a.add(int(ad))
        out[n["counter"]] = sorted(a)
    return out


def site_of(owner, args, rows):
    """(owner unit, site label) from the executed argument list and the census shape rows."""
    shp = [tuple(s) for s, _z in args]
    n4 = [s for s in shp if len(s) == 4]
    if owner == "TriangleMultiplication":
        if rows:
            return (owner, "in-proj")
        wide = [s for s in n4 if s[0] == 1 and s[3] == s[1]]
        return (owner, "reblock-gated" if wide else "reblock-back")
    if owner == "TriangleAttention":
        if len(rows) == 2:
            return (owner, "fused SDPA")
        if rows and rows[0][3] == 128:
            return (owner, "out-proj")
        return (owner, "in-proj")
    return (owner, "other")


def charges(nodes, ops):
    """counts()'s charging, replayed with the provenance of every byte kept.

    Returns (charge, repair, l1_dest). `charge` is what counts(l1,pre,gate) charges, per op, as
    [(role, buffer node, address, bytes)]. `repair` is the DRAM this op demonstrably moves that the
    published rule drops because its destination was reserved in L1.
    """
    _o, rows = itemize({"nodes": nodes})
    dram = [r for r in rows if r["kind"] == "DRAM"]
    addrs = buf_addr(nodes)
    frac = CT.gated_read_fraction(nodes)

    alloc_by_op = defaultdict(int)
    for r in rows:
        if r["alloc_op_i"] is not None:
            alloc_by_op[r["alloc_op_i"]] += r["size"]
    writer_of, pw = {}, set()
    for r in dram:
        i = r["alloc_op_i"]
        if i is None or ops[i]["name"] != CT.ALLOC:
            continue
        cand = [c for c in r["consumers"] if ops[c]["name"] not in CT.NO_TRAFFIC]
        writer_of[r["buffer"]] = cand[0] if cand else None
        if cand:
            pw.add(cand[0])
    # the same rule over L1 rows: who WRITES a destination that was reserved in L1
    l1_writer = set()
    for r in rows:
        if r["kind"] == "DRAM":
            continue
        i = r["alloc_op_i"]
        if i is None or ops[i]["name"] != CT.ALLOC:
            continue
        cand = [c for c in r["consumers"] if ops[c]["name"] not in CT.NO_TRAFFIC]
        if cand:
            l1_writer.add(cand[0])

    def moves(i):
        nm = ops[i]["name"]
        if nm in CT.NO_TRAFFIC or nm == CT.ALLOC:
            return False
        return alloc_by_op[i] > 0 or nm.endswith("_") or i in pw

    charge, repair = defaultdict(list), defaultdict(list)
    for r in dram:
        size, ai, buf = r["size"], r["alloc_op_i"], r["buffer"]
        ad = (addrs.get(buf) or [None])[0]
        wi = writer_of.get(buf, "none")
        if wi != "none":
            charge[wi if wi is not None else ai].append(("W-pre", buf, ad, size))
            for i in r["consumers"]:
                if i == wi:
                    continue
                b = size * frac.get(i, {}).get(size, 1.0)
                (charge if moves(i) else repair)[i].append(("R", buf, ad, b))
            continue
        readers = [i for i in r["consumers"] if moves(i)]
        if ai is not None:
            charge[ai].append(("W", buf, ad, size))
        for i in readers:
            charge[i].append(("R", buf, ad, size * frac.get(i, {}).get(size, 1.0)))
            if ops[i]["name"].endswith("_") and alloc_by_op[i] == 0:
                charge[i].append(("W-inpl", buf, ad, size))
        if not readers:
            for i in r["consumers"]:
                if i in l1_writer or ops[i]["name"] in CT.NO_TRAFFIC:
                    continue
                repair[i].append(("R", buf, ad, size * frac.get(i, {}).get(size, 1.0)))
    # only keep the repair for ops the published rule drops entirely
    repair = {i: v for i, v in repair.items() if not moves(i) and i in l1_writer}
    return charge, repair, l1_writer, addrs, rows


def live_overlap(nodes):
    """Addresses held by two DIFFERENT live buffers at once. Zero means keying on the buffer node
    and keying on (address, epoch) are the same key and the byte total is deduped."""
    ev = defaultdict(list)
    for n in nodes:
        p = n.get("params") or {}
        if n["node_type"] in ("buffer_allocate", "buffer_deallocate") and p.get("address"):
            ev[int(p["address"])].append((n["counter"],
                                          1 if n["node_type"] == "buffer_allocate" else -1,
                                          int(p.get("size", 0) or 0)))
    bad = {}
    for a, es in ev.items():
        live = 0
        for _c, d, _s in sorted(es):
            live += d
            if live > 1:
                bad[a] = max(bad.get(a, 0), live)
    return bad


def main():
    R = TF.setup(PERF, A)
    R["eltwise_rate"] = R["lo_rate"]
    sites = defaultdict(lambda: defaultdict(float))
    rowsrc = {}
    G = defaultdict(float)
    FOLD = defaultdict(float)
    overlap = {}
    l1hole = defaultdict(lambda: defaultdict(float))

    for sig in TF.TOP:
        calls = R["by"][sig]["calls"]
        nodes = cap(sig)
        J = TF.Join(R, sig)
        _ol, owner, _b, _f, _o, _r = R["SU"].split(sig, R["have"], R["edges"])
        ch, rep, l1w, addrs, rows = charges(nodes, J.ops)
        args = CT.generic_op_args(nodes)
        ov = live_overlap(nodes)
        if ov:
            overlap[sig.split("|")[0]] = ov
        for m, kw in CT.MODES.items():
            c = CT.counts({"nodes": nodes}, **kw)
            gi = [i for i in range(len(J.ops)) if J.ops[i]["name"] == GENERIC]
            FOLD["mode_gen_" + m] += calls * sum(c["by_op"][i] for i in gi)
            FOLD["mode_all_" + m] += calls * (sum(c["by_op"]) + c["unattributed_B"])
        # the L1-pre-allocated-destination hole, EVERY op class not just generic_op
        for i, v in rep.items():
            e = l1hole[J.ops[i]["name"]]
            e["calls"] += calls
            e["B"] += calls * sum(b for _k, _bu, _a, b in v)
        for i in range(len(J.ops)):
            if J.ops[i]["name"] != GENERIC:
                continue
            (B, F_mm, _Fel, t_tr, t_ar, *_rest) = TF.op_terms(R, J, i)
            mm = TF.op_shape_rows(EF, GENERIC, J.ins[i], J.outs[i])
            k = site_of(owner[i], args[i], mm)
            rowsrc[k] = mm
            e = sites[k]
            e["calls"] += calls
            e["B"] += calls * B
            e["W"] += calls * sum(b for r, _bu, _a, b in ch[i] if r.startswith("W"))
            e["R"] += calls * sum(b for r, _bu, _a, b in ch[i] if r == "R")
            e["repair_B"] += calls * sum(b for _r, _bu, _a, b in rep.get(i, []))
            e["F"] += calls * F_mm
            e["s_tr"] += calls * t_tr
            e["s_ar"] += calls * t_ar
            e["s_fl"] += calls * max(t_tr, t_ar)
            e["outs_seen"] += len(J.outs[i])
            for key in ("B", "F", "s_tr", "s_ar"):
                G[key] += calls * locals()[{"B": "B", "F": "F_mm", "s_tr": "t_tr",
                                            "s_ar": "t_ar"}[key]]
            G["s_fl"] += calls * max(t_tr, t_ar)
            G["calls"] += calls
            G["repair_B"] += calls * sum(b for _r, _bu, _a, b in rep.get(i, []))

    print("ROOFS  census-measured DRAM %.3f GB/s, cube %.2f TFLOP/s, in-fold matmul class "
          "%.2f TFLOP/s; model stream %.3f GB/s, cube-of-record %.2f TFLOP/s"
          % (DRAM_ROOF / 1e9, CUBE_MEAS / 1e12, MATMUL_CLASS_MEAS / 1e12,
             R["stream"] / 1e9, R["cube"] / 1e12))

    print("\n=== CALL-SITES: 3,920 ttnn.generic_op calls, from the executed graph ===")
    print("%-24s %-14s %6s %8s %8s %8s %8s %8s  %s"
          % ("owner unit", "site", "calls", "W GB", "R GB", "TB/fold", "TFLOP", "s_floor",
             "file:line"))
    for k, e in sorted(sites.items(), key=lambda kv: -kv[1]["s_fl"]):
        print("%-24s %-14s %6d %8.1f %8.1f %8.4f %8.3f %8.4f  %s"
              % (k[0][:24], k[1], e["calls"], e["W"] / 1e9, e["R"] / 1e9, e["B"] / 1e12,
                 e["F"] / 1e12, e["s_fl"], SITE_SRC[k][0]))
        for r in rowsrc[k]:
            hit = R["RATES"].get(r[:4])
            print("      b=%-5d M=%-7d K=%-4d N=%-4d %7.2f GFLOP  %s"
                  % (r[0], r[1], r[2], r[3], r[4] / 1e9,
                     ("%s @ %.2f TF/s" % (hit[0], hit[1] / 1e12)) if hit
                     else "UNCOVERED at hi_rate %.2f TF/s" % (R["hi_rate"] / 1e12)))
        if e["repair_B"]:
            print("      DROPPED by the published rule (L1 pre-allocated destination): "
                  "%.4f TB/fold" % (e["repair_B"] / 1e12))
    print("TOTAL %d calls, %.4f TB, %.3f TFLOP, model floor %.4f s"
          % (G["calls"], G["B"] / 1e12, G["F"] / 1e12, G["s_fl"]))
    print("shape-model outputs seen for this op class: %d (census records out_tiles = 0)"
          % sum(e["outs_seen"] for e in sites.values()))

    print("\n=== BYTES: the audit ===")
    print("corrected charge      W %.4f TB + R %.4f TB = %.4f TB  (census: 0.9127 TB)"
          % (G["B"] / 1e12 - sum(e["R"] for e in sites.values()) / 1e12,
             sum(e["R"] for e in sites.values()) / 1e12, G["B"] / 1e12))
    print("live address overlap  %s"
          % ("none in any capture: buffer node == (address, epoch)" if not overlap else overlap))
    print("repair, this op       +%.4f TB (+%.2f %%) -> %.4f TB"
          % (G["repair_B"] / 1e12, 100 * G["repair_B"] / G["B"],
             (G["B"] + G["repair_B"]) / 1e12))
    print("MODES, generic_op alone vs the whole fold:")
    for m in CT.MODES:
        print("  %-10s %8.4f TB   fold %8.4f TB   %5.1f %% of fold   %.4f s at the measured roof"
              % (m, FOLD["mode_gen_" + m] / 1e12, FOLD["mode_all_" + m] / 1e12,
                 100 * FOLD["mode_gen_" + m] / FOLD["mode_all_" + m],
                 FOLD["mode_gen_" + m] / DRAM_ROOF))
    print("the same hole, EVERY op class (bytes the published rule drops fold-wide):")
    ht = 0.0
    for nm, e in sorted(l1hole.items(), key=lambda kv: -kv[1]["B"]):
        ht += e["B"]
        print("  %-34s %8d calls  %8.4f TB" % (nm, e["calls"], e["B"] / 1e12))
    print("  %-34s %8s   %8.4f TB = %.4f s at the measured roof" % ("TOTAL", "", ht / 1e12,
                                                                    ht / DRAM_ROOF))

    # --- whole-fold floor, recomputed at the census's own measured DRAM roof -------------------
    c = json.load(open(PERF / "roof_launch" / "op_census_512.json"))["by_op"]
    PRICED = {"ttnn.add", "ttnn.add_", "ttnn.layer_norm", "ttnn.linear", "ttnn.matmul",
              "ttnn.multiply", "ttnn.multiply_"}
    allb = sum(v["B"] for v in c.values())
    pb = sum(v["B"] for k, v in c.items() if k in PRICED)
    R["stream"] = DRAM_ROOF
    A2, _pc, _bk, _bo, _caps = TF.run(R, [(t, R["by"][t]["calls"]) for t in TF.TOP])
    floor_meas = A2["floor"]
    gen_tr = G["B"] / DRAM_ROOF
    print("\n=== RESIDUAL ===")
    print("fold bytes %.4f TB; 8 priced classes %.4f TB; unpriced %.4f TB; generic_op %.4f TB "
          "= %.1f %% of unpriced" % (allb / 1e12, pb / 1e12, (allb - pb) / 1e12, G["B"] / 1e12,
                                     100 * G["B"] / (allb - pb)))
    print("unpriced traffic floor at the measured roof: %.4f s total, generic_op %.4f s, "
          "everything else %.4f s" % ((allb - pb) / DRAM_ROOF, gen_tr,
                                      (allb - pb - G["B"]) / DRAM_ROOF))
    print("whole-fold device floor, per-op max(traffic, arithmetic), at the measured roof: "
          "%.4f s / %.1f Mcycles" % (floor_meas, floor_meas * MHZ))
    for nm, v in (("census closure, generic_op at traffic only",
                   10.5368 + gen_tr + (allb - pb - G["B"]) / DRAM_ROOF),
                  ("the same, generic_op at its model floor",
                   10.5368 + G["s_fl"] + (allb - pb - G["B"]) / DRAM_ROOF)):
        print("  %-46s %7.4f s against a %.3f s fold -> %+.4f s for host" % (nm, v, FOLD_S,
                                                                            FOLD_S - v))
    print("AXIS-A-CEILING: fold %.3f s - device floor %.4f s = %.4f s / %.1f Mcycles"
          % (FOLD_S, floor_meas, FOLD_S - floor_meas, (FOLD_S - floor_meas) * MHZ))

    out = {"roofs": {"dram_GBps": DRAM_ROOF / 1e9, "cube_TFLOPs": CUBE_MEAS / 1e12},
           "generic_op": {k: G[k] for k in G},
           "modes_TB": {m: FOLD["mode_gen_" + m] / 1e12 for m in CT.MODES},
           "fold_modes_TB": {m: FOLD["mode_all_" + m] / 1e12 for m in CT.MODES},
           "l1_prealloc_hole_TB": {k: v["B"] / 1e12 for k, v in l1hole.items()},
           "floor_at_measured_roof_s": floor_meas,
           "fold_bytes_TB": allb / 1e12, "priced_bytes_TB": pb / 1e12,
           "live_address_overlap": overlap,
           "sites": [{"owner": k[0], "site": k[1], "src": SITE_SRC[k][0],
                      "what": SITE_SRC[k][1], "rows": rowsrc[k], **dict(e)}
                     for k, e in sorted(sites.items(), key=lambda kv: -kv[1]["s_fl"])]}
    (HERE / "genop_audit.json").write_text(json.dumps(out, indent=1, default=float))

    # --- the bracket, and the native shipped-kernel join --------------------------------------
    # The class rates are `roof_shape`'s fraction of ITS cube applied to a cube of record. The
    # committed cube of record is 104.93 TFLOP/s (roof_budget); the census measured the same part
    # in the same session at 122.29. Both are carried, because the choice moves the floor by 1.2 s
    # and therefore the host ceiling by the same amount.
    print("\n=== AXIS-A-CEILING: the bracket ===")
    brack = {}
    for cor in (R["cube"], CUBE_MEAS):
        R2 = TF.setup(PERF, A)
        R2["eltwise_rate"] = R2["lo_rate"]
        R2["stream"] = DRAM_ROOF
        R2["RATES"] = TF.class_rates(R2["WG"], R2["roofs"], cor)
        Ab, _a, _b, _c, _d = TF.run(R2, [(t, R2["by"][t]["calls"]) for t in TF.TOP])
        brack[cor / 1e12] = Ab["floor"]
        print("  rates scaled to a %.2f TFLOP/s cube: device floor %.4f s / %.1f Mcycles, "
              "host ceiling %.4f s / %.1f Mcycles"
              % (cor / 1e12, Ab["floor"], Ab["floor"] * MHZ, FOLD_S - Ab["floor"],
                 (FOLD_S - Ab["floor"]) * MHZ))
    print("  bytes alone (no arithmetic term): device floor %.4f s, host ceiling %.4f s"
          % (allb / DRAM_ROOF, FOLD_S - allb / DRAM_ROOF))

    # Three of the four generic_op site classes were ALREADY measured natively, on qb2 card 3, on
    # the shipped kernel, at the fold's own FLOP count. Joining them to the fold's call census is
    # a measurement of this op and not a floor -- the only join nobody had made.
    ab = json.loads((PERF / "roof_triatt_rate" / "rate_ab_512_qb2c3.json").read_text())["arms"]
    rr = ab["cube"]["TFLOPs_at_min"] * 1e12 / CUBE_MEAS
    print("\n=== the native join: qb2c3 shipped arms x the fold's call census ===")
    print("  that session's own cube read %.3f TFLOP/s against the census's same-part %.2f in the "
          "same 9-rep style, so its arms are %.2f %% slow and are carried both ways"
          % (ab["cube"]["TFLOPs_at_min"], CUBE_MEAS / 1e12, 100 * (1 - rr)))
    nat = natn = 0.0
    for nm, site in (("ship_in", "in-proj"), ("ship_sdpa", "fused SDPA"),
                     ("ship_out", "out-proj")):
        e = sites[("TriangleAttention", site)]
        s_ = ab[nm]["min_ms"] * 1e-3 * e["calls"]
        nat += s_
        natn += s_ * rr
        print("  %-10s %-11s %8.5f ms x %4d = %7.4f s raw / %7.4f s at 1350 MHz   "
              "(model floor %7.4f s)" % (nm, site, ab[nm]["min_ms"], e["calls"], s_, s_ * rr,
                                         e["s_fl"]))
    tm = sites[("TriangleMultiplication", "in-proj")]["s_fl"]
    rb = (sites[("TriangleMultiplication", "reblock-gated")]["s_fl"]
          + sites[("TriangleMultiplication", "reblock-back")]["s_fl"])
    print("  trimul in-proj  no shipped arm exists: model floor %.4f s on a pc-transferred rate "
          "for trimul_in_flat, a variant the fold does not run" % tm)
    print("  reblock x2      zero FLOPs, pure traffic floor %.4f s" % rb)
    print("  generic_op, measured where measurable: %.4f s raw / %.4f s at 1350 MHz "
          "/ %.1f Mcycles   against a %.4f s model floor"
          % (nat + tm + rb, natn + tm + rb, (natn + tm + rb) * MHZ, G["s_fl"]))
    dev = 10.5368 + (natn + tm + rb) + (allb - pb - G["B"]) / DRAM_ROOF
    print("  closure with it: 10.5368 + %.4f + %.4f = %.4f s against a %.3f s fold -> %+.4f s "
          "for ALL host time" % (natn + tm + rb, (allb - pb - G["B"]) / DRAM_ROOF, dev, FOLD_S,
                                 FOLD_S - dev))

    out["bracket_floor_s"] = brack
    out["native_join_s"] = {"raw": nat + tm + rb, "at_1350": natn + tm + rb,
                            "clock_ratio": rr, "closure_s": dev}

    # --- the band: three unknowns, one measured ----------------------------------------------
    # P (the 8 priced classes' true in-fold cost), G (generic_op) and H (exposed host) plus U
    # (other unpriced device traffic) have to sum to the 14.881 s fold. The census's 10.5368 s is
    # not P: `c12-orchestrator`'s replay_bias measured a DRAM-operand overprice on every class and
    # a grid-110 underprice on linear+matmul, and they cancel in the aggregate. Byte-correcting
    # each class by its own measured ratio brackets P from the replay side; the classes' own
    # roofline floor brackets it from below; the fold identity brackets it from above.
    rbrows = json.loads((HERE / "replay_bias_9d16d1455.json").read_text())["rows"]
    pub = {"linear": 4.6604, "matmul": 1.479, "multiply_": 1.751, "layer_norm_w": 1.357 + 0.176,
           "add_": 0.847, "add": 0.140, "multiply": 0.126}
    print("\n=== the band: byte-correcting the census's replay prices ===")
    print("  %-14s %8s %8s %9s %9s %9s" % ("class", "pub s", "byte x", "pub/ratio",
                                           "bare s", "bare/ratio"))
    plo = phi = 0.0
    for r in sorted(rbrows, key=lambda r: -pub[r["class"]]):
        c, br = r["class"], r["byte_ratio"]
        plo += pub[c] / br
        phi += r["bare_s"] / br
        print("  %-14s %8.4f %8.4f %9.4f %9.4f %9.4f  (published arm: %s)"
              % (c, pub[c], br, pub[c] / br, r["bare_s"], r["bare_s"] / br, r["published_arm"]))
    print("  %-14s %8.4f %8s %9.4f %9.4f %9.4f" % ("TOTAL", sum(pub.values()), "", plo,
                                                   sum(r["bare_s"] for r in rbrows), phi))
    PRICED = {"ttnn.add", "ttnn.add_", "ttnn.layer_norm", "ttnn.linear", "ttnn.matmul",
              "ttnn.multiply", "ttnn.multiply_"}
    band = {}
    for cor in (R["cube"], CUBE_MEAS):
        R2 = TF.setup(PERF, A)
        R2["eltwise_rate"] = R2["lo_rate"]
        R2["stream"] = DRAM_ROOF
        R2["RATES"] = TF.class_rates(R2["WG"], R2["roofs"], cor)
        f = defaultdict(float)
        for sig in TF.TOP:
            calls = R2["by"][sig]["calls"]
            J2 = TF.Join(R2, sig)
            for i in range(len(J2.ops)):
                nm = J2.ops[i]["name"]
                (_B, _Fm, _Fe, t_tr, t_ar, *_r) = TF.op_terms(R2, J2, i)
                f["priced" if nm in PRICED else
                  ("generic" if nm == GENERIC else "other")] += calls * max(t_tr, t_ar)
            f["other"] += calls * J2.unattributed / DRAM_ROOF
        gnat = out["native_join_s"]["at_1350"]
        devf = f["priced"] + gnat + f["other"]
        band[cor / 1e12] = {"P_floor": f["priced"], "G_floor": f["generic"], "G_native": gnat,
                            "U_floor": f["other"], "device_floor": devf,
                            "H_ceiling": FOLD_S - devf,
                            "P_ceiling_at_H0": FOLD_S - gnat - f["other"]}
        print("\n  rates scaled to %.2f TFLOP/s:" % (cor / 1e12))
        print("    P floor %.4f s / %.1f Mc | G native %.4f s (model floor %.4f) | U floor "
              "%.4f s" % (f["priced"], f["priced"] * MHZ, gnat, f["generic"], f["other"]))
        print("    device floor %.4f s -> H ceiling %.4f s / %.1f Mc ; P ceiling at H=0 is "
              "%.4f s" % (devf, FOLD_S - devf, (FOLD_S - devf) * MHZ,
                          FOLD_S - gnat - f["other"]))
    lo = min(b["P_floor"] for b in band.values())
    hi = max(b["P_ceiling_at_H0"] for b in band.values())
    print("\n  P (8 priced classes, in-fold) is in [%.4f, %.4f] s / [%.1f, %.1f] Mc." % (
        lo, hi, lo * MHZ, hi * MHZ))
    print("  The published %.4f s is ABOVE that band's top by %.4f s even with ZERO host time."
          % (sum(pub.values()), sum(pub.values()) - hi))
    print("  Byte-corrected: %.4f s on the published arms (below P's own floor, so the ratio "
          "over-corrects) and %.4f s on the bare arms (%.4f s above the H=0 ceiling, so the bare "
          "arm is refuted too). NEITHER replay arm is a consistent in-fold price."
          % (plo, phi, phi - hi))
    print("  H ceiling across the band: %.4f - %.4f s / %.1f - %.1f Mc, against F = 3.9830 "
          "+/- 0.1181 s." % (min(b["H_ceiling"] for b in band.values()),
                             max(b["H_ceiling"] for b in band.values()),
                             min(b["H_ceiling"] for b in band.values()) * MHZ,
                             max(b["H_ceiling"] for b in band.values()) * MHZ))
    out["band"] = {"P_byte_corrected_published_arms_s": plo,
                   "P_byte_corrected_bare_arms_s": phi,
                   "P_band_s": [lo, hi], "per_cube": band}
    (HERE / "genop_audit.json").write_text(json.dumps(out, indent=1, default=float))
    (HERE / "genop_audit.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
