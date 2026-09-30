#!/usr/bin/env python3
"""Cut the profiled round's ops report into rounds and seams, and place each op family on the roof.

    prof_table.py <ops_perf_results.csv> [--warm 3,4] [--roof-gbs 442.3] [--json out.json]

Per op: DEVICE KERNEL DURATION off the card's own cycle counters, bytes as every DRAM-resident
input and output tensor at its padded shape and stored dtype, FLOPs for matmuls as 2*B*M*K*N.
Bytes are a FLOOR: an op that re-reads an operand (a tiled matmul, a chunked recompute) moves more
than one pass, and L1-resident tensors are counted as zero. So GB/s here is a lower bound on what
the op pulled, and "% of roof" can only understate how close to the DRAM roof an op sits.

Signposts: `bcp_round` at every round entry, `seam_begin:<module>:<phase>` / `seam_end:...`
around every device seam. An op outside every seam is charged to `outside`.
"""
import argparse
import collections
import csv
import json
import re
import sys

DT = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "UINT8": 1,
      "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024}


def dim(v):
    m = re.match(r"\s*(\d+)", v or "")
    return int(m.group(1)) if m else 0


def tensors(row):
    for io in ("INPUT", "OUTPUT"):
        for i in range(32):
            k = f"{io}_{i}_"
            if k + "DATATYPE" not in row or not row[k + "DATATYPE"]:
                if i > 0 or io == "OUTPUT":
                    break
                continue
            shape = [dim(row.get(k + f"{a}_PAD[LOGICAL]", "")) for a in "WZYX"]
            yield io, i, shape, row[k + "DATATYPE"], row.get(k + "MEMORY", "")


def family(code, row):
    c = code.lower()
    for key, fam in (("matmul", "matmul"), ("sdpa", "sdpa"), ("generic", "generic_op"),
                     ("softmax", "softmax"), ("layernorm", "layernorm"), ("moreh", "moreh"),
                     ("binary", "eltwise_binary"), ("unary", "eltwise_unary"),
                     ("typecast", "typecast"), ("permute", "permute"), ("transpose", "transpose"),
                     ("reduce", "reduce"), ("concat", "concat"), ("slice", "slice"),
                     ("pad", "pad"), ("copy", "copy"), ("clone", "copy"), ("reshape", "reshape"),
                     ("tilize", "tilize"), ("untilize", "tilize"), ("fill", "fill"),
                     ("where", "eltwise_ternary"), ("ternary", "eltwise_ternary"),
                     ("interleaved_to_sharded", "reshard"), ("sharded", "reshard")):
        if key in c:
            return fam
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--warm", default="3,4")
    ap.add_argument("--roof-gbs", type=float, default=442.3)
    ap.add_argument("--json")
    a = ap.parse_args()
    warm = {int(x) for x in a.warm.split(",")}
    rnd, seam = 0, "outside"
    agg = collections.defaultdict(lambda: collections.Counter())
    perround = collections.defaultdict(collections.Counter)
    for row in csv.DictReader(open(a.csv)):
        code, typ = row.get("OP CODE", ""), row.get("OP TYPE", "")
        if typ == "signpost":
            if code.startswith("bcp_round"):
                rnd += 1
            elif code.startswith("seam_begin:"):
                seam = code.split(":", 1)[1]
            elif code.startswith("seam_end:"):
                seam = "outside"
            continue
        if typ != "tt_dnn_device":
            continue
        k = float(row.get("DEVICE KERNEL DURATION [ns]") or 0) * 1e-9
        fw = float(row.get("DEVICE FW DURATION [ns]") or 0) * 1e-9
        o2o = float(row.get("OP TO OP LATENCY [ns]") or 0) * 1e-9
        by = 0.0
        shapes = {}
        for io, i, shape, dt, mem in tensors(row):
            n = shape[0] * shape[1] * shape[2] * shape[3]
            shapes[(io, i)] = shape
            if "DRAM" in mem:
                by += n * DT.get(dt, 2)
        fl = 0.0
        fam = family(code, row)
        if fam == "matmul" and ("INPUT", 0) in shapes and ("INPUT", 1) in shapes:
            s0, s1 = shapes[("INPUT", 0)], shapes[("INPUT", 1)]
            b = max(s0[0] * s0[1], s1[0] * s1[1])
            fl = 2.0 * b * s0[2] * s0[3] * s1[3]
        perround[rnd].update(kernel=k, fw=fw, o2o=o2o, calls=1, bytes=by, flops=fl)
        if rnd in warm:
            for key in (("fam", fam), ("seam", seam), ("seamfam", f"{seam}|{fam}")):
                agg[key].update(kernel=k, fw=fw, o2o=o2o, calls=1, bytes=by, flops=fl)
    nw = len(warm)
    out = {"rounds": {r: dict(c) for r, c in sorted(perround.items())}, "warm": sorted(warm)}
    for r, c in sorted(perround.items()):
        print(f"round {r}: calls {c['calls']:.0f} kernel {c['kernel']:.3f} s fw {c['fw']:.3f} s "
              f"op2op {c['o2o']:.3f} s bytes {c['bytes']/1e9:.1f} GB flops {c['flops']/1e12:.2f} T")
    tot = collections.Counter()
    for (kind, name), c in agg.items():
        if kind == "fam":
            tot.update(c)
    for kind in ("fam", "seam", "seamfam"):
        rows = sorted(((n, c) for (k, n), c in agg.items() if k == kind),
                      key=lambda x: -x[1]["kernel"])
        print(f"\n== per warm round, by {kind} (n={nw}) ==")
        print(f"{'name':44s} {'calls':>7s} {'kern s':>7s} {'share':>6s} {'GB':>7s} {'GB/s':>6s} "
              f"{'%roof':>6s} {'TFLOP':>6s} {'op2op s':>7s}")
        tab = []
        for n, c in rows[:40]:
            ks = c["kernel"] / nw
            gbs = c["bytes"] / c["kernel"] / 1e9 if c["kernel"] else 0
            tab.append({"name": n, "calls": c["calls"] / nw, "kernel_s": ks,
                        "share": ks / (tot["kernel"] / nw), "gb": c["bytes"] / nw / 1e9,
                        "gbs": gbs, "pct_roof": gbs / a.roof_gbs, "tflop": c["flops"] / nw / 1e12,
                        "o2o_s": c["o2o"] / nw})
            t = tab[-1]
            print(f"{n[:44]:44s} {t['calls']:7.0f} {ks:7.3f} {t['share']*100:5.1f}% {t['gb']:7.1f} "
                  f"{gbs:6.1f} {t['pct_roof']*100:5.1f}% {t['tflop']:6.2f} {t['o2o_s']:7.3f}")
        out[kind] = tab
    t = {k: v / nw for k, v in tot.items()}
    print(f"\nwarm round: {t['calls']:.0f} calls, kernel {t['kernel']:.3f} s, fw {t['fw']:.3f} s, "
          f"op2op {t['o2o']:.3f} s, {t['bytes']/1e9:.1f} GB (floor {t['bytes']/1e9/a.roof_gbs:.3f} s "
          f"at {a.roof_gbs} GB/s), {t['flops']/1e12:.2f} TFLOP matmul")
    out["total"] = t
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
