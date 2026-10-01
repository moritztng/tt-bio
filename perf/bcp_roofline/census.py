#!/usr/bin/env python3
"""Bytes, FLOPs and kernel seconds per op family for the profiled round, against the roofs.

    census.py <cpp_device_perf_report.csv.gz> <tracy_ops_data.csv> [--rounds 4] [--json out.json]

Joins the card's own kernel durations (cpp report) to each op's tensors (Tracy's op data) on
GLOBAL CALL COUNT. Bytes are every DRAM-resident input and output at its padded shape and stored
dtype: one pass each, so a FLOOR on what the op moved (a tiled matmul re-reads, a chunked recompute
re-reads; L1 tensors count zero). FLOPs: matmul only, 2*B*M*K*N from the two operand shapes. The
profiled run is 4 rounds of identical device programs, so per-round = total / rounds.
"""
import argparse
import collections
import csv
import gzip
import json
import re

DT = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "UINT8": 1,
      "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024}


def ops(path):
    buf, head = [], None
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("`TT_DNN_DEVICE_OP:"):
                head, buf = line, []
                continue
            if head is None:
                continue
            if line.startswith("}`"):
                buf.append("}")
                try:
                    yield json.loads("".join(buf))
                except json.JSONDecodeError:
                    pass
                head = None
                continue
            buf.append(line)


def padded(sh):
    out = []
    for a in "WZYX":
        m = re.match(r"\s*(\d+)", str(sh.get(a, "1")))
        out.append(int(m.group(1)) if m else 1)
    return out


def tensor_info(t):
    sh = padded(t.get("shape", {}))
    st = t.get("storage_type", {})
    buf = st.get("memory_config", {}).get("buffer_type", "") if isinstance(st, dict) else ""
    n = sh[0] * sh[1] * sh[2] * sh[3]
    return sh, n * DT.get(t.get("dtype", ""), 2), buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpp")
    ap.add_argument("opsdata")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--roof-gbs", type=float, default=442.3)
    ap.add_argument("--json")
    a = ap.parse_args()
    meta = {}
    for o in ops(a.opsdata):
        ins = [tensor_info(t) for t in o.get("input_tensors", []) if isinstance(t, dict)]
        outs = [tensor_info(t) for t in o.get("output_tensors", []) if isinstance(t, dict)]
        dram = sum(b for _, b, buf in ins + outs if buf == "DRAM")
        code = o.get("op_code", "")
        ks = o.get("kernel_info", {}).get("compute_kernels", [])
        src = ks[0].get("source", "") if ks else ""
        # Two operand lists overstate the traffic: a slice reads only the window it returns, and
        # rne_add is handed four bf16 views but reads 2 B + 2 B and writes 2 B an element
        # (bcx-p10-rneker's 6 B/element design). Both are counted at what the kernel touches.
        if outs and (code == "SliceDeviceOperation" or "rne_add" in src):
            per = 2 if code == "SliceDeviceOperation" else 3
            dram = per * sum(b for _, b, _buf in outs)
        l1 = sum(b for _, b, buf in ins + outs if buf == "L1")
        fl = 0.0
        if "Matmul" in code and len(ins) >= 2:
            s0, s1 = ins[0][0], ins[1][0]
            fl = 2.0 * max(s0[0] * s0[1], s1[0] * s1[1]) * s0[2] * s0[3] * s1[3]
        name = code
        if code == "GenericOpDeviceOperation":
            name = "Generic:" + (src.rsplit("/", 1)[-1] or "?")
        if code == "BinaryNgDeviceOperation":
            name = "BinaryNg:" + o.get("attributes", {}).get("binary_op_type", "").split("::")[-1]
        meta[int(o.get("global_call_count", -1))] = (name, dram, l1, fl)
    fam = collections.defaultdict(collections.Counter)
    miss = 0
    for r in csv.DictReader(gzip.open(a.cpp, "rt")):
        g = int(r["GLOBAL CALL COUNT"])
        m = meta.get(g)
        if m is None:
            miss += 1
            m = (r["OP NAME"] + "?", 0, 0, 0)
        name, dram, l1, fl = m
        k = float(r["DEVICE KERNEL DURATION [ns]"] or 0) * 1e-9
        fam[name].update(kernel=k, calls=1, dram=dram, l1=l1, flops=fl)
    R = a.rounds
    tot = collections.Counter()
    for c in fam.values():
        tot.update(c)
    rows = []
    for name, c in sorted(fam.items(), key=lambda x: -x[1]["kernel"]):
        ks, gb = c["kernel"] / R, c["dram"] / R / 1e9
        rows.append({"name": name, "calls": c["calls"] / R, "kernel_s": ks, "dram_gb": gb,
                     "l1_gb": c["l1"] / R / 1e9, "tflop": c["flops"] / R / 1e12,
                     "gbs": gb / ks if ks else 0, "floor_s": gb / a.roof_gbs})
    print(f"unmatched cpp rows: {miss}")
    print(f"{'family':42s} {'calls':>7s} {'kern s':>7s} {'share':>6s} {'DRAM GB':>8s} {'GB/s':>6s} "
          f"{'%roof':>6s} {'floor s':>7s} {'x floor':>7s} {'TFLOP':>6s}")
    for t in rows[:45]:
        print(f"{t['name'][:42]:42s} {t['calls']:7.0f} {t['kernel_s']:7.3f} "
              f"{t['kernel_s'] / (tot['kernel'] / R) * 100:5.1f}% {t['dram_gb']:8.1f} {t['gbs']:6.1f} "
              f"{t['gbs'] / a.roof_gbs * 100:5.1f}% {t['floor_s']:7.3f} "
              f"{(t['kernel_s'] / t['floor_s']) if t['floor_s'] else 0:7.2f} {t['tflop']:6.2f}")
    T = {k: v / R for k, v in tot.items()}
    print(f"\nround: {T['calls']:.0f} calls, kernel {T['kernel']:.3f} s, DRAM {T['dram'] / 1e9:.1f} GB "
          f"(floor {T['dram'] / 1e9 / a.roof_gbs:.3f} s at {a.roof_gbs} GB/s), L1 {T['l1'] / 1e9:.1f} GB, "
          f"matmul {T['flops'] / 1e12:.2f} TFLOP")
    if a.json:
        json.dump({"rows": rows, "total": T, "unmatched": miss}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
