#!/usr/bin/env python3
"""Re-derive the campaign's 2.34x from the real dispatched programs, term by term.

The campaign's central number is a subtraction between a measurement and a model:

    measured 41.4152 ms/PairformerLayer  -  (428 ops x 6.36 us  +  6.651 GB / 445 GB/s)

Every term on the right is an estimate. This file re-derives all three from the tt-metal device
profiler CSV that `b2z-grid-utilization` captured of a real `PairformerLayer` (274 dispatched
programs, shapes, dtypes, memory configs and per-op device kernel duration), so the op count, the
bytes and the per-op floor come from what the device actually ran rather than from what Python
called.

No device needed: this is arithmetic over a committed artifact. The CSV is not duplicated here;
it lives on the branch that took it:

    git show origin/wk/b2z-grid-utilization:perf/b2z_grid/out/ops_perf_results_2026_09_12_11_33_40.csv \
        > /tmp/ops.csv
    python3 perf/b2z_redteam/redteam_arith.py /tmp/ops.csv --generic-io drop-one \
        --out perf/b2z_redteam/out/redteam_pairformer_drop-one.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import json
import statistics as st
from pathlib import Path

DUR = "DEVICE KERNEL DURATION [ns]"
CODE = "OP CODE"
CORES, AVAIL = "CORE COUNT", "AVAILABLE WORKER CORE COUNT"

DTYPE_B = {"BFLOAT16": 2.0, "FLOAT32": 4.0, "UINT32": 4.0, "INT32": 4.0,
           "UINT16": 2.0, "UINT8": 1.0, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625}

T_FIXED_US = 6.36          # campaign's per-op device floor (flat end of the add curve, BH p300c)
BW_GBS = 445.0             # campaign's BW_eff (asymptote of the add curve, BH p300c)
BLOCK_MS_QB2 = 41.4152     # campaign's measured PairformerLayer, qb2 p300c trace replay
CAMPAIGN_OPS = 428
CAMPAIGN_GB = 6.651


def find_period(codes, repeats, lo=20):
    n = len(codes)
    for L in range(lo, n // repeats + 1):
        tail = codes[n - repeats * L:]
        if all(tail[i * L:(i + 1) * L] == tail[:L] for i in range(1, repeats)):
            return L
    raise SystemExit("no repeating period")


def dim(v):
    """The profiler writes a dimension as `padded[logical]`. Bytes move at the padded size."""
    m = re.match(r"\s*(\d+)", v or "")
    return int(m.group(1)) if m else None


def dims_of(r, kind, i):
    out = []
    for d in "WZYX":
        v = dim(r.get(f"{kind}_{i}_{d}_PAD[LOGICAL]", ""))
        if v is None:
            return None
        out.append(v)
    return out


def tensors(r, kind, nmax):
    """Yield (elems, dtype_bytes, memory, dims, dtype) for every present tensor of a row."""
    for i in range(nmax):
        dt = r.get(f"{kind}_{i}_DATATYPE", "")
        if not dt or dt == "-":
            continue
        dims = dims_of(r, kind, i)
        if dims is None:
            continue
        el = 1
        for d in dims:
            el *= d
        yield el, DTYPE_B.get(dt, 2.0), r.get(f"{kind}_{i}_MEMORY", ""), dims, dt


def matmul_flops(r):
    """2*B*M*N*K from in0 [.., M, K] and in1 [.., K, N]."""
    a, b = dims_of(r, "INPUT", 0), dims_of(r, "INPUT", 1)
    if a is None or b is None:
        return 0.0
    M, K = a[2], a[3]
    K2, N = b[2], b[3]
    if K != K2:
        return 0.0
    batch = max(a[0] * a[1], b[0] * b[1])
    return 2.0 * batch * M * N * K


def sdpa_flops(r):
    """q [B,H,S,D], k [B,H,S,D]: 2*B*H*S*S*D for QK^T and the same again for PV."""
    q, k = dims_of(r, "INPUT", 0), dims_of(r, "INPUT", 1)
    if q is None or k is None:
        return 0.0
    B, H, S, D = q
    Sk = k[2]
    return 4.0 * B * H * S * Sk * D


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--generic-io", choices=("drop-one", "drop-all"), default="drop-one",
                    help="ttnn.generic_op is handed io_tensors = inputs + outputs, so every "
                         "output also appears as an input. drop-one charges a second "
                         "same-shaped operand as a real read (upper byte bound); drop-all "
                         "treats every same-shaped operand as the preallocated output "
                         "(lower byte bound). The truth is between; the two arms are the "
                         "byte term's error bar.")
    ap.add_argument("--peak-tflops", type=float, default=85.96,
                    help="compute roof; the default is the HiFi4 8192-cube figure the campaign "
                         "prices against (perf/bioir_roofline/roofs_p300c_qb2_card2.json)")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv)) if r.get(DUR)]
    codes = [r[CODE] for r in rows]
    L = find_period(codes, a.repeats)
    n = len(rows)
    reps = [rows[n - a.repeats * L + i * L: n - a.repeats * L + (i + 1) * L]
            for i in range(a.repeats)]

    ops = []
    for j in range(L):
        rs = [rep[j] for rep in reps]
        r0 = rs[0]
        ns = st.median(float(x[DUR]) for x in rs)
        ins = list(tensors(r0, "INPUT", 24))
        outs = list(tensors(r0, "OUTPUT", 3))
        # ttnn hands a preallocated output in as an INPUT (in-place binaries, generic_op io
        # lists). Charging it as both a read and a write is the `published`-rule mistake one
        # scale down. Drop at most one INPUT per OUTPUT with identical dims/dtype/memory.
        drop_all = a.generic_io == "drop-all" and "Generic" in r0[CODE]
        for o in outs:
            while True:
                for k, t in enumerate(ins):
                    if (t[3], t[4], t[2]) == (o[3], o[4], o[2]):
                        ins.pop(k)
                        break
                else:
                    break
                if not drop_all:
                    break
        if "Slice" in r0[CODE]:
            # a slice reads exactly the region it writes, not the whole parent tensor
            ins = [(e, s_, m, d, dt) for e, s_, m, d, dt in outs]
        b_in_dram = sum(e * s for e, s, m, _, _ in ins if "DRAM" in m)
        b_in_l1 = sum(e * s for e, s, m, _, _ in ins if "DRAM" not in m)
        b_out_dram = sum(e * s for e, s, m, _, _ in outs if "DRAM" in m)
        b_out_l1 = sum(e * s for e, s, m, _, _ in outs if "DRAM" not in m)
        code = r0[CODE]
        if "Matmul" in code:
            fl = matmul_flops(r0)
        elif "SDPA" in code:
            fl = sdpa_flops(r0)
        else:
            fl = 0.0
        ops.append({
            "i": j, "code": code, "ns": ns,
            "cores": int(r0[CORES]), "avail": int(r0[AVAIL]),
            "dram_B": b_in_dram + b_out_dram, "l1_B": b_in_l1 + b_out_l1,
            "all_B": b_in_dram + b_out_dram + b_in_l1 + b_out_l1,
            "flops": fl,
            "in0": "x".join(str(d) for d in (ins[0][3] if ins else [])),
            "in0_dt": ins[0][4] if ins else "", "in0_mem": (ins[0][2][:24] if ins else ""),
            "fid": r0.get("MATH FIDELITY", ""),
            "op2op_ns": float(r0.get("OP TO OP LATENCY [ns]") or 0),
        })

    meas_ns = sum(o["ns"] for o in ops)
    gap_ns = sum(o["op2op_ns"] for o in ops)
    dram_B = sum(o["dram_B"] for o in ops)
    all_B = sum(o["all_B"] for o in ops)
    flops = sum(o["flops"] for o in ops)
    peak = a.peak_tflops * 1e12
    # measured 8192-cube ttnn.matmul roofs per fidelity, same file, same part
    FID_ROOF = {"LoFi": 150.34e12, "HiFi2": 121.78e12, "HiFi3": 100.0e12, "HiFi4": 85.96e12}
    for o in ops:
        o["fid_roof"] = FID_ROOF.get(o["fid"], peak)

    def model_add(o, nops_scale=1.0):
        return T_FIXED_US * 1e3 * nops_scale + o["dram_B"] / (BW_GBS * 1e9) * 1e9

    def model_max(o):
        return max(T_FIXED_US * 1e3, o["dram_B"] / (BW_GBS * 1e9) * 1e9)

    def model_max_c(o):
        return max(T_FIXED_US * 1e3, o["dram_B"] / (BW_GBS * 1e9) * 1e9,
                   o["flops"] / peak * 1e9)

    def model_max_cf(o):
        """Same, with the compute roof matched to the op's own recorded math fidelity."""
        return max(T_FIXED_US * 1e3, o["dram_B"] / (BW_GBS * 1e9) * 1e9,
                   o["flops"] / o["fid_roof"] * 1e9)

    res = {
        "source_csv": str(a.csv), "period_ops": L, "repeats": a.repeats,
        "arch_note": "pc card 0, p150a Blackhole, 13x10=130 cores, ttnn 0.72.0-dev Tracy build, "
                     "TT_BIO_TRIATT_PERSISTENT_MASK=0 (stock SDPA). Campaign cell is qb2 p300c "
                     "11x10=110 cores, ttnn 0.68.0, fused SDPA.",
        "measured_ms": meas_ns / 1e6,
        "campaign_measured_ms": BLOCK_MS_QB2,
        "op_count": {"device_programs": L, "campaign_python_calls": CAMPAIGN_OPS},
        "bytes": {"dram_GB": dram_B / 1e9, "dram_plus_l1_GB": all_B / 1e9,
                  "campaign_GB": CAMPAIGN_GB},
        "flops_GFLOP": flops / 1e9,
        "op_to_op_gap_ms": gap_ns / 1e6,
        "peak_tflops_used": a.peak_tflops,
        "models_ms": {
            "campaign_as_published": 1e-3 * (CAMPAIGN_OPS * T_FIXED_US
                                             + CAMPAIGN_GB * 1e9 / (BW_GBS * 1e9) * 1e6),
            "additive_real_opcount_real_bytes":
                1e-6 * (L * T_FIXED_US * 1e3 + dram_B / (BW_GBS * 1e9) * 1e9),
            "per_op_max": 1e-6 * sum(model_max(o) for o in ops),
            "per_op_max_with_compute": 1e-6 * sum(model_max_c(o) for o in ops),
            "per_op_additive": 1e-6 * sum(model_add(o) for o in ops),
            "per_op_max_compute_at_own_fidelity": 1e-6 * sum(model_max_cf(o) for o in ops),
            "campaign_opcount_fixed_only_real_bytes":
                1e-6 * (CAMPAIGN_OPS * T_FIXED_US * 1e3 + dram_B / (BW_GBS * 1e9) * 1e9),
            "real_opcount_campaign_bytes":
                1e-3 * (L * T_FIXED_US + CAMPAIGN_GB * 1e9 / (BW_GBS * 1e9) * 1e6),
        },
    }
    for k, v in res["models_ms"].items():
        res.setdefault("deficit_x", {})[k] = meas_ns / 1e6 / v

    byclass = collections.defaultdict(lambda: {"n": 0, "ns": 0.0, "dram_B": 0.0, "l1_B": 0.0,
                                               "flops": 0.0, "max_ns": 0.0, "maxc_ns": 0.0})
    for o in ops:
        b = byclass[o["code"]]
        b["n"] += 1
        for k in ("ns", "dram_B", "l1_B", "flops"):
            b[k] += o[k]
        b["max_ns"] += model_max(o)
        b["maxc_ns"] += model_max_c(o)
    res["by_class"] = {k: {**v, "ms": v["ns"] / 1e6,
                           "GBs_dram": v["dram_B"] / v["ns"] if v["ns"] else 0,
                           "TFLOPs": v["flops"] / v["ns"] / 1e3 if v["ns"] else 0,
                           "deficit_x_max": v["ns"] / v["max_ns"] if v["max_ns"] else 0,
                           "deficit_x_maxc": v["ns"] / v["maxc_ns"] if v["maxc_ns"] else 0}
                       for k, v in sorted(byclass.items(), key=lambda kv: -kv[1]["ns"])}
    res["ops"] = ops

    print(f"period {L} device programs/unit, {n} rows, measured {meas_ns/1e6:.4f} ms/unit")
    print(f"DRAM bytes {dram_B/1e9:.4f} GB   DRAM+L1 {all_B/1e9:.4f} GB   "
          f"campaign {CAMPAIGN_GB} GB")
    print(f"arithmetic {flops/1e9:.1f} GFLOP = {flops/peak*1e3:.3f} ms at "
          f"{a.peak_tflops} TFLOP/s peak")
    print("\nmodels (ms/unit) and the deficit each leaves:")
    for k, v in res["models_ms"].items():
        print(f"  {k:42s} {v:8.3f} ms   {res['deficit_x'][k]:5.3f}x")
    print(f"\n{'op class':34s} {'n':>4s} {'ms':>8s} {'GB/s':>8s} {'TFLOP/s':>9s} "
          f"{'x(max)':>7s} {'x(max+c)':>9s}")
    for k, v in res["by_class"].items():
        print(f"{k:34s} {v['n']:4d} {v['ms']:8.3f} {v['GBs_dram']:8.1f} {v['TFLOPs']:9.2f} "
              f"{v['deficit_x_max']:7.2f} {v['deficit_x_maxc']:9.2f}")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
