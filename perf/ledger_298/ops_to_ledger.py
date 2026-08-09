#!/usr/bin/env python3
"""Turn a tt-metal device-profiler ops report into a roofline-attributed op ledger.

One row per op INSTANCE CLASS: (op code, math fidelity, every input/output shape+dtype+memory).
For each class the script derives FLOPs and DRAM traffic from the shapes in the report, computes
arithmetic intensity, names WHICH roof binds, and reports the achieved fraction OF THAT ROOF --
never a bare "% of peak".

Roofs are read from a JSON produced by perf/ledger_298/roofs_card.py on the SAME card. Roofs are
per-card and this script refuses to guess them.

Shapes come from `INPUT_i_{W,Z,Y,X}_PAD[LOGICAL]`, whose format is `padded[logical]`; the padded
figure is what the hardware moves and computes, so the padded figure is what is used. Bytes are
attributed to DRAM or L1 by each tensor's `INPUT_i_MEMORY` / `OUTPUT_i_MEMORY` field.

FLOP derivations are explicit per op family (see FLOPS below) and every class carries a
`flop_basis` string so a reader can check the arithmetic rather than trust it.

    python3 perf/ledger_298/ops_to_ledger.py --csv OUT/.../ops_perf_results_*.csv \
        --roofs perf/ledger_298/roofs_c0.json --calls-per-fold 480 --out ledger.json
"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict

DTYPE_BYTES = {"BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625,
               "UINT32": 4, "INT32": 4, "UINT16": 2, "UINT8": 1}

# Op families -> how many FLOPs one output element costs, when the op is not a matmul.
# These are the honest small numbers; an op with a 1-5 FLOP/element budget is on the memory side of
# the roofline by three orders of magnitude and the exact constant never changes the verdict.
ELEMENTWISE = {"BinaryNgDeviceOperation": 1, "UnaryDeviceOperation": 1, "EltwiseUnary": 1,
               "BinaryDeviceOperation": 1, "TernaryNgDeviceOperation": 2}
NORMLIKE = {"LayerNorm": 5, "RMSNorm": 4, "Softmax": 5}
MOVEMENT = ("Permute", "Concat", "Reshape", "Slice", "Transpose", "Clone", "Copy", "Tilize",
            "Untilize", "Pad", "Reduction", "ShardedToInterleaved", "InterleavedToSharded",
            "NlpCreateHeads", "NlpConcatHeads", "Embedding", "Fill", "Repeat", "Split", "Chunk")


def pad_int(v):
    """`padded[logical]` -> padded int. Blank -> None."""
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    m = re.match(r"(\d+)", v)
    return int(m.group(1)) if m else None


def tensors(row, kind, nmax):
    """[(dims tuple, dtype, memory)] for INPUT_* or OUTPUT_* slots that are present."""
    out = []
    for i in range(nmax):
        dims = [pad_int(row.get(f"{kind}_{i}_{d}_PAD[LOGICAL]")) for d in ("W", "Z", "Y", "X")]
        if any(d is None for d in dims):
            continue
        out.append((tuple(dims), (row.get(f"{kind}_{i}_DATATYPE") or "").strip(),
                    (row.get(f"{kind}_{i}_MEMORY") or "").strip()))
    return out


def nbytes(t):
    dims, dt, _ = t
    return math.prod(dims) * DTYPE_BYTES.get(dt, 2)


def is_dram(t):
    return "DRAM" in t[2].upper()


def flops(opcode, ins, outs):
    """(FLOPs, basis string). Returns (0, ...) for pure data movement."""
    oc = opcode
    if "Matmul" in oc or "matmul" in oc or "Linear" in oc:
        if len(ins) < 2:
            return 0, "matmul with <2 inputs: not derivable"
        (aw, az, ay, ax), (bw, bz, by, bx) = ins[0][0], ins[1][0]
        m, k, n = ay, ax, bx
        if by != k:                       # in1 given transposed or as a 2D weight
            k, n = (by, bx) if bw * bz * by == k else (k, bx)
        batch = max(aw * az, 1)
        return 2 * batch * m * k * n, f"2*{batch}*{m}*{k}*{n} (M,K,N padded)"
    if "SDPA" in oc or "ScaledDotProduct" in oc:
        if not ins:
            return 0, "sdpa: no inputs"
        b, h, s, d = ins[0][0]
        return 4 * b * h * s * s * d, f"4*{b}*{h}*{s}^2*{d} (QK^T + PV, 2 FLOP/MAC)"
    fam = next((k for k in ELEMENTWISE if k in oc), None)
    if fam:
        return sum(math.prod(t[0]) for t in outs) * ELEMENTWISE[fam], f"{ELEMENTWISE[fam]} FLOP/out-elem"
    fam = next((k for k in NORMLIKE if k in oc), None)
    if fam:
        return sum(math.prod(t[0]) for t in outs) * NORMLIKE[fam], f"~{NORMLIKE[fam]} FLOP/out-elem (estimate)"
    if any(k in oc for k in MOVEMENT):
        return 0, "data movement, no arithmetic"
    return 0, "unclassified: FLOPs not derived"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--roofs", required=True)
    ap.add_argument("--calls-per-fold", type=float, default=1.0,
                    help="how many times this profiled region runs in one fold")
    ap.add_argument("--regions", type=int, default=1,
                    help="how many copies of the region are inside the CSV (--reps of the target)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    R = json.load(open(args.roofs))
    C_ROOF = R["compute_roof"]["peak_TFLOPs"] * 1e12
    RD_ROOF = R["dram_roofs"]["read_peak_GBs"] * 1e9
    WR_ROOF = R["dram_roofs"]["write_peak_GBs"] * 1e9

    rows = list(csv.DictReader(open(args.csv)))
    rows = [r for r in rows if (r.get("DEVICE KERNEL DURATION [ns]") or "").strip().isdigit()]

    agg = defaultdict(lambda: {"n": 0, "ns": 0, "cores": 0, "trisc1": 0, "brisc": 0, "ncrisc": 0})
    meta = {}
    for r in rows:
        oc = (r.get("OP CODE") or "").strip()
        ins, outs = tensors(r, "INPUT", 10), tensors(r, "OUTPUT", 3)
        key = (oc, (r.get("MATH FIDELITY") or "").strip(), tuple(ins), tuple(outs))
        a = agg[key]
        a["n"] += 1
        a["ns"] += int(r["DEVICE KERNEL DURATION [ns]"])
        for col, k in (("CORE COUNT", "cores"), ("DEVICE TRISC1 KERNEL DURATION [ns]", "trisc1"),
                       ("DEVICE BRISC KERNEL DURATION [ns]", "brisc"),
                       ("DEVICE NCRISC KERNEL DURATION [ns]", "ncrisc")):
            v = (r.get(col) or "").strip()
            if v.isdigit():
                a[k] += int(v)
        meta.setdefault(key, (ins, outs))

    total_ns = sum(a["ns"] for a in agg.values())
    ledger = []
    for key, a in agg.items():
        oc, fid, _, _ = key
        ins, outs = meta[key]
        fl, basis = flops(oc, list(ins), list(outs))
        rd = sum(nbytes(t) for t in ins if is_dram(t))
        wr = sum(nbytes(t) for t in outs if is_dram(t))
        l1 = sum(nbytes(t) for t in ins if not is_dram(t)) + sum(nbytes(t) for t in outs if not is_dram(t))
        secs = a["ns"] / a["n"] / 1e9                      # per call
        t_c = fl / C_ROOF
        t_r = rd / RD_ROOF
        t_w = wr / WR_ROOF
        binding, t_ideal = max((("COMPUTE", t_c), ("READ", t_r), ("WRITE", t_w)), key=lambda x: x[1])
        frac = t_ideal / secs if secs > 0 else 0.0
        if fl == 0 and rd + wr < 1e5:
            verdict = "OVERHEAD"
        elif frac >= 0.70:
            verdict = {"COMPUTE": "COMPUTE-BOUND", "READ": "READ-BOUND", "WRITE": "WRITE-BOUND"}[binding]
        else:
            verdict = "LATENCY-or-OCCUPANCY-BOUND"
        per_fold_ms = a["ns"] / 1e6 / max(args.regions, 1) * args.calls_per_fold
        gap_ms = max(0.0, (secs - t_ideal)) * a["n"] / max(args.regions, 1) * args.calls_per_fold * 1e3
        ledger.append({
            "op": oc, "fidelity": fid, "n_in_region": a["n"],
            "in": [{"shape": list(t[0]), "dtype": t[1], "mem": t[2]} for t in ins],
            "out": [{"shape": list(t[0]), "dtype": t[1], "mem": t[2]} for t in outs],
            "us_per_call": round(secs * 1e6, 2),
            "region_ms": round(a["ns"] / 1e6 / max(args.regions, 1), 3),
            "pct_of_region": round(100 * a["ns"] / total_ns, 2) if total_ns else 0.0,
            "per_fold_ms": round(per_fold_ms, 2),
            "cores_mean": round(a["cores"] / a["n"], 1),
            "trisc1_frac": round(a["trisc1"] / a["ns"], 3) if a["ns"] else None,
            "GFLOP_per_call": round(fl / 1e9, 4), "flop_basis": basis,
            "dram_read_MB": round(rd / 1e6, 3), "dram_write_MB": round(wr / 1e6, 3),
            "l1_MB": round(l1 / 1e6, 3),
            "arith_intensity_FLOP_per_dram_byte": round(fl / (rd + wr), 1) if rd + wr else None,
            "achieved_TFLOPs": round(fl / secs / 1e12, 2) if secs else None,
            "achieved_read_GBs": round(rd / secs / 1e9, 1) if secs else None,
            "achieved_write_GBs": round(wr / secs / 1e9, 1) if secs else None,
            "binding_roof": binding, "pct_of_binding_roof": round(100 * frac, 1),
            "verdict": verdict, "gap_ms_per_fold": round(gap_ms, 2),
        })

    ledger.sort(key=lambda r: -r["per_fold_ms"])
    out = {"label": args.label, "csv": args.csv,
           "roofs": {"compute_TFLOPs": C_ROOF / 1e12, "read_GBs": RD_ROOF / 1e9, "write_GBs": WR_ROOF / 1e9,
                     "machine_balance_FLOP_per_byte": round(C_ROOF / RD_ROOF, 1)},
           "calls_per_fold": args.calls_per_fold, "regions_in_csv": args.regions,
           "region_total_ms": round(total_ns / 1e6 / max(args.regions, 1), 3),
           "region_total_ms_per_fold": round(total_ns / 1e6 / max(args.regions, 1) * args.calls_per_fold, 2),
           "n_classes": len(ledger), "rows": ledger}
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"{args.label}  classes={len(ledger)}  region={out['region_total_ms']} ms  "
          f"per fold={out['region_total_ms_per_fold']} ms", flush=True)
    print(f"{'op':34s} {'n':>4s} {'us/call':>8s} {'ms/fold':>8s} {'cores':>6s} {'AI':>7s} "
          f"{'roof':>7s} {'%roof':>6s} {'gap ms':>7s}  verdict")
    for r in ledger[:args.top]:
        ai = r["arith_intensity_FLOP_per_dram_byte"]
        print(f"{r['op'][:34]:34s} {r['n_in_region']:4d} {r['us_per_call']:8.1f} {r['per_fold_ms']:8.1f} "
              f"{r['cores_mean']:6.1f} {(f'{ai:.1f}' if ai else '-'):>7s} {r['binding_roof']:>7s} "
              f"{r['pct_of_binding_roof']:6.1f} {r['gap_ms_per_fold']:7.1f}  {r['verdict']}")


if __name__ == "__main__":
    main()
