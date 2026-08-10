#!/usr/bin/env python3
"""Roofline-attribute the per-op record from pf_block_ops.py into the 298 aa ledger.

One row per op INSTANCE CLASS: (op name, every input shape+dtype+buffer, output shape+dtype+buffer).
For each class: FLOPs and bytes derived from the PADDED shapes, arithmetic intensity, WHICH roof
binds, and the achieved fraction OF THAT ROOF. Never a bare "% of peak".

Roofs come from a JSON measured on the same card. Three DRAM/compute roofs plus an L1 op roof; a row
whose tensors are all L1-resident is scored against the L1 roof, not silently called compute-bound.

Occupancy: the device profiler's CORE COUNT column is unavailable on this wheel (no tracy capture
binary), so for matmul-family rows the script reports the tile blocking directly -- M/K/N in tiles --
and flags `k_tiles < cores`, the condition that made the tri-attention qkv projection collapse to one
k-tile per core and cost 2.17x. That is the sibling test the sprint asked for, computed from shapes
rather than read from a profiler.

    python3 perf/ledger_298/ledger_from_ops.py --ops ops_pv2_320.json --roofs roofs_c0.json \
        --l1-roof l1_roof_c0.json --calls-per-fold 480 --stage-ms 18676 --out ledger_pv2_320.json
"""
import argparse
import json
import math
from collections import defaultdict

DTYPE_BYTES = {"BFLOAT16": 2, "bfloat16": 2, "FLOAT32": 4, "float32": 4,
               "BFLOAT8_B": 1.0625, "bfloat8_b": 1.0625, "BFLOAT4_B": 0.5625, "bfloat4_b": 0.5625,
               "UINT32": 4, "uint32": 4, "INT32": 4, "int32": 4, "UINT16": 2, "uint16": 2,
               "UINT8": 1, "uint8": 1}
TILE = 32

MATMUL = ("linear", "matmul", "minimal_matmul")
ELEMENTWISE = {"add": 1, "add_": 1, "multiply": 1, "multiply_": 1, "subtract": 1, "relu": 1,
               "sigmoid": 1, "silu": 2, "gelu": 4, "reciprocal": 1, "typecast": 0}
NORMLIKE = {"layer_norm": 5, "rms_norm": 4, "softmax": 5}
MOVEMENT = ("permute", "transpose", "concat", "reshape", "unsqueeze", "squeeze", "chunk", "slice",
            "pad", "clone", "to_layout", "to_memory_config", "reallocate", "repeat", "sum")


def nbytes(t):
    return math.prod(t["shape"]) * DTYPE_BYTES.get(t["dtype"], 2)


def is_dram(t):
    return "DRAM" in (t.get("buf") or "").upper()


def mkn(rec):
    """(M, K, N, batch) in elements for a matmul-family call, or None."""
    ins = [t for t in rec["in"] if t is not None]
    if len(ins) < 2:
        return None
    a, b = ins[0]["shape"], ins[1]["shape"]
    m, k = a[-2], a[-1]
    if b[-2] == k:
        n = b[-1]
    elif b[-1] == k:                 # weight stored transposed
        n = b[-2]
    else:
        return None
    return m, k, n, max(math.prod(a[:-2]), 1)


def flops(rec):
    op = rec["op"]
    outs = [rec["out"]] if rec["out"] else []
    if op in MATMUL:
        g = mkn(rec)
        if not g:
            return 0, "matmul: K not matched between operands"
        m, k, n, batch = g
        return 2 * batch * m * k * n, f"2*{batch}*{m}*{k}*{n} (batch,M,K,N padded)"
    if op == "scaled_dot_product_attention":
        ins = [t for t in rec["in"] if t]
        if not ins:
            return 0, "sdpa: no tensor inputs"
        sh = ins[0]["shape"]
        # Leading dims are the batch; the last three are heads, seq, head_dim. Multiplying the
        # batch by sh[-4] as well squares it and inflated this row 320x on the first pass.
        b, h, s, d = max(math.prod(sh[:-3]), 1), sh[-3], sh[-2], sh[-1]
        return 4 * b * h * s * s * d, f"4*{b}*{h}*{s}^2*{d} (QK^T + PV, 2 FLOP/MAC)"
    if op in ELEMENTWISE:
        e = sum(math.prod(t["shape"]) for t in outs)
        return e * ELEMENTWISE[op], f"{ELEMENTWISE[op]} FLOP/out-elem"
    if op in NORMLIKE:
        e = sum(math.prod(t["shape"]) for t in outs)
        return e * NORMLIKE[op], f"~{NORMLIKE[op]} FLOP/out-elem (estimate)"
    if op in MOVEMENT:
        return 0, "data movement, no arithmetic"
    return 0, "unclassified: FLOPs not derived"


def classkey(rec):
    def k(t):
        return None if t is None else (tuple(t["shape"]), t["dtype"], t["buf"], t["layout"])
    return (rec["op"], rec["site"], tuple(k(t) for t in rec["in"]), k(rec["out"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", required=True)
    ap.add_argument("--roofs", required=True)
    ap.add_argument("--l1-roof", default=None)
    ap.add_argument("--l1-roof-GBs", type=float, default=None,
                    help="override the L1 roof; the binary leg of l1_roof.py is floored, use the "
                         "unary ladder's peak (the only leg whose time scales with size)")
    ap.add_argument("--read-roof-GBs", type=float, default=None,
                    help="override the read roof (roofs_card.py's 64 MB ladder under-reads it)")
    ap.add_argument("--calls-per-fold", type=float, required=True)
    ap.add_argument("--stage-ms", type=float, required=True, help="measured stage ms per fold")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    R = json.load(open(args.roofs))
    C_ROOF = R["compute_roof"]["peak_TFLOPs"] * 1e12
    RD_ROOF = (args.read_roof_GBs or R["dram_roofs"]["read_peak_GBs"]) * 1e9
    WR_ROOF = R["dram_roofs"]["write_peak_GBs"] * 1e9
    if args.l1_roof_GBs:
        L1_ROOF = args.l1_roof_GBs * 1e9
    elif args.l1_roof:
        L1_ROOF = json.load(open(args.l1_roof))["l1_op_roof_GBs"] * 1e9
    else:
        L1_ROOF = None

    D = json.load(open(args.ops))
    agg, meta = defaultdict(lambda: {"n": 0, "s": 0.0}), {}
    # Drop the rows whose re-run failed. They used to arrive as 0.0 s with the reason in `error`,
    # and summing them here dragged every class average down by however many of its calls the
    # instrument had lost -- silently, because the class still reported its full call count.
    dropped = [r for r in D["records"] if r.get("error") or r["s"] is None]
    for rec in D["records"]:
        if rec.get("error") or rec["s"] is None:
            continue
        key = classkey(rec)
        agg[key]["n"] += 1
        agg[key]["s"] += rec["s"]
        meta.setdefault(key, rec)
    if dropped:
        print(f"dropped {len(dropped)} of {len(D['records'])} records the harness could not "
              f"re-run; they are absent from every class below, not scored as free", flush=True)

    measured_block_s = D["block_wall_s"]
    scale = args.calls_per_fold
    rows = []
    for key, a in agg.items():
        rec = meta[key]
        fl, basis = flops(rec)
        ins = [t for t in rec["in"] if t]
        out = rec["out"]
        rd = sum(nbytes(t) for t in ins if is_dram(t))
        wr = nbytes(out) if out and is_dram(out) else 0
        l1 = sum(nbytes(t) for t in ins if not is_dram(t)) + (
            nbytes(out) if out and not is_dram(out) else 0)
        secs = a["s"] / a["n"]

        cands = [("COMPUTE", fl / C_ROOF), ("DRAM-READ", rd / RD_ROOF), ("DRAM-WRITE", wr / WR_ROOF)]
        if L1_ROOF and l1:
            cands.append(("L1", l1 / L1_ROOF))
        binding, t_ideal = max(cands, key=lambda x: x[1])
        frac = t_ideal / secs if secs > 0 else 0.0
        # No L1 roof has been established on this card (see l1_roof.py: the binary leg is pinned at
        # a fixed 22.2 us across a 128x size range, and rows in this very ledger beat the unary
        # ladder's peak by 56%). A row whose traffic is all L1 therefore gets no percentage --
        # its achieved GB/s is reported and it is ranked against its peers, not against a roof.
        l1_only = rd + wr == 0 and l1 > 0

        row = {"op": rec["op"], "site": rec["site"], "n_per_block": a["n"],
               "in": ins, "out": out,
               "us_per_call": round(secs * 1e6, 2),
               "block_ms": round(a["s"] * 1e3, 4),
               "pct_of_block": round(100 * a["s"] / measured_block_s, 2),
               "ms_per_fold": round(a["s"] * scale * 1e3, 2),
               "GFLOP_per_call": round(fl / 1e9, 5), "flop_basis": basis,
               "dram_read_MB": round(rd / 1e6, 3), "dram_write_MB": round(wr / 1e6, 3),
               "l1_MB": round(l1 / 1e6, 3),
               "AI_FLOP_per_dram_byte": round(fl / (rd + wr), 1) if rd + wr else None,
               "achieved_TFLOPs": round(fl / secs / 1e12, 2) if secs and fl else None,
               "achieved_dram_read_GBs": round(rd / secs / 1e9, 1) if secs and rd else None,
               "achieved_dram_write_GBs": round(wr / secs / 1e9, 1) if secs and wr else None,
               "achieved_l1_GBs": round(l1 / secs / 1e9, 1) if secs and l1 else None,
               "binding_roof": binding, "pct_of_binding_roof": round(100 * frac, 1)}

        if rec["op"] in MATMUL:
            g = mkn(rec)
            if g:
                m, k, n, batch = g
                row["tiles_MKN"] = [m // TILE, k // TILE, n // TILE]
                row["k_tiles_lt_cores"] = (k // TILE) < 110
        if fl == 0 and rd + wr + l1 < 1e5:
            row["verdict"] = "OVERHEAD"
        elif l1_only and binding == "L1":
            row["verdict"] = "L1-RESIDENT (no roof established)"
            row["pct_of_binding_roof"] = None
        elif frac >= 1.05:
            row["verdict"] = "ACCOUNTING-SUSPECT (>105% of its roof)"
        elif frac >= 0.70:
            row["verdict"] = {"COMPUTE": "COMPUTE-BOUND", "DRAM-READ": "READ-BOUND",
                              "DRAM-WRITE": "WRITE-BOUND", "L1": "L1-BOUND"}[binding]
        else:
            row["verdict"] = "LATENCY-or-OCCUPANCY-BOUND"
        row["gap_ms_per_fold"] = round(max(0.0, secs - t_ideal) * a["n"] * scale * 1e3, 2)
        rows.append(row)

    rows.sort(key=lambda r: -r["ms_per_fold"])
    summed_fold_ms = sum(r["ms_per_fold"] for r in rows)
    out = {"label": args.label, "ops_json": args.ops, "model": D["model"], "n": D["n"],
           "c_z": D["c_z"],
           "roofs": {"compute_TFLOPs": C_ROOF / 1e12, "dram_read_GBs": RD_ROOF / 1e9,
                     "dram_write_GBs": WR_ROOF / 1e9,
                     "l1_op_GBs": L1_ROOF / 1e9 if L1_ROOF else None,
                     "machine_balance_FLOP_per_byte": round(C_ROOF / RD_ROOF, 1)},
           "block_wall_ms": round(measured_block_s * 1e3, 3),
           "per_op_sum_ms": round(D["sum_s"] * 1e3, 3),
           "per_op_coverage_of_block_pct": round(100 * D["sum_s"] / measured_block_s, 1),
           "calls_per_fold": scale,
           "stage_ms_per_fold_measured": args.stage_ms,
           "stage_ms_per_fold_from_block_wall": round(measured_block_s * scale * 1e3, 1),
           "per_op_sum_ms_per_fold": round(summed_fold_ms, 1),
           "n_classes": len(rows), "n_calls_per_block": D["n_ops"], "rows": rows}
    json.dump(out, open(args.out, "w"), indent=1)

    print(f"{args.label} {D['model']} N={D['n']}  classes={len(rows)}  calls/block={D['n_ops']}")
    print(f"  block wall {out['block_wall_ms']} ms | per-op sum {out['per_op_sum_ms']} ms "
          f"({out['per_op_coverage_of_block_pct']}% of block)")
    print(f"  stage/fold: measured {args.stage_ms} ms | block_wall x {scale} = "
          f"{out['stage_ms_per_fold_from_block_wall']} ms | per-op sum = {summed_fold_ms:.1f} ms")
    print(f"{'op':22s} {'site':28s} {'n':>3s} {'us/call':>8s} {'ms/fold':>8s} {'AI':>7s} "
          f"{'roof':>10s} {'%roof':>6s} {'gap ms':>7s}  verdict")
    for r in rows[:args.top]:
        ai = r["AI_FLOP_per_dram_byte"]
        pr = r["pct_of_binding_roof"]
        print(f"{r['op'][:22]:22s} {r['site'][-28:]:28s} {r['n_per_block']:3d} "
              f"{r['us_per_call']:8.1f} {r['ms_per_fold']:8.1f} "
              f"{(f'{ai:.1f}' if ai else '-'):>7s} {r['binding_roof']:>10s} "
              f"{(f'{pr:.1f}' if pr is not None else '-'):>6s} {r['gap_ms_per_fold']:7.1f}  "
              f"{r['verdict']}")


if __name__ == "__main__":
    main()
