"""Top ops of a census_v2 profile, each against its roofline floor, and the pass_glue split by op code.

    python3 perf/rfd4/land10/top_ops.py <census dir> <out.json> [top]

Same signpost attribution as census_v2.py. Per design = (60 first passes + 120 recycled ones) / batch (the recycle
projections charged to the pass they feed; run.json's batch, designs a pass). Per-pass columns are the whole batched pass. Ops are keyed by (family, op code); bytes are every input and output tensor at its
logical shape and dtype (block-float formats at their packed size, a generic_op's io tensors once each, its returned one
included), which is the DRAM traffic if every operand
streams, so the floor `max(bytes / 436 GB/s, FLOPs / 140 TFLOP/s HiFi2)` is a lower bound even where an operand sits
in L1. FLOPs as census_v2 (2*M*K*N a matmul, 4*b*h*n*n*d an SDPA) plus the projections inside fused generic ops
(op_flops), 0 otherwise.
"""
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e2e"))
from census_v2 import flops  # noqa: E402

BW, PEAK = 436e9, 140e12
BYTES = {"FLOAT32": 4, "UINT32": 4, "INT32": 4, "BFLOAT16": 2, "UINT16": 2, "BFLOAT8_B": 1088 / 1024,
         "BFLOAT4_B": 576 / 1024, "UINT8": 1}
GLUE = ("pass_glue", "outer_residual", "pair_factors", "pair_relu")


def dims(r, k):
    d = [r.get(f"{k}_{a}_PAD[LOGICAL]", "") for a in "WZYX"]
    if not any(d):
        return None
    return [int((x.split("[")[-1].rstrip("]") if "[" in x else x) or 1) for x in d]


def tensors(r):
    # generic_op lists its io tensors as inputs, then its returned (last) io tensor again as one more input and as
    # OUTPUT_0: count that tensor once (outer_residual's io [z, pi, pj4, out] reads INPUT_0..4 + OUTPUT_0)
    gen = r["OP CODE"].startswith("GenericOp")
    ins = [k for k in (f"INPUT_{i}" for i in range(16)) if dims(r, k)]
    for k in (ins[:-1] if gen else ins + [f"OUTPUT_{i}" for i in range(4)]):
        d = dims(r, k)
        if d:
            n = d[0] * d[1] * d[2] * d[3]
            yield k, d, r.get(k + "_DATATYPE", "").replace("DataType::", ""), n


def nbytes(r):
    return sum(n * BYTES.get(dt, 2) for _, _, dt, n in tensors(r))


def op_flops(r):
    """census_v2's FLOPs, plus a fused pair generic_op's projections: 2 * rows * K * H for every [K, H] block-float
    weight it reads (RFD4's pair weights are bfp8; an activation operand such as the write's pi is bf16), rows = the
    pair rows of its first input. A bf16 weight would be missed and its floor read low."""
    if not r["OP CODE"].startswith("GenericOp"):
        return flops(r)
    ts = list(tensors(r))
    if not ts:
        return 0
    d0 = ts[0][1]
    rows = d0[0] * d0[1] * d0[2]
    return sum(2 * rows * d[2] * d[3] for _, d, dt, _ in ts[1:] if d[0] == d[1] == 1 and dt.endswith("_B"))


def shapes(r):
    return " ; ".join(f"{k.lower()} {'x'.join(map(str, d))} {dt}" for k, d, dt, _ in tensors(r))


def passes(prof):
    on, stack, out, pending = False, [], [], []
    for r in csv.DictReader(open(f"{prof}/ops.csv")):
        if r["OP TYPE"] == "signpost":
            c = r["OP CODE"]
            on = {"elt_prof": True, "elt_end": False}.get(c, on)
            if on and c.startswith("B|"):
                stack.append(c[2:])
                if c == "B|pass":
                    out.append(pending)
                    pending = []
            elif on and c.startswith("E|"):
                stack.pop()
            continue
        if on:
            fam = stack[-1] if stack else "outside"
            (out[-1] if "pass" in stack else pending).append(("pass_glue" if fam == "pass" else fam, r))
    return out[0], out[-1]


def table(first, rec, key, nb=1):
    t = collections.defaultdict(lambda: {"ms_design": 0.0, "ms_rec_pass": 0.0, "ops_rec_pass": 0, "bytes_rec_pass": 0.0,
                                         "flop_rec_pass": 0.0, "floor_ms_design": 0.0, "cores": set(), "shapes": {}})
    for w, ops in ((60 / nb, first), (120 / nb, rec)):
        for fam, r in ops:
            e = t[key(fam, r)]
            ms, b, f = float(r["DEVICE KERNEL DURATION [ns]"] or 0) / 1e6, nbytes(r), op_flops(r)
            e["ms_design"] += w * ms
            e["floor_ms_design"] += w * max(b / BW, f / PEAK) * 1e3
            if ops is rec:
                e["ms_rec_pass"] += ms
                e["ops_rec_pass"] += 1
                e["bytes_rec_pass"] += b
                e["flop_rec_pass"] += f
                e["cores"].add(r["CORE COUNT"])
                s = shapes(r)
                e["shapes"][s] = e["shapes"].get(s, 0.0) + ms
    rows = []
    for k, e in t.items():
        ms = e["ms_rec_pass"]
        rows.append({"family": k[0], "op": k[1], **({"shapes": k[2]} if len(k) > 2 else
                    {"main_shapes": max(e["shapes"], key=e["shapes"].get) if e["shapes"] else ""}),
                     "device_s_per_design": round(e["ms_design"] / 1e3, 3),
                     "floor_s_per_design": round(e["floor_ms_design"] / 1e3, 3),
                     "gap_s": round((e["ms_design"] - e["floor_ms_design"]) / 1e3, 3),
                     "x_floor": round(e["ms_design"] / e["floor_ms_design"], 2) if e["floor_ms_design"] else None,
                     "ms_per_recycled_pass": round(ms, 3), "ops_per_recycled_pass": e["ops_rec_pass"],
                     "mb_per_recycled_pass": round(e["bytes_rec_pass"] / 1e6, 1),
                     "gb_s": round(e["bytes_rec_pass"] / ms / 1e6, 1) if ms else None,
                     "tflop_s": round(e["flop_rec_pass"] / ms / 1e9, 2) if ms and e["flop_rec_pass"] else None,
                     "cores": sorted(e["cores"])})
    return sorted(rows, key=lambda x: -x["device_s_per_design"])


def main():
    prof, out = sys.argv[1], sys.argv[2]
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    first, rec = passes(prof)
    nb = json.load(open(f"{prof}/run.json")).get("batch", 1)
    ops = table(first, rec, lambda f, r: (f, r["OP CODE"]), nb)
    glue = table([x for x in first if x[0] in GLUE], [x for x in rec if x[0] in GLUE],
                 lambda f, r: (f, r["OP CODE"], shapes(r)), nb)
    doc = {"run": json.load(open(f"{prof}/run.json")), "roof": {"dram_gb_s": 436, "tflop_s_hifi2": 140},
           "device_s_per_design": round(sum(x["device_s_per_design"] for x in ops), 3),
           "top_ops": ops[:top], "all_ops": ops,
           "pass_glue_split": {"families": GLUE, "device_s_per_design": round(sum(x["device_s_per_design"] for x in glue), 3),
                               "ops": glue}}
    json.dump(doc, open(out, "w"), indent=1)
    for x in ops[:top]:
        print(f"{x['family']:16s} {x['op'][:40]:40s} {x['device_s_per_design']:7.3f} s  floor {x['floor_s_per_design']:6.3f}"
              f"  x{x['x_floor']}  {x['gb_s']} GB/s {x['tflop_s']} TF/s")
    print("pass_glue split", doc["pass_glue_split"]["device_s_per_design"], "s")
    for x in glue[:15]:
        print(f"  {x['family']:15s} {x['op'][:36]:36s} {x['device_s_per_design']:7.3f} s x{x['ops_per_recycled_pass']}"
              f"  floor {x['floor_s_per_design']:6.3f}  {x['gb_s']} GB/s  {x['shapes'][:110]}")


if __name__ == "__main__":
    main()
