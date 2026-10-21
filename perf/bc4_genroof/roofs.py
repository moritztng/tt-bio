#!/usr/bin/env python3
"""generic_op families of a devprof record, each priced at max(DRAM bytes / BW, matmul FLOPs / RATE).

    python3 -I perf/bc4_genroof/roofs.py <u2-devprof.json.gz> [--json out.json]

u2's names.py credits generic_op with 0 FLOPs and counts bytes as `operands + result` of the
`ttnn.generic_op` call. generic_op returns its LAST io tensor, which is also an operand, so every span
counts that tensor twice; and an operand a kernel reads only a slice of (the gated moves' [.., 4C]
projection) or reads once per core (triangle attention's head bias) is counted at its logical size.
This prices each span signature (issuing site, span bytes) from the kernel's own shapes at the BC2
224-token cell (N = 224, c_z = 128, 4 heads x 32, c_t = 64): `MODEL` below, one line per signature,
and the run fails on any signature it does not know.
"""
import argparse
import bisect
import collections
import gzip
import json

N, C, CT, H, D = 224, 128, 64, 4, 32
NN = N * N
P = NN * C * 2             # one [N, N, 128] bf16 pair tensor
PT = NN * CT * 2           # one [N, N, 64] bf16 template pair tensor
MASK = NN * 2
mm = lambda m, k, n: 2 * m * k * n    # noqa: E731
QKV = N * H * N * D * 2    # triatt_bw.byte_census: one of q, k, v, dO
GROUPS, CORES = 27, 108    # triatt_bw.plan on the 11x10 BH grid: groups = 110 // H, cores = groups * H
TRIATT_BW = 7 * QKV + CORES * NN * 2 + GROUPS * H * NN * 4

# (site, span bytes) -> (what, true DRAM bytes, matmul FLOPs)
MODEL = {
    ("trimul_tail.py:861", 25974784): ("gin, 64-ch chunk: x read, a, b 64 ch written", 25974784 - MASK, mm(NN, C, 4 * 64)),
    ("trimul_tail.py:861", 38901760): ("gin, whole: x read, a, b 128 ch written", 38901760 - MASK, mm(NN, C, 4 * C)),
    ("trimul_tail.py:861", 19519488): ("gin, template c 64", 19519488 - MASK, mm(NN, CT, 4 * CT)),
    ("triatt_bw.py:555", 210739200): ("q k v dO + bias per core in, dq dk dv + f32 partial out; S, dP, dV, dQ, dK", TRIATT_BW,
                                      5 * mm(N * H * N, D, N)),
    ("pair_swiglu_ws.py:305", 38838784): ("pair_relu fwd: LN, 128->512 relu ->128", 38838784 - P, mm(NN, C, 4 * C) * 2),
    ("pair_swiglu_ws.py:305", 51806720): ("pair_relu_bw: 3 matmuls", 51806720 - P, mm(NN, C, 4 * C) * 3),
    ("pair_swiglu_ws.py:305", 19312896): ("pair_relu fwd, template 64->128->64", 19312896 - PT, mm(NN, CT, 2 * CT) * 2),
    ("pair_swiglu_ws.py:305", 25747712): ("pair_relu_bw, template", 25747712 - PT, mm(NN, CT, 2 * CT) * 3),
    ("reblock_permute.py:1269", 166985728): ("gated move bw: da + p, g slices in, dp, dg slices out", 5 * P, 0),
    ("reblock_permute.py:1269", 83492864): ("gated move bw, template", 5 * PT, 0),
    ("sdpa_generic.py:985", 64626688): ("triangle attention fwd: q k v bias in, o out; QK^T, PV", 64626688 - P,
                                        2 * mm(N * H * N, D, N)),
    ("mm_generic.py:508", 64323584): ("triangle attention qkv projection 128 -> 3 x 128", 64323584 - P, mm(NN, C, 3 * C)),
    ("pair_transpose.py:150", 3 * P): ("pair transpose", 2 * P, 0),
    ("pair_transpose.py:150", 3 * PT): ("pair transpose, template", 2 * PT, 0),
    ("reblock_permute.py:1049", 77070336): ("gated move fwd: p, g slices in, out", 3 * P, 0),
    ("reblock_permute.py:1049", 38535168): ("gated move fwd, template", 3 * PT, 0),
    ("gate_bw.py:150", 6 * P): ("gate bw: d, o, g in, 2 out", 5 * P, 0),
    ("gate_bw.py:150", 1376256): ("gate bw, [N, 1024]", 1376256 * 5 // 6, 0),
    ("gate_bw.py:150", 6 * PT): ("gate bw, template", 5 * PT, 0),
    ("trimul_tail.py:1413", 64365056): ("T2CM tail: LN, x @ wp, sigmoid(g @ wg), mask, + z", 64365056 - P, 2 * mm(NN, C, C)),
    ("reblock_permute.py:358", 3 * P): ("channel move", 2 * P, 0),
    ("reblock_permute.py:358", 3 * PT): ("channel move, template", 2 * PT, 0),
    ("lead_sum.py:126", 23281664): ("dbias partial sum, f32", GROUPS * H * NN * 4 + H * NN * 4, 0),
    ("permute_back_pk.py:87", 3 * P): ("channel move back", 2 * P, 0),
    ("permute_back_pk.py:87", 3 * PT): ("channel move back, template", 2 * PT, 0),
}



def model(site, b):
    """MODEL, plus the two families whose every shape has the same form: rne_add (a + b [+ small mask] -> out, out
    the size of a) and lnbw (x, dy [+ gamma] -> dx). Both count out twice, so the true bytes are 3/4 of the
    span's large tensors."""
    if (site, b) in MODEL:
        return MODEL[(site, b)]
    if site == "rne_add.py:357":
        return ("a + b -> out (any shape)", b * 3 // 4, 0)
    if site == "lnbw.py:156":
        return ("LayerNorm bw: x, dy in, dx out (any shape)", b * 3 // 4, 0)
    return None


ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--json")
ap.add_argument("--bw", type=float, default=391e9)
ap.add_argument("--rate", type=float, default=45e12)
ap.add_argument("--p150a", type=float, default=0.80)
a = ap.parse_args()
doc = json.load(gzip.open(a.path, "rt"))
Ccol = {c: i for i, c in enumerate(doc["program_cols"])}
spans = sorted(tuple(s) for s in doc["spans"])
starts = [s[0] for s in spans]


def span_of(rid):
    i = bisect.bisect_right(starts, rid >> 10) - 1
    return i if i >= 0 and spans[i][0] <= rid >> 10 < spans[i][1] else None


its = sorted({p[Ccol["iteration"]] for p in doc["programs"]})
K = len(its)
sig = collections.defaultdict(lambda: dict(ns=0, programs=0, spans=set()))
for p in doc["programs"]:
    i = span_of(p[Ccol["runtime_id"]])
    if i is None or spans[i][2] != "generic_op":
        continue
    s = spans[i]
    site = ":".join(s[3].split(":")[:2])
    g = sig[(site, s[5])]
    g["ns"] += p[Ccol["k_ns"]] or 0
    g["programs"] += 1
    g["spans"].add(i)

missing = [k for k in sig if model(*k) is None]
assert not missing, f"unmodelled generic_op signatures: {missing}"
fam = collections.defaultdict(lambda: dict(device_s=0.0, calls=0.0, bytes=0.0, flops=0.0, u2_bytes=0.0, rows=[]))
for (site, b), g in sorted(sig.items()):
    what, tb, fl = model(site, b)
    calls = g["programs"] / K           # one program per call (generic_op is one program)
    f = fam[site]
    f["device_s"] += g["ns"] / 1e9 / K
    f["calls"] += calls
    f["bytes"] += tb * calls
    f["u2_bytes"] += b * calls
    f["flops"] += fl * calls
    f["rows"].append(dict(what=what, calls=calls, ms_per_call=g["ns"] / 1e6 / g["programs"],
                          bytes_mb=tb / 1e6, gflop=fl / 1e9,
                          byte_ms=tb / a.bw * 1e3, flop_ms=fl / a.rate * 1e3))
out = []
for site, f in fam.items():
    tb, tf = f["bytes"] / a.bw, f["flops"] / a.rate
    roof = max(tb, tf)
    out.append(dict(family=site, calls=f["calls"], device_s=f["device_s"], byte_s=tb, flop_s=tf, roof_s=roof,
                    binds="FLOPs" if tf > tb else "bytes", above_s=f["device_s"] - roof,
                    above_p150a=a.p150a * (f["device_s"] - roof), u2_roof_s=f["u2_bytes"] / a.bw,
                    tflops=f["flops"] / f["device_s"] / 1e12 if f["device_s"] else 0, gbs=f["bytes"] / f["device_s"] / 1e9,
                    rows=f["rows"]))
out.sort(key=lambda r: -r["above_s"])
print(f"{doc['sha'][:9]} chip {doc['chip']} iterations {its}; BW {a.bw / 1e9:.0f} GB/s, {a.rate / 1e12:.0f} TFLOP/s, p150a x{a.p150a}")
print(f"{'family':32} {'calls':>5} {'dev s':>7} {'u2 roof':>7} {'bytes s':>7} {'FLOP s':>7} {'roof':>7} {'binds':>5} "
      f"{'above':>7} {'p150a':>7} {'TF/s':>5} {'GB/s':>5}")
for r in out:
    print(f"{r['family']:32} {r['calls']:5.0f} {r['device_s']:7.4f} {r['u2_roof_s']:7.4f} {r['byte_s']:7.4f} "
          f"{r['flop_s']:7.4f} {r['roof_s']:7.4f} {r['binds']:>5} {r['above_s']:+7.4f} {r['above_p150a']:+7.4f} "
          f"{r['tflops']:5.1f} {r['gbs']:5.0f}")
    for w in r["rows"]:
        print(f"    {w['calls']:5.0f} x {w['ms_per_call']:.4f} ms  bytes {w['byte_ms']:.4f} ms  FLOPs {w['flop_ms']:.4f} ms  {w['what']}")
tot = lambda k: sum(r[k] for r in out)  # noqa: E731
print(f"{'total':32} {tot('calls'):5.0f} {tot('device_s'):7.4f} {tot('u2_roof_s'):7.4f} {'':7} {'':7} {tot('roof_s'):7.4f} "
      f"{'':5} {tot('above_s'):+7.4f} {tot('above_p150a'):+7.4f}")
if a.json:
    json.dump(dict(source=a.path, sha=doc["sha"], chip=doc["chip"], iterations=its, bw=a.bw, rate=a.rate,
                   families=out), open(a.json, "w"), indent=1)
