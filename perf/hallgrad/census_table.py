#!/usr/bin/env python3
"""Turn a census.py JSON into the per-op table for one pair unit. No device.

Per signature (autograd verb, ttnn op, phase, argument shapes): unit count = calls at K=2 minus
calls at K=1; unit time = unit count x median replay time of that signature. Roof time per op is
its own class's bound at its own bytes and FLOPs, from roofs measured in the same process:

  matmul         max(FLOPs / HiFi4 4096^3 matmul rate, bytes / best stream rate)
  data-movement  bytes / clone rate at the pair size     (one stream in, one out)
  layernorm      bytes / clone rate at the pair size
  softmax        bytes / clone rate at the pair size
  reduction      bytes / clone rate at the pair size
  elementwise    bytes / add rate at the pair size        (two streams in, one out)
"""
import collections
import json
import statistics
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from census import CLASS  # noqa: E402


def b_device(blob):
    """The unit's device-side slope: median fwd and bwd differences between K=2 and K=1.

    The whole-step paired difference also carries the host loss seed, which is outside the
    unit and, on a loaded host, is where the noise lives.
    """
    t = blob["timed"]
    m = statistics.median
    return (m(t["fwd_K2"]) - m(t["fwd_K1"])) + (m(t["bwd_K2"]) - m(t["bwd_K1"]))


def family(r):
    s = r["sig"]
    if r["class"] == "matmul" and r["origin"] == "triangle_attention":
        return "tri-attention batched matmuls (QK^T, PV and their grads)"
    if r["class"] == "matmul" and s.startswith("[256, 256,"):
        return "3D pair-tensor linears (fwd + dX)"
    if r["class"] == "matmul":
        return "trimul per-channel matmuls"
    if r["op"] == "reshape" and ("4,32" in s or "4, 32" in s):
        return "heads reshape [N,N,128] <-> [N,N,4,32]"
    if r["op"] == "permute":
        return "permutes"
    return f"all other {r['class']}"


def key(c):
    return (c["origin"], c["name"], c["phase"], c["sig"])


def analyze(blob, b_ref=None):
    R = blob["roofs"]
    F = R["matmul_hifi4_4096cube"]["tflops"] * 1e12
    BW_best = max(v["gbs"] for v in R.values() if v["gbs"]) * 1e9
    BW_copy = R["clone_pair"]["gbs"] * 1e9
    BW_elt = R["add_pair"]["gbs"] * 1e9
    k1 = collections.defaultdict(list)
    k2 = collections.defaultdict(list)
    for c in blob["records"]["1"]["calls"]:
        k1[key(c)].append(c)
    for c in blob["records"]["2"]["calls"]:
        k2[key(c)].append(c)
    rows, anomalies = [], []
    for k in set(k1) | set(k2):
        cnt = len(k2.get(k, [])) - len(k1.get(k, []))
        if cnt == 0:
            continue
        pool = k2.get(k, []) + k1.get(k, [])
        if cnt < 0:
            anomalies.append((k, cnt))
            continue
        c0 = pool[0]
        rep = [c["replay_s"] for c in pool if c["replay_s"] is not None]
        syn = [c["synced_s"] for c in pool]
        t_call = statistics.median(rep) if rep else statistics.median(syn)
        cls = CLASS.get(c0["name"], "elementwise")
        by, fl = c0["bytes"], c0["flops"]
        if cls == "matmul":
            roof = max(fl / F, by / BW_best)
        elif cls == "elementwise":
            roof = by / BW_elt
        elif cls == "dealloc":
            roof = 0.0
        else:
            roof = by / BW_copy
        rows.append({
            "origin": k[0], "op": k[1], "phase": k[2], "sig": k[3], "out": c0["out"],
            "class": cls, "count": cnt, "t_call": t_call,
            "t_synced_call": statistics.median(syn),
            "time": cnt * t_call, "synced_time": cnt * statistics.median(syn),
            "roof_call": roof, "roof_time": cnt * roof,
            "pct_roof": (roof / t_call) if t_call else None,
            "bytes": by, "flops": fl, "replayed": bool(rep),
        })
    rows.sort(key=lambda r: -r["time"])
    b = b_ref if b_ref is not None else b_device(blob)
    ops_sum = sum(r["time"] for r in rows)
    synced_sum = sum(r["synced_time"] for r in rows)
    return {"rows": rows, "anomalies": anomalies, "b": b, "ops_sum": ops_sum,
            "synced_sum": synced_sum, "overhead": b - ops_sum,
            "roof_sum": sum(r["roof_time"] for r in rows),
            "F": F, "BW_best": BW_best, "BW_copy": BW_copy, "BW_elt": BW_elt}


def fmt_shape(sig):
    return sig.replace(":bfloat16", ":bf16").replace(":float32", ":fp32").replace(":DRAM", "")


def main():
    blob = json.load(open(sys.argv[1]))
    cover = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
    a = analyze(blob)
    b = a["b"]
    print(f"b = {b:.5f} s; sum of per-op replay = {a['ops_sum']:.5f} s ({a['ops_sum'] / b:.1%}); "
          f"sum of per-op roofs = {a['roof_sum']:.5f} s; ops >= 200 us per call = "
          f"{sum(r['time'] for r in a['rows'] if r['t_call'] >= 200e-6) / b:.1%} of b\n")
    print("| # | op (verb:ttnn op, phase) | shape | dtype | calls | time ms | share of b | cum | class | "
          "its own roof, us/call | % of its roof |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    cum = 0.0
    for i, r in enumerate(a["rows"], 1):
        cum += r["time"]
        dt = "fp32" if "FLOAT32" in r["sig"] else "bf16"
        shape = fmt_shape(r["sig"]).replace("BFLOAT16", "bf16").replace("FLOAT32", "fp32").replace("|", "; ")
        print(f"| {i} | {r['origin']}:{r['op']} {r['phase']} | `{shape}` | {dt} | {r['count']} | "
              f"{r['time'] * 1e3:.2f} ({r['t_call'] * 1e6:.0f} us/call) | {r['time'] / b:.1%} | {cum / b:.1%} | "
              f"{r['class']} | {r['roof_call'] * 1e6:.1f} | {(r['pct_roof'] or 0):.1%} |")
        if cum / b >= cover:
            break
    rest = a["rows"][i:]
    print(f"| | {len(rest)} further signatures | | | {sum(r['count'] for r in rest)} | "
          f"{sum(r['time'] for r in rest) * 1e3:.2f} | {sum(r['time'] for r in rest) / b:.1%} | | | | |")
    print(f"| | **OVERHEAD** = b - sum(per-op) | | | | {a['overhead'] * 1e3:.2f} | "
          f"**{a['overhead'] / b:.1%}** | | overhead | | |")
    for title, fn in (("class", lambda r: r["class"]), ("family", family)):
        agg = collections.defaultdict(lambda: [0.0, 0.0, 0])
        for r in a["rows"]:
            x = agg[fn(r)]
            x[0] += r["time"]
            x[1] += r["roof_time"]
            x[2] += r["count"]
        print(f"\n| {title} | calls | time ms | share of b | roof ms | % of roof | brought to roof saves ms | of b |")
        print("|---|---|---|---|---|---|---|---|")
        for c, (t, rt, n) in sorted(agg.items(), key=lambda x: -x[1][0]):
            print(f"| {c} | {n} | {t * 1e3:.2f} | {t / b:.1%} | {rt * 1e3:.2f} | {rt / t if t else 0:.1%} | "
                  f"{(t - rt) * 1e3:.2f} | {(t - rt) / b:.1%} |")


if __name__ == "__main__":
    main()
