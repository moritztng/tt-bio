"""Card-free bfp8_b inventory of the Evoformer pair track from a bc2_evo_devprof census.

For every census row, the bytes held by pair-sized BFLOAT16 tensors (>= 224*224*4 elements) are
the bytes a bfp8_b pack format could halve (1088 of 2048 bytes per tile). Two prices per row:
  prop  time scales with bytes at the row's own measured rate (generous: assumes the op is bound
        by its bytes), floored at its FLOPs at 45 TFLOP/s;
  roof  the removed bytes at the 438 GB/s DRAM roof (what deleting them buys an op that moves
        them at full speed).
Times are census synced-per-call minus 30 us ("corr"); the caller maps buckets to device time.
"""
import json, math, re, sys
from collections import defaultdict

BFP8 = 1088 / 2048
PEAK, BW, CORR = 45e12, 438e9, 30e-6
TENSOR = re.compile(r"\[([0-9, ]+)\]:([A-Z0-9_]+):")


def tensors(text):
    for m in TENSOR.finditer(text or ""):
        dims, dt = m.groups()
        n = math.prod(int(x) for x in dims.split(","))
        yield n, dt, bool(INNER.fullmatch(m.group(0)[:m.group(0).index("]") + 1]))


#: Intermediates that live inside one sublayer: the trimul in-projection (512 wide), its
#: channel-major a/b and contraction, triangle-attention heads, qkv and bias logits. The pair
#: residual [.., 224, 224, 128] and its cotangent are excluded in scope "inner".
INNER = re.compile(r"\[(?:1, )?224, 224, 512\]|\[1, 128, 224, 224\]|\[224, 4, 224, 32\]|\[1, 4, 224, 224\]|\[224, 1, 224, 384\]|\[224, 128, 224\]")


#: Fraction of a byte cut that turned into time, measured on Blackhole p300 @1350, as (low, high):
#:   mm      ttnn matmul family: in-projection bfp8 out 0.43 at 512 (perf/bfp8_qkv/prod.json),
#:           0.92 at 224 (k128 raw/mm_bound_fbd79f252: 0.281 -> 0.180 ms for a 0.61 byte ratio)
#:   mmk     tt-bio's own matmul-class kernels (triatt qkv): 0.05-0.15 (prod.json, prodclean.json)
#:   stream  streaming element-wise kernels: 0.93-0.98 (c14-bfp8-fastpath, quoted in qkvladder.py)
#:   move    channel moves / transposes at 40-47 % of the DRAM roof: unmeasured, bracketed 0.3-0.95
REALIZE = {"mm": (0.43, 0.92), "mmk": (0.05, 0.15), "stream": (0.93, 0.98), "move": (0.3, 0.95)}
MM = ("matmul", "linear", "experimental.minimal_matmul")
MOVE = ("permute", "transpose", "experimental.nlp_concat_heads", "concat", "to_layout")


def kind(o):
    n, sig, out = o["name"], o["sig"], o["out"]
    if n in MM:
        return "mm"
    if n in MOVE:
        return "move"
    if n != "generic_op":
        return "stream"
    if "[224, 4, 224, 32]" in sig or "[128, 512]" in sig or "[128, 1024]" in sig:
        return "mmk"      # triangle attention qkv / sdpa / backward, pair transition
    if "512]" in sig or "[1, 128, 224, 224]" in sig or sig.count("[224, 224, 128]") == 2 and sig.count("[") == 2:
        return "move"     # gated channel move (+bw), permute_back (+vjp), pair_transpose
    return "stream"       # rne_add(_masked), lnbw, gate_bw, msa_add, lead_sum


def main(census, phases=("evo.taped", "evo.backward"), scope="all"):
    ops = json.load(open(census))["census"]["ops"]
    rows = []
    for o in ops:
        if o["phase"] not in phases:
            continue
        t = o["synced_s"] / o["calls"] - CORR
        if t <= 0:
            continue
        seen = list(tensors(o["sig"])) + list(tensors(o["out"]))
        pair = 224 * 224 * 4
        if scope == "brief" and o["name"] == "layer_norm":   # LN output feeds a projection
            seen = list(tensors(o["sig"])) + [(n, dt, True) for n, dt, _ in tensors(o["out"])]
        elig = sum(2 * n for n, dt, inner in seen
                   if dt == "BFLOAT16" and n >= pair and (scope == "all" or inner))
        tot = sum(n * (2 if dt == "BFLOAT16" else 4 if dt == "FLOAT32" else 1) for n, dt, _ in seen)
        if not tot or not elig:
            continue
        saved_b = elig * (1 - BFP8)
        floor = o.get("flops", 0) / PEAK
        prop = min(t * saved_b / tot, max(0.0, t - floor))
        roof = min(saved_b / BW, prop)
        lo, hi = REALIZE[kind(o)]
        rows.append(dict(phase=o["phase"], name=o["name"], kind=kind(o), lo_s=prop * lo * o["calls"],
                         hi_s=prop * hi * o["calls"], saved_gb=saved_b * o["calls"] / 1e9, calls=o["calls"], ms=t * 1e3,
                         s_it=t * o["calls"], elig_mb=elig / 1e6, tot_mb=tot / 1e6,
                         prop_s=prop * o["calls"], roof_s=roof * o["calls"],
                         sig=o["sig"][:110], out=o["out"]))
    fam = lambda n: "generic_op" if n == "generic_op" else (
        "matmul" if n in ("matmul", "linear", "experimental.minimal_matmul") else "other")
    agg = defaultdict(lambda: [0.0] * 6)
    for r in rows:
        for key in (fam(r["name"]), "kind:" + r["kind"], "total"):
            a = agg[key]
            for i, f in enumerate(("s_it", "prop_s", "roof_s", "lo_s", "hi_s", "saved_gb")):
                a[i] += r[f]
    names = ("corr_s_it", "prop_s", "roof_s", "realized_lo_s", "realized_hi_s", "saved_gb_it")
    return rows, {k: dict(zip(names, (round(x, 4) for x in v))) for k, v in sorted(agg.items())}


if __name__ == "__main__":
    scope = sys.argv[3] if len(sys.argv) > 3 else "all"
    rows, agg = main(sys.argv[1], scope=scope)
    rows.sort(key=lambda r: -r["prop_s"])
    print("| prop s/it | roof s/it | corr s/it | calls | ms | elig/tot MB | phase | kind | op | sig |")
    for r in rows[:40]:
        print(f"| {r['prop_s']:.4f} | {r['roof_s']:.4f} | {r['s_it']:.4f} | {r['calls']} | {r['ms']:.3f} | "
              f"{r['elig_mb']:.0f}/{r['tot_mb']:.0f} | {r['phase']} | {r['kind']} | {r['name']} | {r['sig']} -> {r['out']} |")
    print(json.dumps(agg, indent=1))
    if len(sys.argv) > 2 and sys.argv[2] != "-":
        json.dump(dict(rows=rows, buckets=agg), open(sys.argv[2], "w"), indent=1)
