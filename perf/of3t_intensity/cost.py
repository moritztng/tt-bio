#!/usr/bin/env python3
"""FLOP and bytes per step part, as arithmetic over `census.py`'s shapes.

Every rule is in COST below and every rule is visible. That is the point: the campaign burned
18 passes on a headroom figure whose denominator nobody could inspect, so this file makes the
denominator a table you can read and argue with rather than a number in a state doc.

THREE THINGS THE OUTPUT SEPARATES, because collapsing them is how the earlier mistake happened.

  FLOP     matmul FLOP is exact from the shapes: an [M,K]x[K,N] is 2*M*N*K and there is no
           modelling in it. Non-matmul FLOP carries a per-element multiplicity I assign (a
           softmax is 5, a layernorm 8), so it is reported SEPARATELY and every conclusion
           below is checked against the matmul-only column too. If a part's verdict changed
           when the assigned constants moved, the verdict would not be worth having.

  BYTES    operands in plus results out, at PADDED volume, split by where the tensor lives.
           Only DRAM bytes are the bandwidth roof's denominator; L1 operands never cross it
           and host operands cross PCIe instead. This is the program's MINIMUM traffic: a
           matmul that re-reads a blocked operand moves more. So the arithmetic intensity
           here is an UPPER BOUND, which is the safe direction -- a part that is below the
           machine balance at its best-case AI is below it, full stop.

  CALLS    how many issue no arithmetic at all. A permute, a to_layout, a slice and a
           deallocate have zero FLOP by construction, so their %-of-compute-peak is not small,
           it is undefined, and a part made of them cannot be ranked on a roofline at all.

    cost.py --census out/census_384.counts.json --out out/cost_384.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ESIZE = {"BFLOAT16": 2.0, "FLOAT32": 4.0, "UINT32": 4.0, "INT32": 4.0, "UINT16": 2.0,
         "UINT8": 1.0, "BFLOAT8_B": 17.0 / 16.0, "BFLOAT4_B": 9.0 / 16.0, "?": 2.0}

# --- the cost rules -------------------------------------------------------------------------
# name -> (class, per-element multiplicity). MATMUL is 2*out_volume*K and takes no constant.
MATMUL = re.compile(r"(^|\.)(matmul|linear|bmm|baddbmm|mm_split|minimal_matmul|"
                    r"group_attn_matmul|attn_matmul|ssm_.*matmul)$")
REDUCE = re.compile(r"(^|\.)(sum|mean|max|min|prod|std|var|logsumexp|argmax|argmin|"
                    r"cumsum|topk|moreh_sum|moreh_mean)$")
SOFTMAX = re.compile(r"(^|\.)(softmax|log_softmax|scale_mask_softmax|.*_softmax|"
                     r"softmax_in_place|scale_causal_mask_hw_dims_softmax)$")
NORM = re.compile(r"(^|\.)(layer_norm|layernorm|rms_norm|rmsnorm|group_norm|batch_norm|"
                  r"moreh_layer_norm|normalize_.*)$")
SDPA = re.compile(r"(^|\.)(scaled_dot_product_attention.*|sdpa.*|flash.*attention.*)$")
# Zero-FLOP by construction: these move, reshape, allocate or free bytes and compute nothing.
MOVE = re.compile(r"(^|\.)(permute|transpose|reshape|view|slice|concat|cat|pad|unpad|tilize|"
                  r"untilize|tilize_with_.*|to_layout|typecast|clone|copy|assign|"
                  r"to_memory_config|from_torch|to_torch|to_device|from_device|deallocate|"
                  r"reallocate|repeat|repeat_interleave|expand|broadcast_to|split|chunk|"
                  r"squeeze|unsqueeze|fill|zeros.*|ones.*|empty.*|arange|full|embedding|"
                  r"gather|index_select|nonzero|sharded_to_interleaved|interleaved_to_sharded|"
                  r"reshard|all_gather|reduce_scatter|line_all_gather|mesh_.*|"
                  r"as_tensor|allocate_tensor_on_device|copy_host_to_device_tensor)$")
# Everything else that touches a tensor is elementwise: one unit of arithmetic per output
# element. Transcendentals are one SFPU op per element on this machine, so they are not
# weighted up; if they were, the non-matmul column would grow and the matmul column -- which
# every verdict is also checked against -- would not move.
ELT_MULT = 1.0
SOFTMAX_MULT = 5.0        # max, subtract, exp, sum, divide
NORM_MULT = 8.0           # mean, centre, square, mean, rsqrt, scale, gamma, beta


def parse_sig(sig: str):
    ins, _, outs = sig.partition(" -> ")
    f = lambda part: [t for t in (_one(x) for x in part.split(";")) if t]
    return f(ins), f(outs)


def _one(x: str):
    if not x:
        return None
    try:
        shp, pad, dt, bt = x.split("|")
    except ValueError:
        return None
    vol = lambda s: math.prod(int(d) for d in s.split("x")) if s else 1
    return {"vol": vol(shp), "pad": vol(pad), "dtype": dt, "buf": bt,
            "dims": [int(d) for d in shp.split("x")] if shp else []}


# Verbs that move no data at all: they free an address, read a flag or drain a queue. Counting
# their operand bytes as traffic put `deallocate` at the top of the trunk's byte table and
# `is_tensor_storage_on_device` fourth, which is nonsense -- neither touches a byte. They stay
# in the CALL count, because a call that costs time while moving nothing is precisely the
# shape of this backward's problem.
META = re.compile(r"(^|\.)(deallocate|reallocate|is_tensor_storage_on_device|memory_config|"
                  r"get_memory_view|synchronize_device|dump_.*|buffer_type|storage_type|"
                  r"allocate_tensor_on_device|get_.*|set_.*|.*_config|is_.*|has_.*)$")


def classify(verb: str) -> str:
    if META.search(verb):
        return "meta"
    if MATMUL.search(verb):
        return "matmul"
    if SDPA.search(verb):
        return "sdpa"
    if SOFTMAX.search(verb):
        return "softmax"
    if NORM.search(verb):
        return "norm"
    if REDUCE.search(verb):
        return "reduce"
    if MOVE.search(verb):
        return "move"
    return "eltwise"


def flops(kind: str, ins, outs) -> float:
    if not outs:
        return 0.0
    ov = sum(o["vol"] for o in outs)
    if kind == "matmul":
        # K is the contracted dim: the last dim of the first operand that has one.
        k = next((i["dims"][-1] for i in ins if len(i["dims"]) >= 2), 0)
        return 2.0 * outs[0]["vol"] * k
    if kind == "sdpa":
        # Q[...,S,D] K[...,T,D] V[...,T,D]: scores 2*S*T*D, softmax 5*S*T, out 2*S*T*D.
        d = ins[0]["dims"][-1] if ins and ins[0]["dims"] else 0
        s = ins[0]["vol"] // max(d, 1) if d else 0
        t = ins[1]["dims"][-2] if len(ins) > 1 and len(ins[1]["dims"]) >= 2 else 0
        return (4.0 * d + 5.0) * s * t
    if kind == "softmax":
        return SOFTMAX_MULT * ov
    if kind == "norm":
        return NORM_MULT * ov
    if kind == "reduce":
        return float(sum(i["vol"] for i in ins))
    if kind in ("move", "meta"):
        return 0.0
    return ELT_MULT * ov


def byts(ins, outs):
    """Padded bytes in and out, per buffer type. DRAM is the roof's denominator."""
    b = defaultdict(float)
    for t in list(ins) + list(outs):
        b[t["buf"]] += t["pad"] * ESIZE.get(t["dtype"], 2.0)
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=18)
    a = ap.parse_args()

    rows = json.loads(a.census.read_text())
    parts: dict = defaultdict(lambda: {
        "calls": 0, "flop": 0.0, "flop_matmul": 0.0, "dram_b": 0.0, "l1_b": 0.0,
        "host_b": 0.0, "calls_zero_flop": 0, "by_kind": defaultdict(
            lambda: {"calls": 0, "flop": 0.0, "dram_b": 0.0}),
        "by_verb": defaultdict(lambda: {"calls": 0, "flop": 0.0, "dram_b": 0.0})})

    for r in rows:
        ins, outs = parse_sig(r["sig"])
        kind = classify(r["verb"])
        f = flops(kind, ins, outs) * r["n"]
        b = {} if kind == "meta" else byts(ins, outs)
        p = parts[r["part"]]
        p["calls"] += r["n"]
        p["flop"] += f
        if kind in ("matmul", "sdpa"):
            p["flop_matmul"] += f
        p["dram_b"] += b.get("DRAM", 0.0) * r["n"]
        p["l1_b"] += b.get("L1", 0.0) * r["n"]
        p["host_b"] += b.get("HOST", 0.0) * r["n"]
        if f == 0.0:
            p["calls_zero_flop"] += r["n"]
        for d, key in ((p["by_kind"], kind), (p["by_verb"], r["verb"])):
            d[key]["calls"] += r["n"]
            d[key]["flop"] += f
            d[key]["dram_b"] += b.get("DRAM", 0.0) * r["n"]

    out = {"census": str(a.census), "rules": {
        "matmul_flop": "2 * out_volume * K, exact from the shapes",
        "softmax_per_element": SOFTMAX_MULT, "norm_per_element": NORM_MULT,
        "eltwise_per_element": ELT_MULT,
        "bytes": "operands in + results out at PADDED volume, split by buffer type; the "
                 "program's MINIMUM traffic, so the AI below is an upper bound",
        "zero_flop": "permute/slice/concat/to_layout/typecast/deallocate and friends compute "
                     "nothing by construction, so they have no compute roof to be a "
                     "fraction of",
        "meta": "deallocate, is_tensor_storage_on_device and friends move NO bytes; they are "
                "counted as calls and as zero traffic"},
        "parts": {}}
    for name, p in parts.items():
        dram = p["dram_b"]
        out["parts"][name] = {
            "calls": p["calls"],
            "calls_zero_flop": p["calls_zero_flop"],
            "zero_flop_call_share": round(p["calls_zero_flop"] / max(p["calls"], 1), 4),
            "gflop": round(p["flop"] / 1e9, 3),
            "gflop_matmul_only": round(p["flop_matmul"] / 1e9, 3),
            "dram_gb": round(dram / 1e9, 3),
            "l1_gb": round(p["l1_b"] / 1e9, 3),
            "pcie_gb": round(p["host_b"] / 1e9, 3),
            "ai_flop_per_dram_byte": round(p["flop"] / dram, 3) if dram else None,
            "ai_matmul_only": round(p["flop_matmul"] / dram, 3) if dram else None,
            "by_kind": {k: {"calls": v["calls"], "gflop": round(v["flop"] / 1e9, 3),
                            "dram_gb": round(v["dram_b"] / 1e9, 3)}
                        for k, v in sorted(p["by_kind"].items(),
                                           key=lambda kv: -kv[1]["calls"])},
            "top_verbs_by_calls": [
                {"verb": k, **{kk: (round(vv / 1e9, 3) if kk != "calls" else vv)
                               for kk, vv in v.items()}}
                for k, v in sorted(p["by_verb"].items(), key=lambda kv: -kv[1]["calls"])[:a.top]],
            "top_verbs_by_dram": [
                {"verb": k, "calls": v["calls"], "dram_gb": round(v["dram_b"] / 1e9, 3),
                 "gflop": round(v["flop"] / 1e9, 3)}
                for k, v in sorted(p["by_verb"].items(),
                                   key=lambda kv: -kv[1]["dram_b"])[:a.top]],
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    for name, p in out["parts"].items():
        print(f"{name:24s} calls {p['calls']:8d}  zeroFLOP {p['zero_flop_call_share']*100:5.1f}%  "
              f"{p['gflop']:10.1f} GFLOP ({p['gflop_matmul_only']:9.1f} mm)  "
              f"DRAM {p['dram_gb']:9.2f} GB  AI {p['ai_flop_per_dram_byte']}")
    print("WROTE", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
