#!/usr/bin/env python3
"""Turn the ttnn graph captures into a replayable op-instance manifest.

Reads perf/b2x-baseline-attrib/captures/, groups every ``*DeviceOperation`` dispatch by
(op, operand specs, op params), and writes bench_manifest.json: one record per distinct
instance with everything ``op_replay.py`` needs to rebuild and call it standalone.

calls/fold = (occurrences inside the captured unit) x (that unit's calls per fold).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_manifest import CAPS, key_of, walk  # noqa: E402

# The two captures that carry the fold. PairformerLayer runs 264 times (256 trunk + 8
# standalone, ATTRIBUTION.md); the diffusion step runs 200 times; MSALayer 16.
UNITS = {
    "PairformerLayer__1x512x384,1x512x512x128": ("PairformerLayer", 264),
    "Diffusion__1x7168x3,1": ("DiffusionStep", 200),
    "MSALayer__1x512x512x128,1x1024x512x64": ("MSALayer", 16),
}

DT = {"BFLOAT16": "bfloat16", "FLOAT32": "float32", "BFLOAT8_B": "bfloat8_b",
      "UINT32": "uint32", "INT32": "int32", "UINT16": "uint16", "BFLOAT4_B": "bfloat4_b"}

CKC_RE = re.compile(
    r"ComputeKernelConfig\(math_fidelity=(\w+),math_approx_mode=(\d),fp32_dest_acc_en=(\d),"
    r"packer_l1_acc=(\d),dst_full_sync_en=(\d)")
BIN_RE = re.compile(r"BinaryOpType::(\w+)")
ACT_RE = re.compile(r"UnaryOpType::(\w+)")
TRANSPOSE_RE = re.compile(r"TransposeOpDim::(\w+)")
PERMUTE_RE = re.compile(r"SmallVector\(\[([\d,]*)\]\)")
SLICE_RE = re.compile(r"SliceParams\(Shape\(\[([\d,]*)\]\),Shape\(\[([\d,]*)\]\),Shape\(\[([\d,]*)\]\)")
RESHAPE_RE = re.compile(r"ReshapeViewParams\(Shape\(\[([-\d,]*)\]\),Shape\(\[([-\d,]*)\]\)")
SOFTMAX_RE = re.compile(r"SoftmaxParams\(SoftmaxOperationType::(\w+),(-?\d+)")
CONCAT_RE = re.compile(r"ConcatParams\((\d+),")
OUTDT_RE = re.compile(r"DataType::(\w+)")


def mem_of(params: str) -> dict | None:
    m = re.search(r"MemoryConfig\(memory_layout=TensorMemoryLayout::(\w+),buffer_type=BufferType::(\w+)",
                  params)
    return {"layout": m.group(1), "buffer": m.group(2)} if m else None


def ckc_of(params: str) -> dict | None:
    m = CKC_RE.search(params)
    if not m:
        return None
    return {"math_fidelity": m.group(1), "math_approx_mode": int(m.group(2)),
            "fp32_dest_acc_en": int(m.group(3)), "packer_l1_acc": int(m.group(4)),
            "dst_full_sync_en": int(m.group(5))}


def in_spec(i: dict) -> dict:
    return {"shape": i["shape"], "dtype": DT.get(i["dtype"], i["dtype"].lower()),
            "layout": i["layout"], "buffer": i["mem"], "bytes": i["bytes"]}


def lower(rec: dict) -> dict | None:
    """Map one captured device op onto a replayable call descriptor, or None if unsupported."""
    op, api, p = rec["op"], rec["api"], rec["params"]
    ins = [in_spec(i) for i in rec["inputs"]]
    d: dict = {"op": op, "api": api, "unit_path": rec["unit"], "inputs": ins,
               "out_mem": mem_of(p), "ckc": ckc_of(p)}
    if op == "Matmul":
        d["kind"] = "matmul"
        # MatmulParams(program_config, bcast_batch, out_mem, out_dtype, ckc, ...)
        m = re.search(r"MemoryConfig\([^)]*\),DataType::(\w+)", p)
        d["out_dtype"] = DT.get(m.group(1), "bfloat16") if m else "bfloat16"
        d["transpose_a"] = False
        d["has_bias"] = len(ins) == 3
    elif op == "BinaryNg":
        d["kind"] = "binary"
        b = BIN_RE.search(p)
        d["binop"] = b.group(1).lower() if b else "add"
        a = ACT_RE.search(p)
        d["post_act"] = a.group(1).lower() if a else None
        dts = OUTDT_RE.findall(p)
        d["out_dtype"] = DT.get(dts[0], "bfloat16") if dts else "bfloat16"
        d["inplace"] = api.endswith("_")
    elif op == "LayerNorm":
        d["kind"] = "layernorm"
        e = re.search(r"LayerNormType::(\w+),DistributedLayerNormStage::\w+,([0-9.e-]+)", p)
        d["norm"] = e.group(1).lower() if e else "layernorm"
        d["eps"] = float(e.group(2)) if e else 1e-5
    elif op == "Softmax":
        d["kind"] = "softmax"
        m = SOFTMAX_RE.search(p)
        d["dim"] = int(m.group(2)) if m else -1
    elif op == "Transpose":
        d["kind"] = "transpose"
        m = TRANSPOSE_RE.search(p)
        d["dim"] = m.group(1) if m else "WH"
    elif op == "Permute":
        d["kind"] = "permute"
        m = PERMUTE_RE.search(p)
        d["perm"] = [int(x) for x in m.group(1).split(",")] if m and m.group(1) else None
    elif op == "Slice":
        d["kind"] = "slice"
        m = SLICE_RE.search(p)
        if not m:
            return None
        d["begins"] = [int(x) for x in m.group(1).split(",")]
        d["ends"] = [int(x) for x in m.group(2).split(",")]
        d["steps"] = [int(x) for x in m.group(3).split(",")]
    elif op == "Concat":
        d["kind"] = "concat"
        m = CONCAT_RE.search(p)
        d["dim"] = int(m.group(1)) if m else 0
    elif op == "ReshapeView":
        d["kind"] = "reshape"
        m = RESHAPE_RE.search(p)
        if not m:
            return None
        d["out_shape"] = [int(x) for x in m.group(2).split(",")]
    elif op == "NlpCreateHeads":
        d["kind"] = "nlp_create_qkv_heads"
        nums = re.search(r"operation_attributes_t\((\d+),(\d+),(\d+),(\d)", p)
        if not nums:
            return None
        d["num_heads"] = int(nums.group(1))
        d["num_kv_heads"] = int(nums.group(2))
        d["head_dim"] = int(nums.group(3))
        d["transpose_k"] = bool(int(nums.group(4)))
    elif op == "NLPConcatHeads":
        d["kind"] = "nlp_concat_heads"
    elif op == "SDPAOperation":
        d["kind"] = "sdpa"
        m = re.search(r"SDPAParams\(([0-9.e-]+)", p)
        d["scale"] = float(m.group(1)) if m else None
        q = re.search(r"SDPAProgramConfig\(<xy_pair>,[^,]*,(\d+),(\d+),(\d),(\d+)\)", p)
        if q:
            d["q_chunk"], d["k_chunk"] = int(q.group(1)), int(q.group(2))
    elif op == "UnaryNg":
        d["kind"] = "unary"
        a = ACT_RE.search(p)
        d["unop"] = a.group(1).lower() if a else None
    elif op == "Copy":
        d["kind"] = "copy"
    elif op == "Pad":
        d["kind"] = "pad"
    elif op == "GenericOp":
        d["kind"] = "generic"        # custom fused kernel, not replayable from the capture
    else:
        return None
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "bench_manifest.json"))
    a = ap.parse_args()

    out = []
    for tag, (unit, calls) in UNITS.items():
        path = CAPS / f"cap_{tag}.json.gz"
        recs = walk(path)
        groups: dict[str, dict] = {}
        for r in recs:
            k = key_of(r)
            g = groups.setdefault(k, {"n": 0, "rec": r})
            g["n"] += 1
        for i, (k, g) in enumerate(sorted(groups.items(), key=lambda kv: -kv[1]["n"])):
            low = lower(g["rec"])
            if low is None:
                low = {"kind": "unsupported", "op": g["rec"]["op"], "api": g["rec"]["api"],
                       "unit_path": g["rec"]["unit"],
                       "inputs": [in_spec(x) for x in g["rec"]["inputs"]]}
            low.update({"id": f"{unit}#{i:03d}", "unit": unit, "per_unit": g["n"],
                        "unit_calls_per_fold": calls, "calls_per_fold": g["n"] * calls})
            out.append(low)

    json.dump(out, open(a.out, "w"), indent=1)
    n_ok = sum(1 for r in out if r["kind"] not in ("unsupported", "generic"))
    print(f"{len(out)} instances, {n_ok} replayable, "
          f"{sum(r['calls_per_fold'] for r in out)} device dispatches/fold -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
