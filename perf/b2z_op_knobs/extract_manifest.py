#!/usr/bin/env python3
"""Extract the device-op instance manifest for Boltz-2 512 aa from the ttnn graph
captures in perf/b2x-baseline-attrib/captures/.

Every ``*DeviceOperation`` node in a capture is one real dispatch. Group them by
(op, operand specs, op params) and count occurrences per captured unit. Multiply by
the unit's calls/fold to get calls/fold per instance.

Usage: extract_manifest.py [--out manifest.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
CAPS = ROOT / "perf" / "b2x-baseline-attrib" / "captures"

# Units we replay, and how many times each runs in one 512 aa fold.
# PairformerLayer: 256 in the trunk (16 layers x 16 recycles... measured 256) + 8 in the
# standalone PairformerModule = 264 in ATTRIBUTION.md; the op-cost curve counts 280.
# MSALayer: 16. DiffusionTransformerLayer: the diffusion step is captured whole.
UNIT_CALLS = {
    "PairformerLayer|1x512x384,1x512x512x128": 264,
    "MSALayer|1x512x512x128,1x1024x512x64": 16,
    "DiffusionTransformerLayer|1x224x32x128,1x224x32x128": 0,  # counted via the step capture
    "Diffusion|1x7168x3,1": 200,
}

TENSOR_RE = re.compile(
    r"logical_shape=Shape\(\[([^\]]*)\]\).*?dtype=DataType::(\w+).*?"
    r"TilePageConfig\(tile=Tile\(tile_shape=\{(\d+), (\d+)\}.*?"
    r"memory_config=MemoryConfig\(memory_layout=TensorMemoryLayout::(\w+),"
    r"buffer_type=BufferType::(\w+)",
    re.S,
)


def spec_of(node: dict) -> dict:
    p = node["params"]
    shape = [int(x) for x in re.findall(r"-?\d+", p.get("shape", ""))]
    mc = p.get("memory_config", "")
    return {
        "shape": shape,
        "dtype": p.get("dtype", "").replace("DataType::", ""),
        "layout": p.get("layout", "").replace("Layout::", ""),
        "mem": "L1" if "BufferType::L1" in mc else "DRAM",
        "sharded": "shard_spec=std::nullopt" not in mc,
        "bytes": p.get("size", 0),
    }


def compact_params(arg: str) -> str:
    """Strip the noisy bits from a Params string so equal configs hash equal."""
    s = re.sub(r"\s+", "", arg)
    s = s.replace("shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0", "")
    s = s.replace("Tile(tile_shape={32,32},face_shape={16,16},num_faces=4)", "T32")
    return s


def walk(path: pathlib.Path):
    nodes = json.load(gzip.open(path))
    by_id = {n["counter"]: n for n in nodes}
    stack: list[str] = []
    out = []
    for n in nodes:
        t = n["node_type"]
        if t == "function_start":
            name = n["params"].get("name", "")
            stack.append(name)
            if name.endswith("DeviceOperation") or name.endswith("Operation"):
                # nearest enclosing ttnn.* python api
                api = next((s for s in reversed(stack[:-1]) if s.startswith("ttnn.")), "")
                unit = "/".join(s[6:] for s in stack if s.startswith("unit::"))
                ins = []
                for tid in n.get("input_tensors", []):
                    tn = by_id.get(tid)
                    if tn is not None and tn["node_type"] == "tensor":
                        ins.append(spec_of(tn))
                params = compact_params(n["arguments"][0]) if n["arguments"] else ""
                out.append(
                    {
                        "op": name.replace("DeviceOperation", ""),
                        "api": api,
                        "unit": unit,
                        "inputs": ins,
                        "params": params,
                    }
                )
        elif t == "function_end":
            if stack:
                stack.pop()
    return out


def key_of(rec: dict) -> str:
    ins = ";".join(
        f"{i['dtype']}:{'x'.join(map(str, i['shape']))}:{i['layout']}:{i['mem']}{'/S' if i['sharded'] else ''}"
        for i in rec["inputs"]
    )
    return f"{rec['op']}|{rec['api']}|{ins}|{rec['params']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "manifest.json"))
    ap.add_argument("--captures", nargs="*", default=None)
    args = ap.parse_args()

    caps = args.captures or sorted(str(p) for p in CAPS.glob("*.json.gz"))
    per_unit = {}
    for c in caps:
        p = pathlib.Path(c)
        tag = p.name[len("cap_") : -len(".json.gz")]
        recs = walk(p)
        groups = defaultdict(lambda: {"n": 0})
        for r in recs:
            k = key_of(r)
            g = groups[k]
            g["n"] += 1
            g.update({kk: r[kk] for kk in ("op", "api", "unit", "inputs", "params")})
        per_unit[tag] = {"total_device_ops": len(recs), "instances": groups}

    json.dump(per_unit, open(args.out, "w"), indent=1)
    for tag, u in sorted(per_unit.items(), key=lambda kv: -kv[1]["total_device_ops"]):
        print(f"{u['total_device_ops']:6d} device ops  {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
