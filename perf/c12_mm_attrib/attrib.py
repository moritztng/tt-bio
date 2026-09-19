#!/usr/bin/env python3
"""Attribute the census's `matmul`-arm launch keys to the code that issues them.

CPU only: reads the 26 committed graph captures under `perf/roof_budget/captures`, which are
executed-graph recordings of the 512 aa Boltz-2 fold's bracketed modules, and pulls out every
`ttnn.matmul` / `ttnn.linear` with its operand specs, its output memory config, its compute
kernel config and the `unit::` module it ran inside. Op ownership uses the committed
`itemize.top_level_spans`, so the op set is the one roof-budget and roof-residual already count.

The key is built with c10_fold_census's own convention: `<arm>|out=<out shape>|K=<inner>`, where
for `linear` the activation is out[:-1] x K and for `matmul` both operands are batched. That is
how a capture row is matched to a census price.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))
from itemize import top_level_spans                                          # noqa: E402

CAPDIR = ROOT / "perf" / "roof_budget" / "captures"
MM = ("ttnn.matmul", "ttnn.linear")

SHAPE_RE = re.compile(r"logical_shape=Shape\(\[([0-9, ]*)\]\)")
TENSOR_RE = re.compile(r"Tensor\(storage=")
DTYPE_RE = re.compile(r"dtype=(DataType::\w+)")
BUF_RE = re.compile(r"buffer_type=BufferType::(\w+)")
LAYOUT_RE = re.compile(r"memory_layout=TensorMemoryLayout::(\w+)")
SHARD_RE = re.compile(r"shard_spec=(std::nullopt|ShardSpec)")
CKC_RE = re.compile(r"ComputeKernelConfig\(([^)]*)\)")


def load(p):
    return json.load(gzip.open(p, "rt"))


def _tensor_blobs(s):
    """Split a stringified tensor list into one blob per Tensor(...)."""
    idx = [m.start() for m in TENSOR_RE.finditer(s)]
    return [s[a:b] for a, b in zip(idx, idx[1:] + [len(s)])]


def _spec(blob):
    m = SHAPE_RE.search(blob)
    shape = [int(x) for x in m.group(1).split(",") if x.strip()] if m else None
    d = DTYPE_RE.search(blob)
    b = BUF_RE.search(blob)
    l = LAYOUT_RE.search(blob)
    sh = SHARD_RE.search(blob)
    return {"shape": shape,
            "dtype": d.group(1).split("::")[-1] if d else None,
            "buffer": b.group(1) if b else None,
            "mem_layout": l.group(1) if l else None,
            "sharded": bool(sh and sh.group(1) != "std::nullopt")}


def _params_and_operands(nodes, by_counter, op):
    """The device-op params string and the operand tensor specs for one top-level matmul."""
    params, operands = None, []
    for c in range(op["start"], (op["end"] or op["start"]) + 1):
        n = by_counter.get(c)
        if not n or n.get("node_type") != "function_start":
            continue
        for a in n.get("arguments") or []:
            if a.startswith("MatmulParams("):
                params = a
            elif a.startswith("{Tensor(") or a.startswith("Tensor("):
                for blob in _tensor_blobs(a):
                    operands.append(_spec(blob))
        if params and operands:
            break
    return params, operands


def _out_shape(name, operands, transpose_a, transpose_b):
    """The output shape, from the operands and the transpose flags. Deterministic, not a guess.

    `linear`: out[:-1] = a[:-1], out[-1] = w[-1]. `matmul`: leading dims broadcast, M from a,
    N from b, each read through its own transpose flag.
    """
    if not operands or operands[0]["shape"] is None:
        return None, None
    a = list(operands[0]["shape"])
    b = list(operands[1]["shape"]) if len(operands) > 1 and operands[1]["shape"] else None
    if name == "ttnn.linear":
        if b is None:
            return None, None
        return a[:-1] + [b[-1]], a[-1]
    if b is None:
        return None, None
    am, ak = (a[-1], a[-2]) if transpose_a else (a[-2], a[-1])
    bk, bn = (b[-1], b[-2]) if transpose_b else (b[-2], b[-1])
    lead = a[:-2] if len(a) >= len(b) else b[:-2]
    return lead + [am, bn], ak


def _tensor_nodes(by_counter, op):
    """Every tensor node inside the span, as (shape, buffer, address) -- the byte evidence."""
    out = []
    for c in range(op["start"], (op["end"] or op["start"]) + 1):
        n = by_counter.get(c)
        if not n or n.get("node_type") != "tensor":
            continue
        q = n.get("params") or {}
        m = re.search(r"Shape\(\[([0-9, ]*)\]\)", str(q.get("shape", "")))
        if not m:
            continue
        out.append({"shape": [int(x) for x in m.group(1).split(",") if x.strip()],
                    "buffer": str(q.get("buffer_type", "")).split("::")[-1],
                    "address": q.get("address"), "dtype": str(q.get("dtype", "")).split("::")[-1]})
    return out


def census_key(name, out_shape, k):
    """c10_fold_census's launch key, rebuilt from the executed operands."""
    if out_shape is None:
        return None
    return "%s|out=%s|K=%d" % (name.split(".")[-1], "x".join(str(d) for d in out_shape), k)


def _fields(params):
    """The Matmul device-op attributes this row needs, read off the reflected params string.

    Field order is `ttnn::prim::MatmulParams` (matmul_device_operation_types.hpp): program_config, bcast_batch, output_mem_config,
    output_dtype, compute_kernel_config, untilize_out, user_core_coord, user_fused_activation,
    transpose_a, transpose_b, ... A variant prints as `<variant>` whether or not the optional
    holds one, so position 0 is NOT readable -- `core_grid` is, because a present
    `user_core_coord` prints `<xy_pair>` where an absent one prints `std::nullopt`.
    """
    if not params:
        return {"core_grid": None, "out_mem": None, "out_layout": None, "out_dtype": None,
                "compute_kernel": None, "transpose_a": False, "transpose_b": False,
                "user_run_batched": None, "bcast_batch": None, "fused_activation": None,
                "untilize_out": None}
    inner = params[len("MatmulParams("):-1]
    toks, depth, cur = [], 0, ""
    for ch in inner:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            toks.append(cur.strip())
            cur = ""
        else:
            cur += ch
    toks.append(cur.strip())
    g = lambda i: toks[i] if i < len(toks) else None
    mc = g(2) or ""
    ckc = CKC_RE.search(params)
    return {"core_grid": ("passed" if g(6) not in (None, "std::nullopt") else "absent"),
            "out_mem": (BUF_RE.search(mc).group(1) if BUF_RE.search(mc) else None),
            "out_layout": (LAYOUT_RE.search(mc).group(1) if LAYOUT_RE.search(mc) else None),
            "out_dtype": (g(3) or "").split("::")[-1] or None,
            "compute_kernel": ckc.group(1) if ckc else None,
            "bcast_batch": g(1),
            "fused_activation": g(7),
            "untilize_out": g(5),
            "user_run_batched": g(8) == "1",
            "transpose_a": g(9) == "1",
            "transpose_b": g(10) == "1"}


def walk(path):
    nodes = load(path)
    by_counter = {n["counter"]: n for n in nodes}
    ops, owner = top_level_spans(nodes)
    # unit:: markers in counter order, so an op is attributed to the innermost unit opened before it
    marks = []
    for n in nodes:
        if n.get("node_type") == "function_start":
            nm = str((n.get("params") or {}).get("name", ""))
            if nm.startswith("unit::"):
                marks.append((n["counter"], nm[len("unit::"):], n.get("stacking_level")))
    rows = []
    for i, op in enumerate(ops):
        if op["name"] not in MM:
            continue
        params, operands = _params_and_operands(nodes, by_counter, op)
        unit = None
        for c, nm, _lvl in marks:
            if c <= op["start"]:
                unit = nm
        f = _fields(params)
        ta, tb = f["transpose_a"], f["transpose_b"]
        out_shape, k = _out_shape(op["name"], operands, ta, tb)
        rows.append({"capture": path.name, "op_index": i, "name": op["name"], "unit": unit,
                     "key": census_key(op["name"], out_shape, k), "K": k,
                     "out_shape": out_shape,
                     "operands": operands, "tensors": _tensor_nodes(by_counter, op),
                     "params": params, **f})
    return rows, len(ops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capdir", type=Path, default=CAPDIR)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "mm_sites.json.gz",
                    help="gzipped when the name ends .gz -- pc / is at 94 %%")
    ap.add_argument("--full-params", action="store_true")
    a = ap.parse_args()
    allrows, per_cap = [], {}
    for p in sorted(a.capdir.glob("cap_*.json.gz")):
        rows, nops = walk(p)
        per_cap[p.name] = {"top_level_ops": nops, "matmuls": len(rows)}
        allrows += rows
    if not a.full_params:
        for r in allrows:
            r.pop("params", None)
            r.pop("tensors", None)
    blob = json.dumps({"captures": per_cap, "rows": allrows}, indent=1)
    if a.out.name.endswith(".gz"):
        with gzip.open(a.out, "wt") as fh:
            fh.write(blob)
    else:
        a.out.write_text(blob)
    bykey = defaultdict(list)
    for r in allrows:
        bykey[r["key"]].append(r)
    for key in sorted(bykey, key=lambda k: (k is None, k)):
        rs = bykey[key]
        print("%-42s n=%-3d %s" % (key, len(rs),
              sorted({"%s/%s:%s" % (r["capture"].replace("cap_", "").replace(".json.gz", ""),
                                    r["unit"], r["name"].split(".")[-1]) for r in rs})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
