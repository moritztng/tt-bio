#!/usr/bin/env python3
"""Turn the fold's own recorded (class, shape, K) launch keys into runnable arms, and count
their FLOPs and minimum bytes.

The keys come from `perf/roof_launch/fold_shapes.json`, written by the same `op_census.py`
that produced `op_census_512.json`, so the arm set, the call weights and the byte column all
share one definition of a shape. Nothing here opens a device or reads a clock.

Two counters carry the whole table, so both are known-answer controlled against the dense cube
the brief names: a bf16 8192^3 matmul is exactly 1,099,511,627,776 matrix FLOPs and exactly
402,653,184 bytes of unavoidable DRAM traffic. A byte counter in this campaign has now been
wrong three times, in two directions, so `byte_identity` re-runs the census's own internal
identity B == calls * (in_tiles + out_tiles) * 2048 before any byte number is quoted.
"""
from __future__ import annotations

import json
from math import ceil, prod
from pathlib import Path

BF16 = 2                       # bytes per element, the dtype every arm below runs in
TILE_BYTES = 32 * 32 * BF16    # 2048, the census's own tile constant

CUBE_N = 8192
CUBE_FLOPS = 1_099_511_627_776
CUBE_BYTES = 402_653_184

MATMUL_ARMS = ("linear", "matmul")
# in-place elementwise: the destination is read AND written, so three tensor traversals
INPLACE_ARMS = ("multiply_", "add_")
ELTWISE_ARMS = ("multiply", "add")
NORM_ARMS = ("layer_norm", "layer_norm_w")

# classes whose zero byte column is semantics, not a counting hole: an allocation moves nothing,
# and a metadata-only reshape/unsqueeze/squeeze/view moves nothing on device.
SEMANTIC_ZERO = ("ttnn.allocate_tensor_on_device", "ttnn.reshape", "ttnn.unsqueeze",
                 "ttnn.squeeze", "ttnn.deallocate", "ttnn.Tensor.__getitem__")


def matmul_flops(m_dims, k):
    """2*prod(out)*K -- the matrix FLOP count of a matmul with output `m_dims` and inner K."""
    return 2 * prod(m_dims) * k


def tensor_bytes(shape):
    return prod(shape) * BF16 if shape else 0


def tiles(shape):
    """Verbatim from perf/roof_launch/op_census.py, so the identity check uses its convention."""
    if not shape:
        return 0
    if len(shape) == 1:
        return ceil(shape[0] / 32)
    n = 1
    for d in shape[:-2]:
        n *= d
    return n * ceil(shape[-2] / 32) * ceil(shape[-1] / 32)


def cube_control():
    """The known-answer control both counters must pass before a table row is written."""
    n = CUBE_N
    flops = matmul_flops([n, n], n)
    byts = tensor_bytes([n, n]) * 3          # two operands in, one result out, no reuse
    return {"n": n, "flops": flops, "bytes": byts,
            "flops_expected": CUBE_FLOPS, "bytes_expected": CUBE_BYTES,
            "flops_pass": flops == CUBE_FLOPS, "bytes_pass": byts == CUBE_BYTES}


def byte_identity(census):
    """B == calls * (in_tiles + out_tiles) * 2048, per recorded shape, on the census we weight by.

    Returns one row per shape that records tiles, plus the holes: nonzero tiles against zero
    bytes. Direction matters -- an overcount and an undercount cannot both be corrected by one
    scale factor -- so ratios are reported signed, never aggregated into a single factor.
    """
    rows, holes = [], []
    for key, e in census["top_shapes"].items():
        it, ot, calls, b = e["in_tiles"], e["out_tiles"], e["calls"], e["B"]
        implied = calls * (it + ot) * TILE_BYTES
        if it + ot == 0:
            continue
        if b == 0:
            holes.append({"key": key, "calls": calls, "tiles_per_call": it + ot,
                          "implied_bytes": implied})
            continue
        rows.append({"key": key, "calls": calls, "recorded_bytes": b,
                     "implied_bytes": implied, "ratio": implied / b})
    ratios = sorted(r["ratio"] for r in rows)
    med = (ratios[len(ratios) // 2] if len(ratios) % 2
           else 0.5 * (ratios[len(ratios) // 2 - 1] + ratios[len(ratios) // 2])) if ratios else None
    for h in holes:
        h["semantic"] = h["key"].split("|")[0] in SEMANTIC_ZERO
    holes.sort(key=lambda h: -h["implied_bytes"])
    real = [h for h in holes if not h["semantic"]]
    recorded = sum(e["B"] for e in census["by_op"].values())
    real_bytes = sum(h["implied_bytes"] for h in real)
    return {"shapes_checked": len(rows), "median_ratio": med,
            "min_ratio": ratios[0] if ratios else None,
            "max_ratio": ratios[-1] if ratios else None,
            "holes": holes, "real_holes": real,
            "hole_bytes": sum(h["implied_bytes"] for h in holes),
            "real_hole_bytes": real_bytes,
            "recorded_total_bytes": recorded,
            "real_hole_pct_of_recorded": 100 * real_bytes / recorded if recorded else None,
            "rows": rows}


def _operands(arm, out, k):
    """The actual operand shapes an arm consumes, in call order, from (class, out shape, K).

    `linear` writes out[:-1] x N from an activation out[:-1] x K and a weight K x N, which is
    exactly what the recorded signatures show (`out=1x16x512x512|in=1x16x512x128,128x512`).
    `matmul` keeps both operands batched over the leading dims.
    """
    if arm == "linear":
        return [("a", list(out[:-1]) + [k]), ("w", [k, out[-1]])]
    if arm == "matmul":
        lead = list(out[:-2])
        return [("a", lead + [out[-2], k]), ("b", lead + [k, out[-1]])]
    if arm in INPLACE_ARMS or arm in ELTWISE_ARMS:
        return [("a", list(out)), ("b", list(out))]
    if arm == "layer_norm":
        return [("a", list(out))]
    if arm == "layer_norm_w":
        return [("a", list(out)), ("w", [out[-1]]), ("b", [out[-1]])]
    return None


def _bytes(arm, out, ops):
    """Minimum DRAM traffic per call: every distinct operand read once, every result written once.

    An in-place op reads its destination and writes it back, so its destination is counted twice
    and its other operand once. This is a floor on traffic, not a claim about what the kernel did.
    """
    if arm in INPLACE_ARMS:
        return 2 * tensor_bytes(ops[0][1]) + tensor_bytes(ops[1][1])
    return sum(tensor_bytes(s) for _n, s in ops) + tensor_bytes(list(out))


def build_arms(shapes, include=None):
    """One runnable spec per recorded launch key, with its call weight and both counters."""
    arms, refused = [], []
    for row in shapes:
        arm, out, k, calls = row["arm"], row["shape"], row["K"], row["calls"]
        if include and arm not in include:
            continue
        ops = _operands(arm, out, k)
        if ops is None:
            refused.append({**row, "reason": "no operand model for this class"})
            continue
        if arm in MATMUL_ARMS and not k:
            refused.append({**row, "reason": "no recorded K, arithmetic not derivable"})
            continue
        if arm in MATMUL_ARMS and len(out) < 2:
            refused.append({**row, "reason": "output rank below 2, not a matmul key"})
            continue
        arms.append({
            "key": "%s|out=%s|K=%s" % (arm, "x".join(str(d) for d in out), k),
            "arm": arm, "out": list(out), "K": k, "calls": calls,
            "operands": ops,
            "flops_per_call": matmul_flops(out, k) if arm in MATMUL_ARMS else 0,
            "min_bytes_per_call": _bytes(arm, out, ops),
        })
    arms.sort(key=lambda a: -(a["flops_per_call"] * a["calls"] + a["min_bytes_per_call"] * a["calls"]))
    return arms, refused


def load(perf: Path):
    return (json.loads((perf / "roof_launch" / "fold_shapes.json").read_text()),
            json.loads((perf / "roof_launch" / "op_census_512.json").read_text()))
