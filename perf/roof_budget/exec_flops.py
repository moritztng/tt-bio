#!/usr/bin/env python3
"""Executed FLOPs per captured call, counted off the device's own operand shapes.

torch's ``FlopCounterMode`` counts the LOGICAL matmul the model asks for. The device runs the one
the padded tensors describe: the token axis bucketed, the MSA axis padded to
``MSA_PAD_MULTIPLE = 1024``, the atom axis bucketed to a window multiple, and every matmul
dimension rounded up to a 32-tile. This file counts what the device was actually asked to issue,
from the same ttnn.graph capture the byte counter reads, so the FLOP column and the byte column
come off one instrument on one capture.

Per top-level ttnn op, from the distinct operand buffers reaching it:

  ttnn.linear / ttnn.matmul
      output shape comes from the ``create_device_tensor`` inside the op, so M, N and the batch are
      the device's. K is the last dim of the operand whose leading dims multiply to batch*M.
      FLOPs = 2 * batch * M * N * K.

  ttnn.generic_op -- a hand-written kernel, three kinds, told apart by operand rank alone:
      a rank-2 operand (K, N) present   a fused matmul (mm_generic / triatt_qkv / trimul_tail).
                                        rows = max numel over operands whose last dim is K,
                                        divided by K. FLOPs = 2 * rows * K * N. A kernel that
                                        splits N between two activations does the same MACs as one
                                        that applies all of N to one, so this is exact either way.
      a (1, h, S, S) mask operand       sdpa_generic. With q, k, v, out all (b, h, S, d):
                                        FLOPs = 4 * b * h * S * S * d (QK^T and AV).
      neither                           reblock_permute. Layout only, zero.

  everything else   one FLOP per output element. Eltwise, layernorm, softmax. Reported in its own
                    column, never mixed into the matmul term.

Free, charged nothing: reshape, permute, transpose, squeeze, unsqueeze, chunk, slice, concat,
getitem, deallocate, allocate_tensor_on_device, to_layout, typecast, clone, to_memory_config,
nlp_create_qkv_heads, nlp_concat_heads.

The known-answer control for this file is ``instrument_control.py``: a dense matmul whose FLOP
count and minimum byte count are exact by arithmetic.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b2x_difflayer"))
from itemize import top_level_spans                                          # noqa: E402

TILE = 32
MATMUL = {"ttnn.linear", "ttnn.matmul"}
FREE = {"ttnn.reshape", "ttnn.unsqueeze", "ttnn.squeeze", "ttnn.deallocate", "ttnn.permute",
        "ttnn.transpose", "ttnn.chunk", "ttnn.slice", "ttnn.concat", "ttnn.Tensor.__getitem__",
        "ttnn.allocate_tensor_on_device", "ttnn.to_layout", "ttnn.typecast", "ttnn.clone",
        "ttnn.to_memory_config", "ttnn.experimental.nlp_create_qkv_heads",
        "ttnn.experimental.nlp_concat_heads"}
SHAPE_RE = re.compile(r"Shape\(\[([0-9,\s]+)\]\)")


def _shape(s):
    m = SHAPE_RE.search(s or "")
    return tuple(int(x) for x in m.group(1).split(",")) if m else None


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def pad2(shape):
    """Round the last two dims of a shape up to the 32-tile."""
    s = list(shape)
    for j in (-1, -2):
        if len(s) >= -j:
            s[j] = -(-s[j] // TILE) * TILE
    return tuple(s)


def operands(nodes):
    """op index -> {(address, shape)} of every distinct tensor buffer reaching it, and
    op index -> [output shapes created inside it]."""
    ops, owner = top_level_spans(nodes)
    ins = defaultdict(set)
    outs = defaultdict(list)
    for n in nodes:
        p = n.get("params") or {}
        if n["node_type"] == "tensor" and p.get("shape"):
            sh = _shape(p["shape"])
            if sh:
                for c in (n.get("connections") or []):
                    o = owner.get(c)
                    if o is not None:
                        ins[o].add((p.get("address"), sh))
        elif n["node_type"] == "function_start" and \
                str(p.get("name", "")).endswith("create_device_tensor"):
            o = owner.get(n["counter"])
            sh = _shape((n.get("arguments") or [""])[0])
            if o is not None and sh:
                outs[o].append(sh)
    return ops, ins, outs


def _mm(batch, M, N, K):
    return 2 * batch * M * N * K


def op_flops(name, ins, outs):
    """(logical, tile_padded, kind) for one op."""
    if name in FREE:
        return 0, 0, "free"
    shapes = [s for _, s in ins]

    if name in MATMUL:
        out = max(outs, key=_prod) if outs else None
        if out is None or len(out) < 2:
            return 0, 0, "matmul-unresolved"
        M, N = out[-2], out[-1]
        batch = _prod(out[:-2])
        acts = [s for s in shapes if len(s) >= 2 and _prod(s[:-1]) == batch * M]
        if not acts:
            w = [s for s in shapes if len(s) == 2 and s[-1] == N]
            if not w:
                return 0, 0, "matmul-unresolved"
            K = max(s[0] for s in w)
        else:
            K = max(s[-1] for s in acts)
        Mp, Np = pad2((M, N))
        Kp = -(-K // TILE) * TILE
        return _mm(batch, M, N, K), _mm(batch, Mp, Np, Kp), "matmul"

    if name == "ttnn.generic_op":
        w2 = [s for s in shapes if len(s) == 2]
        if w2:
            log = padf = 0
            for K, N in w2:
                rows_c = [_prod(s) for s in shapes if s[-1] == K and len(s) >= 2]
                if not rows_c:
                    continue
                rows = max(rows_c) // K
                log += _mm(1, rows, N, K)
                padf += _mm(1, -(-rows // TILE) * TILE, -(-N // TILE) * TILE,
                            -(-K // TILE) * TILE)
            if log:
                return log, padf, "matmul"
            return 0, 0, "generic-unresolved"
        mask = [s for s in shapes if len(s) == 4 and s[0] == 1 and s[-1] == s[-2]]
        qkv = [s for s in shapes if len(s) == 4 and s[0] != 1 and s[-1] != s[-2]]
        if mask and qkv:
            b, h, S, d = max(qkv, key=_prod)
            f = 4 * b * h * S * S * d
            return f, f, "matmul"
        return 0, 0, "layout"

    if not outs:
        return 0, 0, "noshape"
    out = max(outs, key=_prod)
    return _prod(out), _prod(pad2(out)), "eltwise"


def per_op(nodes):
    ops, ins, outs = operands(nodes)
    return [(o["name"],) + op_flops(o["name"], ins[i], outs[i]) for i, o in enumerate(ops)]


def totals(nodes):
    rows = per_op(nodes)
    t = defaultdict(int)
    t["n_ops"] = len(rows)
    for _name, log, padf, kind in rows:
        if kind == "matmul":
            t["matmul_logical"] += log
            t["matmul_padded"] += padf
        elif kind in ("eltwise", "noshape"):
            t["eltwise_logical"] += log
            t["eltwise_padded"] += padf
        elif kind.endswith("unresolved"):
            t["unresolved"] += 1
    t["logical"] = t["matmul_logical"] + t["eltwise_logical"]
    t["padded"] = t["matmul_padded"] + t["eltwise_padded"]
    t["tile_pad_factor"] = (t["matmul_padded"] / t["matmul_logical"]) if t["matmul_logical"] else 1.0
    return dict(t)


def load(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.load(open(path))


def nodes_of(path):
    d = load(path)
    return d["nodes"] if isinstance(d, dict) and "nodes" in d else d


if __name__ == "__main__":
    for p in sys.argv[1:]:
        t = totals(nodes_of(p))
        print("%-58s matmul %10.3f GF -> %10.3f GF padded (%.4fx)  eltwise %7.3f GF  ops %d"
              " unresolved %d"
              % (Path(p).name, t["matmul_logical"] / 1e9, t["matmul_padded"] / 1e9,
                 t["tile_pad_factor"], t["eltwise_padded"] / 1e9, t["n_ops"],
                 t.get("unresolved", 0)))
