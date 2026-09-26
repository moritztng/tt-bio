#!/usr/bin/env python3
"""bcx-p10-mmlay: the round's matmuls keyed by SHAPE and PLACEMENT, not by op name.

`bcx-p10-calls` ranked the device column by ttnn op name and found `matmul` + `linear` +
`experimental.minimal_matmul` at 2.269 s and a weighted ~40 % of the DRAM roof. A name is the
wrong key for that question: one `matmul` row is a hundred different calls, and a 288x512x128
against a DRAM-interleaved bf16 weight and a 288x288x288 against an L1 activation want opposite
fixes. This widens `OpTimer`'s second key from the op name to

    op # batch x M x K x N # transpose flags # a:<buf>/<dtype>/<layout>/<memlayout> # b:... # o:...

and changes NOTHING else: the same `device = synced - free - lambda * calls` subtraction, the
same records, the same medians. The cost of the wider key is more per-key zero clamping, which
is why `report.py` prints A_shape against `bcx-p10-calls`'s A_verb with the difference named.

FLOPs are COUNTED here, per call, off the padded shapes the card actually ran -- not the
analytic family model `bcx-p10-devmap` uses -- so a shape's arithmetic intensity is measured
on both axes and the roofline placement in leg 1 is not an estimate.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_devmap import devmap as D          # noqa: E402

#: The three arithmetic verbs `bcx-p10-calls` put at 2.269 s. Only these get the wide key: a
#: shape key on all 50,868 calls would multiply the record size and clamp far more keys to
#: zero, and the question is about matmuls.
MM_VERBS = ("matmul", "linear", "experimental.minimal_matmul")

_SHORT = {"BFLOAT16": "bf16", "FLOAT32": "fp32", "BFLOAT8_B": "bfp8", "BFLOAT4_B": "bfp4",
          "UINT32": "u32", "INT32": "i32", "UINT16": "u16", "UINT8": "u8"}


def _enum(v):
    """`BufferType.DRAM` -> `DRAM`, `TensorMemoryLayout.INTERLEAVED` -> `INTERLEAVED`."""
    s = str(v)
    return s.rsplit(".", 1)[-1]


def describe(t):
    """`<buffer>/<dtype>/<page layout>/<memory layout>[/<grid>]` for one operand."""
    if t is None:
        return "-"
    try:
        dt = _SHORT.get(str(t.dtype).rsplit(".", 1)[-1].upper(),
                        str(t.dtype).rsplit(".", 1)[-1].lower())
    except Exception:
        return "?"
    lay = _enum(getattr(t, "layout", "?"))[:4]
    try:
        mc = t.memory_config()
        buf, ml = _enum(mc.buffer_type), _enum(mc.memory_layout)
        sh = mc.shard_spec
        if sh is not None:
            ml += "[%s]" % "x".join(str(int(d)) for d in sh.shape)
    except Exception:
        buf, ml = "?", "?"
    return f"{buf}/{dt}/{lay}/{ml}"


def _pshape(t):
    try:
        return [int(d) for d in t.padded_shape]
    except Exception:
        try:
            return [int(d) for d in t.shape]
        except Exception:
            return []


def mnk(a, b, ta, tb):
    """(batch, M, K, N) off the PADDED shapes, which is what the card multiplied.

    Returns None when either operand is not rank>=2, which is the signal to fall back to the
    bare op name rather than emit a key that claims a shape it did not verify.
    """
    sa, sb = _pshape(a), _pshape(b)
    if len(sa) < 2 or len(sb) < 2:
        return None
    M, Ka = (sa[-1], sa[-2]) if ta else (sa[-2], sa[-1])
    Kb, N = (sb[-1], sb[-2]) if tb else (sb[-2], sb[-1])
    if Ka != Kb:
        return None
    batch = 1
    for d in sa[:-2]:
        batch *= d
    bb = 1
    for d in sb[:-2]:
        bb *= d
    return max(batch, bb), M, Ka, N


class ShapeOpTimer(D.OpTimer):
    """`OpTimer` with the matmul family's second key widened to shape + placement."""

    def verb_key(self, path, args, kwargs, out):
        if path not in MM_VERBS:
            return path
        T = self.ttnn.Tensor
        pos = [x for x in args if isinstance(x, T)]
        # `tenstorrent.py` calls `minimal_matmul` entirely by keyword (`input_tensor=`,
        # `weight_tensor=`), which is 624 calls and 0.145 s of triangle attention -- the
        # positional read alone left them keyless and shapeless in the table.
        for slot in ("input_tensor_a", "input_tensor", "input"):
            if len(pos) < 1 and isinstance(kwargs.get(slot), T):
                pos.append(kwargs[slot])
        for slot in ("input_tensor_b", "weight_tensor", "weight", "other"):
            if len(pos) < 2 and isinstance(kwargs.get(slot), T):
                pos.append(kwargs[slot])
        if len(pos) < 2:
            return path
        a, b = pos[0], pos[1]
        ta = bool(kwargs.get("transpose_a", False))
        tb = bool(kwargs.get("transpose_b", False))
        dims = mnk(a, b, ta, tb)
        if dims is None:
            return path
        batch, M, K, N = dims
        outs = []
        self._walk(out, outs)
        o = outs[0] if outs else None
        flags = ("ta" if ta else "") + ("tb" if tb else "") or "-"
        extra = []
        if kwargs.get("program_config") is not None or kwargs.get("config") is not None:
            extra.append("pc")
        cg = kwargs.get("core_grid")
        if cg is not None:
            try:
                extra.append("g%dx%d" % (int(cg.x), int(cg.y)))
            except Exception:
                extra.append("g?")
        if kwargs.get("bias") is not None or kwargs.get("bias_tensor") is not None:
            extra.append("bias")
        if kwargs.get("memory_config") is not None:
            extra.append("omc=" + _enum(kwargs["memory_config"].buffer_type))
        return "%s#%dx%dx%dx%d#%s#a:%s#b:%s#o:%s%s" % (
            path, batch, M, K, N, flags, describe(a), describe(b), describe(o),
            "#" + ",".join(extra) if extra else "")


def parse(key):
    """A wide key back into its parts. A bare op name parses as shape-less."""
    if "#" not in key:
        return {"op": key, "shape": None}
    parts = key.split("#")
    op, dims, flags = parts[0], parts[1], parts[2]
    batch, M, K, N = (int(x) for x in dims.split("x"))
    d = {"op": op, "batch": batch, "M": M, "K": K, "N": N, "flags": flags,
         "shape": dims, "extra": ""}
    for p in parts[3:]:
        if p.startswith("a:"):
            d["a"] = p[2:]
        elif p.startswith("b:"):
            d["b"] = p[2:]
        elif p.startswith("o:"):
            d["o"] = p[2:]
        else:
            d["extra"] = p
    d["flop"] = 2.0 * batch * M * K * N
    return d
