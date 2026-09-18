#!/usr/bin/env python3
"""Runnable arms for the launch keys `c10-fold-census` refused, built from the fold's own capture.

The refusal was uniform and it was not structural: `census._operands` had no branch for these
classes. The shapes were in the capture the whole time. What the capture does NOT record is op
ARGUMENTS -- every `function_start` node carries `arguments: []` -- so a permutation order, a
slice range, a concat axis and a to_layout target are not read, they are DERIVED from the
operand and result shapes the capture does record. Each arm therefore carries a `derivation`
field saying which of its parameters were read and which were inferred, and `pinned` is False
for any arm whose parameter set is not uniquely determined by the shapes.

Where derivation is ambiguous the arm is still built, because a tile-count-equivalent stand-in
bounds the cost even when the exact offsets differ, but the ambiguity is published beside the
number rather than hidden inside it. One case is known to matter: a tile-misaligned slice or pad
routes a different kernel from an aligned one, so alignment is recorded per arm.
"""
from __future__ import annotations

from math import prod

BF16 = 2
TILE = 32


def tile_aligned(shape):
    """Are the last two dims tile multiples? A ragged tail routes a different kernel."""
    if len(shape) < 2:
        return shape and shape[-1] % TILE == 0
    return shape[-2] % TILE == 0 and shape[-1] % TILE == 0


def _numel(s):
    return prod(s) if s else 0


def _ins(site):
    return [tuple(r["shape"]) for r in site["ins"]]


def _out(site):
    return tuple(max(site["outs"], key=_numel)) if site["outs"] else None


def _perm_of(src, dst):
    """The unique permutation taking `src` to `dst`, or None if it is not unique."""
    if len(src) != len(dst) or sorted(src) != sorted(dst):
        return None
    used, perm = set(), []
    for d in dst:
        cand = [i for i, s in enumerate(src) if s == d and i not in used]
        if not cand:
            return None
        perm.append(cand[0])
        used.add(cand[0])
    ambiguous = len(set(src)) != len(src)
    return tuple(perm), ambiguous


def build(key, entry):
    """One arm spec for a refused key, or a dict with `refused` naming what could not be derived.

    Picks the site carrying the most calls as the representative, because a key aggregates sites
    with identical launch shape and the per-call cost is what the weight multiplies.
    """
    op, arm = entry["op"], entry["arm"]
    site = max(entry["sites"], key=lambda s: s["calls"])
    ins, out = _ins(site), _out(site)
    base = {"key": key, "op": op, "arm": arm, "calls": entry["calls"], "B": entry["B"],
            "in_shapes": [list(s) for s in ins], "out": list(out) if out else None,
            "n_sites": entry["n_sites"], "layouts": [r["layout"] for r in site["ins"]],
            "dtypes": [r["dtype"] for r in site["ins"]],
            "buffers": [r["buffer"] for r in site["ins"]],
            "pinned": True, "derivation": "", "params": {}}

    def refuse(why):
        return {**base, "refused": why, "pinned": False}

    if op == "ttnn.generic_op":
        return refuse("hand-written kernel, no stock ttnn op reproduces it; "
                      "c12-genericop-rate already measured it bandwidth-bound at all six sites")

    if out is None:
        return refuse("no output tensor recorded for this op in the capture")

    n_out = _numel(out)

    if arm in ("permute", "transpose"):
        cand = [s for s in ins if sorted(s) == sorted(out) and len(s) == len(out) and s != out]
        if not cand:
            cand = [s for s in ins if _numel(s) == n_out and len(s) == len(out)]
        if not cand:
            return refuse("no input shape is a permutation of the output")
        src = max(cand, key=_numel)
        got = _perm_of(src, out)
        if got is None:
            return refuse("output is not a permutation of any recorded input")
        perm, ambiguous = got
        if tuple(perm) == tuple(range(len(out))):
            # A shape-preserving permute reads as the identity here. It is not the identity --
            # the fold would not launch a program for that -- so the real order swaps two
            # equal-sized dims and the capture cannot say which. Take the first equal pair.
            pair = next(((i, j) for i in range(len(out)) for j in range(i + 1, len(out))
                         if out[i] == out[j] and out[i] > 1), None)
            if pair is None:
                return refuse("shape-preserving permute with no equal dim pair to swap")
            i, j = pair
            p2 = list(range(len(out)))
            p2[i], p2[j] = p2[j], p2[i]
            perm, ambiguous = tuple(p2), True
        base["params"] = {"dims": list(perm), "src": list(src)}
        base["pinned"] = not ambiguous
        base["derivation"] = ("permutation order INFERRED from %s -> %s%s"
                              % (src, out, "; repeated dim sizes make it non-unique"
                                 if ambiguous else "; unique"))
        return base

    if arm == "matmul":
        # the census refused this key for having no recorded K, which is true of
        # `op_shape_rows` and not of the capture: the two operands contract on a dim neither
        # of them carries as its last, i.e. the call transposes one of them.
        pair = [(x, y) for x in ins for y in ins
                if x is not y and len(x) == len(y) == len(out)
                and x[-1] == out[-2] and y[-1] == out[-1] and x[-2] == y[-2]]
        if not pair:
            return refuse("no operand pair contracts to the output shape")
        x, y = pair[0]
        K = x[-2]
        base["params"] = {"a": list(out[:-1]) + [K], "b": list(out[:-2]) + [K, out[-1]],
                          "K": K, "recorded_a": list(x), "recorded_b": list(y),
                          "flops": 2 * _numel(out) * K}
        base["pinned"] = False
        base["derivation"] = ("K=%d DERIVED from the operand pair %s x %s -> %s, which "
                              "contracts on a dim neither operand carries last, so the call "
                              "transposes one of them. The arm runs it untransposed."
                              % (K, x, y, out))
        return base

    if arm == "reshape":
        cand = [s for s in ins if _numel(s) == n_out and tuple(s) != out]
        if not cand:
            return refuse("no input shape has the output's element count")
        base["params"] = {"src": list(max(cand, key=_numel)), "dst": list(out)}
        base["derivation"] = "source shape READ, target shape READ; no argument needed"
        return base

    if arm == "slice":
        cand = [s for s in ins
                if len(s) == len(out) and all(a >= b for a, b in zip(s, out)) and s != out]
        if not cand:
            # Every recorded operand already has the sliced extent. The capture keeps distinct
            # (address, shape) pairs, so the PARENT tensor this was a view of never appears:
            # its extent is structurally unrecorded. corrected_traffic charges this key one
            # read plus one write of the OUT shape, so a full-extent slice is the arm that
            # matches the byte column, and it is a lower bound on the read side.
            if any(tuple(s) == out for s in ins):
                base["params"] = {"src": list(out), "starts": [0] * len(out),
                                  "ends": list(out)}
                base["pinned"] = False
                base["derivation"] = ("source extent NOT RECORDED -- every operand already "
                                      "carries the sliced shape %s, so the parent tensor is "
                                      "outside the capture. Run at full extent, which matches "
                                      "the byte column and is a lower bound on the read side. "
                                      "aligned_out=%s" % (list(out), tile_aligned(out)))
                return base
            return refuse("no input shape dominates the output on every axis")
        src = max(cand, key=_numel)
        base["params"] = {"src": list(src), "starts": [0] * len(src), "ends": list(out)}
        base["pinned"] = False
        base["derivation"] = ("extent READ from %s -> %s; OFFSETS NOT RECORDED, taken as zero. "
                              "aligned_src=%s aligned_out=%s"
                              % (src, out, tile_aligned(src), tile_aligned(out)))
        return base

    if arm == "pad":
        cand = [s for s in ins
                if len(s) == len(out) and all(a <= b for a, b in zip(s, out)) and s != out]
        if not cand:
            return refuse("no input shape is dominated by the output on every axis")
        src = min(cand, key=_numel)
        base["params"] = {"src": list(src), "padding": [[0, b - a] for a, b in zip(src, out)]}
        base["pinned"] = False
        base["derivation"] = ("extent READ from %s -> %s; the split between leading and trailing "
                              "pad is NOT recorded, taken as all-trailing. aligned_src=%s "
                              "aligned_out=%s" % (src, out, tile_aligned(src), tile_aligned(out)))
        return base

    if arm == "concat":
        for axis in range(len(out)):
            parts = [s for s in ins
                     if len(s) == len(out)
                     and all(d == o for j, (d, o) in enumerate(zip(s, out)) if j != axis)
                     and s[axis] < out[axis]]
            if parts and sum(p[axis] for p in parts) == out[axis]:
                base["params"] = {"dim": axis, "parts": [list(p) for p in parts]}
                base["derivation"] = ("axis %d INFERRED: the recorded inputs sum to the output "
                                      "on exactly that axis" % axis)
                return base
        # inputs may be deduped to one buffer reused n times
        for axis in range(len(out)):
            parts = [s for s in ins
                     if len(s) == len(out)
                     and all(d == o for j, (d, o) in enumerate(zip(s, out)) if j != axis)
                     and s[axis] < out[axis] and out[axis] % s[axis] == 0]
            if parts:
                p = max(parts, key=_numel)
                n = out[axis] // p[axis]
                base["params"] = {"dim": axis, "parts": [list(p)] * n}
                base["pinned"] = False
                base["derivation"] = ("axis %d INFERRED; the capture dedupes on buffer address so "
                                      "the %d parts are taken as %d copies of %s"
                                      % (axis, n, n, list(p)))
                return base
        return refuse("no axis on which the recorded inputs tile the output")

    if arm == "chunk":
        for axis in range(len(out)):
            cand = [s for s in ins
                    if len(s) == len(out)
                    and all(d == o for j, (d, o) in enumerate(zip(s, out)) if j != axis)
                    and s[axis] > out[axis] and s[axis] % out[axis] == 0]
            if cand:
                src = max(cand, key=_numel)
                base["params"] = {"dim": axis, "chunks": src[axis] // out[axis],
                                  "src": list(src)}
                base["derivation"] = ("axis %d and chunk count %d INFERRED from %s -> %s"
                                      % (axis, src[axis] // out[axis], src, out))
                return base
        return refuse("no axis on which a recorded input is an integer multiple of the output")

    if arm in ("to_layout", "to_memory_config_l1", "softmax", "cos"):
        cand = [r for r in site["ins"] if tuple(r["shape"]) == out]
        if not cand:
            cand = [r for r in site["ins"] if _numel(r["shape"]) == n_out]
        if not cand:
            return refuse("no input shape matches the output")
        r = cand[0]
        base["params"] = {"src": list(r["shape"]), "src_layout": r["layout"],
                          "src_buffer": r["buffer"]}
        if arm == "to_layout":
            tgt = "ROW_MAJOR" if r["layout"] == "TILE" else "TILE"
            base["params"]["target_layout"] = tgt
            base["pinned"] = False
            base["derivation"] = ("source layout %s READ from the capture; target layout NOT "
                                  "recorded, taken as the other one (%s)" % (r["layout"], tgt))
        elif arm == "to_memory_config_l1":
            base["params"]["target"] = "L1_INTERLEAVED"
            base["pinned"] = False
            base["derivation"] = ("target memory config NOT recorded per call; the fold's class "
                                 "name in true_floor.LAUNCH_ARM is to_memory_config_l1, so DRAM "
                                 "-> L1 interleaved is taken")
        else:
            base["params"]["dim"] = -1
            base["pinned"] = arm == "cos"
            base["derivation"] = ("shape READ; softmax reduction axis NOT recorded, taken as -1"
                                  if arm == "softmax" else "shape READ, no argument needed")
        return base

    if arm == "sdpa":
        four = [s for s in ins if len(s) == 4]
        q = [s for s in four if s == out]
        kv = [s for s in four if s[:2] == out[:2] and s[-1] == out[-1]]
        if not q or not kv:
            return refuse("could not identify q and k/v among the recorded 4D operands")
        Sk = max(s[2] for s in kv)
        mask = [s for s in ins if len(s) == 4 and s not in four[:0] and s[-1] == s[-2] == Sk
                and s != out]
        base["params"] = {"q": list(out), "kv": [out[0], out[1], Sk, out[3]],
                          "mask": list(max(mask, key=_numel)) if mask else None}
        base["derivation"] = ("q READ as the output shape; k/v S_k=%d READ from the recorded "
                              "operands; mask %s. scale and is_causal NOT recorded, taken as "
                              "default and False" % (Sk, "present" if mask else "absent"))
        base["pinned"] = False
        return base

    if arm == "nlp_create_qkv_heads":
        # one fused activation in, three head-major tensors out; heads and head_dim from the out
        cand = [s for s in ins if len(s) in (3, 4) and _numel(s) >= n_out]
        if not cand:
            return refuse("no recorded input large enough to be the fused qkv activation")
        src = max(cand, key=_numel)
        base["params"] = {"src": list(src), "out": list(out),
                          "num_heads": out[1], "head_dim": out[-1]}
        base["pinned"] = False
        base["derivation"] = ("fused input READ as %s, num_heads=%d and head_dim=%d INFERRED "
                              "from the output %s; whether the call was the fused-qkv or the "
                              "separate-kv overload is NOT recorded"
                              % (src, out[1], out[-1], out))
        return base

    if arm == "nlp_concat_heads":
        cand = [s for s in ins if len(s) == 4 and _numel(s) == n_out]
        if not cand:
            cand = [s for s in ins if len(s) == 4]
        if not cand:
            return refuse("no recorded 4D input to concat heads from")
        base["params"] = {"src": list(max(cand, key=_numel))}
        base["derivation"] = "shape READ, no argument needed"
        return base

    return refuse("no arm builder for class %r" % arm)


def build_all(keys):
    arms, refused = [], []
    for key, e in sorted(keys.items(), key=lambda kv: -kv[1]["B"]):
        if e["bucket"] not in ("refused", "noarm"):
            continue
        spec = build(key, e)
        (refused if "refused" in spec else arms).append(spec)
    return arms, refused
