#!/usr/bin/env python3
"""bcx-p10-devmap: where one AF2 block's DEVICE seconds go, at the shipped configuration.

The device side of a BindCraft 2 round is 52 copies of one pair block (48 Evoformer + 4
extra-MSA), so this attributes ONE block and multiplies. Three numbers per op family, all
measured in the same process on the same card:

  synced   `ttnn.synchronize_device` after every verb, so the host wall around the verb is
           enqueue + device execution + one sync round trip. This is the honest device timer
           (`ttnn-perf-profiling` SKILL.md section 0) and it is what the attribution table uses.
  free     the same verbs with no sync at all. The wall is then the ENQUEUE cost alone, which
           is the ttnn dispatch loop HOSTMAP priced at 3.55 s inside the device column.
  lambda   `synchronize_device` on an already-idle device, measured directly, so the synced
           timer's own floor is subtracted rather than assumed.

  device_i = synced_i - free_i - lambda

Bytes are the operand and result bytes on the PADDED device shape, which is what the card
moves. FLOPs are analytic, from AF2's own dimensions, not counted per verb: the fused kernels
issue one `generic_op` whose arithmetic a verb counter cannot see.

Nothing here changes what the model computes. Every wrapper calls through.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_stack"))
from perf.bcx_stack import stack as S            # noqa: E402
from perf.bcx_afgrad import afgrad as A          # noqa: E402

OUT = ROOT / "perf" / "bcx_p10_devmap"

#: bytes per element on the card. bfp8_b is a 16-element block sharing one exponent byte.
ELEM = {"BFLOAT16": 2.0, "FLOAT32": 4.0, "BFLOAT8_B": 17.0 / 16.0, "BFLOAT4_B": 9.0 / 16.0,
        "UINT32": 4.0, "INT32": 4.0, "UINT16": 2.0, "UINT8": 1.0, "FLOAT64": 8.0}

VERBS = [
    "matmul", "linear", "add", "add_", "subtract", "subtract_", "multiply", "multiply_",
    "div", "divide", "layer_norm", "rms_norm", "softmax", "softmax_in_place", "sigmoid",
    "relu", "silu", "gelu", "typecast", "clone", "permute", "transpose", "concat", "reshape",
    "to_layout", "slice", "pad", "sum", "mean", "max", "mul", "neg", "exp", "reciprocal",
    "rsqrt", "sqrt", "abs", "clamp", "cos", "pow", "eq", "lt", "ge", "gtz", "where", "rsub",
    "addcmul", "addalpha", "repeat", "repeat_interleave", "embedding", "embedding_bw",
    "unsqueeze", "squeeze", "chunk", "split", "zeros", "zeros_like", "ones_like", "empty",
    "full", "arange", "to_memory_config", "reallocate", "fill_implicit_tile_padding",
    "moreh_softmax_backward", "relu_bw", "silu_bw", "sigmoid_bw", "mul_bw", "generic_op",
    "experimental.minimal_matmul", "experimental.nlp_concat_heads",
    "experimental.nlp_create_qkv_heads", "experimental.view", "experimental.rotary_embedding",
    "transformer.scaled_dot_product_attention", "transformer.concatenate_heads",
    "transformer.split_query_key_value_and_split_heads",
]

# ------------------------------------------------------------------ the label context

#: What the op currently running belongs to. The forward sets it from the block's own op
#: names; the backward restores each tape node's tag from when it was recorded.
CUR = {"stack": "-", "block": -1, "dir": "fwd", "family": "outside"}


def _tag():
    return f'{CUR["stack"]}|{CUR["block"]}|{CUR["dir"]}|{CUR["family"]}'


def install_labels():
    """Tag every ttnn verb with the AF2 op that issued it, in both directions.

    Forward: `AF2PairBlock._update` already names every op class the block runs, and
    `_residual` is the wide add between them. Backward: a tape node captures the tag that was
    current when it was RECORDED, and restores it around its own closure, so a verb issued by
    `tri_att_start`'s vjp is charged to `tri_att_start` no matter where in the tape it fires.
    """
    from tt_bio import af2, autograd

    PB = af2.AF2PairBlock
    orig_update, orig_res = PB._update, PB._residual

    def _update(self, name, device, x, *args):
        prev, CUR["family"] = CUR["family"], name
        try:
            return orig_update(self, name, device, x, *args)
        finally:
            CUR["family"] = prev
    PB._update = _update

    def _residual(self, x, update):
        prev, CUR["family"] = CUR["family"], "residual"
        try:
            return orig_res(self, x, update)
        finally:
            CUR["family"] = prev
    PB._residual = _residual

    EB = af2.AF2EvoformerBlock
    orig_bias = EB._mask_biases

    def _mask_biases(self, msa_mask):
        prev, CUR["family"] = CUR["family"], "mask_bias"
        try:
            return orig_bias(self, msa_mask)
        finally:
            CUR["family"] = prev
    EB._mask_biases = _mask_biases

    D = A.Dev
    orig_evo, orig_extra = D.evo, D.extra

    def evo(self, i, *a, **kw):
        CUR["stack"], CUR["block"] = "evo", i
        try:
            return orig_evo(self, i, *a, **kw)
        finally:
            CUR["stack"], CUR["block"], CUR["family"] = "-", -1, "outside"
    D.evo = evo

    def extra(self, i, *a, **kw):
        CUR["stack"], CUR["block"] = "extra", i
        try:
            return orig_extra(self, i, *a, **kw)
        finally:
            CUR["stack"], CUR["block"], CUR["family"] = "-", -1, "outside"
    D.extra = extra

    Base = autograd._Node

    class TaggedNode(Base):
        __slots__ = ()

        def __init__(self, fn, parents, group=None):
            tag = (CUR["stack"], CUR["block"], CUR["family"])

            def wrapped(g, _fn=fn, _tag=tag):
                prev = (CUR["stack"], CUR["block"], CUR["family"])
                CUR["stack"], CUR["block"], CUR["family"] = _tag
                try:
                    return _fn(g)
                finally:
                    CUR["stack"], CUR["block"], CUR["family"] = prev
            super().__init__(wrapped, parents, group)
    autograd._Node = TaggedNode

    orig_backward = autograd._backward

    def _backward(roots, seeds):
        prev = (CUR["stack"], CUR["block"], CUR["dir"], CUR["family"])
        CUR["stack"], CUR["block"], CUR["dir"], CUR["family"] = "-", -1, "bwd", "tape_engine"
        try:
            return orig_backward(roots, seeds)
        finally:
            (CUR["stack"], CUR["block"], CUR["dir"], CUR["family"]) = prev
    autograd._backward = _backward


# ------------------------------------------------------------------ the op timer


def _nbytes(t):
    try:
        dt = str(t.dtype).rsplit(".", 1)[-1].upper()
    except Exception:
        return 0.0
    per = ELEM.get(dt)
    if per is None:
        return 0.0
    try:
        shape = list(t.padded_shape)
    except Exception:
        try:
            shape = list(t.shape)
        except Exception:
            return 0.0
    n = 1
    for d in shape:
        n *= int(d)
    return n * per


class OpTimer:
    """Every ttnn verb: wall, operand bytes, result bytes, by (stack, block, dir, family).

    `sync=True` puts a `synchronize_device` inside the timed region, which is the only way a
    host clock around an asynchronous enqueue means anything. `sync=False` times the same
    verbs with nothing after them, which is the enqueue cost on its own. The bookkeeping runs
    OUTSIDE the timed region in both modes, so the instrument does not charge itself.
    """

    def __init__(self, ttnn, device):
        self.ttnn, self.device = ttnn, device
        self.sync = False
        self.on = False
        self.wall = collections.Counter()
        self.calls = collections.Counter()
        self.read = collections.Counter()
        self.written = collections.Counter()
        self.read_dtype = collections.Counter()
        self.verb_wall = collections.Counter()
        self.verb_calls = collections.Counter()
        self._saved = []

    def _walk(self, obj, out):
        T = self.ttnn.Tensor
        if isinstance(obj, T):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                self._walk(o, out)
        elif isinstance(obj, dict):
            for o in obj.values():
                self._walk(o, out)

    def install(self):
        for path in VERBS:
            parent, _, leaf = path.rpartition(".")
            mod = self.ttnn
            ok = True
            for part in parent.split(".") if parent else []:
                mod = getattr(mod, part, None)
                if mod is None:
                    ok = False
                    break
            if not ok or not hasattr(mod, leaf):
                continue
            real = getattr(mod, leaf)
            if not callable(real):
                continue
            self._saved.append((mod, leaf, real))
            setattr(mod, leaf, self._wrap(path, real))
        from tt_bio import taped_ttnn as T
        T.forget_shim_bindings()
        return self

    def _wrap(self, path, real):
        def w(*args, **kwargs):
            if not self.on:
                return real(*args, **kwargs)
            ins = []
            self._walk(args, ins)
            self._walk(kwargs, ins)
            rb = 0.0
            dts = collections.Counter()
            for t in ins:
                b = _nbytes(t)
                rb += b
                try:
                    dts[str(t.dtype).rsplit(".", 1)[-1].upper()] += b
                except Exception:
                    pass
            tag = _tag()
            sync = self.sync
            dev = self.device
            t0 = time.perf_counter()
            out = real(*args, **kwargs)
            if sync:
                self.ttnn.synchronize_device(dev)
            t1 = time.perf_counter()
            outs = []
            self._walk(out, outs)
            self.wall[tag] += t1 - t0
            self.calls[tag] += 1
            self.read[tag] += rb
            self.written[tag] += sum(_nbytes(t) for t in outs)
            for d, b in dts.items():
                self.read_dtype[(tag, d)] += b
            self.verb_wall[(tag, path)] += t1 - t0
            self.verb_calls[(tag, path)] += 1
            return out
        w.__name__ = getattr(real, "__name__", path)
        return w

    def uninstall(self):
        for mod, leaf, real in self._saved:
            setattr(mod, leaf, real)
        from tt_bio import taped_ttnn as T
        T.forget_shim_bindings()

    def take(self):
        snap = {"wall": dict(self.wall), "calls": dict(self.calls), "read": dict(self.read),
                "written": dict(self.written),
                "read_dtype": {f"{k}||{d}": v for (k, d), v in self.read_dtype.items()},
                "verb_wall": {f"{k}||{p}": v for (k, p), v in self.verb_wall.items()},
                "verb_calls": {f"{k}||{p}": v for (k, p), v in self.verb_calls.items()}}
        for c in (self.wall, self.calls, self.read, self.written, self.read_dtype,
                  self.verb_wall, self.verb_calls):
            c.clear()
        return snap


def sync_floor(ttnn, device, reps=400):
    """`synchronize_device` on an idle device: the synced timer's own floor, measured."""
    ttnn.synchronize_device(device)
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.synchronize_device(device)
        xs.append(time.perf_counter() - t0)
    return S.dist(xs)


# ------------------------------------------------------------------ analytic FLOPs

#: AF2's own dimensions for the trunk this campaign runs (`af2_reference.EvoformerBlock`).
DIMS = {"c_m": 256, "c_z": 128, "trimul_hidden": 128, "pair_heads": 4, "pair_head_dim": 32,
        "msa_heads": 8, "msa_head_dim": 32, "opm_hidden": 32, "factor": 4}


def analytic_flops(n, depth=1, d=DIMS):
    """FLOPs of one Evoformer block's forward, per op family, from AF2's arithmetic.

    A matmul of [M,K]x[K,N] is 2MNK. Only the matmuls are counted: the layer norms, the
    sigmoids and the softmaxes are O(elements) and three orders under the contractions.
    The backward of a matmul is two matmuls of the same size, so a taped block's
    fwd+fwd+bwd (2 forwards, 1 backward per round) is 2*F + 2*F = 4*F to a good approximation,
    stated as a separate column rather than folded in.
    """
    cz, cm = d["c_z"], d["c_m"]
    h, hd = d["pair_heads"], d["pair_head_dim"]
    out = {}
    # triangle multiplication: two projections in (n^2 cz -> n^2 hidden, x2 for a+b and gates),
    # the n^3 contraction, the output projection and its gate.
    hid = d["trimul_hidden"]
    proj = 4 * 2 * (n * n) * cz * hid          # a, b, gate_a, gate_b
    contract = 2 * n * n * n * hid
    outproj = 2 * (n * n) * hid * cz + 2 * (n * n) * cz * cz   # proj_o + output gate
    out["tri_mul_out"] = out["tri_mul_in"] = proj + contract + outproj
    # triangle attention: q/k/v/g projections, the n^3 logits, the n^3 weighted sum, the
    # pair-bias projection and the output projection.
    qkvg = 4 * 2 * (n * n) * cz * (h * hd)
    logits = 2 * n * n * n * h * hd
    weighted = 2 * n * n * n * h * hd
    bias = 2 * (n * n) * cz * h
    o = 2 * (n * n) * (h * hd) * cz
    out["tri_att_start"] = out["tri_att_end"] = qkvg + logits + weighted + bias + o
    # pair transition: two matmuls through 4x
    out["pair_transition"] = 2 * 2 * (n * n) * cz * (d["factor"] * cz)
    # MSA row attention with pair bias: q/k/v/g over depth rows, the n^2 logits per row,
    # and the pair-bias projection which is n^2 and does NOT scale with depth.
    mh, mhd = d["msa_heads"], d["msa_head_dim"]
    out["msa_row_attn"] = (4 * 2 * depth * n * cm * (mh * mhd)
                           + 2 * 2 * depth * n * n * mh * mhd
                           + 2 * (n * n) * cz * mh
                           + 2 * depth * n * (mh * mhd) * cm)
    # MSA column attention attends over the depth axis: at depth 1 it is projections only.
    out["msa_col_attn"] = (4 * 2 * depth * n * cm * (mh * mhd)
                           + 2 * 2 * depth * depth * n * mh * mhd
                           + 2 * depth * n * (mh * mhd) * cm)
    out["msa_transition"] = 2 * 2 * depth * n * cm * (d["factor"] * cm)
    # outer product mean: two projections to the hidden width, then the n^2 outer product.
    oh = d["opm_hidden"]
    out["opm"] = (2 * 2 * depth * n * cm * oh + 2 * (n * n) * depth * oh * oh
                  + 2 * (n * n) * (oh * oh) * cz)
    out["residual"] = 0
    return out


# ------------------------------------------------------------------ the run


def _round(d, p=6):
    return {k: round(v, p) for k, v in d.items()}


def cmd_block(args):
    """Per-family device seconds for one block of each stack, both directions.

    K = 1 and K = 2 are both run so a per-block number can be taken as the difference and
    everything outside the blocks cancels; the tags already separate the blocks, so the
    difference is a control on the tags rather than the only route to them.
    """
    import ttnn
    lv, dev, ref = S.open_all(args)
    lv.mask = True                      # the design path hands the Evoformer an MSA mask
    install_labels()
    timer = OpTimer(ttnn, dev.device).install()
    clock = S.Clock()
    m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)
    stacks = args.stacks.split(",")
    ks = [int(k) for k in args.ks.split(",")]

    blob = {"stamp": S.stamp(args, clock), "n": args.n, "seed": args.seed,
            "stacks": stacks, "ks": ks, "reps": args.reps,
            "dims": DIMS, "flops_fwd_analytic": analytic_flops(args.n, args.depth),
            "flops_fwd_analytic_padded": analytic_flops(args.pad or args.n, args.depth),
            "loadavg_start": os.getloadavg(), "records": [], "verb": {}}

    for stack in stacks:                                   # warm: JIT + program cache
        for k in ks:
            S.block_step(dev, lv, m0, z0, wm, wz, stack, k=k)
    blob["sync_floor_s"] = sync_floor(ttnn, dev.device)
    print(json.dumps({"sync_floor_median_s": blob["sync_floor_s"]["median"]}), flush=True)

    verb_wall = collections.Counter()
    verb_calls = collections.Counter()
    for rep in range(args.reps):
        modes = ["free", "sync"] if rep % 2 == 0 else ["sync", "free"]
        for mode in modes:
            timer.sync = (mode == "sync")
            for stack in stacks:
                for k in ks:
                    timer.on = True
                    r, _ = S.block_step(dev, lv, m0, z0, wm, wz, stack, k=k)
                    timer.on = False
                    snap = timer.take()
                    rec = {"rep": rep, "mode": mode, "stack": stack, "K": k,
                           "wall_fwd": round(r["fwd"], 6), "wall_bwd": round(r["bwd"], 6),
                           "bwd_cpu": round(r["bwd_cpu"], 6),
                           "loadavg": os.getloadavg()[0],
                           "aiclk": clock.window(r["spans"]),
                           "wall": _round(snap["wall"]), "calls": snap["calls"],
                           "read": _round(snap["read"], 1),
                           "written": _round(snap["written"], 1),
                           "read_dtype": _round(snap["read_dtype"], 1)}
                    blob["records"].append(rec)
                    if mode == "sync" and k == max(ks):
                        for key, v in snap["verb_wall"].items():
                            verb_wall[key] += v
                        for key, v in snap["verb_calls"].items():
                            verb_calls[key] += v
                    print(json.dumps({"rep": rep, "mode": mode, "stack": stack, "K": k,
                                      "fwd": round(r["fwd"], 3), "bwd": round(r["bwd"], 3),
                                      "op_wall": round(sum(snap["wall"].values()), 3),
                                      "ops": sum(snap["calls"].values()),
                                      "aiclk": rec["aiclk"]}), flush=True)
    blob["verb"] = {"wall": _round(verb_wall), "calls": dict(verb_calls),
                    "reps": args.reps, "note": "sync mode, K=max, summed over reps"}
    blob["loadavg_end"] = os.getloadavg()
    clock.stop()
    timer.uninstall()
    S.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (args.out or f"block_n{args.n}.json")).write_text(json.dumps(blob, indent=1,
                                                                       default=str))
    print(f"wrote {OUT / (args.out or f'block_n{args.n}.json')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["block"])
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--n", type=int, default=275)
    ap.add_argument("--pad", type=int, default=288,
                    help="the padded device axis the card actually computes on")
    ap.add_argument("--depth", type=int, default=1, help="MSA rows; design is single-sequence")
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"block": cmd_block}[args.cmd](args)


if __name__ == "__main__":
    main()
