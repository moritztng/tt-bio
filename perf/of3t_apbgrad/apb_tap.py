#!/usr/bin/env python3
"""Every op inside AttentionPairBias, recorded where the tape already sees it.

`of3t-bwdaccum` located the trunk's backward defect to this module and did not open it. This
does. The shipped `AttentionPairBias.__call__` is NOT copied and NOT rewritten -- the recorder
sits on `taped_ttnn._taped_verb`, which is the factory the shim builds every taped verb from,
so the arithmetic that runs is the shipped arithmetic and the only cost is a host read per
recorded tensor. `--no-tap` runs the identical chain with the recorder off, which is the A/A.

Per op, at the chosen blocks, it writes:

  the forward operand VALUES the device actually had, the cotangent the device handed that op's
  backward, and every `add_grad` contribution that backward made, tagged to the operand it
  landed on.

That is the full input to two scorings `score_apb.py` performs on the host in float64: each
op's backward recomputed from its own captured operands (ISOLATION), and the whole module's
backward recomputed from the module's captured operands and incoming cotangent (COMPOSED WALK).

WHERE THE BLOCK INDEX COMES FROM. The trunk runs each block through
`ops.checkpoint_segment`, so the taped forward of a block happens inside its own BACKWARD --
`autograd.checkpoint` runs `fn` untaped, then re-runs it taped from the node closure. This
wrapper therefore sets the current block on the checkpoint node's `fn`, and the recompute that
runs inside it carries that index. It is installed BEFORE `dev_cot`'s, so `dev_cot`'s
teacher-forcing wrapper ends up OUTSIDE this one: the reference cotangent is substituted first,
and what this records is the backward of the correct input.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_bwdaccum"))
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkg043"))
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_gradients"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap-blocks", default="4,15,22,31,39,47")
    ap.add_argument("--tap-out", default="")
    ap.add_argument("--no-tap", action="store_true",
                    help="A/A: the identical chain with the recorder not installed")
    a, rest = ap.parse_known_args()
    t0 = time.perf_counter()

    import torch
    import ttnn
    import tt_bio.ops as ops
    from tt_bio import autograd as ag
    import tt_bio.taped_ttnn as tt
    import tt_bio.tenstorrent as T

    CAP = {int(x) for x in a.tap_blocks.split(",") if x.strip() != ""}
    CUR = {"blk": None, "apb": False, "seq": None, "rec": None}
    RECS: dict = {}
    META: dict = {}
    ERR: list = []
    STATE = {"segments": 0, "taped": 0, "apb_calls": 0, "ops": 0}

    def th(t):
        if t is None:
            return None
        return ttnn.to_torch(t).to(torch.float64).clone()

    # ---- 1. every add_grad contribution, tagged to the op that made it ------------------
    _real_add_grad = ag.Tensor.add_grad

    def add_grad(self, grad):
        rec = CUR["rec"]
        if rec is not None and self.requires_grad:
            try:
                rec["contrib"].append([id(self), th(grad)])
            except Exception as e:                                          # noqa: BLE001
                ERR.append(f"contrib: {type(e).__name__}: {e}")
        return _real_add_grad(self, grad)

    # ---- 2. every taped op, recorded from the factory the shim builds them with ---------
    def _tensor_in(t):
        """The tape object for one operand, without creating one that did not exist."""
        if isinstance(t, ag.Tensor):
            return t, id(t)
        if isinstance(t, ttnn.Tensor):
            p = ag._param(t)
            if p is not None:
                return p, id(p)
            w = ag._WRAPPED.get(id(t))
            if w is not None and w.value is t:
                return w, id(w)
            return None, None
        return False, None

    def _record(qual, args, kwargs, out):
        ins = []
        flat = list(args) + [kwargs[k] for k in sorted(kwargs)]
        items = []
        for v in flat:
            items += list(v) if isinstance(v, (list, tuple)) else [v]
        for t in items:
            obj, oid = _tensor_in(t)
            if obj is False:
                continue
            raw = t.value if isinstance(t, ag.Tensor) else t
            ins.append({"id": oid, "val": th(raw), "dtype": str(raw.dtype),
                        "shape": [int(d) for d in raw.shape]})
        outs = [t for t in (out if isinstance(out, (tuple, list)) else [out])
                if isinstance(t, ag.Tensor)]
        rec = {"qual": qual, "idx": len(CUR["seq"]), "in": ins,
               "kw": {k: str(v) for k, v in kwargs.items()
                      if isinstance(v, (int, float, str, bool, type(None)))},
               "out": [{"id": id(t), "val": th(t.value), "dtype": str(t.value.dtype),
                        "shape": [int(d) for d in t.value.shape], "g": [], "contrib": []}
                       for t in outs]}
        CUR["seq"].append(rec)
        STATE["ops"] += 1
        for j, t in enumerate(outs):
            if t.node is None:
                continue
            orig = t.node.fn
            slot = rec["out"][j]

            def fn(g, orig=orig, slot=slot, rec=rec):
                slot["g"].append(th(g))
                prev, CUR["rec"] = CUR["rec"], slot
                try:
                    return orig(g)
                finally:
                    CUR["rec"] = prev

            t.node.fn = fn

    _real_tv = tt._taped_verb

    def _tv(qual, shipped):
        call = _real_tv(qual, shipped)

        def wrapped(*args, **kwargs):
            if not CUR["apb"]:
                return call(*args, **kwargs)
            out = call(*args, **kwargs)
            try:
                _record(qual, args, kwargs, out)
            except Exception as e:                                          # noqa: BLE001
                ERR.append(f"record {qual}: {type(e).__name__}: {e}")
            return out

        return wrapped

    def _clear_shim(sh):
        """Drop the shim's memoised verbs so the patched factory is the one that builds them."""
        for k in [k for k in list(sh.__dict__) if k not in ("_real", "_prefix")]:
            v = sh.__dict__[k]
            if isinstance(v, tt._Ttnn):
                _clear_shim(v)
            del sh.__dict__[k]

    # ---- 3. the module gate, and the weights the float64 walk needs ---------------------
    _real_apb = T.AttentionPairBias.__call__

    def apb_call(self, *ar, **kw):
        blk = CUR["blk"]
        if blk is None or blk not in CAP or CUR["apb"] or self.atom_level:
            return _real_apb(self, *ar, **kw)
        STATE["apb_calls"] += 1
        CUR["apb"], CUR["seq"] = True, []
        try:
            out = _real_apb(self, *ar, **kw)
        finally:
            CUR["apb"] = False
        seq = CUR["seq"]
        CUR["seq"] = None
        s_norm = ar[0] if ar else kw.get("s")
        z_arg = ar[1] if len(ar) > 1 else kw.get("z")
        if blk not in META:
            W = {n: th(getattr(self, n)) for n in
                 ("qkv_weight", "qkv_bias", "z_norm_weight", "z_norm_bias", "z_weight",
                  "g_weight", "o_weight") if getattr(self, n, None) is not None}
            META[blk] = {
                "weights": W,
                "n_heads": int(self.n_heads), "head_dim": int(self.head_dim),
                "padded_head_dim": int(getattr(self, "padded_head_dim", self.head_dim)),
                "bias_scale": float(self._bias_scale),
                "concat_heads": bool(self._concat_heads),
                "kq_norm": bool(getattr(self, "kq_norm", False)),
                "compute_pair_bias": bool(self.compute_pair_bias),
                "dtype": str(self.dtype),
                "accurate_softmax": bool(self.accurate_softmax),
                "fp32_softmax": bool(self.fp32_softmax),
                "seq_mask_given": kw.get("seq_mask") is not None,
            }
        RECS.setdefault(blk, []).append({
            "ops": seq,
            "s_in_id": id(s_norm) if isinstance(s_norm, ag.Tensor) else None,
            "s_in": th(s_norm.value if isinstance(s_norm, ag.Tensor) else s_norm),
            "z_in_id": id(z_arg) if isinstance(z_arg, ag.Tensor) else None,
            "z_in": th(z_arg.value if isinstance(z_arg, ag.Tensor) else z_arg),
            "x_out": th(out.value if isinstance(out, ag.Tensor) else out),
            "x_out_id": id(out) if isinstance(out, ag.Tensor) else None,
        })
        return out

    # ---- 4. the block index, from the checkpoint node --------------------------------
    _real_cs = ops.checkpoint_segment
    TAPED = []

    def _cs(fn, *inputs):
        out = _real_cs(fn, *inputs)
        STATE["segments"] += 1
        if not any(isinstance(t, ag.Tensor) for t in inputs):
            return out
        blk = len(TAPED)
        TAPED.append(blk)
        STATE["taped"] = len(TAPED)
        for t in (out if isinstance(out, (tuple, list)) else [out]):
            if not isinstance(t, ag.Tensor) or t.node is None:
                continue
            orig = t.node.fn

            def nfn(g, orig=orig, blk=blk):
                prev, CUR["blk"] = CUR["blk"], blk
                try:
                    return orig(g)
                finally:
                    CUR["blk"] = prev

            t.node.fn = nfn
        return out

    if not a.no_tap:
        ag.Tensor.add_grad = add_grad
        tt._taped_verb = _tv
        _clear_shim(tt._SHIM)
        T.AttentionPairBias.__call__ = apb_call
        ops.checkpoint_segment = _cs

    # ---- run of3t-bwdaccum's harness, which runs of3t-trunkg043's -----------------------
    import dev_cot
    sys.argv = ["dev_cot.py"] + [x for x in rest if x != "--"]
    rc = dev_cot.main()

    if a.tap_out and not a.no_tap:
        torch.save({"recs": RECS, "meta": META, "state": STATE, "errors": ERR,
                    "tap_blocks": sorted(CAP)}, a.tap_out)
    print(json.dumps({"tap_out": a.tap_out, "blocks": sorted(RECS),
                      "passes_per_block": {str(k): len(v) for k, v in sorted(RECS.items())},
                      "ops_per_block": {str(k): [len(p["ops"]) for p in v]
                                        for k, v in sorted(RECS.items())},
                      "state": STATE, "errors": ERR[:6],
                      "seconds": round(time.perf_counter() - t0, 1)}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
