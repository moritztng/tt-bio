#!/usr/bin/env python3
"""of3t-cotterm: `of3t-apbleaf/armln.py`, run unchanged, with the APB BOUNDARY captured too.

armln.py is not copied and not rewritten. This imports it, installs three patches on the
shipped classes BEFORE calling its `main()`, and lets it run -- so the arm, the levers, the
fused-five placement, the config spy, the LayerNorm capture and the whole scoring path stay the
ones that produced the banked trunk gradient, and `--lever none` must still come back
bit-identical to it. That equality is this row's instrument control.

The three patches, none of which touches arithmetic:

  1. `Pairformer.__init__` tags every `AttentionPairBias` instance with its block index, so the
     capture is keyed on the module rather than on call order. Call order is not usable here:
     the trunk runs each block through `ops.checkpoint_segment`, so the taped forward happens
     inside that block's own BACKWARD.
  2. `AttentionPairBias.__call__` reads its two inputs to the host BEFORE calling the shipped
     one -- `s_norm` (F's single-track activation) and `z` (the pair track), the latter
     projected to the raw additive bias in float64 by the same `apbmath.proj_bias` the
     reference arm uses. It also registers the returned tensor so the next patch can see it.
  3. `autograd.Tensor.add_grad` records the cotangent landing on those registered outputs,
     which is `do` -- the inherited channel. A tensor hook is not available here and
     `_retire` releases gradients, so the accumulation point is where it has to be read.

Only calls whose `s` is an `autograd.Tensor` are recorded: `dev_grad.py` runs a DISCOVERY
forward on raw handles first, and that one is not on the tape.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in ("", "perf/of3t_apbleaf", "perf/of3t_trunkg043", "perf/of3t_gradients", "perf/of3t_cotterm"):
    sys.path.insert(0, os.path.join(_ROOT, _p) if _p else _ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apb-out", required=True)
    ap.add_argument("--real-rows", type=int, default=0,
                    help="slice the stored pair-track activation to the first R rows/columns; "
                         "the bias projection is always stored at full width")
    a, rest = ap.parse_known_args()
    t0 = time.perf_counter()

    import torch
    import ttnn
    from tt_bio import autograd as ag
    import tt_bio.tenstorrent as T

    import apbmath
    import armln

    CKPT = "/home/ttuser/of3-weights/of3-p2-155k.pt"
    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    PRE = "pairformer_stack.blocks."

    CAP: dict = {}
    OUTREG: dict = {}
    HOLD: list = []
    ERR: list = []
    STATE = {"apb_calls": 0, "taped_calls": 0, "tagged": 0, "do_hits": 0}

    def th(t, dtype=torch.float64):
        return ttnn.to_torch(t).to(dtype).clone()

    # ---- 1. the block index, from the module rather than from call order ----------------
    _real_pf_init = T.Pairformer.__init__

    def pf_init(self, *args, **kwargs):
        _real_pf_init(self, *args, **kwargs)
        for i, blk in enumerate(self.blocks):
            apb = getattr(blk, "attention_pair_bias", None)
            if apb is not None:
                apb._cotterm_blk = i
                STATE["tagged"] += 1

    T.Pairformer.__init__ = pf_init

    # ---- 2. the two inputs, read before the shipped call --------------------------------
    _real_call = T.AttentionPairBias.__call__

    def apb_call(self, s, z, keys_indexing=None, seq_mask=None, bias_precomputed=False):
        STATE["apb_calls"] += 1
        i = getattr(self, "_cotterm_blk", None)
        taped = isinstance(s, ag.Tensor) and i is not None
        if taped:
            STATE["taped_calls"] += 1
            try:
                zv = z.value if isinstance(z, ag.Tensor) else z
                zt = th(zv)
                W = apbmath.weights_from_sd(sd, f"{PRE}{i}.attn_pair_bias.")
                e = CAP.setdefault(i, {})
                e["a"] = th(s.value)
                e["bias"] = apbmath.proj_bias(zt, W["w_ln_z"], W["b_ln_z"], W["w_lz"])
                e["z_real"] = (zt[:, :a.real_rows, :a.real_rows] if a.real_rows
                               else zt).to(torch.float32).clone()
                del zt
            except Exception as exc:                                       # noqa: BLE001
                ERR.append(f"apb in {i}: {type(exc).__name__}: {exc}")
        out = _real_call(self, s, z, keys_indexing, seq_mask, bias_precomputed)
        if taped and isinstance(out, ag.Tensor):
            OUTREG[id(out)] = i
            HOLD.append(out)          # keep the object alive so its id cannot be reused
        return out

    T.AttentionPairBias.__call__ = apb_call

    # ---- 3. the cotangent landing on the module's output --------------------------------
    _real_add_grad = ag.Tensor.add_grad

    def add_grad(self, grad):
        i = OUTREG.get(id(self))
        if i is not None:
            try:
                g = th(grad.value if isinstance(grad, ag.Tensor) else grad)
                e = CAP.setdefault(i, {})
                e["do"] = g if "do" not in e else e["do"] + g
                STATE["do_hits"] += 1
            except Exception as exc:                                       # noqa: BLE001
                ERR.append(f"do in {i}: {type(exc).__name__}: {exc}")
        return _real_add_grad(self, grad)

    ag.Tensor.add_grad = add_grad

    sys.argv = ["armln.py"] + rest
    rc = armln.main()

    sites = {i: e for i, e in sorted(CAP.items())
             if all(k in e for k in ("a", "bias", "do"))}
    torch.save({"sites": sites, "host": os.uname().nodename, "real_rows": a.real_rows,
                "state": STATE, "errors": ERR}, a.apb_out)
    print(json.dumps({"apb_out": a.apb_out, "sites": len(sites),
                      "host": os.uname().nodename, "state": STATE,
                      "missing": [i for i in range(48) if i not in sites],
                      "errors": ERR[:4],
                      "seconds": round(time.perf_counter() - t0, 1)}), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
