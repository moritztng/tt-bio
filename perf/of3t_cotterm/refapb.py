#!/usr/bin/env python3
"""of3t-cotterm: `of3t-apbleaf/refln.py`, run unchanged, with the APB BOUNDARY captured too.

refln.py is not copied and not rewritten. This imports it, patches its `build()` to register
three hooks per block on the modules it just constructed, and calls its `main()` -- so the
reference arithmetic, the tree, the boundary, the cotangent, the checkpointing and the 2,736
parameter gradients are the ones that produced `REF_LN_N384.json`, and the control is that
they come back bit-identical to it.

What the hooks add, per block, on the float64 reference:

  a_ref     `attn_pair_bias.layer_norm_a`'s OUTPUT -- the single-track activation the
            attention actually reads, which is F's first argument
  bias_ref  the raw additive pair bias, `permute(linear_z(layer_norm_z(z)))`, evaluated in
            float64 from the reference's own `z` at APB's input. 16 channels per pair instead
            of 128, and `z` reaches the cotangent through nothing else
  do_ref    the cotangent arriving at APB's OUTPUT -- the inherited channel

`refln.py --ln-out` already writes `x`, `g`, `gamma`, `beta` and the two parameter gradients
per site, so `g_ref` is taken from there rather than captured twice.

The forward hook fires twice per site under `--checkpoint` (refln's `fires` field records it):
the first firing is inside `torch.no_grad`, so the guard is the same one refln uses -- record
only the graph-connected call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import trees                                                              # noqa: E402

_ROOT = trees.ROOT
_PATHS = trees.install(("", "perf/of3t_apbleaf", "perf/of3t_cotterm"))

import torch                                                              # noqa: E402

import apbmath                                                            # noqa: E402
import refln                                                              # noqa: E402

PRE = refln.PRE
CAP: dict = {}
ROWS = [0]
FIRES: dict = {}
ERR: list = []


def install(mods, sd, keep_z_block):
    for i, m in enumerate(mods):
        apb = m.get_submodule("attn_pair_bias")
        W = apbmath.weights_from_sd(sd, f"{PRE}{i}.attn_pair_bias.")

        def _apb_hook(mod, args, kwargs, out, i=i, W=W):
            FIRES[i] = FIRES.get(i, 0) + 1
            if not (torch.is_grad_enabled() and torch.is_tensor(out) and out.requires_grad):
                return
            z = kwargs["z"] if "z" in kwargs else args[1]
            try:
                e = CAP.setdefault(i, {})
                e["bias"] = apbmath.proj_bias(z.detach(), W["w_ln_z"], W["b_ln_z"], W["w_lz"])
                if i == keep_z_block:
                    e["z_full"] = z.detach().to(torch.float64).clone()
                zr = z.detach()
                if ROWS[0]:
                    zr = zr[:, :ROWS[0], :ROWS[0]]
                e["z_real"] = zr.to(torch.float32).clone()

                def _dh(gr, i=i):
                    CAP.setdefault(i, {})["do"] = gr.detach().to(torch.float64).clone()

                out.register_hook(_dh)
            except Exception as exc:                                       # noqa: BLE001
                ERR.append(f"apb {i}: {type(exc).__name__}: {exc}")

        def _ln_hook(mod, args, out, i=i):
            if not (torch.is_grad_enabled() and out.requires_grad):
                return
            CAP.setdefault(i, {})["a"] = out.detach().to(torch.float64).clone()

        apb.register_forward_hook(_apb_hook, with_kwargs=True)
        apb.get_submodule("layer_norm_a").register_forward_hook(_ln_hook)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apb-out", required=True)
    ap.add_argument("--keep-z-block", type=int, default=44,
                    help="the one block whose full float64 z is kept, so split.py can check "
                         "apbmath.proj_bias against upstream's own two modules")
    ap.add_argument("--real-rows", type=int, default=0,
                    help="if set, z_real is sliced to the first R rows/columns before storing; "
                         "0 keeps the pair-track activation reading off this file")
    a, rest = ap.parse_known_args()
    ROWS[0] = a.real_rows
    t0 = time.perf_counter()

    real_build = refln.build

    def build(sd, n, dt):
        out = real_build(sd, n, dt)
        install(out[0], sd, a.keep_z_block)
        return out

    refln.build = build
    sys.argv = ["refln.py"] + rest
    rc = refln.main()

    # `refln.main()` is what puts the reference tree on sys.path and imports openfold3, so the
    # resolution is read back AFTER it rather than from the constant that asked for it. refln
    # already refuses a tree that is not the one it was pointed at; this records which one
    # answered, in this row's own artifact (D149).
    TREES = trees.resolved()

    R = a.real_rows
    sites = {i: e for i, e in sorted(CAP.items())
             if all(k in e for k in ("a", "bias", "do"))}
    torch.save({"sites": sites, "policy": "f64", "host": os.uname().nodename,
                "keep_z_block": a.keep_z_block, "real_rows": R, "trees": TREES,
                "fires": sorted(set(FIRES.values())), "errors": ERR}, a.apb_out)
    print(json.dumps({"apb_out": a.apb_out, "sites": len(sites),
                      "host": os.uname().nodename, "trees": TREES,
                      "missing": [i for i in range(48) if i not in sites],
                      "fires": sorted(set(FIRES.values())), "errors": ERR[:4],
                      "seconds": round(time.perf_counter() - t0, 1)}), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
