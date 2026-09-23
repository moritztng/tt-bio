#!/usr/bin/env python3
"""of3t-cotcoh pass 2: the reference hook control, at full strength.

Pass 1 compared the hooked arm's per-tensor squared norms against the published FULL-MODEL
float64 bundle and read a worst relative difference of 34.998. That comparison is between a
trunk-on-captured-boundary run and a full-model run, so it cannot isolate the hook and pass 1
said so. This runs the SAME producer on the SAME boundary twice, once with `refcot.py`'s hooks
installed and once without, and compares the gradients tensor by tensor. At `--blocks 2` it
costs about a minute and answers the question the 34.998 could not.

Only `grads` is saved: `s` and `z` are 151 MB of float64 apiece and nothing reads them here.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "of3t_trunkg043"))
sys.path.insert(0, os.getcwd())
print("SYS_PATH resolved: " + repr(sys.path[:2]), flush=True)

FAM = {"A": "attn_pair_bias.layer_norm_a",
       "B": "pair_stack.pair_transition.layer_norm"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hooks", choices=("on", "off"), required=True)
    ap.add_argument("--grads-out", required=True)
    a, rest = ap.parse_known_args()
    passthrough = [x for x in rest if x != "--"]
    t0 = time.perf_counter()

    import torch
    import ref_grad

    OUTPATH = None
    for i, x in enumerate(passthrough):
        if x == "--out":
            OUTPATH = passthrough[i + 1]
    _real_save = torch.save

    def _save(obj, f, *ar, **kw):
        if isinstance(f, str) and OUTPATH is not None and \
                os.path.abspath(f) == os.path.abspath(OUTPATH):
            return _real_save({"grads": obj["grads"], "loss": obj.get("loss")}, a.grads_out,
                              *ar, **kw)
        return _real_save(obj, f, *ar, **kw)

    torch.save = _save

    if a.hooks == "on":
        CAP, FIRES = {}, {}
        R = 56
        _real_build = ref_grad.build

        def _mask(fam, t):
            t = t.detach()
            if fam == "A":
                return t.reshape(-1, t.shape[-1])[:R].contiguous()
            C = t.shape[-1]
            t = t.reshape(-1, t.shape[-2], C)
            return t[:R, :R].reshape(-1, C).contiguous()

        def build(sd, n, dtype):
            mods, dims, load = _real_build(sd, n, dtype)
            for i, m in enumerate(mods):
                for fam, sub in FAM.items():
                    def _mk(i=i, fam=fam):
                        def hook(mod, inp, out):
                            FIRES[(fam, i)] = FIRES.get((fam, i), 0) + 1
                            xm = _mask(fam, inp[0]).to(torch.float32).clone()

                            def _gh(gr, i=i, fam=fam, xm=xm):
                                CAP[(fam, i)] = {"g": _mask(fam, gr).to(torch.float64).clone(),
                                                 "x": xm}
                            out.register_hook(_gh)
                        return hook
                    m.get_submodule(sub).register_forward_hook(_mk())
            return mods, dims, load

        ref_grad.build = build

    sys.argv = ["ref_grad.py"] + passthrough
    rc = ref_grad.main()
    torch.save = _real_save
    print(json.dumps({"hooks": a.hooks, "grads_out": a.grads_out,
                      "host": socket.gethostname(),
                      "seconds": round(time.perf_counter() - t0, 1)}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
