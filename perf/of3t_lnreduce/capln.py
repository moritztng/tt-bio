#!/usr/bin/env python3
"""of3t-lnreduce step 2a: the real LayerNorm affine operands on the MODEL-FRAME boundary, plus
a RUNTIME census of every reduction the backward actually performs.

This does not rebuild `perf/of3t_modelframe/`. It drives `of3t-trunkg043/dev_grad.py` -- the same
harness `perf/of3t_modelframe/runarm.sh` drives through `perf/of3t_bwdaccum/dev_cot.py --lever
none` -- over the boundary and cotangent that row captured from the reference's own full-model
float64 backward, and installs two instruments and no arithmetic:

  1. `ttnn.sum` is wrapped. Every call is recorded with its operand shape, dtype, buffer type and
     compute_kernel_config, and its OUTPUT dtype. That is the ACCUM count, taken at runtime on a
     real backward, because a source read is not the answer.
  2. `ag._taped_layer_norm` is WRAPPED, not replaced: the wrapper calls the shipped verb, so the
     forward and the backward arithmetic are the shipped ones bit for bit, and only the tape
     node's function is decorated to write out `x`, `g`, `gamma` and the device's own dW/dB at
     the sites asked for. `--census-only` writes the census and no operands.

Operands are stored in the dtype the card holds them in, so the file carries the exact values the
reduction consumed and not a re-rounding of them.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkg043"))
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_gradients"))
# D149: read the resolution back rather than trusting the insert.
print("SYS_PATH resolved: " + repr(sys.path[:3]), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ln-out", default="")
    ap.add_argument("--census", required=True)
    ap.add_argument("--match", default="",
                    help="capture operands only at sites whose resolved weight path contains "
                         "this substring; empty means capture nothing")
    ap.add_argument("--max-sites", type=int, default=8)
    a, rest = ap.parse_known_args()
    passthrough = [x for x in rest if x != "--"]
    t0 = time.perf_counter()

    import torch
    import ttnn
    from tt_bio import autograd as ag
    import tt_bio.taped_ttnn as tt
    import tt_bio.tenstorrent as T

    # ---- instrument 1: the reduction census ---------------------------------------------
    CENSUS = {}
    _real_sum = ttnn.sum

    def _cfg_fields(c):
        if c is None:
            return None
        out = {}
        for f in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc",
                  "dst_full_sync_en"):
            try:
                out[f] = str(getattr(c, f))
            except Exception:
                pass
        return out or str(type(c).__name__)

    def _sum(t, *ar, **kw):
        out = _real_sum(t, *ar, **kw)
        try:
            key = json.dumps({
                "operand_shape": [int(d) for d in t.shape],
                "operand_dtype": str(t.dtype),
                "buffer_type": str(t.memory_config().buffer_type),
                "layout": str(t.layout),
                "dim": kw.get("dim", ar[0] if ar else None),
                "keepdim": kw.get("keepdim"),
                "dtype_requested": str(kw.get("dtype")),
                "compute_kernel_config": _cfg_fields(kw.get("compute_kernel_config")),
                "out_shape": [int(d) for d in out.shape],
                "out_dtype": str(out.dtype),
            }, sort_keys=True)
            CENSUS[key] = CENSUS.get(key, 0) + 1
        except Exception as e:                      # an instrument must not break the arm
            CENSUS["INSTRUMENT_ERROR: " + type(e).__name__] = \
                CENSUS.get("INSTRUMENT_ERROR: " + type(e).__name__, 0) + 1
        return out

    ttnn.sum = _sum

    # ---- the module handle, so a gamma can be named --------------------------------------
    MOD = []
    _RealPF = T.Pairformer

    class _PF(_RealPF):
        def __init__(self, *ar, **kw):
            super().__init__(*ar, **kw)
            MOD.append(self)

    T.Pairformer = _PF
    WPATH = {}

    def _name_of(v):
        if not WPATH and MOD:
            from tt_bio.tenstorrent import device_weights
            for p, w in device_weights(MOD[0]).items():
                try:
                    WPATH[w.buffer_address()] = p
                except Exception:
                    pass
        try:
            return WPATH.get(v.buffer_address())
        except Exception:
            return None

    # ---- instrument 2: wrap the shipped verb, do not replace it ---------------------------
    CAPTURED = []
    SITES = {}
    _shipped_verb = ag._taped_layer_norm

    def _verb(shipped, args, kwargs):
        argl = list(args) + [None] * (3 - len(args))
        xw = ag._wrap(argl[0])
        gw = ag._wrap(kwargs.get("weight", argl[1]))
        bw_ = ag._wrap(kwargs.get("bias", argl[2]))
        out = _shipped_verb(shipped, args, kwargs)
        if out is None or getattr(out, "node", None) is None or gw is None:
            return out
        path = _name_of(gw.value)
        nm = path or ("unnamed_gamma_w%d" % int(gw.value.shape[-1]))
        want = bool(a.match) and (path is not None) and (a.match in path) \
            and len(CAPTURED) < a.max_sites
        orig = out.node.fn

        def fn(g, orig=orig, xw=xw, gw=gw, bw_=bw_, nm=nm, want=want):
            if not want:
                return orig(g)
            xv, gv = xw.value, g
            rec = {"gamma_path": nm,
                   "x": ttnn.to_torch(xv).clone(), "x_dtype": str(xv.dtype),
                   "g": ttnn.to_torch(gv).clone(), "g_dtype": str(gv.dtype),
                   "gamma": ttnn.to_torch(gw.value).to(torch.float32).clone(),
                   "beta": (None if bw_ is None
                            else ttnn.to_torch(bw_.value).to(torch.float32).clone()),
                   "gamma_before": (None if gw.grad is None
                                    else ttnn.to_torch(gw.grad).to(torch.float64).clone()),
                   "beta_before": (None if bw_ is None or bw_.grad is None
                                   else ttnn.to_torch(bw_.grad).to(torch.float64).clone())}
            r = orig(g)
            rec["gamma_after"] = (None if gw.grad is None
                                 else ttnn.to_torch(gw.grad).to(torch.float64).clone())
            rec["beta_after"] = (None if bw_ is None or bw_.grad is None
                                 else ttnn.to_torch(bw_.grad).to(torch.float64).clone())
            CAPTURED.append(rec)
            print("CAPTURED %s x=%s g=%s" % (nm, tuple(rec["x"].shape),
                                             tuple(rec["g"].shape)), flush=True)
            return r

        out.node.fn = fn
        SITES[nm] = SITES.get(nm, 0) + 1
        return out

    ag._TAPED["layer_norm"] = _verb
    tt._VERBS["layer_norm"] = _verb

    import dev_grad
    sys.argv = ["dev_grad.py"] + passthrough
    rc = dev_grad.main()

    cen = {"what": "every ttnn.sum the trunk backward performed, at runtime, on the "
                   "model-frame boundary. The gamma/beta gradients of every LayerNorm land "
                   "through tt_bio/autograd.py::_sum_leading, which is this call.",
           "host": socket.gethostname(),
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
           "argv": passthrough, "match": a.match,
           "seconds": round(time.perf_counter() - t0, 1),
           "layer_norm_sites_taped": len(SITES),
           "distinct_sum_signatures": len(CENSUS),
           "total_sum_calls": sum(v for v in CENSUS.values() if isinstance(v, int)),
           "signatures": [{"count": v, **json.loads(k)} for k, v in
                          sorted(CENSUS.items(), key=lambda kv: -kv[1])
                          if not k.startswith("INSTRUMENT_ERROR")],
           "instrument_errors": {k: v for k, v in CENSUS.items()
                                 if k.startswith("INSTRUMENT_ERROR")},
           "site_names_sample": sorted(SITES)[:400]}
    json.dump(cen, open(a.census, "w"), indent=1)
    print(json.dumps({"census": a.census, "signatures": len(CENSUS),
                      "sum_calls": cen["total_sum_calls"],
                      "ln_sites": len(SITES), "captured": len(CAPTURED)}))
    if a.ln_out and CAPTURED:
        torch.save({"sites": CAPTURED, "match": a.match,
                    "boundary_argv": passthrough}, a.ln_out)
        print(json.dumps({"ln_out": a.ln_out, "sites": len(CAPTURED),
                          "paths": [c["gamma_path"] for c in CAPTURED]}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
