#!/usr/bin/env python3
"""of3t-cotcoh step 1b: the REFERENCE cotangent at the same two affine leaf families, all 48
blocks, masked to the real rows, on the MODEL frame.

This is not a copy of `perf/of3t_trunkg043/ref_grad.py`. It imports it and wraps exactly one
function, `build`, which is the seam that hands back the 48 constructed PairFormerBlocks. The
hooks are registered on the two LayerNorms of each block after the real `build` returns, so
every line of arithmetic, the strict load, the checkpointing and the loss are upstream's own and
this row's diff against the producer is a registration and a mask.

`--checkpoint` recomputes each block in the backward, so the forward hook fires twice per site.
The record is written from inside the TENSOR hook, which only fires on the graph the backward
actually traverses, so a double forward cannot produce a double record or a stale one.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
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
    ap.add_argument("--cot-out", required=True)
    ap.add_argument("--cot-report", required=True)
    ap.add_argument("--real-rows", type=int, default=56)
    a, rest = ap.parse_known_args()
    passthrough = [x for x in rest if x != "--"]
    R = a.real_rows
    t0 = time.perf_counter()

    import torch
    import ref_grad

    OUTPATH = None
    for i, x in enumerate(passthrough):
        if x == "--out":
            OUTPATH = passthrough[i + 1]
    _real_save = torch.save

    def _norms(o):
        if torch.is_tensor(o):
            return {"__sqnorm__": float(torch.linalg.vector_norm(o.to(torch.float64))) ** 2,
                    "shape": list(o.shape), "dtype": str(o.dtype)}
        if isinstance(o, dict):
            return {k: _norms(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(_norms(v) for v in o)
        return o

    def _save(obj, f, *ar, **kw):
        if isinstance(f, str) and OUTPATH is not None and \
                os.path.abspath(f) == os.path.abspath(OUTPATH):
            return _real_save({"REDIRECTED_TO_SQNORMS_BY": "perf/of3t_cotcoh/refcot.py",
                               "sqnorms": _norms(obj)}, f, *ar, **kw)
        return _real_save(obj, f, *ar, **kw)

    torch.save = _save

    CAP, FIRES = {}, {}
    _real_build = ref_grad.build

    def _mask(fam, t):
        t = t.detach()
        if fam == "A":                      # [1, N, c_s] -> [R, c_s]
            return t.reshape(-1, t.shape[-1])[:R].contiguous()
        C = t.shape[-1]                     # [1, N, N, c_z] -> [R*R, c_z]
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
                            CAP[(fam, i)] = {
                                "g": _mask(fam, gr).to(torch.float64).clone(),
                                "x": xm,
                                "g_full_sqnorm": float(
                                    torch.linalg.vector_norm(gr.detach().to(torch.float64))) ** 2,
                            }
                        out.register_hook(_gh)
                    return hook
                m.get_submodule(sub).register_forward_hook(_mk())
        print(json.dumps({"hooks_registered": len(mods) * len(FAM), "blocks": len(mods)}),
              flush=True)
        return mods, dims, load

    ref_grad.build = build
    sys.argv = ["ref_grad.py"] + passthrough
    rc = ref_grad.main()

    torch.save = _real_save
    torch.save({"sites": {f"{f}:{b}": v for (f, b), v in CAP.items()},
                "real_rows": R, "argv": passthrough}, a.cot_out)
    rep = {"what": "upstream 0.4.3's own float64 cotangent at the two affine leaf families, "
                   "masked to the real rows, at every block, on the model frame",
           "host": socket.gethostname(), "device_involved": False,
           "why_no_aiclk": "CPU only, no card is opened",
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
           "argv": passthrough, "real_rows": R, "families": FAM,
           "seconds": round(time.perf_counter() - t0, 1),
           "n_sites": len(CAP), "n_A": sum(1 for k in CAP if k[0] == "A"),
           "n_B": sum(1 for k in CAP if k[0] == "B"),
           "forward_hook_fires": {f"{f}:{b}": v for (f, b), v in sorted(FIRES.items())
                                  if b in (0, 47)},
           "missing": [f"{f}:{b}" for f in FAM for b in range(48) if (f, b) not in CAP],
           "cot_out": a.cot_out,
           "cot_out_bytes": os.path.getsize(a.cot_out) if os.path.exists(a.cot_out) else None}
    json.dump(rep, open(a.cot_report, "w"), indent=1)
    print(json.dumps({"cot_out": a.cot_out, "n_A": rep["n_A"], "n_B": rep["n_B"],
                      "missing": len(rep["missing"]), "seconds": rep["seconds"]}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
