#!/usr/bin/env python3
"""of3t-cotcoh pass 2: upstream 0.4.3's OWN bf16 autocast trunk gradient on the MODEL frame,
scored per block against the same float64 reference, so the walk-back can print both absolute
curves and not only their ratio.

The amendment's `ENTRY_OR_ACCUMULATION.json` prior reads twelve of 48 blocks. This measures all
48 on the model frame. It imports `ref_grad.py` and changes nothing: the only additions are a
`torch.save` interception, which scores the gradient in process and writes per-block sums
instead of a ~1 GB float64 dump onto a host with 1.5 GB free, and nothing else.
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

SEC = "pairformer_stack.blocks."
F64REF = "/home/ttuser/of3t-campaign-refs/bundle_min_043/grads_f64_043.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-out", required=True)
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
    RES = {}

    def _sq(t):
        return float(torch.linalg.vector_norm(t.to(torch.float64))) ** 2

    def _save(obj, f, *ar, **kw):
        if isinstance(f, str) and OUTPATH is not None and \
                os.path.abspath(f) == os.path.abspath(OUTPATH):
            ref = torch.load(F64REF, map_location="cpu", weights_only=False)
            if isinstance(ref, dict) and "grads" in ref:
                ref = ref["grads"]
            g = obj["grads"]
            blk, per = {}, {}
            errs = refs = 0.0
            n = miss = 0
            for k, v in ref.items():
                if not k.startswith(SEC) or v is None:
                    continue
                o = g.get(k)
                if o is None:
                    miss += 1
                    continue
                r = v.to(torch.float64)
                e2, r2 = _sq(o.to(torch.float64) - r), _sq(r)
                per[k] = {"err_sq": e2, "ref_sq": r2}
                errs += e2
                refs += r2
                n += 1
                b = int(k[len(SEC):].split(".")[0])
                d = blk.setdefault(b, [0.0, 0.0, 0])
                d[0] += e2
                d[1] += r2
                d[2] += 1
            RES.update({"n_tensors": n, "absent": miss,
                        "trunk_mass_weighted_rel_l2_vs_float64":
                            (errs / refs) ** 0.5 if refs else None,
                        "banked_upstream_bf16_trunk_vs_float64": 0.3147698293887927,
                        "by_block": {str(b): {"err_sq": v[0], "ref_sq": v[1], "n": v[2],
                                              "rel_l2_vs_float64": (v[0] / v[1]) ** 0.5
                                              if v[1] else None}
                                     for b, v in sorted(blk.items())},
                        "per_tensor": per})
            print("UPSTREAM BF16 TRUNK vs float64: %.16f (banked 0.3147698293887927)"
                  % RES["trunk_mass_weighted_rel_l2_vs_float64"], flush=True)
            return _real_save({"REDIRECTED": "perf/of3t_cotcoh/refbf16.py",
                               "trunk": {k: v for k, v in RES.items() if k != "per_tensor"}},
                              f, *ar, **kw)
        return _real_save(obj, f, *ar, **kw)

    torch.save = _save
    sys.argv = ["ref_grad.py"] + passthrough
    rc = ref_grad.main()
    torch.save = _real_save

    out = {"what": "upstream 0.4.3's own bf16 autocast trunk gradient on the model frame, "
                   "scored per block against the published float64",
           "host": socket.gethostname(), "device_involved": False,
           "why_no_aiclk": "CPU only, no card is opened",
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
           "argv": passthrough, "float64_reference": F64REF,
           "seconds": round(time.perf_counter() - t0, 1), **RES}
    json.dump({k: v for k, v in out.items() if k != "per_tensor"},
              open(a.score_out, "w"), indent=1)
    json.dump({"host": socket.gethostname(), "per_tensor": RES.get("per_tensor", {})},
              open(a.score_out.replace(".json", "_PERTENSOR.json"), "w"))
    print(json.dumps({"out": a.score_out,
                      "trunk": RES.get("trunk_mass_weighted_rel_l2_vs_float64"),
                      "seconds": out["seconds"]}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
