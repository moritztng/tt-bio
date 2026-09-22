#!/usr/bin/env python3
"""of3t-cotcoh step 1a: the cotangent ARRIVING at the two worst affine leaf families, masked to
the real rows, at all 48 blocks, from the shipped device backward on the MODEL frame.

`perf/of3t_lnreduce/capln.py`'s instrument with three changes and no arithmetic:

  1. it captures at EVERY block rather than a handful of sites, because the deliverable is a
     curve over block index and not a point;
  2. it stores the MASKED operands only. The pair-track sites are [1, 64, 384, 128] chunks and
     the real signal lives in the 56x56 block of one chunk, so a masked store is 0.8 MB where
     the chunk is 6.3 MB. Disk on qb1 is the binding constraint, not RAM;
  3. `torch.save` to the harness's own `--out` is redirected to per-tensor squared norms. This
     row never reads that 1.3 GB file and the host has 1.9 GB free.

`ag._taped_layer_norm` is WRAPPED, not replaced: the wrapper calls the shipped verb, so the
forward and the backward are the shipped ones bit for bit and only the tape node's function is
decorated to copy out `x` and `g`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkg043"))
# D149: read the resolution back rather than trusting the insert.
print("SYS_PATH resolved: " + repr(sys.path[:2]), flush=True)

FAM_A = "pre_norm_s_weight"                 # upstream attn_pair_bias.layer_norm_a
FAM_B = "transition_z.norm_weight"          # upstream pair_stack.pair_transition.layer_norm
BLK = re.compile(r"blocks\.(\d+)\.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--real-rows", type=int, default=56)
    a, rest = ap.parse_known_args()
    passthrough = [x for x in rest if x != "--"]
    t0 = time.perf_counter()
    R = a.real_rows

    import torch
    import ttnn
    from tt_bio import autograd as ag
    import tt_bio.taped_ttnn as tt
    import tt_bio.tenstorrent as T

    # ---- keep the harness's 1.3 GB gradient dump off a disk with 1.9 GB on it -------------
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
        if isinstance(f, str) and OUTPATH is not None and os.path.abspath(f) == \
                os.path.abspath(OUTPATH):
            return _real_save({"REDIRECTED_TO_SQNORMS_BY": "perf/of3t_cotcoh/capcot.py",
                               "why": "this row reads the report and the masked cotangents, "
                                      "never the tensor dump; qb1 has 1.9 GB free",
                               "sqnorms": _norms(obj)}, f, *ar, **kw)
        return _real_save(obj, f, *ar, **kw)

    torch.save = _save

    # ---- the module handle, so a gamma can be named (capln.py, unchanged) -----------------
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

    CAP = {}          # (fam, block) -> record
    CHUNKS = {}       # (fam, block) -> list of per-fire chunk norms, the B control
    SHAPES = {}
    SITES = {}
    _shipped_verb = ag._taped_layer_norm

    def _mask(fam, gt, C):
        """gt is the device tensor as torch. Return the masked [P, C] view, or None if this
        chunk carries no real signal."""
        t = gt.reshape(-1, C)
        if fam == "A":
            return t[:R].contiguous()
        rows = t.shape[0] // 384                      # the chunk's i-extent
        t = t.reshape(rows, 384, C)
        if rows < R:
            return None
        return t[:R, :R].reshape(-1, C).contiguous()

    def _verb(shipped, args, kwargs):
        argl = list(args) + [None] * (3 - len(args))
        xw = ag._wrap(argl[0])
        gw = ag._wrap(kwargs.get("weight", argl[1]))
        out = _shipped_verb(shipped, args, kwargs)
        if out is None or getattr(out, "node", None) is None or gw is None:
            return out
        path = _name_of(gw.value)
        fam = None
        if path:
            if FAM_A in path:
                fam = "A"
            elif FAM_B in path:
                fam = "B"
        m = BLK.search(path or "")
        blk = int(m.group(1)) if m else -1
        orig = out.node.fn

        def fn(g, orig=orig, xw=xw, gw=gw, fam=fam, blk=blk, path=path):
            if fam is None or blk < 0:
                return orig(g)
            xv = xw.value
            gt = ttnn.to_torch(g)
            xt = ttnn.to_torch(xv)
            C = int(gw.value.shape[-1])
            SHAPES.setdefault((fam, blk), []).append(
                [list(gt.shape), list(xt.shape), C])
            full = float(torch.linalg.vector_norm(gt.to(torch.float64)))
            CHUNKS.setdefault((fam, blk), []).append(full)
            gm = _mask(fam, gt, C)
            if gm is not None and float(torch.linalg.vector_norm(gm.to(torch.float64))) > 0.0:
                xm = _mask(fam, xt, C)
                # the pad control: everything in this chunk that the mask throws away
                outside = full ** 2 - float(torch.linalg.vector_norm(gm.to(torch.float64))) ** 2
                CAP[(fam, blk)] = {
                    "g": gm.clone(), "g_dtype": str(g.dtype),
                    "x": xm.clone(), "x_dtype": str(xv.dtype),
                    "gamma": ttnn.to_torch(gw.value).to(torch.float32).clone(),
                    "gamma_path": path, "chunk_full_sqnorm": full ** 2,
                    "outside_mask_sqnorm": outside,
                }
            return orig(g)

        out.node.fn = fn
        if fam:
            SITES[path] = SITES.get(path, 0) + 1
        return out

    ag._TAPED["layer_norm"] = _verb
    tt._VERBS["layer_norm"] = _verb

    import dev_grad
    sys.argv = ["dev_grad.py"] + passthrough
    rc = dev_grad.main()

    sites = {f"{fam}:{blk}": {"gamma_path": v["gamma_path"],
                              "P": int(v["g"].shape[0]), "C": int(v["g"].shape[1]),
                              "chunk_full_sqnorm": v["chunk_full_sqnorm"],
                              "outside_mask_sqnorm": v["outside_mask_sqnorm"],
                              "fires": len(CHUNKS[(fam, blk)]),
                              "nonzero_fires": sum(1 for x in CHUNKS[(fam, blk)] if x > 0.0)}
             for (fam, blk), v in sorted(CAP.items())}
    torch.save = _real_save
    torch.save({"sites": {f"{fam}:{blk}": v for (fam, blk), v in CAP.items()},
                "real_rows": R, "argv": passthrough}, a.cot_out)
    rep = {"what": "the cotangent arriving at the two worst affine leaf families, masked to the "
                   "real rows, at every block, from the shipped device backward",
           "host": socket.gethostname(),
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "board_class_note": "read by the launcher, written into this file by it",
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
           "argv": passthrough, "real_rows": R,
           "seconds": round(time.perf_counter() - t0, 1),
           "families": {"A": FAM_A, "B": FAM_B},
           "n_sites_captured": len(CAP),
           "n_A": sum(1 for k in CAP if k[0] == "A"),
           "n_B": sum(1 for k in CAP if k[0] == "B"),
           "taped_family_sites": len(SITES),
           "fires_per_site": {f"{f}:{b}": len(v) for (f, b), v in sorted(CHUNKS.items())},
           "nonzero_chunks_per_B_site": {f"B:{b}": sum(1 for x in v if x > 0.0)
                                         for (f, b), v in sorted(CHUNKS.items()) if f == "B"},
           "shapes_seen": {f"{f}:{b}": v[:2] for (f, b), v in sorted(SHAPES.items())
                           if b in (0, 47)},
           "sites": sites,
           "cot_out": a.cot_out,
           "cot_out_bytes": os.path.getsize(a.cot_out) if os.path.exists(a.cot_out) else None}
    json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps({"cot_out": a.cot_out, "n_A": rep["n_A"], "n_B": rep["n_B"],
                      "seconds": rep["seconds"]}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
