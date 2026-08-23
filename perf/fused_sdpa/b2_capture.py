#!/usr/bin/env python3
"""Capture real Boltz-2 trunk tensors out of a real fold, from inside the fold's own process.

`tt-bio predict` folds in a spawned worker, so monkeypatching `tt_bio.tenstorrent` in the launcher
patches nothing (`scripts/lever_census.py` root-caused the same thing for counters). This runs the
CLI as a subprocess with a generated `sitecustomize.py`; the child installs a `sys.meta_path`
finder that wraps `tt_bio.tenstorrent`'s loader, so the patch lands the moment the module executes,
in whichever process executes it.

Two modes, both keyed to the construction site by `PairformerModule.n_blocks` so the 64-block trunk
is not confused with the 8-block confidence head or the 2-block template stack:

    sdpa    a reservoir of triangle-attention (q, k, v, bias) calls, spread evenly over the whole
            call stream, in the blob layout `errstruct.py --capture-in` reads
    trunk   the (s, z) the 64-block trunk pairformer is handed on its first recycling iteration,
            which is what turns a synthetic-input screen into a fold-level verdict

`trunk` taps `Pairformer.__call__`, not `PairformerModule.forward`: Boltz-2's trunk hands the
device-resident `TrunkModule` the INNER `Pairformer` (boltz2.py, `TrunkModule(..., pairformer_
module.module, ...)`), so the wrapper's forward never runs for the 512 trunk calls. Measured, not
assumed -- a first pass tagged every trunk call `?` and only the 48 confidence-head calls `pf8`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


# ----------------------------------------------------------------------- child

def _reservoir(n):
    from errstruct import Reservoir
    return Reservoir(n)


def _patch(T):
    import atexit
    import torch

    out = Path(os.environ["B2_CAPTURE_OUT"])
    mode = os.environ.get("B2_CAPTURE_MODE", "sdpa")
    n_keep = int(os.environ.get("B2_CAPTURE_N", "10"))
    rows = int(os.environ.get("B2_CAPTURE_ROWS", "64"))
    site = ["?"]

    # Tag every triangle attention with the pairformer that is running, so the 512 trunk calls can
    # be told from the 48 that belong to the confidence head and the template stack.
    orig_fwd = T.PairformerModule.forward

    def fwd(self, s, z, mask=None, pair_mask=None, use_kernels=False):
        prev, site[0] = site[0], f"pf{self.n_blocks}"
        try:
            return orig_fwd(self, s, z, mask, pair_mask, use_kernels)
        finally:
            site[0] = prev

    T.PairformerModule.forward = fwd

    import ttnn

    orig_call = T.Pairformer.__call__

    def call(self, s, z, mask=None, attn_mask_start=None, attn_mask_end=None,
             extra_attn_bias=None):
        n = len(self.blocks)
        prev, site[0] = site[0], f"pf{n}"
        if mode == "trunk" and n == 64 and not out.exists():
            torch.save({"s": None if s is None else ttnn.to_torch(s).float().clone(),
                        "z": ttnn.to_torch(z).float().clone(),
                        "mask": None if mask is None else ttnn.to_torch(mask).float().clone(),
                        "attn_mask_start": None if attn_mask_start is None
                                           else ttnn.to_torch(attn_mask_start).float().clone(),
                        "attn_mask_end": None if attn_mask_end is None
                                         else ttnn.to_torch(attn_mask_end).float().clone(),
                        "extra_attn_bias": None if extra_attn_bias is None
                                           else ttnn.to_torch(extra_attn_bias).float().clone(),
                        "n_blocks": n}, out)
            print(f"[b2_capture] trunk input saved: z={tuple(z.shape)}", flush=True)
        try:
            return orig_call(self, s, z, mask, attn_mask_start, attn_mask_end, extra_attn_bias)
        finally:
            site[0] = prev

    T.Pairformer.__call__ = call

    if mode != "sdpa":
        return
    res, counter, orig = _reservoir(n_keep), [0], T._tri_att_sdpa

    def spy(q, k, v, bias, scale, ckc=None):
        shp = tuple(int(d) for d in q.shape)
        # Triangle attention is the only caller whose batch dim IS the sequence dim.
        if len(shp) == 4 and shp[0] == shp[2] and shp[0] > 1:
            i = counter[0]
            counter[0] += 1
            if res.want(i):
                import torch as _t
                qt, kt, vt = ttnn.to_torch(q), ttnn.to_torch(k), ttnn.to_torch(v)
                bt = ttnn.to_torch(bias)
                sub = _t.arange(0, qt.shape[0], max(1, qt.shape[0] // rows))
                res.add(dict(call=i, site=site[0],
                             q=qt[sub].clone(), k=kt[sub].clone(), v=vt[sub].clone(),
                             bias=(bt[sub].clone() if bt.shape[0] == qt.shape[0]
                                   else bt.clone()),
                             scale_inv=float(scale), bias_scale_inv=float(scale),
                             shape=list(shp)))
                del qt, kt, vt, bt
        return orig(q, k, v, bias, scale, ckc)

    T._tri_att_sdpa = spy

    def dump():
        if not res.items:
            return
        torch.save({"grabs": res.items, "seen": counter[0], "model": "boltz2"}, out)
        print(f"[b2_capture] wrote {out}: {len(res.items)} of {counter[0]} calls", flush=True)

    atexit.register(dump)


def install():
    """Wrap `tt_bio.tenstorrent`'s loader so the patch lands wherever the module executes."""
    class Finder:
        def find_spec(self, name, path=None, target=None):
            if name != "tt_bio.tenstorrent":
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            inner = spec.loader.exec_module

            def exec_module(module):
                inner(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, Finder())


# ---------------------------------------------------------------------- parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("sdpa", "trunk"), default="sdpa")
    ap.add_argument("--n-keep", type=int, default=10)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--pythonpath", required=True)
    ap.add_argument("cli", nargs="*")
    args = ap.parse_args()
    if not args.cli:
        ap.error("give the CLI after `--`")

    here = Path(__file__).resolve().parent
    hookdir = Path(args.out).resolve().parent / (".b2cap-" + Path(args.out).stem)
    hookdir.mkdir(parents=True, exist_ok=True)
    (hookdir / "sitecustomize.py").write_text(
        "import sys\n"
        f"sys.path.append({str(here)!r})\n"
        "try:\n"
        "    from b2_capture import install\n"
        "    install()\n"
        "except Exception as exc:\n"
        "    print('b2_capture install failed:', exc)\n")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(hookdir), args.pythonpath])
    env["B2_CAPTURE_OUT"] = str(Path(args.out).resolve())
    env["B2_CAPTURE_MODE"] = args.mode
    env["B2_CAPTURE_N"] = str(args.n_keep)
    env["B2_CAPTURE_ROWS"] = str(args.rows)
    rc = subprocess.call(args.cli, env=env)
    ok = Path(args.out).exists()
    print(f"--- rc={rc} capture={'written' if ok else 'MISSING'} {args.out}")
    sys.exit(0 if ok else (rc or 1))


if __name__ == "__main__":
    main()
