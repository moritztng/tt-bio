#!/usr/bin/env python3
"""Our shipped trunk Pairformer at padded N=384, dumping EVERY block's output state.

The dump is taken by wrapping `ops.checkpoint_segment`, which is the seam
`Pairformer.__call__` composes through (`tt_bio/tenstorrent.py`, the `for i, block in
enumerate(self.blocks)` loop). Block k's recorded output is therefore the stack's own running
state and not a re-composition of it -- the distinction `of3t-trunkfwd` had to measure
separately, and here it holds by construction.

The configuration is read off `OF3Trunk`'s own construction by spying on the Pairformer
constructor, never hardcoded: `of3t-trunkfwd` found an instrument calling
`scale_pair_bias=True` "shipped" while the trunk shipped False, and since `701ddcf63` the
trunk ships True, so a hardcoded copy would be wrong in both directions at different times.

Every block output is stored at the device's own width (bfloat16 for both tracks with
`s_fp32_residual=False`), asserted lossless against what `ttnn.to_torch` returned.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# The venv at /home/ttuser/tt-bio-dev/env carries an EDITABLE install of tt_bio pointing at the
# SHARED checkout /home/ttuser/tt-bio-dev, which sits on `main`. `python3 perf/<ns>/x.py` puts the
# SCRIPT's directory on sys.path and not the repo root, so without this line `import tt_bio`
# silently resolves to main -- a different trunk configuration from this branch's. Every other
# of3t device script opens with the same insert; the assertion below is this row's addition,
# because the first arm of this row was measured on main before anyone noticed.
sys.path.insert(0, os.getcwd())

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


def tt_bio_file(mod):
    return getattr(mod, "__file__", "<none>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--dump", required=True, help="directory for per-block state")
    ap.add_argument("--report", required=True)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--set", action="append", default=[],
                    help="Pairformer kwarg override, k=v. Empty means shipped.")
    a = ap.parse_args()
    t0 = time.perf_counter()

    import torch
    import ttnn
    import tt_bio.ops as ops
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_pairformer_stack
    import tt_bio.openfold3_trunk as OT

    root = os.getcwd()
    if not os.path.realpath(tt_bio_file(OT)).startswith(os.path.realpath(root)):
        raise SystemExit(f"tt_bio resolved to {tt_bio_file(OT)}, outside {root} -- refusing to "
                         f"measure a tree this row is not on")

    os.makedirs(a.dump, exist_ok=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    spy = {}

    class _Stop(Exception):
        pass

    def _spy(*ar, **kw):
        spy["n_blocks"] = ar[0]
        spy["dims"] = list(ar[1:5])
        spy["transform_s"] = ar[5]
        spy["kwargs"] = {k: (v if isinstance(v, (bool, int, float, str, type(None))) else str(v))
                         for k, v in kw.items()}
        raise _Stop()

    real_pf = OT.Pairformer
    OT.Pairformer = _spy
    try:
        OT.OF3Trunk(sd, ckc)
    except _Stop:
        pass
    finally:
        OT.Pairformer = real_pf
    if "kwargs" not in spy:
        raise SystemExit("the spy never reached Pairformer -- a hardcoded default is refused")
    shipped = dict(spy["kwargs"])
    kwargs = dict(shipped)
    for kv in a.set:
        k, _, v = kv.partition("=")
        kwargs[k] = {"true": True, "false": False}.get(v.lower(), v)

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    s_in, z_in = b["s_in"], b["z_in"]
    sm, pm = b["single_mask"], b["pair_mask"]
    N = int(z_in.shape[1])

    flat_all = remap_pairformer_stack(sd, prefix="pairformer_stack")
    flat = (flat_all if a.blocks == spy["n_blocks"]
            else {k: v for k, v in flat_all.items() if int(k.split(".")[1]) < a.blocks})
    mod = T.Pairformer(a.blocks, *spy["dims"], spy["transform_s"], flat, ckc, **kwargs)

    s_dtype = ttnn.float32 if kwargs.get("s_fp32_residual", False) else ttnn.bfloat16
    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
    fts = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=s_dtype)
    attn = (1.0 - sm.reshape(1, 1, 1, N)) * -1e9

    seen = []
    flag = [True]
    _real_cs = ops.checkpoint_segment

    def _cs(fn, *inputs):
        out = _real_cs(fn, *inputs)
        k = len(seen)
        st = {}
        for nm, t in zip("sz", out):
            h = ttnn.to_torch(t).clone()
            n16 = h.to(torch.bfloat16)
            if not torch.equal(n16.to(torch.float64), h.to(torch.float64)):
                flag[0] = False
            st[nm] = n16
        torch.save(st, os.path.join(a.dump, f"blk{k:02d}.pt"))
        seen.append({"block": k,
                     "s_norm": float(st["s"].to(torch.float64).norm()),
                     "z_norm": float(st["z"].to(torch.float64).norm())})
        return out

    ops.checkpoint_segment = _cs
    try:
        so, zo = mod(fts(s_in), ft(z_in), ft(pm), ft(attn), ft(attn))
    finally:
        ops.checkpoint_segment = _real_cs
    lossless = flag[0]
    s = ttnn.to_torch(so).to(torch.float64)
    z = ttnn.to_torch(zo).to(torch.float64)

    rep = {"what": __doc__.strip().splitlines()[0],
           "host": os.uname().nodename, "blocks": a.blocks,
           "tt_bio_module": tt_bio_file(OT),
           "config": kwargs, "shipped_config": shipped, "overrides": a.set,
           "tokens": N, "real_tokens": int(sm.sum()),
           "boundary": a.boundary,
           "segments_seen": len(seen),
           "bf16_storage_lossless": bool(lossless),
           "s_norm": float(s.norm()), "z_norm": float(z.norm()),
           "final_equals_block47": {
               "s": bool(torch.equal(torch.load(os.path.join(a.dump, f"blk{len(seen)-1:02d}.pt"))
                                     ["s"].to(torch.float64), s)),
               "z": bool(torch.equal(torch.load(os.path.join(a.dump, f"blk{len(seen)-1:02d}.pt"))
                                     ["z"].to(torch.float64), z))},
           "per_block": seen, "dump": a.dump,
           "seconds": time.perf_counter() - t0}
    with open(a.report, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_block"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
