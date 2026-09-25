#!/usr/bin/env python3
"""Our port's trunk forward over the captured boundary, tensors written out so they can be
re-scored against any reference.

`of3t-trunkfwd` ran these arms and published the ratios but kept no output tensor, so this row
cannot re-score its numbers against a different reference without running the device again. It
writes the tensors this time.

Three arms, all the ordinary untaped inference path:

  shipped   the configuration READ OFF `OF3Trunk`'s own construction by spying on the Pairformer
            constructor, not a hardcoded copy of somebody's default. For `of3-p2-155k` that is
            `transpose_bias=True`, the preview2 orientation, which is the convention this
            checkpoint was trained with.
  lever     the same with `transpose_bias=False`, the port deliberately mis-set to agree with a
            0.5.0 reference. Nothing else moves.
  break     shipped, with the 56 real token positions permuted in the input. Weights, masks,
            flags and kernels untouched. A comparison that does not move under this is saturated
            and carries no information.
"""
from __future__ import annotations

import argparse
import json
import os
import time

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--break-seed", type=int, default=20260920)
    a = ap.parse_args()
    t0 = time.perf_counter()

    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_pairformer_stack
    import tt_bio.openfold3_trunk as OT

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    # ---- the SHIPPED configuration, read off the shipped construction site -------------------
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
        raise SystemExit("the spy never reached Pairformer -- a hardcoded default would be the "
                         "only alternative and this row refuses to keep one")
    shipped_kw = dict(spy["kwargs"])
    print(f"[{time.perf_counter()-t0:.0f}s] shipped config {shipped_kw}", flush=True)

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    s_in, z_in = b["s_in"], b["z_in"]
    sm, pm = b["single_mask"], b["pair_mask"]
    N = int(z_in.shape[1])

    flat_all = remap_pairformer_stack(sd, prefix="pairformer_stack")
    flat = (flat_all if a.blocks == spy["n_blocks"]
            else {k: v for k, v in flat_all.items() if int(k.split(".")[1]) < a.blocks})

    def build(kw):
        return T.Pairformer(a.blocks, *spy["dims"], spy["transform_s"], flat, ckc, **kw)

    s_fp32 = bool(shipped_kw.get("s_fp32_residual", False))
    s_dtype = ttnn.float32 if s_fp32 else ttnn.bfloat16
    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
    fts = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=s_dtype)

    def run(mod, s, z, single_mask):
        attn = (1.0 - single_mask.reshape(1, 1, 1, N)) * -1e9
        so, zo = mod(fts(s), ft(z), ft(pm), ft(attn), ft(attn))
        return (ttnn.to_torch(so).to(torch.float64), ttnn.to_torch(zo).to(torch.float64))

    rep = {"what": __doc__.strip().splitlines()[0], "blocks": a.blocks, "tokens": N,
           "real_tokens": int(sm.sum()), "boundary": a.boundary,
           "shipped_config": {"source": "tt_bio/openfold3_trunk.py OF3Trunk.__init__, read by "
                                        "spying on the Pairformer constructor",
                              "n_blocks": spy["n_blocks"], "dims": spy["dims"],
                              "transform_s": spy["transform_s"], "kwargs": shipped_kw},
           "s_input_dtype": str(s_dtype), "arms": {}}
    os.makedirs(a.outdir, exist_ok=True)

    # ---- shipped -----------------------------------------------------------------------------
    mod = build(shipped_kw)
    so, zo = run(mod, s_in, z_in, sm)
    torch.save({"s": so, "z": zo, "arm": "shipped", "config": shipped_kw},
               os.path.join(a.outdir, "device_shipped.pt"))
    rep["arms"]["shipped"] = {"transpose_bias": shipped_kw.get("transpose_bias"),
                              "s_norm": float(so.norm()), "z_norm": float(zo.norm())}
    print(f"[{time.perf_counter()-t0:.0f}s] shipped done", flush=True)

    # ---- break control: permute the real token positions, same module ------------------------
    g = torch.Generator().manual_seed(a.break_seed)
    real = int(sm.sum())
    perm = torch.randperm(real, generator=g)
    idx = torch.arange(N)
    idx[:real] = perm
    moved = int((perm != torch.arange(real)).sum())
    s_p = s_in[:, idx].contiguous()
    z_p = z_in[:, idx][:, :, idx].contiguous()
    so_b, zo_b = run(mod, s_p, z_p, sm)
    torch.save({"s": so_b, "z": zo_b, "arm": "break", "perm": idx, "moved": moved},
               os.path.join(a.outdir, "device_break.pt"))
    rep["arms"]["break"] = {"seed": a.break_seed, "real_tokens_permuted": moved,
                            "of": real, "s_norm": float(so_b.norm()),
                            "z_norm": float(zo_b.norm())}
    print(f"[{time.perf_counter()-t0:.0f}s] break done, {moved} of {real} moved", flush=True)
    del mod

    # ---- lever: transpose_bias flipped, nothing else ------------------------------------------
    lever_kw = dict(shipped_kw)
    lever_kw["transpose_bias"] = not shipped_kw.get("transpose_bias", True)
    mod2 = build(lever_kw)
    so_l, zo_l = run(mod2, s_in, z_in, sm)
    torch.save({"s": so_l, "z": zo_l, "arm": "lever", "config": lever_kw},
               os.path.join(a.outdir, "device_lever.pt"))
    rep["arms"]["lever"] = {"transpose_bias": lever_kw["transpose_bias"],
                            "s_norm": float(so_l.norm()), "z_norm": float(zo_l.norm())}
    print(f"[{time.perf_counter()-t0:.0f}s] lever done", flush=True)

    rep["seconds"] = time.perf_counter() - t0
    with open(a.report, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep["arms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
