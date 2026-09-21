#!/usr/bin/env python3
"""D141: which layer_norm_z is each side running?

AMENDMENT 2 of the of3t-ditcot brief. Upstream 0.4.3/0.5.0 carry layer_norm_z in two places:
one shared LayerNorm(c_z) on the transformer (diffusion_transformer.py:254, applied :313) and
one per AttentionPairBias module (attention_pair_bias.py:107, applied :156). Which one is built
decides whether `of3-p2-155k.pt` (which stores it PER BLOCK) lands its trained tensors or has
them dropped as unexpected_keys with the shared weight left at its all-ones init.

This loads the reference module EXACTLY the way perf/of3t_diffusion/sub_boundary.py:62 and
perf/of3t_condtrans/floor_bf16.py:148 do -- same build call, same strict=False -- and reports
what load_state_dict actually accepted. CPU only, no card.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/ttuser/of3t_gradients/ref")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_gradients"))

import capture_trunk_boundary as CTB  # noqa: E402
import bundle_min as BM  # noqa: E402

OUT = Path(__file__).resolve().parent / "D141.json"
LNZ = re.compile(r"layer_norm_z")


def main() -> int:
    rep = {"what": __doc__.strip().splitlines()[0], "ckpt": str(CTB.CKPT)}

    # --- where does the INSTALLED upstream put layer_norm_z? ------------------------------
    import openfold3
    from openfold3.core.model.layers import diffusion_transformer as DT
    from openfold3.core.model.layers import attention_pair_bias as APB
    rep["upstream"] = {
        "version": getattr(openfold3, "__version__", "?"),
        "root": str(Path(openfold3.__file__).resolve().parent),
        "dit_has_layer_norm_z_src": "layer_norm_z" in Path(DT.__file__).read_text(),
        "apb_has_layer_norm_z_src": "layer_norm_z" in Path(APB.__file__).read_text(),
    }

    # --- build the reference the way every diffusion builder does -------------------------
    dtype = torch.float64
    built = BM.build(dtype, 20260919, "cpu", num_recycles=0)
    model = built[1]
    ck = torch.load(CTB.CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
          for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)

    miss = list(inc.missing_keys)
    unexp = list(inc.unexpected_keys)
    rep["load"] = {"missing_total": len(miss), "unexpected_total": len(unexp)}
    rep["load"]["missing_lnz"] = sorted(k for k in miss if LNZ.search(k))
    rep["load"]["unexpected_lnz"] = sorted(k for k in unexp if LNZ.search(k))
    # the brief asks specifically for attention_pair_bias.layer_norm_z.weight in unexpected
    rep["load"]["unexpected_apb_lnz"] = sorted(
        k for k in unexp if "attention_pair_bias.layer_norm_z" in k)
    rep["load"]["missing_sample"] = miss[:20]
    rep["load"]["unexpected_sample"] = unexp[:20]

    # --- what the checkpoint actually STORES ---------------------------------------------
    ck_lnz = sorted(k for k in sd if LNZ.search(k) and "diffusion_transformer" in k)
    rep["ckpt_dit_lnz_keys"] = {"n": len(ck_lnz), "sample": ck_lnz[:4]}

    # --- what the BUILT reference is RUNNING at every layer_norm_z ------------------------
    dm = model.diffusion_module
    dit = dm.diffusion_transformer
    running = []
    for name, mod in dit.named_modules():
        if not name.endswith("layer_norm_z"):
            continue
        w = getattr(mod, "weight", None)
        if w is None:
            running.append({"site": name, "weight": None})
            continue
        wd = w.detach().double().reshape(-1)
        running.append({
            "site": name,
            "n": int(wd.numel()),
            "mean": float(wd.mean()), "std": float(wd.std()),
            "min": float(wd.min()), "max": float(wd.max()),
            "all_ones": bool(torch.equal(wd, torch.ones_like(wd))),
            "max_abs_dev_from_one": float((wd - 1.0).abs().max()),
        })
    rep["reference_running"] = running
    rep["reference_lnz_sites"] = len(running)
    rep["reference_all_ones_sites"] = sum(1 for r in running if r.get("all_ones"))

    # --- what OUR port would run at the same sites, straight from the checkpoint ----------
    ours = []
    for k in ck_lnz[:4] + ck_lnz[-1:]:
        if not k.endswith(".weight"):
            continue
        wd = sd[k].detach().double().reshape(-1)
        ours.append({"key": k, "n": int(wd.numel()), "mean": float(wd.mean()),
                     "std": float(wd.std()), "min": float(wd.min()), "max": float(wd.max()),
                     "all_ones": bool(torch.equal(wd, torch.ones_like(wd))),
                     "max_abs_dev_from_one": float((wd - 1.0).abs().max())})
    rep["ours_running"] = ours

    mismatch = (rep["reference_all_ones_sites"] > 0 and len(rep["load"]["unexpected_apb_lnz"]) > 0)
    rep["ARCHITECTURE_MISMATCH"] = mismatch
    OUT.write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps(rep, indent=1, sort_keys=True))
    print(f"\nARCHITECTURE_MISMATCH: {mismatch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
