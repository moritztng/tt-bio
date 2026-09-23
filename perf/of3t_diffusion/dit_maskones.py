#!/usr/bin/env python3
"""Their DiT at the same shape with an all-ones token mask, so the device arm can ask whether
the defect is the 85 % padding or the 384-token shape itself.

Our fixture gate runs 76 real tokens of 96 (79 % occupancy) and is bit-exact. The boundary here
is 56 of 384 (15 %). If our DiT disagrees only under heavy padding the exposure is the training
crop; if it disagrees at 384 tokens with every token real, shipped folds of ~384-residue targets
are in scope too, and that is a production question rather than a training one.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "/home/ttuser/of3t_gradients/ref")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_gradients"))
import capture_trunk_boundary as CTB  # noqa: E402

CAP = Path("/home/ttuser/of3t_diffusion_cap")


def main() -> int:
    t0 = time.time()
    import bundle_min as BM
    S = torch.load(CAP / "sub_boundary.pt", map_location="cpu", weights_only=False)
    dit_args, dit_kw = S["dit_in"]
    built = BM.build(torch.float64, 20260919, "cpu", num_recycles=0)
    model = built[1]
    ck = torch.load(CTB.CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: v.to(torch.float64) if torch.is_tensor(v) and v.is_floating_point() else v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    dit = model.diffusion_module.diffusion_transformer
    print(f"[{time.time()-t0:.0f}s] model built, mask {tuple(dit_kw['mask'].shape)} "
          f"sum {float(dit_kw['mask'].sum())}", flush=True)

    kw = dict(dit_kw)
    kw["mask"] = torch.ones_like(dit_kw["mask"])
    with torch.no_grad(), BM.no_autocast():
        out = dit(*dit_args, **kw)
    torch.save({"dit_out_maskones": out.detach(), "mask": kw["mask"]},
               CAP / "dit_out_maskones.pt")
    print(f"[{time.time()-t0:.0f}s] wrote dit_out_maskones.pt {tuple(out.shape)}", flush=True)
    Path("perf/of3t_diffusion/dit_maskones.json").write_text(json.dumps(
        {"shape": list(out.shape), "mask_sum": float(kw["mask"].sum()),
         "orig_mask_sum": float(dit_kw["mask"].sum()),
         "seconds": time.time() - t0}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
