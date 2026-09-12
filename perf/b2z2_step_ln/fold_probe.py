#!/usr/bin/env python3
"""Is the folded AdaLN conditioning closer to fp32 than the shipped one, or further?

The step-level output moves 6.9 % rms when the shared `layer_norm(s)` is on, which is far more
than a bf16 re-rounding should buy. This asks the question where the arithmetic actually changes:
at ONE AdaLN, against a torch fp32 reference of the same weights and the same real `s` lifted off
a settled diffusion step. Both device arms are scored against that reference, so the answer is
"which one is wrong", not "they differ".
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FUSION = ROOT / "perf" / "b2z2_step_fusion"
for p in (str(FUSION), str(ROOT), str(ROOT / "scripts" / "gpu_vs_tt"),
          str(ROOT / "perf" / "b2x_difflayer")):
    sys.path.insert(0, p)


def rel(x, ref):
    import torch
    return float((x - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keep-blocks", type=int, default=2)
    ap.add_argument("--layers", type=int, default=24)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import step_probe as SP

    a.out.parent.mkdir(parents=True, exist_ok=True)
    SP.OUT_PATH = a.out
    SP.OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "card": os.environ.get("TT_VISIBLE_DEVICES"),
                     "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip()}
    SP.dump()
    dev, g = SP.grab_step(ttnn, T, B, a.size, a.keep_blocks)

    # lift the token stack's real `s` (and the stack itself) out of one replay
    grabbed = {}
    cls, orig = T.DiffusionTransformer, T.DiffusionTransformer.__dict__["__call__"]

    def wrapper(self_obj, a_, s_, z_, *args, **kw):
        if getattr(self_obj, "shared_s_norm", False) and "s" not in grabbed:
            grabbed["stack"], grabbed["s"] = self_obj, ttnn.clone(s_)
        return orig(self_obj, a_, s_, z_, *args, **kw)
    cls.__call__ = wrapper
    try:
        g["obj"](*g["args"], **g["kwargs"])
        ttnn.synchronize_device(dev)
    finally:
        cls.__call__ = orig
    stack, s = grabbed["stack"], grabbed["s"]
    s_t = ttnn.to_torch(s).float()
    SP.OUT["s_shape"] = list(s.shape)

    s_normed = ttnn.layer_norm(s, epsilon=1e-5,
                               compute_kernel_config=stack.compute_kernel_config)
    rows = []
    for i, layer in enumerate(stack.layers[:a.layers]):
        for which, adaln in (("attn", layer.adaln), ("transition", layer.transition.adaln)):
            w = adaln.weights["s_norm.weight"].float()
            n = torch.nn.functional.layer_norm(s_t, (s_t.shape[-1],), eps=1e-5)
            ref_scale = (n * w) @ adaln.weights["s_scale.weight"].float().t() \
                + adaln.weights["s_scale.bias"].float()
            ref_bias = (n * w) @ adaln.weights["s_bias.weight"].float().t()
            b_scale, b_bias = adaln.s_terms(s)
            f_scale, f_bias = adaln.s_terms(s, s_normed=s_normed)
            ttnn.synchronize_device(dev)
            rows.append({
                "layer": i, "which": which,
                "base_scale": rel(ttnn.to_torch(b_scale).float(), ref_scale),
                "fold_scale": rel(ttnn.to_torch(f_scale).float(), ref_scale),
                "base_bias": rel(ttnn.to_torch(b_bias).float(), ref_bias),
                "fold_bias": rel(ttnn.to_torch(f_bias).float(), ref_bias),
            })
            for t in (b_scale, b_bias, f_scale, f_bias):
                ttnn.deallocate(t)
    SP.OUT["rows"] = rows
    SP.OUT["summary"] = {k: {"median": round(st.median([r[k] for r in rows]), 6),
                             "max": round(max(r[k] for r in rows), 6)}
                         for k in ("base_scale", "fold_scale", "base_bias", "fold_bias")}
    # the shared normed s against fp32, on its own
    SP.OUT["s_normed_rel"] = rel(ttnn.to_torch(s_normed).float(),
                                 torch.nn.functional.layer_norm(s_t, (s_t.shape[-1],), eps=1e-5))
    SP.dump()
    print(json.dumps(SP.OUT["summary"], indent=1), flush=True)
    print("s_normed rel", SP.OUT["s_normed_rel"], flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
