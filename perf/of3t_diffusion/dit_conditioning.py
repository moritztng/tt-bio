#!/usr/bin/env python3
"""How much does THEIR diffusion transformer amplify a perturbation of its input?

The device bisect localises our forward gap to the DiT: `ai` goes in at 2.63e-03 relative L2
against their float64 activation and comes out at 7.35e-02, a 28x amplification, and the error
is larger on the 56 real tokens (1.59e-01) than on the 328 pad rows (4.12e-02), so masking is
not the cause.

Two readings fit that: our DiT computes a different function, or the DiT is simply
ill-conditioned at this shape and faithfully amplifies the 2.63e-03 it is handed. Those demand
opposite responses, and the difference is measurable rather than arguable: perturb THEIR input
by the same relative size with random noise, run THEIR DiT in float64, and see how far the
output moves. If their own amplification is ~28x then our number is what a correct
implementation must produce given its input, and the thing to fix is upstream of the DiT.

Costs one DiT forward per arm. The trunk is never touched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/ttuser/of3t_gradients/ref")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_gradients"))
import capture_trunk_boundary as CTB  # noqa: E402

CAP = Path("/home/ttuser/of3t_diffusion_cap")
REPORT = Path("perf/of3t_diffusion/dit_conditioning.json")


def main() -> int:
    t0 = time.time()
    import bundle_min as BM

    S = torch.load(CAP / "sub_boundary.pt", map_location="cpu", weights_only=False)
    dit_args, dit_kw = S["dit_in"]
    ref_out = S["dit_out"]
    print(f"[{time.time()-t0:.0f}s] dit_in kwargs: {sorted(dit_kw)}", flush=True)

    built = BM.build(torch.float64, 20260919, "cpu", num_recycles=0)
    model = built[1]
    ck = torch.load(CTB.CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: v.to(torch.float64) if torch.is_tensor(v) and v.is_floating_point() else v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    dit = model.diffusion_module.diffusion_transformer
    print(f"[{time.time()-t0:.0f}s] model built", flush=True)

    def run(kw):
        with torch.no_grad(), BM.no_autocast():
            return dit(*dit_args, **kw)

    rel = lambda x, y: float(torch.linalg.vector_norm(x - y)
                             / (torch.linalg.vector_norm(y) + 1e-300))

    base = run(dict(dit_kw))
    rep = {"reproduces_capture": rel(base, ref_out),
           "a_shape": tuple(dit_kw["a"].shape),
           "observed": {"ours_in": 2.6259512880580068e-03,
                        "ours_out": 7.349989198116111e-02,
                        "ours_out_real": 1.5859073376536942e-01,
                        "ours_out_pad": 4.117744277770751e-02}}
    print(f"[{time.time()-t0:.0f}s] unperturbed re-run vs captured dit_out: "
          f"{rep['reproduces_capture']:.3e}", flush=True)

    gen = torch.Generator().manual_seed(20260919)
    arms = {}
    for eps in (2.6259512880580068e-03, 1.0e-02):
        kw = dict(dit_kw)
        a = kw["a"]
        d = torch.randn(a.shape, generator=gen, dtype=a.dtype)
        d = d / torch.linalg.vector_norm(d) * torch.linalg.vector_norm(a) * eps
        kw["a"] = a + d
        out = run(kw)
        mask = dit_kw.get("mask")
        r = {"in_rel": rel(kw["a"], a), "out_rel": rel(out, base)}
        r["amplification"] = r["out_rel"] / r["in_rel"] if r["in_rel"] else None
        if torch.is_tensor(mask):
            m = mask.reshape(-1).bool()
            o2 = out.reshape(-1, out.shape[-1])
            b2 = base.reshape(-1, base.shape[-1])
            mm = m.repeat(o2.shape[0] // m.numel()) if o2.shape[0] % m.numel() == 0 else None
            if mm is not None:
                r["out_rel_real"] = rel(o2[mm], b2[mm])
                r["out_rel_pad"] = rel(o2[~mm], b2[~mm])
        arms[f"eps_{eps:.3e}"] = r
        print(f"[{time.time()-t0:.0f}s] eps {eps:.3e}: out {r['out_rel']:.3e}, "
              f"amplification {r['amplification']:.1f}x, "
              f"real {r.get('out_rel_real')}, pad {r.get('out_rel_pad')}", flush=True)

    rep["control"] = arms
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[{time.time()-t0:.0f}s] wrote {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
