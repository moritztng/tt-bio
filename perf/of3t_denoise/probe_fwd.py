#!/usr/bin/env python3
"""of3t-denoise: is the taped denoise arm's forward the shipped diffusion module's forward?

    probe_fwd.py --batch B --out F.json

Runs the adapter's training forward once with the denoise arm on, captures the arguments of the
taped `OF3DiffusionModule` call, and re-runs the same module untaped (the inference path) on the
same device inputs. Both outputs, and the noisy input itself, are scored against the ground truth
at atom scope (RMS per atom, Angstrom). Forward only; no gradient.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    argv = sys.argv[1:]
    bpath = Path(argv[argv.index("--batch") + 1])
    out = Path(argv[argv.index("--out") + 1])
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.train.openfold3 import OpenFold3Dataset, OpenFold3Forward, denoise_draw
    import tt_bio.openfold3_diffusion_module as dmm

    ds = OpenFold3Dataset(bpath)
    fwd = OpenFold3Forward(Path.home() / "of3-weights/of3-p2-155k.pt", rollout=20,
                           num_cycles=1, seed=20260922)
    batch = ds.batch([0])
    cls = dmm.OF3DiffusionModule
    orig = cls.__call__
    cap = {}

    def spy(self, *args, **kw):
        y = orig(self, *args, **kw)
        if any(isinstance(a, ag.Tensor) for a in args):
            cap["args"], cap["kw"], cap["self"], cap["y"] = args, kw, self, y
        return y

    cls.__call__ = spy
    outputs = fwd(batch)
    cls.__call__ = orig
    raw = lambda t: t.value if isinstance(t, ag.Tensor) else t  # noqa: E731
    host = lambda t: ttnn.to_torch(raw(t)).double()  # noqa: E731
    args = [raw(a) for a in cap["args"]]
    kw = dict(cap["kw"], cache={})
    y_untaped = orig(cap["self"], *args, **kw)
    y_untaped2 = orig(cap["self"], *args, **dict(kw, cache={}))
    to32 = lambda t: ttnn.typecast(t, ttnn.float32) if t.dtype == ttnn.bfloat16 else t  # noqa
    a32 = list(args)
    a32[1], a32[2] = to32(args[1]), to32(args[2])
    y_fp32c = orig(cap["self"], *a32, **dict(kw, cache={}))

    f = batch["features"]
    amask = torch.as_tensor(np.asarray(f["atom_mask"])).double().reshape(-1)
    n_atom = amask.numel()
    gt = torch.as_tensor(np.asarray(f["ground_truth"]["atom_positions"])).double().reshape(n_atom, 3)
    sigma, eps = denoise_draw(fwd.seed, n_atom)
    noisy = (gt + sigma * torch.as_tensor(eps)) * amask[:, None]
    m = amask > 0

    def rms(x, ref=gt):
        d = (x.reshape(-1, 3)[:n_atom] - ref)[m]
        return float((d ** 2).sum(1).mean().sqrt())

    yt, yu = host(cap["y"]), host(y_untaped)
    yu2, yf = host(y_untaped2), host(y_fp32c)
    # centred: the same noise on the centred truth, uploaded fp32 as the sampler does
    NP = args[20]
    dev = args[5].device()
    cen = gt[m].mean(0)
    gtc = (gt - cen) * amask[:, None]
    noisyc = (gtc + sigma * torch.as_tensor(eps)) * amask[:, None]
    sd = 16.0
    rl = torch.nn.functional.pad(noisyc / (sigma * sigma + sd * sd) ** 0.5, (0, 0, 0, NP - n_atom))
    up = lambda x: ttnn.from_torch(x.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev,  # noqa
                                   dtype=ttnn.float32)
    ac = list(a32)
    ac[5], ac[6] = up(rl), up(noisyc)
    ycen = host(orig(cap["self"], *ac, **dict(kw, cache={})))
    au = list(a32)
    au[5], au[6] = up(torch.nn.functional.pad((noisy / (sigma * sigma + sd * sd) ** 0.5),
                                              (0, 0, 0, NP - n_atom))), up(noisy)
    yf32in = host(orig(cap["self"], *au, **dict(kw, cache={})))
    rec = {"batch": str(bpath), "n_atom": n_atom, "n_real_atom": int(m.sum()), "sigma": sigma,
           "rms_A": {"noisy_input_vs_truth": rms(noisy),
                     "taped_vs_truth": rms(yt), "untaped_vs_truth": rms(yu),
                     "taped_vs_untaped": rms(yt, yu.reshape(-1, 3)[:n_atom]),
                     "untaped_AA": rms(yu, yu2.reshape(-1, 3)[:n_atom]),
                     "untaped_si_zij_fp32_vs_truth": rms(yf),
                     "untaped_fp32_inputs_vs_truth": rms(yf32in),
                     "centred_noisy_input_vs_centred_truth": rms(noisyc, gtc),
                     "untaped_centred_vs_centred_truth": rms(ycen, gtc)},
           "centroid_A": cen.tolist(),
           "dtypes": {"taped_out": str(raw(cap["y"]).dtype), "untaped_out": str(y_untaped.dtype),
                      "args": [str(getattr(a, "dtype", type(a).__name__)) for a in args[:20]]},
           "pred_xyz_host_rms_vs_true_xyz": None}
    tx = batch.get("true_xyz")
    if tx is not None and "pred_xyz" in outputs:
        p = host(outputs["pred_xyz"]).reshape(-1, 3)
        tm = torch.as_tensor(np.asarray(f["token_mask"])).reshape(-1) > 0
        t = torch.as_tensor(np.asarray(tx)).double().reshape(-1, 3)
        rec["pred_xyz_host_rms_vs_true_xyz"] = float(((p - t)[tm] ** 2).sum(1).mean().sqrt())
    print(json.dumps(rec, indent=1), flush=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
