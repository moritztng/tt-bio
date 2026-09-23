#!/usr/bin/env python3
"""Upstream 0.4.3's own forward AND gradient at the msa_module boundary, out of ONE process.

`of3t-tapeamp`'s amp_arm.py moved to the other leg of D58. The identity pin, the scorer and the
median convention are imported from it rather than copied, so the two legs are scored by the same
code. What differs is the boundary: `boundary_msa_module.pt`, the float64 capture written by
perf/of3t_auxheads/capture_boundary.py, whose 227 msa_module parameter gradients are
bit-identical to grads_f64_043.pt.

The forward statistic is the one msa_instrument.py reports for our device arm: rel_l2 of z_out on
the real token block (diag of pair_mask), so the two forwards are comparable without a
conversion. The full 384x384 figure is published beside it for the upstream arms only.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_tapeamp"))
from amp_arm import cast, median, rel, score_grads, upstream_identity  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=("bf16auto", "f32", "f64"))
    ap.add_argument("--boundary", type=Path, required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--ckpt", type=Path, default=Path("/home/ttuser/of3-weights/of3-p2-155k.pt"))
    ap.add_argument("--bf16-inputs", action="store_true", dest="bf16_inputs",
                    help="round m and z to bf16 before entry, which is what our device arm "
                         "uploads. Separates input precision at the boundary from the module.")
    ap.add_argument("--break-cot", action="store_true", dest="break_cot",
                    help="roll the cotangent's token axis by one. The forward does not read it, "
                         "so the forward must come back bit-identical while the gradient moves.")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    t0 = time.time()

    root, ver, meta, ln_sig, dig = upstream_identity()
    print(f"upstream openfold3 {ver} at {root} (from {meta}), LayerNorm: {ln_sig}", flush=True)
    if ver != a.expect_version or dig["version_from_digest"] != a.expect_version:
        print(f"HARD FAILURE: declared {a.expect_version}, resolved {ver}, digest {dig}")
        return 3

    import bundle_min as BM

    B = torch.load(a.boundary, map_location="cpu", weights_only=False)
    args, kwargs = B["inputs"]["args"], B["inputs"]["kwargs"]
    z_ref, cot, ref_grad = B["outputs"], B["cotangents"]["out"], B["param_grads"]
    if z_ref.dtype != torch.float64:
        print(f"HARD FAILURE: the capture's output is {z_ref.dtype}, not a float64 reference")
        return 4
    n = int(z_ref.shape[-2])
    tokm = torch.diagonal(kwargs["pair_mask"].reshape(n, n)) > 0
    print(f"[{time.time()-t0:.0f}s] boundary: z {tuple(z_ref.shape)}, {int(tokm.sum())} real "
          f"tokens, m {tuple(args[0].shape)}, {len(ref_grad)} reference tensors", flush=True)

    dt = torch.float64 if a.policy == "f64" else torch.float32
    model = BM.build(dt, 20260919, "cpu", num_recycles=0)[1]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: (v.to(dt) if torch.is_tensor(v) and v.is_floating_point() else v)
          for k, v in sd.items()}
    got = model.load_state_dict(sd, strict=False)
    mm = model.msa_module
    del ck, sd
    # D141's guard, scoped to the one module this arm measures.
    unexpected = [k for k in got.unexpected_keys if k.startswith("msa_module.")]
    missing = [k for k in got.missing_keys if k.startswith("msa_module.")]
    tree = {f"msa_module.{k}" for k, _ in mm.named_parameters()}
    fp = {"n_parameters": len(tree), "capture_keys_equal_tree_parameters": set(ref_grad) == tree,
          "msa_unexpected_keys": len(unexpected), "msa_missing_keys": len(missing),
          "training_mode": mm.training}
    print(f"[{time.time()-t0:.0f}s] msa_module at {dt}, fingerprint {json.dumps(fp)}", flush=True)
    if unexpected or missing or not fp["capture_keys_equal_tree_parameters"]:
        print("HARD FAILURE: the tree cannot express the checkpoint's msa_module (D141)")
        return 5

    probe = {}

    def dtype_hook(mod, inp, out):
        if torch.is_tensor(out):
            probe.setdefault("first_linear_out_dtype", str(out.dtype))

    for name, mod in mm.named_modules():
        if isinstance(mod, torch.nn.Linear):
            probe["probed_module"] = f"msa_module.{name}"
            h = mod.register_forward_hook(dtype_hook)
            break

    args, kwargs = cast(args, dt), cast(kwargs, dt)
    if a.bf16_inputs:
        args = tuple(x.to(torch.bfloat16).to(dt) for x in args)
    ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if a.policy == "bf16auto"
           else BM.no_autocast() if a.policy == "f64" else torch.autocast("cpu", enabled=False))
    params = [p for _, p in mm.named_parameters()]
    names = [f"msa_module.{k}" for k, _ in mm.named_parameters()]
    load0 = os.getloadavg()
    t1 = time.time()
    with ctx:
        z = mm(*args, **kwargs)
    t_fwd = time.time() - t1
    h.remove()
    if a.policy == "bf16auto" and probe.get("first_linear_out_dtype") != "torch.bfloat16":
        print(f"HARD FAILURE: the bf16 recipe did not reach the kernels: {probe}")
        return 2
    if tuple(z.shape) != tuple(z_ref.shape):
        print(f"HARD FAILURE: z {tuple(z.shape)} is not the reference's {tuple(z_ref.shape)}")
        return 6
    zd, zr = z.detach().reshape(n, n, -1), z_ref.reshape(n, n, -1)
    f_real, f_real_norm = rel(zd[tokm][:, tokm], zr[tokm][:, tokm])
    f_full, _ = rel(zd, zr)
    print(f"[{time.time()-t0:.0f}s] forward {t_fwd:.1f}s, z {z.dtype}, real-block rel_l2 "
          f"{f_real:.10e}, full {f_full:.6e}, probe {probe}", flush=True)

    cot_used = torch.roll(cot, 1, dims=1) if a.break_cot else cot
    t1 = time.time()
    g = torch.autograd.grad(z, params, grad_outputs=cot_used.to(z.dtype), allow_unused=True)
    t_bwd = time.time() - t1
    load1 = os.getloadavg()
    grads = {nm: (x.detach().to(torch.float64) if x is not None else None)
             for nm, x in zip(names, g)}
    print(f"[{time.time()-t0:.0f}s] backward {t_bwd:.1f}s, "
          f"{sum(v is None for v in grads.values())} parameters with no gradient", flush=True)
    if a.dump:
        torch.save({"policy": a.policy, "grads": grads, "boundary": str(a.boundary)}, a.dump)

    full, per = score_grads(grads, ref_grad)
    rep = {
        "what": __doc__.strip().splitlines()[0], "tag": a.tag, "policy": a.policy,
        "bf16_inputs": a.bf16_inputs, "break_cot": a.break_cot,
        "boundary": str(a.boundary), "ckpt": str(a.ckpt),
        "upstream_pkg": root, "upstream_version": ver, "upstream_version_source": meta,
        "LAYERNORM_SIGNATURE": ln_sig, "TREE_DIGEST": dig, "FINGERPRINT": fp,
        "DTYPE_PROBE": probe, "z_dtype": str(z.dtype),
        "host": socket.gethostname(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch": torch.__version__, "loadavg_before_forward": load0,
        "loadavg_after_backward": load1, "forward_seconds": t_fwd, "backward_seconds": t_bwd,
        "cot_norm": float(cot.double().norm()), "cot_used_norm": float(cot_used.double().norm()),
        "FORWARD": {"statistic": "rel_l2 of z_out on the real token block, msa_instrument.py's "
                                 "rel_l2_real_block", "real_block": f_real,
                    "real_block_reference_norm": f_real_norm, "full_384": f_full,
                    "n_real_tokens": int(tokm.sum())},
        "GRADIENT_full_scope": full, "PER_TENSOR": per,
        # None when the forward is exact: the f64 arm reproduces the capture to 0.0.
        "RATIO_full_scope": {"mass_weighted": full["mass_weighted_rel"] / f_real if f_real else None,
                             "median": full["median_rel"] / f_real if f_real else None},
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "PER_TENSOR"}, indent=1,
                     sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
