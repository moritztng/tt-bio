#!/usr/bin/env python3
"""Capture the diffusion-module boundary of BUNDLE-MIN's own r = 0 step.

Why a boundary and not the bundle entries directly. `diffusion_module.*` is 738 of 4,147
tensors and 91.21 % of the bundle's squared gradient norm, so it is the scope this row must
reach. But our stack cannot run their trunk, and the trunk is where D19 lives: a CPU float64
replay of their step reads loss 1.6311432393241485 against the bundle's 1.6422035029890711,
6.73e-03 apart with every draw replayed. A gradient comparison cannot be tighter than the
forward it is taken at, so comparing our diffusion gradient against the bundle entries would
measure D19, which is another row's defect, and not ours.

The diffusion module's parameters appear nowhere else in the graph, so the gradient of the loss
with respect to them is EXACTLY the gradient of the diffusion module's own local function driven
by the cotangent arriving at its output. Capturing (kwargs, xl_out, dL/dxl_out) of their run
therefore turns this into a self-contained per-parameter reference that both stacks can be
driven by, and D19 cancels because it is entirely upstream of the captured inputs.

Two numbers come out of one run and they answer different questions:

  * `vs_bundle`: the diffusion gradient recomputed HERE against the bundle's own entries. That
    is D19's reach into this row's 91.21 %, measured rather than assumed, and it belongs to
    `of3t-reference`.
  * the saved boundary: inputs plus cotangent plus the float64 gradient, which is the reference
    the device comparison in compare_diffusion_gradient.py is run against.

Reuses `of3t-gradients`' harness rather than restating it: ReplayDraws, verify, manifest_from_git
and sha256_file are imported from capture_trunk_boundary.py, and bundle_min.py builds the model.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "of3t_gradients"))
sys.path.insert(0, "/home/ttuser/of3t_gradients/ref")

import capture_trunk_boundary as CTB  # noqa: E402  the harness, not a second copy of it

BUNDLE = CTB.BUNDLE
CKPT = CTB.CKPT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/home/ttuser/of3t_diffusion_cap"),
                    help="durable. /tmp is swept and this capture costs minutes of float64 CPU.")
    ap.add_argument("--report", type=Path,
                    default=Path("perf/of3t_diffusion/capture_diffusion_boundary.json"))
    ap.add_argument("--seed", type=int, default=20260919)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import bundle_min as BM

    man = CTB.manifest_from_git()
    files = ["batch_step003.pt", "draws_recycles0.pt", "grads_f64_r0.pt", "grad_presence_r0.json"]
    rep = {"instrument": "capture of the diffusion-module boundary of BUNDLE-MIN's r = 0 step",
           "manifest": f"{CTB.REF_BRANCH}:{CTB.MANIFEST_GIT}",
           "manifest_commit": subprocess.run(["git", "rev-parse", CTB.REF_BRANCH],
                                             capture_output=True, text=True).stdout.strip(),
           "bundle_dir": str(BUNDLE)}
    rep["hashes"] = CTB.verify(man, files)
    vg = man["r0_replay_gradient"]
    rep["reference"] = {"file": vg["file"], "num_recycles": vg["num_recycles"],
                        "loss": vg["loss"], "global_norm": vg["gradient_global_norm"],
                        "n_parameters": vg["n_parameters"],
                        "fd_max_rel": vg["finite_difference"]["max_rel_err"]}
    print(f"[{time.time()-t0:.0f}s] bundle verified, {len(files)} files", flush=True)

    dtype = torch.float64
    raw = torch.load(BUNDLE / "batch_step003.pt", weights_only=False)
    draws = torch.load(BUNDLE / "draws_recycles0.pt", map_location="cpu", weights_only=False)
    rep["draws"] = {"num_recycles": int(draws["num_recycles"]),
                    "n_torch_randn": len(draws["torch_randn"]),
                    "n_python_random": len(draws["python_random"])}

    built = BM.build(dtype, a.seed, "cpu", num_recycles=0)
    cfg, model, loss_fn = built[0], built[1], built[2]
    rep["their_build_disabled"] = (built[3] if len(built) > 3 else None)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
          for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)
    rep["checkpoint"] = {"file": CKPT.name, "n_loaded": len(sd),
                         "n_missing": len(inc.missing_keys),
                         "missing": list(inc.missing_keys)}
    print(f"[{time.time()-t0:.0f}s] model built, checkpoint loaded", flush=True)

    dm = model.diffusion_module
    rep["no_samples"] = int(model.shared.diffusion.no_samples)
    cap: dict = {}

    def pre_hook(mod, args, kwargs):
        cap["args"] = [x.detach().clone() if torch.is_tensor(x) else x for x in args]
        cap["kwargs"] = {k: CTB_clone(v) for k, v in kwargs.items()}

    def CTB_clone(v):
        if torch.is_tensor(v):
            return v.detach().clone()
        if isinstance(v, dict):
            return {k: CTB_clone(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(CTB_clone(x) for x in v)
        return v

    def post_hook(mod, args, kwargs, out):
        cap["out_live"] = out
        cap["out"] = out.detach().clone() if torch.is_tensor(out) else \
            [x.detach().clone() for x in out]

    h1 = dm.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = dm.register_forward_hook(post_hook, with_kwargs=True)

    BM.set_rng_state(draws["rng_state_before_step"], model)
    replay = CTB.ReplayDraws(draws["torch_randn"], draws["python_random"])
    batch = BM.move(raw, "cpu", dtype)
    t1 = time.time()
    with BM.no_autocast(), replay:
        private = copy.deepcopy(batch)
        b, out = model(private)
        loss, breakdown = loss_fn(b, out, _return_breakdown=True)
    t_fwd = time.time() - t1
    h1.remove(); h2.remove()
    rep["replay"] = {"randn_consumed": replay.i_randn, "random_consumed": replay.i_rnd,
                     "mismatches": replay.mismatch[:8], "n_mismatch": len(replay.mismatch)}
    rep["forward"] = {"loss": float(loss), "bundle_loss": vg["loss"],
                      "rel": abs(float(loss) - vg["loss"]) / abs(vg["loss"]),
                      "seconds": t_fwd, "drawn_recycles": int(out["recycles"])}
    print(f"[{time.time()-t0:.0f}s] forward {t_fwd:.0f}s loss {float(loss):.16f} vs bundle "
          f"{vg['loss']:.16f} rel {rep['forward']['rel']:.3e} mism {len(replay.mismatch)}",
          flush=True)

    # The backward is PRUNED to the diffusion subgraph: autograd walks only what leads to the
    # requested inputs, so the 48-block pairformer is never traversed. The whole-model backward
    # in this configuration costs 1040 s and none of it is needed here.
    xl = cap["out_live"]
    names, params = zip(*[(n, p) for n, p in dm.named_parameters()])
    t1 = time.time()
    g = torch.autograd.grad(loss, (xl,) + tuple(params), allow_unused=True, retain_graph=False)
    rep["backward_seconds"] = time.time() - t1
    cot, gp = g[0], g[1:]
    ours_here = {n: (x.detach().clone() if x is not None else None) for n, x in zip(names, gp)}
    print(f"[{time.time()-t0:.0f}s] pruned backward {rep['backward_seconds']:.0f}s, "
          f"cotangent norm {float(cot.norm()):.6e}", flush=True)

    # ---- the capture is CHECKED against the bundle, not asserted -----------------------------
    ref = torch.load(BUNDLE / "grads_f64_r0.pt", map_location="cpu", weights_only=False)

    def rel(x, y):
        return float(torch.linalg.vector_norm(x - y) / (torch.linalg.vector_norm(y) + 1e-300))

    worst, worst_n, n_cmp, n_absent = -1.0, None, 0, 0
    sq_here = sq_ref = 0.0
    per = {}
    for n, gg in ours_here.items():
        r = ref.get("diffusion_module." + n)
        if gg is None or r is None:
            n_absent += 1
            continue
        v = rel(gg.double(), r.double())
        n_cmp += 1
        sq_here += float(gg.double().pow(2).sum())
        sq_ref += float(r.double().pow(2).sum())
        per[n] = v
        if v > worst:
            worst, worst_n = v, n
    tot_ref_sq = sum(float(v.double().pow(2).sum()) for v in ref.values() if v is not None)
    rep["vs_bundle"] = {
        "compared": n_cmp, "absent_either_side": n_absent, "worst_rel": worst,
        "worst_tensor": "diffusion_module." + str(worst_n),
        "median_rel": float(torch.tensor(sorted(per.values())).median()) if per else None,
        "sq_norm_here": sq_here, "sq_norm_bundle": sq_ref,
        "share_of_bundle_sq_norm": sq_ref / tot_ref_sq,
        "bundle_total_sq_norm": tot_ref_sq,
        "what": "D19's reach into this row's scope: their model on CPU float64 at a replayed "
                "forward, against the bundle's own diffusion entries. Not this row's number."}
    print(f"[{time.time()-t0:.0f}s] vs bundle: {n_cmp} tensors, worst {worst:.3e} on "
          f"{worst_n}, median {rep['vs_bundle']['median_rel']:.3e}, scope "
          f"{100*rep['vs_bundle']['share_of_bundle_sq_norm']:.4f} % of squared norm", flush=True)

    blob = {"kwargs": cap["kwargs"], "args": cap["args"], "out": cap["out"], "cot": cot.detach(),
            "grad_f64": ours_here, "loss": float(loss), "no_samples": rep["no_samples"]}
    p = a.out / "diffusion_boundary.pt"
    torch.save(blob, p)
    rep["saved"] = {"file": str(p), "bytes": p.stat().st_size,
                    "out_shape": tuple(cap["out"].shape),
                    "cot_norm": float(cot.norm()),
                    "kwarg_shapes": {k: tuple(v.shape) for k, v in cap["kwargs"].items()
                                     if torch.is_tensor(v)},
                    "n_grad": len(ours_here)}
    rep["breakdown"] = {k: float(v) for k, v in breakdown.items()
                        if torch.is_tensor(v) and v.numel() == 1}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[{time.time()-t0:.0f}s] wrote {a.report} and {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
