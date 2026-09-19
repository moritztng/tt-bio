#!/usr/bin/env python3
"""Capture the pairformer-block boundary of the frozen bundle's own step, so instrument A's
magnitude half can be measured against BUNDLE-MIN rather than against a self-generated reference.

The bundle publishes a per-parameter float64 gradient for the whole model. Our stack has no OF3
training forward, so the whole-model comparison it invites cannot be run here. But the gradient
of the loss with respect to pairformer block i's parameters is EXACTLY the gradient of that one
block's local function driven by the cotangent arriving at its outputs, because block i's
parameters appear nowhere else in the graph and, at num_recycles 0, the trunk runs once. So
capturing (s_in, z_in, masks) and (ds_out, dz_out) at the block boundary of THEIR run turns the
bundle's `pairformer_stack.blocks.i.*` entries into a usable per-parameter reference for a single
block we can actually run on a card.

What makes the capture theirs rather than a plausible neighbour of theirs, in order of how badly
each one bites:

  * the weights are their checkpoint, loaded the way `bundle_min.py` loads it;
  * `num_recycles` is pinned to 0, the same pinning the validated gradient was taken under;
  * every stochastic draw is REPLAYED from `draws_recycles0.pt` rather than re-drawn. Their run
    was on CUDA, so a CPU replay that merely restores the RNG states draws different numbers from
    the same seeds. Replaying the recorded values is the only way a CPU run evaluates the same
    function;
  * and the capture is checked, not asserted: the loss is compared against the manifest's, and
    the block's own parameter gradients recomputed here are compared entry-by-entry against the
    bundle's. If those two agree, the captured cotangent is the one the bundle was taken with.

Nothing here writes to the bundle. Every artifact it reads is hashed against MANIFEST.json first.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

BUNDLE = Path("/home/ttuser/of3t/bundle_min")
CKPT = Path(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"))
REF_BRANCH = "origin/wk/of3t-reference"
MANIFEST_GIT = "perf/of3t_reference/bundle_min/MANIFEST.json"


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def manifest_from_git() -> dict:
    """The manifest as `of3t-reference` published it, never a working-tree copy of it."""
    out = subprocess.run(["git", "show", f"{REF_BRANCH}:{MANIFEST_GIT}"],
                         capture_output=True, check=True)
    return json.loads(out.stdout)


def verify(man: dict, names: list[str]) -> dict:
    decl = {a["file"]: a for a in man["artifacts"] if "sha256" in a}
    rep = {}
    for n in names:
        p = BUNDLE / n
        want = decl.get(n, {}).get("sha256")
        got = sha256_file(p)
        rep[n] = {"bytes": p.stat().st_size, "sha256": got, "declared": want,
                  "match": (want == got)}
        if want is not None and want != got:
            raise SystemExit(f"{n}: sha256 {got} != manifest {want} -- a drifted bundle still "
                             f"produces numbers, which is worse than no bundle")
        if want is None:
            raise SystemExit(f"{n}: not declared in {REF_BRANCH}:{MANIFEST_GIT}")
    return rep


class ReplayDraws:
    """Consume the recorded draws in order instead of sampling new ones.

    PROTOCOL 4a: the draws are inputs to the update rule. Their run drew `torch.randn` on CUDA;
    restoring the CPU RNG state on this box reproduces the STATE and not the VALUES, so a replay
    that only restores state evaluates a different function and every comparison downstream is
    void. The shape of each call is checked against the recorded draw, which is what turns a
    silent misalignment into a failure.
    """

    def __init__(self, randn, rnd):
        self.randn, self.rnd = list(randn), list(rnd)
        self.i_randn = self.i_rnd = 0
        self.mismatch = []

    def __enter__(self):
        self._torch_randn, self._random_random = torch.randn, random.random

        def randn(*args, **kwargs):
            shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
            if self.i_randn >= len(self.randn):
                self.mismatch.append(f"call {self.i_randn}: no recorded draw left")
                return self._torch_randn(*args, **kwargs)
            v = self.randn[self.i_randn]
            self.i_randn += 1
            if tuple(v.shape) != tuple(int(x) for x in shape):
                self.mismatch.append(
                    f"call {self.i_randn - 1}: recorded {tuple(v.shape)} != asked "
                    f"{tuple(int(x) for x in shape)}")
                return self._torch_randn(*args, **kwargs)
            dt = kwargs.get("dtype", None) or torch.get_default_dtype()
            return v.to(device=kwargs.get("device", "cpu"), dtype=dt).clone()

        def rnd():
            if self.i_rnd >= len(self.rnd):
                self.mismatch.append(f"random.random call {self.i_rnd}: no recorded value left")
                return self._random_random()
            v = self.rnd[self.i_rnd]
            self.i_rnd += 1
            return v

        torch.randn, random.random = randn, rnd
        return self

    def __exit__(self, *exc):
        torch.randn, random.random = self._torch_randn, self._random_random
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="0,23,47",
                    help="which pairformer blocks to capture the boundary of")
    ap.add_argument("--out", type=Path, default=Path("/home/ttuser/of3t_gradients/cap"),
                    help="durable by default. /tmp is swept, and this capture costs 26 minutes "
                         "of CPU to rebuild.")
    ap.add_argument("--no-dropout", action="store_true",
                    help="put every Dropout module in eval for the whole step. The bundle was "
                         "taped at r = 0.25 with a mask drawn from the CUDA stream and never "
                         "recorded, so a replay on another device cannot reproduce it and its "
                         "draw-to-draw floor is median 0.48-0.55 on the block gradient "
                         "(dropout_floor_block0.json). At r = 0 the whole capture is "
                         "bit-reproducible, and r = 0 is also the only function our tape can "
                         "compute -- tt_bio/train/lora.py:46, the tape has no dropout op.")
    ap.add_argument("--report", type=Path,
                    default=Path("perf/of3t_gradients/capture_trunk_boundary.json"))
    ap.add_argument("--seed", type=int, default=20260919)
    a = ap.parse_args()
    want_blocks = [int(x) for x in a.blocks.split(",") if x != ""]
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Durable, not /tmp. The campaign's slug scratch under /tmp/of3t/<slug>/ is swept, and
    # it took this capture (26 min of CPU) and another row's vendored openfold3 0.5.0 with
    # it mid-pass. Expensive inputs live under ~/of3t_gradients/ now.
    sys.path.insert(0, "/home/ttuser/of3t_gradients/ref")
    import bundle_min as BM

    man = manifest_from_git()
    files = ["batch_step003.pt", "draws_recycles0.pt", "grads_f64_recycles0.pt",
             "grad_presence_recycles0.json"]
    rep = {"instrument": "capture of the pairformer block boundary of BUNDLE-MIN's own step",
           "manifest": f"{REF_BRANCH}:{MANIFEST_GIT}",
           "manifest_commit": subprocess.run(["git", "rev-parse", REF_BRANCH],
                                             capture_output=True, text=True).stdout.strip(),
           "bundle_dir": str(BUNDLE), "blocks": want_blocks}
    rep["hashes"] = verify(man, files)
    vg = man["validated_gradient"]
    rep["reference"] = {"file": vg["file"], "num_recycles": vg["num_recycles"],
                        "loss": vg["loss"], "global_norm": vg["gradient_global_norm"],
                        "n_nonzero": vg["n_nonzero_gradient_tensors"],
                        "n_parameters": vg["n_parameters"],
                        "fd_max_rel": vg["finite_difference"]["max_rel_err"]}
    print(f"[{time.time()-t0:.0f}s] bundle verified, {len(files)} files", flush=True)

    dtype = torch.float64
    raw = torch.load(BUNDLE / "batch_step003.pt", weights_only=False)
    draws = torch.load(BUNDLE / "draws_recycles0.pt", map_location="cpu", weights_only=False)
    rep["draws"] = {"num_recycles": int(draws["num_recycles"]),
                    "n_torch_randn": len(draws["torch_randn"]),
                    "n_python_random": len(draws["python_random"])}

    # `of3t-reference` acted on the dropout finding and its `build` now pins every Dropout to
    # r = 0 itself, returning a fourth value describing what it disabled. Taking the first
    # three keeps this working against both revisions rather than pinning to one.
    built = BM.build(dtype, a.seed, "cpu", num_recycles=0)
    cfg, model, loss_fn = built[0], built[1], built[2]
    rep["their_build_disabled"] = (built[3] if len(built) > 3 else None)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
          for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)
    rep["checkpoint"] = {"file": CKPT.name, "sha256": sha256_file(CKPT), "n_loaded": len(sd),
                         "n_missing": len(inc.missing_keys),
                         "n_unexpected": len(inc.unexpected_keys),
                         "missing": list(inc.missing_keys)}
    print(f"[{time.time()-t0:.0f}s] model built, checkpoint loaded "
          f"({len(sd)} / {len(inc.missing_keys)} missing)", flush=True)

    if a.no_dropout:
        from openfold3.core.model.primitives.dropout import Dropout
        n_drop = 0
        for m in model.modules():
            if isinstance(m, Dropout):
                m.eval()
                n_drop += 1
        rep["dropout"] = {"disabled": True, "modules": n_drop,
                          "why": "the bundle's masks were drawn from the CUDA stream and are not "
                                 "in draws_recycles0.pt; at r = 0 the capture is reproducible and "
                                 "matches the function our tape computes"}
    else:
        rep["dropout"] = {"disabled": False,
                          "why": "as published, r = 0.25, mask drawn locally and therefore NOT "
                                 "the mask the published gradient was taken with"}

    stack = model.pairformer_stack
    blocks = stack.blocks
    rep["n_blocks"] = len(blocks)
    cap: dict[int, dict] = {i: {} for i in want_blocks}

    def pre_hook(i):
        def h(mod, args, kwargs):
            d = cap[i]
            d["args"] = [x.detach().clone() if torch.is_tensor(x) else x for x in args]
            d["kwargs"] = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                           for k, v in kwargs.items()}
        return h

    def post_hook(i):
        def h(mod, args, kwargs, out):
            d = cap[i]
            d["out"] = [x.detach().clone() if torch.is_tensor(x) else x for x in out]
            d["cot"] = [None] * len(out)
            for j, t in enumerate(out):
                if torch.is_tensor(t) and t.requires_grad:
                    t.register_hook(lambda g, j=j, d=d: d["cot"].__setitem__(j, g.detach().clone()))
        return h

    handles = []
    for i in want_blocks:
        handles.append(blocks[i].register_forward_pre_hook(pre_hook(i), with_kwargs=True))
        handles.append(blocks[i].register_forward_hook(post_hook(i), with_kwargs=True))

    BM.set_rng_state(draws["rng_state_before_step"], model)
    replay = ReplayDraws(draws["torch_randn"], draws["python_random"])
    batch = BM.move(raw, "cpu", dtype)
    t1 = time.time()
    with BM.no_autocast(), replay:
        private = copy.deepcopy(batch)
        b, out = model(private)
        loss, breakdown = loss_fn(b, out, _return_breakdown=True)
    t_fwd = time.time() - t1
    rep["replay"] = {"randn_consumed": replay.i_randn, "random_consumed": replay.i_rnd,
                     "mismatches": replay.mismatch[:8], "n_mismatch": len(replay.mismatch)}
    rep["forward"] = {"loss": float(loss), "their_loss": vg["loss"],
                      "rel": abs(float(loss) - vg["loss"]) / abs(vg["loss"]),
                      "seconds": t_fwd,
                      "drawn_recycles": int(out["recycles"])}
    print(f"[{time.time()-t0:.0f}s] forward {t_fwd:.0f}s, loss {float(loss):.15f} vs their "
          f"{vg['loss']:.15f} rel {rep['forward']['rel']:.3e}, "
          f"randn {replay.i_randn}/{len(draws['torch_randn'])}, "
          f"mismatches {len(replay.mismatch)}", flush=True)

    t1 = time.time()
    loss.backward()
    rep["backward_seconds"] = time.time() - t1
    for h in handles:
        h.remove()
    print(f"[{time.time()-t0:.0f}s] backward {rep['backward_seconds']:.0f}s", flush=True)

    # ---- the capture is checked against the bundle, not asserted ---------------------------
    ref = torch.load(BUNDLE / "grads_f64_recycles0.pt", map_location="cpu", weights_only=False)
    ours_here = {n: (p.grad.detach().clone() if p.grad is not None else None)
                 for n, p in model.named_parameters()}
    def rel(a_, b_):
        return float(torch.linalg.vector_norm(a_ - b_) / (torch.linalg.vector_norm(b_) + 1e-300))
    check = {}
    for i in want_blocks:
        pre = f"pairformer_stack.blocks.{i}."
        worst, worst_n = -1.0, None
        n_cmp = n_absent = 0
        for n, g in ours_here.items():
            if not n.startswith(pre):
                continue
            r = ref.get(n)
            if g is None or r is None:
                n_absent += 1
                continue
            v = rel(g.to(torch.float64), r.to(torch.float64))
            n_cmp += 1
            if v > worst:
                worst, worst_n = v, n
        check[str(i)] = {"compared": n_cmp, "absent_either_side": n_absent,
                         "worst_rel": worst, "worst_tensor": worst_n}
        print(f"  block {i}: {n_cmp} tensors, worst {worst:.3e} on {worst_n}", flush=True)
    rep["capture_vs_bundle"] = check
    gn = float(torch.linalg.vector_norm(torch.stack(
        [torch.linalg.vector_norm(g.double()) for g in ours_here.values() if g is not None])))
    rep["global_norm_here"] = gn
    rep["global_norm_bundle"] = vg["gradient_global_norm"]
    rep["global_norm_rel"] = abs(gn - vg["gradient_global_norm"]) / vg["gradient_global_norm"]
    rep["n_nonzero_here"] = sum(1 for g in ours_here.values()
                                if g is not None and bool(g.abs().max() > 0))
    rep["n_parameters_here"] = len(ours_here)
    print(f"[{time.time()-t0:.0f}s] global norm {gn:.12f} vs {vg['gradient_global_norm']:.12f}, "
          f"non-zero {rep['n_nonzero_here']}/{rep['n_parameters_here']}", flush=True)

    # ---- save the boundary -------------------------------------------------------------------
    saved = {}
    for i in want_blocks:
        d = cap[i]
        blob = {"args": d.get("args"), "kwargs": d.get("kwargs"), "out": d.get("out"),
                "cot": d.get("cot"),
                "grad": {n[len(f"pairformer_stack.blocks.{i}."):]: g
                         for n, g in ours_here.items()
                         if n.startswith(f"pairformer_stack.blocks.{i}.")}}
        p = a.out / f"block{i}_boundary.pt"
        torch.save(blob, p)
        saved[str(i)] = {"file": str(p), "bytes": p.stat().st_size,
                         "arg_shapes": [tuple(x.shape) for x in blob["args"] if torch.is_tensor(x)],
                         "kwarg_shapes": {k: tuple(v.shape) for k, v in (blob["kwargs"] or {}).items()
                                          if torch.is_tensor(v)},
                         "out_shapes": [tuple(x.shape) for x in blob["out"]],
                         "cot_present": [c is not None for c in blob["cot"]],
                         "cot_norms": [None if c is None else float(c.norm()) for c in blob["cot"]]}
    rep["saved"] = saved
    rep["breakdown"] = {k: float(v) for k, v in breakdown.items()
                        if torch.is_tensor(v) and v.numel() == 1}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(f"[{time.time()-t0:.0f}s] wrote {a.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
