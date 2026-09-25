#!/usr/bin/env python3
"""Capture upstream 0.4.3's cotangent at the INPUT EMBEDDER's atom-encoder boundaries.

`READABLE_MASS.json` files nine tensors under HOST_APPLIED whose second blocker is NO_ARM: no
device arm covers `input_embedder` on the model's own batch, so wiring the leg onto the card
yields no reading on its own. An arm needs a cotangent at the leg's own output, which is how
`perf/of3t_conditioning/device_cond_gradient.py` is seeded at the conditioning's. This produces
it, from one `loss.backward()` of the reference's own model on `batch_step003`.

TWO boundaries, not one, and the reason is in the shipped code rather than a preference. The
leg's pair completion (`linear_l`, `linear_m`, `pair_mlp`) runs on the HOST between `cl`/`plm`
and `ai` (`openfold3_host_prep.run_input_atom_encoder`), so one end-to-end tape from `ai` back to
`cl` does not exist on our side:

  * `(cl, plm)` -- the eight `ref_atom_feature_embedder` linears. Their gradient is a function of
    the module's own inputs and the cotangent here, and nothing downstream.
  * `(ai, ql)` -- `linear_q.0.weight`. `ai = aggregate_mean(relu(linear_q(ql * atom_mask)))`, so
    the cotangent at `ai` plus the value of `ql` is the whole of it.

Everything is upstream's: their model, their loss, their batch, their replayed draws, float64
with the autocast blocks removed -- the same recipe `grads_f64_043.pt` was taken with, which is
why the cotangent below can be compared against that file in one denominator. The published
diffusion capture's cotangent is float64 (`diffcap043/diffusion_boundary.pt`), so this one is too.

The parameter-free block inputs (`dlm`, `vlm`, `inv_sq_dists`) are NOT captured: both sides
derive them from the same batch with the shipped `openfold3_host_prep.ref_atom_block_inputs`, so
capturing them would replace a shipped derivation with a frozen copy of it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "of3t_reference"))

BATCH_SHA = "3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f"
WANT_FEATURES = ("ref_pos", "ref_charge", "ref_mask", "ref_element",
                 "ref_atom_name_chars", "ref_space_uid", "atom_mask",
                 "atom_to_token_index", "token_mask")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=Path("/home/ttuser/of3t_hostleg/bundle_ref"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.path.expanduser("~/of3-weights/of3-p2-155k.pt")))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/ttuser/of3t_hostleg/ie_boundary.pt"))
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--seed", type=int, default=20260919)
    a = ap.parse_args()
    t0 = time.time()

    import torch
    import bundle_min as BM

    got = sha256(a.bundle / "batch_step003.pt")
    if got != BATCH_SHA:
        raise SystemExit(f"STOP: batch is {got}, the pin says {BATCH_SHA}")
    dtype = getattr(torch, a.dtype)
    device = "cpu"   # a STRING: cast_policy hands it to torch.is_autocast_enabled, which refuses a torch.device

    det = BM.pin_deterministic_kernels(True)
    raw = torch.load(a.bundle / "batch_step003.pt", weights_only=False)
    cfg, model, loss_fn, dropout = BM.build(dtype, a.seed, device, 0)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
          for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)
    if inc.unexpected_keys:
        raise SystemExit(f"KEY GATE FAILED: {len(inc.unexpected_keys)} unexpected")
    print(f"[{time.time()-t0:.0f}s] model built, {len(sd)} tensors loaded, "
          f"{len(inc.missing_keys)} missing", flush=True)
    del ck, sd

    # --- the two hooks -------------------------------------------------------------------
    enc = dict(model.named_modules()).get("input_embedder.atom_attn_enc")
    if enc is None:
        cand = [n for n in dict(model.named_modules()) if "atom_attn_enc" in n]
        raise SystemExit(f"STOP: no input_embedder.atom_attn_enc; candidates {cand[:8]}")
    rafe = dict(model.named_modules())["input_embedder.atom_attn_enc.ref_atom_feature_embedder"]
    grab = {}

    def on_rafe(mod, inp, kwargs, out):
        # with_kwargs, because their encoder calls this module by KEYWORD
        # (batch=..., n_query=..., n_key=...) and a positional-only hook sees an empty tuple.
        cl, plm = out
        grab["cl"], grab["plm"] = cl, plm
        pos = list(inp)
        b = kwargs.get("batch", pos[0] if pos else None)
        if b is None:
            raise SystemExit(f"STOP: no batch in the RAFE call; args={len(pos)} "
                             f"kwargs={sorted(kwargs)}")
        grab["features"] = {k: b[k].detach().clone() for k in WANT_FEATURES if k in b}
        grab["features_absent"] = [k for k in WANT_FEATURES if k not in b]
        grab["n_query"] = kwargs.get("n_query", pos[1] if len(pos) > 1 else 32)
        grab["n_key"] = kwargs.get("n_key", pos[2] if len(pos) > 2 else 128)
        grab["rafe_calls"] = grab.get("rafe_calls", 0) + 1

    def on_enc(mod, inp, out):
        ai, ql = out[0], out[1]
        grab["ai"], grab["ql"] = ai, ql
        grab["enc_calls"] = grab.get("enc_calls", 0) + 1

    h1 = rafe.register_forward_hook(on_rafe, with_kwargs=True)
    h2 = enc.register_forward_hook(on_enc)

    batch = BM.move(raw, device, dtype)
    pinned = BM.rng_state(model)
    replay = torch.load(a.bundle / "draws_recycles0.pt", map_location="cpu", weights_only=False)
    BM.set_rng_state(pinned, model)
    rec = BM.DrawRecorder(replay)
    policy = BM.cast_policy("removed" if dtype is torch.float64 else "upstream", device)

    print(f"[{time.time()-t0:.0f}s] forward, dtype={a.dtype}", flush=True)
    loss, breakdown, out = BM.forward_loss(model, loss_fn, batch, rec, cast_ctx=policy)
    print(f"[{time.time()-t0:.0f}s] loss {float(loss):.15f}  "
          f"rafe_calls={grab.get('rafe_calls')} enc_calls={grab.get('enc_calls')}", flush=True)
    if grab.get("rafe_calls") != 1 or grab.get("enc_calls") != 1:
        raise SystemExit(f"STOP: the leg ran {grab.get('rafe_calls')} / {grab.get('enc_calls')} "
                         f"times, not once. A cotangent captured from the last of several calls "
                         f"is not the cotangent of the leg.")
    # `torch.autograd.grad`, not `retain_grad` inside the forward hook. The first attempt used
    # retain_grad and read `cl.grad is None` after `loss.backward()` -- and a None there says
    # nothing about WHY: it looks identical whether the tensor is off the path to the loss,
    # whether it was recomputed under gradient checkpointing, or whether the hook grabbed a
    # different object. `autograd.grad` with allow_unused answers the question instead of
    # reporting its symptom, and it returns the cotangents in the SAME backward that produces
    # the parameter gradients, so this costs one pass rather than two.
    enc_names = [n for n, _ in model.named_parameters()
                 if n.startswith("input_embedder.atom_attn_enc.")]
    enc_params = [dict(model.named_parameters())[n] for n in enc_names]
    for k in ("cl", "plm", "ai", "ql"):
        t = grab[k]
        print(f"    {k}: shape {tuple(t.shape)} requires_grad={t.requires_grad} "
              f"is_leaf={t.is_leaf} grad_fn={type(t.grad_fn).__name__ if t.grad_fn else None}",
              flush=True)
    wrt = [grab["cl"], grab["plm"], grab["ai"]] + enc_params
    gs = torch.autograd.grad(loss, wrt, allow_unused=True, retain_graph=False)
    h1.remove(); h2.remove()
    cot = {"cl": gs[0], "plm": gs[1], "ai": gs[2]}
    pgrad = {n: g for n, g in zip(enc_names, gs[3:])}
    print(f"[{time.time()-t0:.0f}s] backward done; cotangents "
          + ", ".join(f"{k}={'None' if v is None else f'{float(v.norm()):.6e}'}"
                      for k, v in cot.items())
          + f"; {sum(1 for g in pgrad.values() if g is not None)}/{len(pgrad)} encoder "
            f"parameters have a gradient", flush=True)
    unused = [k for k, v in cot.items() if v is None]
    if unused:
        raise SystemExit(
            f"STOP: {unused} are not used in the graph that produces the loss. That is a fact "
            f"about the model, not about this script: allow_unused returned None for them while "
            f"{sum(1 for g in pgrad.values() if g is not None)} encoder parameters DID get a "
            f"gradient in the same call.")

    # the nine, straight from the checkpoint names, for the arm to score
    pay = {
        "loss": float(loss),
        "dtype": a.dtype,
        "seed": a.seed,
        "n_query": int(grab["n_query"]),
        "n_key": int(grab["n_key"]),
        "features": grab["features"],
        "cl": grab["cl"].detach().to(torch.float64).clone(),
        "plm": grab["plm"].detach().to(torch.float64).clone(),
        "ai": grab["ai"].detach().to(torch.float64).clone(),
        "ql": grab["ql"].detach().to(torch.float64).clone(),
        "cot_cl": cot["cl"].detach().to(torch.float64).clone(),
        "cot_plm": cot["plm"].detach().to(torch.float64).clone(),
        "cot_ai": cot["ai"].detach().to(torch.float64).clone(),
        "grad_f64": {n: g.detach().to(torch.float64).cpu().clone()
                     for n, g in pgrad.items() if g is not None},
        "provenance": {
            "batch": str(a.bundle / "batch_step003.pt"), "batch_sha256": got,
            "draws": str(a.bundle / "draws_recycles0.pt"),
            "draws_sha256": sha256(a.bundle / "draws_recycles0.pt"),
            "checkpoint": str(a.checkpoint), "checkpoint_sha256": sha256(a.checkpoint),
            "upstream_tree": __import__("openfold3").__file__,
            "determinism": det, "dropout": dropout,
            "autocast": "removed" if dtype is torch.float64 else "upstream",
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pay, a.out)
    meta = {k: (list(v.shape) if hasattr(v, "shape") else v)
            for k, v in pay.items() if k not in ("features", "grad_f64", "provenance")}
    meta["features"] = {k: list(v.shape) for k, v in pay["features"].items()}
    meta["grad_f64_n"] = len(pay["grad_f64"])
    meta["cot_norms"] = {k: float(pay[k].norm()) for k in ("cot_cl", "cot_plm", "cot_ai")}
    meta["out"] = str(a.out)
    meta["sha256"] = sha256(a.out)
    meta["provenance"] = pay["provenance"]
    meta["elapsed_s"] = round(time.time() - t0, 1)
    Path(__file__).with_name("IE_BOUNDARY.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "provenance"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
