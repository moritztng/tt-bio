"""Reference capture for the OpenBind token-level DiffusionTransformer (A1, the hoisted
pair LayerNorm).

The one architectural difference between OF3-preview2 and OpenBind: preview2 keeps a
weight-only ``layer_norm_z`` inside each of the 24 DiT blocks' ``AttentionPairBias``,
OpenBind hoists it to a single ``layer_norm_z`` applied once before the stack
(``DiffusionTransformer.forward``: ``z = self.layer_norm_z(z)``), and its
``DiffusionAttentionPairBias._prep_bias`` then runs ``linear_z(z)`` with no norm at all.
Those 48 keys traded for 1 are the entire reason preview2 weights refuse to load on
v0.5.0+.

Inputs are the REAL DiT inputs already captured for the preview2 leg
(``~/of3_ref_out.pkl["intermediates"]["diffusion_transformer_real"]``: a_in, s, z,
token_mask at N=76, ubiquitin), reused verbatim here with OpenBind weights. That is not a
real OpenBind fold, and it does not need to be: for a component gate what matters is that
the tensors are on-manifold and byte-identical on both sides. Synthetic N(0,1) input was
avoided deliberately -- the preview2 pairformer leg already burned two ticks blaming an
off-manifold golden for a real conditioning effect.

v0.5.0 runs the sampler with ``use_high_precision_attention=True`` (of3_all_atom/model.py
``sample_diffusion``), so the reference does too. That is the same fp32 sampler boundary
tt-bio already defaults to via OF3_DIFFUSION_FP32_DEVICE.

Writes ``~/of3_ob_ref.pkl["intermediates"]["diffusion_transformer_ob0"]``:
  a_in, s, z, token_mask   the reused real inputs
  a_block0                 output of a 1-block stack (shared LN + block 0)
  a_stack                  output of the full 24-block stack
  a_block0_unnormed        block 0, and
  a_stack_unnormed         the 24-block stack, both with the shared norm SKIPPED -- what a
                           device that silently dropped ``layer_norm_z`` would compute.
                           These are the controls that make a match mean something.

Scored on the block UPDATE (``a_out - a_in``), not on ``a_out``. Two weaker controls were
tried first and both are useless here, which is worth recording because either would have
looked like a passing gate:
  * applying the norm TWICE: LN is near-idempotent, so a leftover per-block norm shows up
    at pcc 0.99999 through block 0 -- invisible.
  * comparing ``a_out`` instead of the update: the residual ``a_in`` dominates (std 170 out
    vs an update an order below), so even skipping the norm entirely still reads 0.99503.
The update is where a bias-level change is actually visible.

Run with the upstream v0.5.0 reference env, NOT the tt-bio device env:

    OB0_UPSTREAM=/home/ttuser/ob0_upstream \
    /home/ttuser/pharma_protenix_run/refenv312/bin/python \
        scripts/ob0_diffusion_transformer_golden.py
"""
from __future__ import annotations

import inspect
import os
import pickle
import sys

import torch

UPSTREAM = os.environ.get("OB0_UPSTREAM", os.path.expanduser("~/ob0_upstream"))
CKPT = os.environ.get("OB0_CKPT", os.path.expanduser("~/.boltz/of3-ob-2025-06-30-174k.pt"))
IN_GOLD = os.environ.get("OF3_REF_OUT", os.path.expanduser("~/of3_ref_out.pkl"))
OUT = os.environ.get("OB0_REF_OUT", os.path.expanduser("~/of3_ob_ref.pkl"))
STEM = "diffusion_module.diffusion_transformer"

sys.path.insert(0, UPSTREAM)


def _sub(sd: dict, prefix: str) -> dict:
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def _load_ckpt(path: str) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "ema"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    return sd


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double(); b = b.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def _build(n_blocks: int, w: dict):
    from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C

    cfg = dict(C.architecture.diffusion_module.diffusion_transformer)
    params = set(inspect.signature(DiffusionTransformer.__init__).parameters) - {"self"}
    cfg = {k: v for k, v in cfg.items() if k in params}
    cfg["no_blocks"] = n_blocks
    dt = DiffusionTransformer(**cfg).eval()
    keep = {k: v.float() for k, v in w.items()
            if not k.startswith("blocks.")
            or int(k.split(".")[1]) < n_blocks}
    dt.load_state_dict(keep, strict=True)
    return dt


def main() -> None:
    sd = _load_ckpt(CKPT)
    w = _sub(sd, STEM)
    assert "layer_norm_z.weight" in w, (
        f"{CKPT} has no shared {STEM}.layer_norm_z -- not the OpenBind checkpoint")
    assert not any(k.endswith("attention_pair_bias.layer_norm_z.weight") for k in w), (
        "OpenBind should carry no per-block layer_norm_z")

    with open(IN_GOLD, "rb") as fh:
        g = pickle.load(fh)["intermediates"]["diffusion_transformer_real"]
    a_in, s, z, tok = g["a_in"], g["s"], g["z"], g["token_mask"]

    out = {"a_in": a_in, "s": s, "z": z, "token_mask": tok}
    args = dict(a=a_in.unsqueeze(0).float(), s=s.unsqueeze(0).float(),
                z=z.unsqueeze(0).float(), mask=tok.unsqueeze(0).float(),
                use_high_precision_attention=True)
    with torch.no_grad():
        for label, nb in (("a_block0", 1), ("a_stack", 24)):
            dt = _build(nb, w)
            out[label] = dt(**args).squeeze(0).contiguous()
            print(f"{label}: {tuple(out[label].shape)} "
                  f"std={out[label].std():.4f}")
        # The controls: the shared norm skipped, which is what a device that dropped
        # layer_norm_z would produce. Realised by replacing the norm with the identity
        # rather than by editing the forward, so the rest of the stack is untouched.
        for label, nb in (("a_block0_unnormed", 1), ("a_stack_unnormed", 24)):
            dt = _build(nb, w)
            dt.layer_norm_z = torch.nn.Identity()
            out[label] = dt(**args).squeeze(0).contiguous()

    base = a_in.float()
    for leg in ("a_block0", "a_stack"):
        sep_out = _pcc(out[leg], out[leg + "_unnormed"])
        sep_upd = _pcc(out[leg] - base, out[leg + "_unnormed"] - base)
        out[leg + "_unnormed_separation_pcc"] = sep_upd
        print(f"{leg}: norm-skipped control  out pcc={sep_out:.5f}  "
              f"UPDATE pcc={sep_upd:.5f}")

    prev = {}
    if os.path.exists(OUT):
        with open(OUT, "rb") as fh:
            prev = pickle.load(fh)
    prev.setdefault("intermediates", {})["diffusion_transformer_ob0"] = out
    with open(OUT, "wb") as fh:
        pickle.dump(prev, fh)
    print(f"wrote {OUT} [intermediates][diffusion_transformer_ob0]")


if __name__ == "__main__":
    main()
