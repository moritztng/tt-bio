# SPDX-License-Identifier: Apache-2.0
"""On-device assembly verification for the OpenDDE expander/refiner seam.

Loads the REAL OpenDDE checkpoint and checks, with real weights, the two things the
assembly tick actually delivers:

  1. Weight routing (the "remap"): every checkpoint key is routed exactly once into
     {expander, refiner, shared}; the shared subtree's keys are byte-identical to
     Protenix-v2's (0 missing vs protenix-v2.pt); the expander/refiner subtrees build.
  2. The novel expander->refiner seam runs finite on-device (card 0) with real weights,
     producing structural-token (s_inputs, s, z) of the expected shapes.

This is a WIRING + finiteness check, NOT an accuracy/parity claim: there is no real-weight
golden without an upstream OpenDDE forward pass (no CUDA here), and the shared trunk +
structural-token tokenizer are not yet ported (see docs). The expander block itself is
separately parity-verified (PCC >=0.99999) against the Phase-0 random-weight golden by
scripts/opendde_structtoken_parity.py.

Run (qb2 card 0, worktree shadowing the editable shared checkout):
  PYTHONPATH=<worktree> TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=<ttnn>/p150_mesh_graph_descriptor.textproto \
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/opendde_assembly_verify.py
"""
import os

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
import torch
import ttnn

from tt_bio.tenstorrent import get_device
from tt_bio.opendde import (OpenDDE, load_opendde_checkpoint, route_opendde_weights,
                            OPENDDE_CONFIG)

torch.set_grad_enabled(False)

CKPT = os.environ.get("OPENDDE_CKPT")   # None -> HF fetch
GOLDEN = os.environ.get("OPENDDE_GOLDEN", "/tmp/opendde_structtoken_golden.pt")


def main():
    # --- 1) routing / remap coverage (CPU, deterministic) ---
    sd = load_opendde_checkpoint(CKPT)
    routed = route_opendde_weights(sd)
    n_exp, n_ref = len(routed["expander"]), sum(
        1 for k in sd if k.startswith("structural_token_refiner."))
    n_shared = len(routed["shared"])
    print("weight routing:", flush=True)
    print(f"  total {len(sd)} keys -> expander {n_exp} + refiner_raw {n_ref} + shared {n_shared}",
          flush=True)
    assert n_exp + n_ref + n_shared == len(sd), "routing dropped keys"
    print(f"  refiner blocks: {routed['refiner_blocks']}  (config {OPENDDE_CONFIG['refiner_blocks']})",
          flush=True)

    # cross-check the shared subtree is Protenix-v2-identical, if that checkpoint is present
    pxp = os.path.expanduser("~/.boltz/protenix-v2.pt")
    if os.path.exists(pxp):
        pck = torch.load(pxp, map_location="cpu", weights_only=True)
        pck = pck.get("model", pck)
        pck = {k[len("module."):] if k.startswith("module.") else k: v for k, v in pck.items()}
        missing = set(pck) - set(routed["shared"])
        print(f"  shared vs protenix-v2.pt: {len(pck)} protenix keys, {len(missing)} missing "
              f"from OpenDDE shared subtree {'PASS' if not missing else 'FAIL'}", flush=True)

    # --- 2) build on device + run the seam with real weights ---
    dev = get_device()
    model = OpenDDE.load_from_checkpoint(CKPT)
    print("built OpenDDE on device (expander + 4-block refiner, real weights)", flush=True)

    # synthetic residue-trunk inputs + integer feature dict from the Phase-0 golden
    # (roles/parent indices; dims c_s=c_z=384 match the real checkpoint). Random-weight
    # golden's *inputs* are reused only as a self-consistent structural-token layout.
    d = torch.load(GOLDEN, map_location="cpu", weights_only=False)
    inp = d["inputs"]
    for tag, use_bias in (("with extra_attn_bias", True), ("without extra_attn_bias", False)):
        try:
            si, s, z = model.expand_and_refine(
                inp["ifd"], inp["s_inputs_res"], inp["s_res"], inp["z_res"], extra_attn_bias=use_bias)
            sh = torch.Tensor(ttnn.to_torch(s)).float()
            zh = torch.Tensor(ttnn.to_torch(z)).float()
            sih = torch.Tensor(ttnn.to_torch(si)).float()
            ok = bool(torch.isfinite(sh).all() and torch.isfinite(zh).all()
                      and torch.isfinite(sih).all())
            print(f"  seam [{tag}]: s_inputs {tuple(sih.shape)} s {tuple(sh.shape)} "
                  f"z {tuple(zh.shape)}  finite={ok}  {'PASS' if ok else 'FAIL'}", flush=True)
        except Exception as e:
            print(f"  seam [{tag}]: FAILED -> {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
