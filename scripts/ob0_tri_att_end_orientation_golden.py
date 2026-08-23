"""Reference capture for the OpenBind ending-node triangle-attention bias orientation.

Upstream v0.5.0 passes ``transpose_bias=True`` from ``PairBlock.tri_att_end``, which moves
the ending-node triangle bias off the TRANSPOSED pair (``z_ji``, what preview2 does) and
onto the untransposed one (``z_ij``, AF3 Algorithm 15). No weights change, so nothing in
the checkpoint forces it and the flag has to be selected deliberately.

tt-bio names its flag the other way round: ours says the bias FOLLOWS the pair transpose,
theirs says undo the transpose for the bias. By index algebra tt-bio ``transpose_bias=False``
should equal upstream ``transpose_bias=True``. Algebra of that shape inverts easily, and if
it did invert the OpenBind trunk would be silently wrong with nothing downstream able to
tell. So this script captures the reference under BOTH orientations, and the device test
reports the full 2x2: each device flag against each reference orientation. A single high
PCC proves nothing on its own; the 2x2 shows the two orientations are far apart AND which
pairing is the identity.

Two sites, both read out of the OpenBind checkpoint, because all three trunk pair stacks
share one upstream ``PairBlock`` and therefore move together:

  * ``pairformer_stack.blocks.0``    c_in=128, head_dim=32 -- the tile-aligned device path
  * ``template_embedder.template_pair_stack.blocks.0``  c_in=64, head_dim=16 -- the
    sub-tile head_dim path, which slices and hand-concats the SDPA output

Outputs are stored in the ``z`` frame (transposed back), which is where tt-bio's
``TriangleAttention(ending=True)`` returns its result. Upstream's ``tri_att_end`` is built
with ``starting=True`` and leaves that transpose to its caller.

Run with the upstream v0.5.0 reference env, NOT the tt-bio device env:

    OB0_UPSTREAM=/home/ttuser/ob0_upstream \\
    /home/ttuser/pharma_protenix_run/refenv312/bin/python \\
        scripts/ob0_tri_att_end_orientation_golden.py
"""
from __future__ import annotations

import os
import pickle
import sys

import torch

UPSTREAM = os.environ.get("OB0_UPSTREAM", os.path.expanduser("~/ob0_upstream"))
CKPT = os.environ.get("OB0_CKPT", os.path.expanduser("~/.boltz/of3-ob-2025-06-30-174k.pt"))
OUT = os.environ.get("OB0_REF_OUT", os.path.expanduser("~/of3_ob_ref.pkl"))
N = int(os.environ.get("OB0_REF_N", "96"))
SEED = 0

SITES = {
    "pairformer_block0": "pairformer_stack.blocks.0.pair_stack.tri_att_end",
    "template_block0": "template_embedder.template_pair_stack.blocks.0.tri_att_end",
}

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
    a = a.flatten().double() - a.flatten().double().mean()
    b = b.flatten().double() - b.flatten().double().mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def main() -> None:
    from openfold3.core.model.layers.triangular_attention import TriangleAttention

    sd = _load_ckpt(CKPT)
    assert "diffusion_module.diffusion_transformer.layer_norm_z.weight" in sd, (
        f"{CKPT} is not the OpenBind checkpoint (no shared diffusion layer_norm_z)")

    out: dict = {}
    for name, prefix in SITES.items():
        w = _sub(sd, prefix)
        assert w, f"no weights under {prefix}"
        c_in = w["linear_z.weight"].shape[1]
        no_heads = w["linear_z.weight"].shape[0]
        c_hidden = w["mha.linear_q.weight"].shape[0] // no_heads

        # starting=True is how v0.5.0 builds tri_att_end: the caller transposes the pair.
        ta = TriangleAttention(c_in, c_hidden, no_heads, starting=True).eval()
        ta.load_state_dict({k: v.float() for k, v in w.items()}, strict=True)

        torch.manual_seed(SEED)
        z = torch.randn(1, N, N, c_in, dtype=torch.float32)

        site = {"z": z, "c_in": c_in, "c_hidden": c_hidden, "no_heads": no_heads}
        with torch.no_grad():
            for label, flag in (("v050", True), ("preview2", False)):
                # Exactly what PairBlock.tri_att_start_end does around the call.
                y = ta(z.transpose(-2, -3), mask=None, transpose_bias=flag)
                site[label] = y.transpose(-2, -3).contiguous()
        # The two orientations must actually differ, or the device test cannot
        # discriminate and a high PCC would be meaningless.
        sep = _pcc(site["v050"], site["preview2"])
        print(f"{name}: c_in={c_in} head_dim={c_hidden} heads={no_heads} N={N} "
              f"pcc(v050, preview2)={sep:.5f}")
        assert sep < 0.9, (
            f"{name}: the two bias orientations agree to pcc={sep:.5f}, so this input "
            f"cannot discriminate them")
        site["orientation_separation_pcc"] = sep
        out[name] = site

    prev = {}
    if os.path.exists(OUT):
        with open(OUT, "rb") as fh:
            prev = pickle.load(fh)
    prev.setdefault("intermediates", {})["tri_att_end_orientation"] = out
    with open(OUT, "wb") as fh:
        pickle.dump(prev, fh)
    print(f"wrote {OUT} [intermediates][tri_att_end_orientation] ({len(out)} sites)")


if __name__ == "__main__":
    main()
