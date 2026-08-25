"""Which ending-node triangle-attention bias orientation does the device actually run?

OpenBind (upstream v0.5.0) builds the ending-node triangle bias from the UNtransposed pair
(AF3 Algorithm 15, z_ij); OF3-preview2 builds it from the transposed one (z_ji). No weights
change between them, so ``openfold3_trunk.py`` selects it from ``is_openbind`` and the
selection is pure index algebra -- the kind that inverts without any downstream signal.
tt-bio and upstream also name the flag in opposite directions (ours: the bias follows the
pair transpose; theirs: undo the transpose for the bias), so ``transpose_bias=False`` is
claimed to be upstream v0.5.0.

This gates the claim as a 2x2 rather than a single number: each device flag against each
reference orientation. One high PCC alone would prove nothing, since a bug that ignored the
flag entirely would still match one of them. Passing needs all four cells: the diagonal
high, the off-diagonal low, and the two references far enough apart to tell.

Golden: ``~/of3_ob_ref.pkl["intermediates"]["tri_att_end_orientation"]``, written by
``scripts/ob0_tri_att_end_orientation_golden.py`` against upstream v0.5.0 in fp32 on CPU.
Both trunk head_dim regimes are covered, because all three trunk pair stacks share one
upstream ``PairBlock`` and move together: the pairformer at head_dim=32 (tile-aligned) and
the template pair stack at head_dim=16 (the sub-tile slice-and-concat path).
"""
import os

import pytest
import torch
import ttnn

import of3_golden

_CKPT = os.path.expanduser("~/.boltz/of3-ob-2025-06-30-174k.pt")
_GOLD = os.path.expanduser("~/of3_ob_ref.pkl")

pytestmark = [
    pytest.mark.device,
    pytest.mark.skipif(
        not (os.path.exists(_CKPT) and os.path.exists(_GOLD)),
        reason="OpenBind ckpt or ~/of3_ob_ref.pkl missing "
          "(scripts/ob0_tri_att_end_orientation_golden.py writes it)"),
]

_SITES = {
    "pairformer_block0": "pairformer_stack.blocks.0.pair_stack.tri_att_end",
    "template_block0": "template_embedder.template_pair_stack.blocks.0.tri_att_end",
}


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def _cfg(dev):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def _run_device(weights, z, head_dim, n_heads, transpose_bias, dev):
    from tt_bio.tenstorrent import TriangleAttention
    ta = TriangleAttention(
        head_dim, n_heads, True, weights, _cfg(dev),
        scale_pair_bias=False, fp32_softmax=True, transpose_bias=transpose_bias)
    zt = ttnn.from_torch(z.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16)
    out = ta(zt)
    return torch.Tensor(ttnn.to_torch(out)).float().reshape(z.shape)


@pytest.mark.parametrize("site", sorted(_SITES))
def test_openbind_tri_att_end_bias_orientation(site):
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import _rename_tri_att, _sub

    g = of3_golden.intermediates(_GOLD)["tri_att_end_orientation"][site]
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "ema"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    # TriangleAttention is scoped with an "mha." prefix at its call sites, so the mha
    # weights arrive unprefixed.
    w = _rename_tri_att(_sub(sd, _SITES[site]))
    weights = {k[len("mha."):] if k.startswith("mha.") else k: v for k, v in w.items()}

    dev = get_device()
    z, head_dim, n_heads = g["z"], g["c_hidden"], g["no_heads"]
    got = {flag: _run_device(weights, z, head_dim, n_heads, flag, dev)
           for flag in (False, True)}

    m = {(flag, ref): _pcc(got[flag], g[ref])
         for flag in (False, True) for ref in ("v050", "preview2")}
    sep = g["orientation_separation_pcc"]
    print(f"\nOF3/OB0 tri_att_end bias orientation [{site}, head_dim={head_dim}]")
    print(f"  reference separation pcc(v050, preview2) = {sep:.5f}")
    for flag in (False, True):
        print(f"  device transpose_bias={str(flag):5s} -> "
              f"v050 {m[(flag, 'v050')]:.5f}   preview2 {m[(flag, 'preview2')]:.5f}")

    assert sep < 0.9, f"references are not discriminating (pcc={sep:.5f})"
    # The diagonal: tt-bio False is upstream v0.5.0, tt-bio True is preview2.
    assert m[(False, "v050")] > 0.99, f"transpose_bias=False vs v050 {m[(False, 'v050')]:.5f}"
    assert m[(True, "preview2")] > 0.99, (
        f"transpose_bias=True vs preview2 {m[(True, 'preview2')]:.5f}")
    # The off-diagonal, which is what makes the diagonal mean something.
    assert m[(False, "preview2")] < 0.9, "transpose_bias=False also matches preview2"
    assert m[(True, "v050")] < 0.9, "transpose_bias=True also matches v0.5.0"
