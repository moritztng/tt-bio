"""RFD4's pair transition as one program (`pair_block`) on the card against float64.

The lever runs LayerNorm, fc1 + bias, ReLU and fc2 + bias as one `pair_swiglu_ws.pair_relu` program with bfp8
weights. Graded on the update itself (module bar 0.05, as the campaign's per-sublayer bar), against the float64
transition of the same bf16 pair stream, at a padded length (the mask multiply runs after the program) and a full one
(the mask is all ones and skipped). Padded pairs must come out exactly zero.

Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> python3 -m pytest tests/test_rfd4_pairblock.py
"""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TAIL
from tt_bio.rfd4 import trunk as TR

import rfd4_reference as REF

BAR = 5e-2
PCC_BAR = 0.999


def rel_rms(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def _dev(t, dev):
    return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _transition64(w, z):
    g = lambda k: w[f"pu.transition_out.{k}"]  # noqa: E731
    h = torch.nn.functional.layer_norm(z, z.shape[-1:], g("layer_norm.weight"), g("layer_norm.bias"), TR.EPS)
    h = torch.relu(h @ g("linear_1.weight").t() + g("linear_1.bias"))
    return h @ g("linear_2.weight").t() + g("linear_2.bias")


@pytest.mark.device
@pytest.mark.parametrize("length", [96, 128], ids=["padded", "full"])
def test_pair_block_transition(length):
    dev = T.get_device()
    w = REF.pair_update_weights(768, 384, 256, seed=5, prefix="pu")
    _, z, _, mask = REF.block_inputs(1, 128, 768, 384, 256, lengths=[length], seed=1)
    pair = mask[:, :, None] * mask[:, None, :]
    zb = REF.bf16(z * pair[..., None])          # the trunk hands the transition a masked bf16 stream
    ref = _transition64(w, zb) * pair[..., None]
    with T.lever_scope(frozenset({"pair_block"})):
        mod = TR.PairUpdate(T.WeightScope({k: v.to(torch.float32) for k, v in w.items()}).child("pu"),
                            TR.compute_kernel_config())
        assert mod.tr_ws is not None
        zt, pmc = _dev(zb, dev), _dev(pair[..., None], dev)
        out = mod._transition(zt, pmc if length < 128 else None)
    got = ttnn.to_torch(out).double().reshape(ref.shape)
    real = pair.bool()
    g, r = got[real], ref[real]
    err = ((g - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
    pcc = torch.corrcoef(torch.stack([g.flatten(), r.flatten()]))[0, 1].item()
    print(f"\npair_block length {length}: rel_rms {err:.3e} vs float64, pcc {pcc:.6f}")
    assert err < BAR and pcc > PCC_BAR
    assert bool((got[~real] == 0).all())


@pytest.mark.device
def test_pair_block_residual_in_place():
    """With every token real the transition's residual write is in the same program: z (float32) + update, written
    over z, and its bf16 copy written over the input. Graded on the update against float64 (bar 0.05) and the copy
    against the float32 result's own bf16 rounding."""
    dev = T.get_device()
    w = REF.pair_update_weights(768, 384, 256, seed=5, prefix="pu")
    _, z, _, _ = REF.block_inputs(1, 128, 768, 384, 256, seed=1)
    z32 = z.to(torch.float32)
    zb = REF.bf16(z32.double())
    ref = _transition64(w, zb)
    with T.lever_scope(frozenset({"pair_block"})):
        mod = TR.PairUpdate(T.WeightScope({k: v.to(torch.float32) for k, v in w.items()}).child("pu"),
                            TR.compute_kernel_config())
        zt = ttnn.from_torch(z32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        zbt = _dev(zb, dev)
        both = mod._transition_block(zbt, resid=zt)
    assert both is not None and both[0] is zt and both[1] is zbt
    z_new = ttnn.to_torch(zt).double().reshape(ref.shape)
    zb_new = ttnn.to_torch(zbt).double().reshape(ref.shape)
    upd = z_new - z32.double()
    err = ((upd - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    same = (zb_new == z_new.to(torch.bfloat16).double()).double().mean().item()
    print(f"\npair_block residual: update rel_rms {err:.3e} vs float64, bf16 copy equal to rounding {same:.6f}")
    assert err < BAR
    assert same > 0.999


@pytest.mark.device
@pytest.mark.parametrize("outgoing", [True, False], ids=["out", "in"])
def test_pair_tail_residual_in_place(outgoing):
    """`pair_tail`: the triangle multiplication's output tail and its residual add into the float32 stream as one
    T2 program (T2_RES32), z and its bf16 copy written in place. 256 tokens, where trimul_t1 leaves the g_out T2
    reads. Graded on the update against float64 (bar 0.05) and the copy against the float32 sum's own rounding."""
    dev = T.get_device()
    n = 256
    w = REF.pair_update_weights(768, 384, 256, seed=5, prefix="pu")
    _, z, _, _ = REF.block_inputs(1, n, 768, 384, 256, seed=1)
    z32 = z.to(torch.float32)
    zb = REF.bf16(z32.double())
    name = "tri_mult_out" if outgoing else "tri_mult_in"
    ref = REF.tri_mult({k: v.double() for k, v in w.items()}, f"pu.{name}", zb, torch.ones(1, n, n, dtype=torch.float64),
                       outgoing)
    with T.lever_scope(frozenset({"trimul_t1", "pair_tail"})):
        mod = TR.PairUpdate(T.WeightScope({k: v.to(torch.float32) for k, v in w.items()}).child("pu"),
                            TR.compute_kernel_config())
        tri = mod.tri_out if outgoing else mod.tri_in
        # A process's first trimul at a shape can meet an L1 clash and retry without T1 for that call only, and
        # without T1's g_out the tail is not offered. The trunk pays that once at warmup; so does the test.
        tri(_dev(zb, dev), None)
        zt = ttnn.from_torch(z32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        zbt = _dev(zb, dev)
        got = tri(zbt, None, stream32=zt)
    if T.arch_name() not in TAIL.T2_RES32_ARCHS:
        # The float32-stream form declines here (wrong on Wormhole, see T2_RES32_ARCHS): nothing is written in
        # place and the update comes back for the caller's residual, graded the same way.
        assert not tri.took_residual and got is not zt
        err = rel_rms(ttnn.to_torch(got).double().reshape(ref.shape), ref)
        print(f"\npair_tail {name}: declined on {T.arch_name()}, unfused update rel_rms {err:.3e} vs float64")
        assert err < BAR
        return
    assert tri.took_residual and got is zt
    z_new = ttnn.to_torch(zt).double().reshape(ref.shape)
    zb_new = ttnn.to_torch(zbt).double().reshape(ref.shape)
    upd = z_new - z32.double()
    err = ((upd - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    same = (zb_new == z_new.to(torch.bfloat16).double()).double().mean().item()
    print(f"\npair_tail {name}: update rel_rms {err:.3e} vs float64, bf16 copy equal to rounding {same:.6f}")
    assert err < BAR
    assert same > 0.999
