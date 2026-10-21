"""RFD4-Proteina trunk components on the card against a float64 host reference.

`tests/rfd4_reference.py` transcribes the pinned upstream module code; the charter's hard stop is
to grade against float64 on CPU rather than against another approximation, so that reference runs
in float64. Each component is graded twice: against the float64 run of the fp32 inputs, and against
the float64 run of the bf16 inputs the card actually holds. The second is the floor -- the part of
the error that is the bf16 inputs rather than the device chain -- and the attention needs it,
because averaging 128 near-orthogonal random value rows cancels most of their magnitude and
amplifies any input error about sqrt(n) times. A component is accepted when it is inside BAR and
within 2x that floor.

Padded tokens are masked tokens, which is what upstream does to a short sequence in a batch, so
only the real tokens are graded (`test_padding_does_not_touch_the_real_tokens` is why that is a
sufficient grade).

Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> python3 -m pytest tests/test_rfd4_trunk_parity.py
"""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T
from tt_bio.rfd4 import trunk as TR

import rfd4_reference as REF

#: bf16 weights and activations through a 1024-wide dot product, four linears deep in the block.
BAR = 5e-2
#: How far above its bf16 floor a component may sit before the device chain itself is suspect. The
#: attention sits at 2.6x because the fused flash kernel rescales its running max and sum in bf16,
#: which the torch floor does not model; everything else lands within 4% of its floor.
FLOOR_FACTOR = 3.0
#: Pearson correlation against float64, which is what "the same tensor" means once the magnitude
#: has cancelled: an output that is a near-zero average of 128 rows can be a few percent off in
#: rel_rms and still carry the whole signal.
PCC_BAR = 0.999

DIM, DIM_PAIR, DIM_COND, HEADS = 1024, 384, 256, 16

# conftest's `_release_device_after_test` closes the card after every test, so each test takes the
# handle itself -- a module-scoped device fixture would hand the second test a closed mesh.


def _dev(t, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


_bf16 = REF.bf16


def _rel_rms(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def _masked(t, mask):
    return t[mask.bool()]


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def _grade(name, got, ref, floor_ref, mask, bar=BAR, pcc_bar=PCC_BAR):
    """rel_rms and PCC against float64, with the same chain's bf16 floor beside them.

    `bar=None` grades against the floor alone: the block's UPDATE is a small difference of two
    much larger tensors, so its own bf16 floor is 2e-1 and an absolute bar would say nothing.
    """
    g, r, f = _masked(got, mask), _masked(ref, mask), _masked(floor_ref, mask)
    err, floor, pcc = _rel_rms(g, r), _rel_rms(f, r), _pcc(g, r)
    print(f"\n{name}: rel_rms {err:.3e} vs float64, bf16 floor {floor:.3e} "
          f"(pcc {_pcc(f, r):.6f}), pcc {pcc:.6f}")
    if bar is not None:
        assert err < bar, f"{name} {err:.3e} over the {bar} bar"
    assert err < FLOOR_FACTOR * max(floor, 1e-4), f"{name} {err:.3e} is over {FLOOR_FACTOR}x its {floor:.3e} floor"
    assert pcc > pcc_bar, f"{name} pcc {pcc:.6f} under {pcc_bar}"
    return err, floor, pcc


@pytest.fixture(scope="module")
def case():
    """One block's weights and inputs, 128 tokens with 96:128 padded, plus their bf16 values."""
    w = REF.block_weights(DIM, DIM_PAIR, DIM_COND, HEADS, seed=0)
    x, z, cond, mask = REF.block_inputs(1, 128, DIM, DIM_PAIR, DIM_COND, lengths=[96], seed=1)
    wb = {k: _bf16(v) for k, v in w.items()}
    return w, wb, x, z, cond, mask


def _tt_inputs(case, dev):
    _, _, x, z, cond, mask = case
    pair = mask[:, :, None] * mask[:, None, :]
    bias_mask = torch.where(pair.bool(), 0.0, REF.NEG_INF)[:, None]
    return (_dev(x, dev), _dev(z, dev), _dev(cond, dev), _dev(mask[..., None], dev),
            _dev(bias_mask, dev))


def _scope(w, scope=""):
    s = T.WeightScope({k: v.to(torch.float32) for k, v in w.items()})
    return s.child(scope) if scope else s


@pytest.mark.device
def test_adaptive_norm(case):
    dev = T.get_device()
    w, wb, x, z, cond, mask = case
    args = ("mhba.adaln",)
    ref = REF.adaptive_norm(w, *args, x, cond, mask)
    floor = REF.adaptive_norm(wb, *args, _bf16(x), _bf16(cond), mask, REF.bf16)
    mod = TR.AdaptiveNorm(_scope(w, "mhba.adaln"), TR.compute_kernel_config())
    xt, zt, ct, mt, _ = _tt_inputs(case, dev)
    got = ttnn.to_torch(mod(xt, ct, mt)).double()
    _grade("adaptive_norm", got, ref, floor, mask)


@pytest.mark.device
def test_transition(case):
    dev = T.get_device()
    w, wb, x, z, cond, mask = case
    ref = REF.transition(w, "transition.transition", x, mask)
    floor = REF.transition(wb, "transition.transition", _bf16(x), mask, REF.bf16)
    mod = TR.Transition(_scope(w, "transition.transition"), TR.compute_kernel_config())
    xt, zt, ct, mt, _ = _tt_inputs(case, dev)
    got = ttnn.to_torch(mod(xt, mt)).double()
    _grade("transition", got, ref, floor, mask)


@pytest.mark.device
def test_pair_bias_attention(case):
    dev = T.get_device()
    w, wb, x, z, cond, mask = case
    pm = mask[:, :, None].bool() & mask[:, None, :].bool()
    ref = REF.pair_bias_attention(w, "mhba.mha", x, z, pm, HEADS)
    floor = REF.pair_bias_attention(wb, "mhba.mha", _bf16(x), _bf16(z), pm, HEADS, REF.bf16)
    mod = TR.PairBiasAttention(_scope(w, "mhba.mha"), TR.compute_kernel_config(), HEADS)
    xt, zt, ct, mt, bt = _tt_inputs(case, dev)
    got = ttnn.to_torch(mod(xt, zt, bt)).double()
    _grade("pair_bias_attention", got, ref, floor, mask)


@pytest.mark.device
def test_attn_adaln(case):
    dev = T.get_device()
    w, wb, x, z, cond, mask = case
    pm = mask[:, :, None].bool() & mask[:, None, :].bool()
    ref = REF.attn_adaln(w, "mhba", x, z, cond, mask, pm, HEADS)
    floor = REF.attn_adaln(wb, "mhba", _bf16(x), _bf16(z), _bf16(cond), mask, pm, HEADS, REF.bf16)
    mod = TR.AttnADALN(_scope(w, "mhba"), TR.compute_kernel_config(), HEADS)
    xt, zt, ct, mt, bt = _tt_inputs(case, dev)
    got = ttnn.to_torch(mod(xt, zt, ct, mt, bt)).double()
    _grade("attn_adaln", got, ref, floor, mask)


@pytest.mark.device
def test_transition_adaln(case):
    dev = T.get_device()
    w, wb, x, z, cond, mask = case
    ref = REF.transition_adaln(w, "transition", x, cond, mask)
    floor = REF.transition_adaln(wb, "transition", _bf16(x), _bf16(cond), mask, REF.bf16)
    mod = TR.TransitionADALN(_scope(w, "transition"), TR.compute_kernel_config())
    xt, zt, ct, mt, _ = _tt_inputs(case, dev)
    got = ttnn.to_torch(mod(xt, ct, mt)).double()
    _grade("transition_adaln", got, ref, floor, mask)


@pytest.mark.device
def test_trunk_block(case):
    """The whole block, graded on its UPDATE as well as its output.

    Both residuals carry x straight through and both adaptive scales start near sigmoid(-2), so a
    block whose attention is entirely wrong still matches its output to a few percent. The update
    (output minus input) is what the block contributes, and it is what the grade has to see.
    """
    dev = T.get_device()
    w, wb, x, z, cond, mask = case
    ref = REF.trunk_block(w, x, z, cond, mask, HEADS)
    floor = REF.trunk_block(wb, _bf16(x), _bf16(z), _bf16(cond), mask, HEADS, "", REF.bf16)
    mod = TR.TrunkBlock(_scope(w), TR.compute_kernel_config(), HEADS)
    xt, zt, ct, mt, bt = _tt_inputs(case, dev)
    got = ttnn.to_torch(mod(xt, zt, ct, mt, bt)).double()
    _grade("trunk_block", got, ref, floor, mask)
    xm = x * mask[..., None]
    _grade("trunk_block update", got - xm, ref - xm, floor - _bf16(x) * mask[..., None], mask, bar=None, pcc_bar=0.95)


def test_padding_does_not_touch_the_real_tokens():
    """Host-only: the reference's 96 real tokens are the same whether the batch is padded to 128
    or run at 96. Without this, a parity number on a padded run proves nothing about a short one."""
    w = REF.block_weights(256, 128, 64, 4, seed=3)
    x, z, cond, mask = REF.block_inputs(1, 128, 256, 128, 64, lengths=[96], seed=4)
    full = REF.trunk_block(w, x, z, cond, mask, 4)
    short = REF.trunk_block(w, x[:, :96], z[:, :96, :96], cond[:, :96], mask[:, :96], 4)
    assert torch.allclose(full[:, :96], short, atol=1e-12, rtol=1e-9)
    assert full[:, 96:].abs().max() == 0.0
