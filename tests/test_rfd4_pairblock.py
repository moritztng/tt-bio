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
from tt_bio.rfd4 import trunk as TR

import rfd4_reference as REF

BAR = 5e-2
PCC_BAR = 0.999


def _dev(t, dev):
    return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _transition64(w, z):
    g = lambda k: w[f"pu.transition_out.{k}"]  # noqa: E731
    h = torch.nn.functional.layer_norm(z, z.shape[-1:], g("layer_norm.weight"), g("layer_norm.bias"), TR.EPS)
    h = torch.relu(h @ g("linear_1.weight").t() + g("linear_1.bias"))
    return h @ g("linear_2.weight").t() + g("linear_2.bias")


@pytest.mark.device
@pytest.mark.parametrize("hifi2", [False, True], ids=["trunk-fidelity", "hifi2"])
@pytest.mark.parametrize("length", [96, 128], ids=["padded", "full"])
def test_pair_block_transition(length, hifi2):
    dev = T.get_device()
    w = REF.pair_update_weights(768, 384, 256, seed=5, prefix="pu")
    _, z, _, mask = REF.block_inputs(1, 128, 768, 384, 256, lengths=[length], seed=1)
    pair = mask[:, :, None] * mask[:, None, :]
    zb = REF.bf16(z * pair[..., None])          # the trunk hands the transition a masked bf16 stream
    ref = _transition64(w, zb) * pair[..., None]
    levers = frozenset({"pair_block"} | ({"pair_block_hifi2"} if hifi2 else set()))
    with T.lever_scope(levers):
        mod = TR.PairUpdate(T.WeightScope({k: v.to(torch.float32) for k, v in w.items()}).child("pu"),
                            TR.compute_kernel_config())
        assert mod.tr_ws is not None
        zt, pmc = _dev(zb, dev), _dev(pair[..., None], dev)
        out = mod._transition(zt, pmc, length < 128)
    got = ttnn.to_torch(out).double().reshape(ref.shape)
    real = pair.bool()
    g, r = got[real], ref[real]
    err = ((g - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
    pcc = torch.corrcoef(torch.stack([g.flatten(), r.flatten()]))[0, 1].item()
    print(f"\npair_block length {length} hifi2 {hifi2}: rel_rms {err:.3e} vs float64, pcc {pcc:.6f}")
    assert err < BAR and pcc > PCC_BAR
    assert bool((got[~real] == 0).all())
