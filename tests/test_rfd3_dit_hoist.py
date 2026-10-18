"""RFD3_DIT_SHOIST's wide weights, host only: block i's columns of the three wide linears reproduce that block's own
adaLN (rms_norm(s, ln_s) -> sigmoid gain, shift) and output gates in float64."""

import pytest
import torch

M = pytest.importorskip("tt_bio.rfd3.model")


def _rms(x, w=None):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    return y if w is None else y * w


def test_hoist_columns_match_per_block():
    torch.manual_seed(0)
    nb, cs, c = 3, 8, 12
    W = {}
    for i in range(nb):
        for ada, out in (("attention_pair_bias.ada_ln_1.", "attention_pair_bias."),
                         ("transition_block.ada_ln.", "transition_block.")):
            p = f"blocks.{i}."
            W[p + ada + "ln_s.weight"] = torch.randn(cs, dtype=torch.float64)
            W[p + ada + "to_gain.0.weight"] = torch.randn(c, cs, dtype=torch.float64)
            W[p + ada + "to_gain.0.bias"] = torch.randn(c, dtype=torch.float64)
            W[p + ada + "to_bias.weight"] = torch.randn(c, cs, dtype=torch.float64)
            W[p + out + "linear_output_project.0.weight"] = torch.randn(c, cs, dtype=torch.float64)
            W[p + out + "linear_output_project.0.bias"] = torch.randn(c, dtype=torch.float64)
    W = {k: v.double() for k, v in W.items()}
    sc_w, sc_b, sb_w, g_w, g_b = (t.double() for t in M.dit_hoist_weights(W, nb))
    assert sc_w.shape == (cs, 2 * nb * c) and sc_b.shape == (1, 2 * nb * c)
    s = torch.randn(5, cs, dtype=torch.float64)
    sn = _rms(s)
    gain, shift, gate = torch.sigmoid(sn @ sc_w + sc_b), sn @ sb_w, torch.sigmoid(s @ g_w + g_b)
    for i in range(nb):
        for j, (ada, out) in enumerate((("attention_pair_bias.ada_ln_1.", "attention_pair_bias."),
                                        ("transition_block.ada_ln.", "transition_block."))):
            p, cols = f"blocks.{i}.", slice((2 * i + j) * c, (2 * i + j + 1) * c)
            sl = _rms(s, W[p + ada + "ln_s.weight"])
            ref_gain = torch.sigmoid(sl @ W[p + ada + "to_gain.0.weight"].t() + W[p + ada + "to_gain.0.bias"])
            ref_shift = sl @ W[p + ada + "to_bias.weight"].t()
            ref_gate = torch.sigmoid(s @ W[p + out + "linear_output_project.0.weight"].t()
                                     + W[p + out + "linear_output_project.0.bias"])
            torch.testing.assert_close(gain[:, cols], ref_gain, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(shift[:, cols], ref_shift, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(gate[:, cols], ref_gate, rtol=1e-12, atol=1e-12)
