"""The taped softmax forward runs a precise compute kernel config, and inference does not.

`of3t-p10grad` graded the shipped softmax `dx` at 2.201e-02 relative L2 against float64 while the
backward closure alone read 9.750e-05: the error is the forward `y` the closure multiplies by, and
the one field responsible is `math_approx_mode` (`perf/of3t_d116_verify/APPROX.json` -- HiFi4,
`fp32_dest_acc_en` and `packer_l1_acc` are bit-exactly inert at the trunk's shapes).

The gate on `is_grad_enabled` is the whole reason this is not release-gated the way
`TT_BIO_SOFTMAX_CKC` is. If it ever fires with taping off, every shipped inference path moves and
so does the `eval_before` that `of3t-p10trainout`'s instrument refuses to compare without.
No device: this is the routing decision, which is host-side.
"""
import ttnn

from tt_bio import autograd as ag
from tt_bio import taped_ttnn as tp


def _approx(cfg):
    return bool(getattr(cfg, "math_approx_mode"))


def test_inference_leaves_the_config_alone():
    with ag.no_grad():
        assert tp._softmax_fw_config({}) is None
        assert tp._softmax_fw_config({"compute_kernel_config": None}) is None


def test_a_taped_forward_gets_a_precise_config():
    assert ag.is_grad_enabled(), "the module default is taping on; this test reads that default"
    cfg = tp._softmax_fw_config({})
    assert cfg is not None
    assert _approx(cfg) is False, (
        "math_approx_mode is the only field that moves ttnn.softmax: with it set the forward "
        "reads 1.8e-02 from float64, cleared it reads 5.9e-04")
    assert cfg.math_fidelity == ttnn.MathFidelity.HiFi4
    assert cfg.fp32_dest_acc_en is True


def test_a_caller_that_brought_its_own_config_keeps_it():
    mine = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=False)
    assert tp._softmax_fw_config({"compute_kernel_config": mine}) is mine


def test_the_ops_level_softmax_already_did_this():
    """`autograd.softmax` has defaulted to `precise_config()` since it was written. The taped verb
    now carries the same rule, which is the point: one rule, not two."""
    src = open(ag.__file__).read()
    i = src.index("def softmax(x: Tensor")
    assert "config or precise_config()" in src[i:i + 400]
