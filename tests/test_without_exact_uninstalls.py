"""`without_exact` takes the instrument out; `exact_training(False)` only changes its answer.

`install()` reads `exact_training_ops()` ONCE, when a run starts, and installs what it finds.
Entering `exact_training(False)` after that changes what the switch reports and leaves the
installed ops in place, so a section that has to run on the device's own arithmetic -- the
detached rollout, where upstream computes nothing that reaches a gradient -- was still running
the host float64 layer norm. It did not merely cost: `_ln_forward64` downloads its input and
the shipped diffusion decoder has already deallocated that buffer, so the reference arm died on
`tensor.is_allocated()` 398 s in, before step 0.
"""
import pytest

pytest.importorskip("ttnn")

from tt_bio import autograd as ag


def test_exact_training_false_does_not_uninstall():
    with ag.exact(("layer_norm", "softmax")):
        assert ag.exact_layer_norm_installed() and ag.exact_softmax_installed()
        with ag.exact_training(False):
            assert ag.exact_layer_norm_installed(), \
                "exact_training(False) uninstalled; this test encodes that it does not"
            assert ag.exact_training_ops() == ()
    assert not ag.exact_layer_norm_installed()


def test_without_exact_takes_them_out_and_puts_them_back():
    with ag.exact(("layer_norm", "softmax")):
        with ag.without_exact():
            assert not ag.exact_layer_norm_installed()
            assert not ag.exact_softmax_installed()
        assert ag.exact_layer_norm_installed(), "the ops were not restored"
        assert ag.exact_softmax_installed()
    assert not ag.exact_layer_norm_installed()


def test_without_exact_is_a_no_op_when_nothing_is_installed():
    assert not ag.exact_layer_norm_installed()
    with ag.without_exact():
        assert not ag.exact_layer_norm_installed()
    assert not ag.exact_layer_norm_installed()
