"""The scalar a fused activation carries, and the one thing that can silently go wrong with it.

`_fp32_softmax_tail` rides the attention score scale on the add's input-a activation:
`ttnn.add_(sc, bias_f, input_tensor_a_activations=[UnaryWithParam(MUL_UNARY_SFPU, c)])` is
`sc * c + bias_f`. openfold3's trunk, template and MSA stacks all take that path, so the
tape has to differentiate it, and its derivative is the constant.

ttnn does not bind `params` on `UnaryWithParam` -- `op_type` is the only attribute -- so the
constant is read out of the repr. That is a format dependency and this file is what keeps it
honest. If ttnn ever prints the fp32 BIT PATTERN instead of the value (which is how
`rfd3_bias._scale_bits` has to pass the same scalar to a hand-written kernel), the parse
would return ~1e9 and the tape would scale a gradient by it. `_check_param` catches that on
device against the kernel; these tests catch it without one.
"""
import math

import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import taped_ttnn as tp                                    # noqa: E402


@pytest.mark.parametrize("c", [0.25, 2.0, 1.0 / math.sqrt(32.0), 1e-3, -0.5, 1.0])
def test_scalar_round_trips_through_the_repr(c):
    op = ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, c)
    got = tp._unary_scalar(op)
    assert got is not None, f"no scalar parsed from {op!r}"
    # fp32 storage, so compare at fp32 resolution rather than exactly.
    assert abs(got - c) <= 1e-6 * max(abs(c), 1.0), f"{got} != {c} from {op!r}"


def test_a_bare_op_type_carries_no_scalar():
    assert tp._unary_scalar(ttnn.UnaryOpType.SIGMOID) is None


def test_the_parameterised_entry_is_the_constant_and_its_derivative():
    c = 0.17677669529663687
    fwd, dfn = tp._FUSED_UNARY_PARAM[ttnn.UnaryOpType.MUL_UNARY_SFPU](c)
    assert dfn(None, None) == c
    assert fwd is not None


def test_an_unmodelled_parameterised_activation_refuses():
    """A fused unary the tape cannot differentiate must raise, not be forwarded."""
    op = ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, 1.0)
    with pytest.raises(NotImplementedError, match="no backward for the fused activation"):
        tp._activation({"input_tensor_a_activations": [op]}, "input_tensor_a_activations")


def test_a_plain_modelled_activation_still_resolves():
    got = tp._activation({"input_tensor_b_activations": [ttnn.UnaryOpType.SIGMOID]},
                         "input_tensor_b_activations")
    assert got is tp._FUSED_UNARY[ttnn.UnaryOpType.SIGMOID]
