"""The gate that decides whether the A/B may take the wheel's fused softmax backward.

It can fail in two directions and both are expensive. Dropping a good arm wastes the card window
this row waited eighteen turns for. ADMITTING a bad one produces a fast number from a wrong
gradient, which is the worst outcome available on this row -- the sibling op
`moreh_layer_norm_backward` is wrong on Blackhole at dx 2.741e+06 rel L2 in bf16 (upstream
#12349), so the failure mode is real rather than theoretical. Both directions are pinned, and the
default on anything ambiguous is DROP.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perf" / "bcx_bwbytes"))
from moreh_verdict import verdict  # noqa: E402

BAR = 5.0e-2


def rows(*pairs):
    return {"rows": [{"n": n, "moreh_rn_bf16": rec} for n, rec in pairs]}


def test_admits_an_arm_that_clears_the_bar_at_every_n():
    ok, why = verdict(rows((224, {"rel_l2_vs_f64": 1.1e-3}),
                           (256, {"rel_l2_vs_f64": 2.2e-3})), BAR)
    assert ok, why
    assert "2.200e-03" in why


def test_drops_an_arm_that_fails_at_any_n():
    """One n past the bar is enough: the round runs 224 and the census was taken at 256."""
    ok, why = verdict(rows((224, {"rel_l2_vs_f64": 1.1e-3}),
                           (256, {"rel_l2_vs_f64": 9.9e-1})), BAR)
    assert not ok and "n=256" in why


def test_drops_when_the_arm_was_refused_and_recorded_as_an_error():
    """An fp32 refusal is recorded as an error string, never as a timing -- it is not a pass."""
    ok, why = verdict(rows((256, {"error": "moreh_softmax_backward: unsupported dtype FLOAT32"})),
                      BAR)
    assert not ok and "no rel_l2_vs_f64" in why


def test_drops_when_the_arm_is_absent_from_every_row():
    ok, why = verdict({"rows": [{"n": 256, "chain": {"rel_l2_vs_f64": 0.0}}]}, BAR)
    assert not ok and "appears in no row" in why


def test_drops_on_an_empty_artifact():
    ok, why = verdict({}, BAR)
    assert not ok


def test_the_bar_is_the_thing_being_tested_not_a_constant_in_the_code():
    """The control. A gate that ignored its bar would pass every case above; this separates them
    by moving ONLY the bar across a fixed reading."""
    blob = rows((256, {"rel_l2_vs_f64": 1.0e-2}))
    assert verdict(blob, 5.0e-2)[0] is True
    assert verdict(blob, 1.0e-3)[0] is False
