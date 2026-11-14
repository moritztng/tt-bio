"""RFD4's trunk must take its matmul fidelity through the shared `trunk_compute_kernel_config`.

It did not: `engine._build_trunk` passed the bare HiFi4 default, so `trunk_hifi3` and `trunk_hifi2`
were silently inert for RFD4 while every other trunk in tt-bio honoured them. These pin that the
wiring is there and that it is opt-in -- no lever, no change.
"""
import ast
import pathlib

import pytest

import tt_bio.tenstorrent as T

ENGINE = pathlib.Path(T.__file__).parent / "rfd4" / "engine.py"


def test_the_trunk_is_built_through_the_shared_fidelity_wrapper():
    """A grep-level pin: the lever is unreachable again the moment this call is unwrapped."""
    src = ENGINE.read_text()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Trunk"]
    assert calls, "no Trunk(...) call in engine.py"
    for call in calls:
        first = call.args[1] if len(call.args) > 1 else None
        assert isinstance(first, ast.Call) and getattr(first.func, "id", None) == "trunk_compute_kernel_config", (
            "RFD4's Trunk must receive trunk_compute_kernel_config(compute_kernel_config()); "
            "passing the bare default makes trunk_hifi3/trunk_hifi2 inert for RFD4"
        )


def test_without_a_lever_the_fidelity_is_unchanged():
    ttnn = pytest.importorskip("ttnn")
    from tt_bio.rfd4.trunk import compute_kernel_config
    base = compute_kernel_config()
    assert base.math_fidelity == ttnn.MathFidelity.HiFi4
    wrapped = T.trunk_compute_kernel_config(base)
    assert wrapped.math_fidelity == ttnn.MathFidelity.HiFi4, "the wrapper must be a no-op with no lever set"
    for field in ("fp32_dest_acc_en", "packer_l1_acc", "math_approx_mode", "dst_full_sync_en"):
        assert getattr(wrapped, field) == getattr(base, field), f"{field} was not carried through"


def test_trunk_hifi3_is_not_in_rfd4s_graded_set():
    """It changes arithmetic, so it owes an RFD4 binder grade before it may ship on by default."""
    from tt_bio.rfd4.trunk import FAST_LEVERS, NORMAL_LEVERS
    assert "trunk_hifi3" not in NORMAL_LEVERS
    assert "trunk_hifi3" not in FAST_LEVERS
    assert "trunk_hifi2" not in NORMAL_LEVERS
