"""The pre-registered speed bar (scripts/speed_bar.py) fails what it claims to fail.

Synthetic curves, CPU only. A curve that bends from N^2 toward the algorithm's own N^3 must pass;
the same curve with a 2x step at 1536 (a chunked path that went to the host) must fail.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import speed_bar as sb  # noqa: E402

FIT = (512, 640, 768, 896, 1024)


def _curve(k, c=1e-4):
    return {n: c * n ** k for n in FIT}


def test_a_bend_up_to_the_algorithms_own_order_passes():
    rt = _curve(2.0)
    t1536 = rt[1024] * 1.5 ** 3  # cubic from 1024 on, the worst a pair model can legitimately do
    assert sb.judge(rt, 1536, t1536)["verdict"] == "PASS"


def test_a_step_above_the_envelope_fails():
    rt = _curve(2.0)
    assert sb.judge(rt, 1536, rt[1024] * 1.5 ** 3 * 2)["verdict"] == "FAIL"


def test_a_sequence_model_is_held_to_quadratic():
    rt = _curve(2.0)
    assert sb.judge(rt, 1536, rt[1024] * 1.5 ** 3, order=2)["verdict"] == "FAIL"


def test_noise_widens_the_margin_not_the_curvature():
    rt = _curve(3.0)
    t = rt[1024] * 1.5 ** 3 * 1.3
    assert sb.judge(rt, 1536, t)["verdict"] == "FAIL"
    assert sb.judge(rt, 1536, t, sigma=0.12)["verdict"] == "PASS"


def test_a_clock_that_moved_voids_the_comparison():
    rt = _curve(2.0)
    clk = {n: 1000.0 for n in FIT} | {1536: 800.0}
    assert sb.judge(rt, 1536, rt[1024] * 3, aiclk=clk)["verdict"] == "VOID"


def test_mixed_hosts_void_the_comparison():
    rt = _curve(2.0)
    ids = {n: ("whglx", 3, "abc") for n in FIT} | {1536: ("whglx", 4, "abc")}
    assert sb.judge(rt, 1536, rt[1024] * 3, identity=ids)["verdict"] == "VOID"


def test_too_few_fit_rungs_is_ungated_not_a_pass():
    assert sb.judge({768: 1.0, 1024: 2.0}, 1536, 1e9)["verdict"] == "UNGATED"
