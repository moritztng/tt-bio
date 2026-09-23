"""The triangle product runs HiFi3 on Wormhole and keeps its config everywhere else.

Wormhole's HiFi4 fp32-dest matmul returns rare -2^k elements at large K, and the triangle
product's K is the sequence (see `_triangle_mul_compute_config`). These pin who gets HiFi3.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ttnn = pytest.importorskip("ttnn")

import tt_bio.tenstorrent as T  # noqa: E402


def _cfg(fid):
    return ttnn.types.WormholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)


def test_wormhole_hifi4_becomes_hifi3_with_every_other_field_kept(monkeypatch):
    monkeypatch.setattr(T, "is_wormhole", lambda: True)
    base = _cfg(ttnn.MathFidelity.HiFi4)
    out = T._triangle_mul_compute_config(base)
    assert out is not base
    assert out.math_fidelity == ttnn.MathFidelity.HiFi3
    assert base.math_fidelity == ttnn.MathFidelity.HiFi4
    for f in ("math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc", "dst_full_sync_en", "throttle_level"):
        assert getattr(out, f) == getattr(base, f), f


@pytest.mark.parametrize("wormhole,fid", [(False, "HiFi4"), (True, "HiFi3"), (True, "HiFi2")])
def test_every_other_case_is_returned_unchanged(monkeypatch, wormhole, fid):
    monkeypatch.setattr(T, "is_wormhole", lambda: wormhole)
    base = _cfg(getattr(ttnn.MathFidelity, fid))
    assert T._triangle_mul_compute_config(base) is base
