"""sdpa_generic's program cache: two calls that compile differently never share a program.

Opens no device: `build` and `generic_op` are stubbed, so what runs is the cache lookup.
"""
import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import sdpa_generic as SG                                       # noqa: E402


class _T:
    """A stand-in tensor with the three attributes the cache key and the address check read."""
    padded_shape, dtype = (736, 4, 736, 32), "BFLOAT16"

    def buffer_address(self):
        return 0


def test_a_different_scale_is_a_different_program(monkeypatch):
    built = []
    monkeypatch.setattr(SG, "_CACHE", {})
    monkeypatch.setattr(SG, "build", lambda *a, **k: built.append(a[-1]) or {"addrs": (0,) * 5, "pd": None})
    monkeypatch.setattr(SG.ttnn, "generic_op", lambda *a, **k: None)
    t = _T()
    for scale in (0.25, 32 ** -0.5, 0.25):
        SG.sdpa(None, t, t, t, t, t, 256, 736, (8, 8), (1, 0, 1, 0), scale)
    assert built == [0.25, 32 ** -0.5], "the head_dim-32 call reused the head_dim-16 program"
