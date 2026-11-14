"""`triatt_ln` LN_SHARE applies only with two plain-read column groups writing bfp8 q|k|v. Bit-exactness of the pair
handshake, odd rows per pair included (608), is a device check: perf/spd_m_protenix/tl_attr.py exact_vs_first."""
import pytest

pytest.importorskip("ttnn")
from tt_bio import triatt_ln as TL  # noqa: E402


def test_gate(monkeypatch):
    monkeypatch.setattr(TL, "LN_SHARE", True)
    assert TL._share(2, False, 12, True)
    assert not TL._share(2, False, 12, False)   # bf16 outputs are write-bound: slower shared
    assert not TL._share(1, False, 12, True)
    assert not TL._share(2, True, 12, True)
    assert not TL._share(2, False, 0, True)
    monkeypatch.setattr(TL, "LN_SHARE", False)
    assert not TL._share(2, False, 12, True)
