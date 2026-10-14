"""ESMC's minimal_matmul projections default by arch: on for Wormhole, off for Blackhole; the switch overrides both."""
from __future__ import annotations

import pytest

from tt_bio import esmc, tenstorrent


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("switch", [None, True, False])
def test_esmc_mm_default_by_arch(wormhole, switch, monkeypatch):
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(esmc, "_ESMC_MM", switch)
    assert esmc._esmc_mm() is (wormhole if switch is None else switch)


class _T:
    def __init__(self, rows, cols, dtype):
        self.shape, self.dtype, self._v = (rows, cols), dtype, rows * cols

    def volume(self):
        return self._v


@pytest.mark.parametrize("act", [None, "silu"])
def test_mm_carries_fc1_silu(act, monkeypatch):
    """The SwiGLU's fc1a (activation="silu") takes minimal_matmul with the SiLU fused, not the linear."""
    calls = []
    monkeypatch.setattr(esmc, "_ESMC_MM", True)
    monkeypatch.setattr(esmc.ttnn.experimental, "minimal_matmul", lambda x, w, **kw: calls.append(kw) or "mm",
                        raising=False)
    lin = esmc._MMLin()
    lin.compute_kernel_config = None
    bf16 = esmc.ttnn.bfloat16
    kw = {"activation": act} if act else {}
    assert lin._lin(_T(esmc.ESMC_MM_MIN_ROWS, 960, bf16), _T(960, 2560, bf16), **kw) == "mm"
    assert ("fused_activation" in calls[0]) is bool(act)
