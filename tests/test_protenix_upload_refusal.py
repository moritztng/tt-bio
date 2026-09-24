"""Protenix's block-0 OPM over the host-resident MSA falls back to depth chunks on a refusal.

Two refusals send it there. protenix-v2 at 1408 tokens against 8192 alignment rows was refused
the whole upload of the pristine m (2952790016 B, 596 MiB per bank free, 202 MiB largest block).
At 1280 tokens against 14743 rows the upload landed and OPM's 160 MiB depth slice of it was
refused (10 MiB largest block per bank). After either, OPM gets host-tiled depth chunks holding
exactly the rows of the whole tensor and the same residual, and the shape is remembered so the
next recycling cycle does not ask again.

Host-only: `_up`, `ttnn.from_torch`, `ttnn.deallocate` and the OPM are stand-ins (a host tilize
still queries the cluster), so what runs is the control flow and the row partition, and no
device is opened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from tt_bio import protenix as P  # noqa: E402

REFUSAL = ("Out of Memory: Not enough space to allocate 2952790016 B DRAM buffer across 12 banks, "
           "where each bank needs to store 246067200 B, but bank size is 1073741792 B (allocated: "
           "449108160 B, free: 624633632 B, largest free block: 211896800 B)")


class _Z:
    def __init__(self, alive=True):
        self.alive = alive

    def is_allocated(self):
        return self.alive


class _Trunk:
    dtype = ttnn.bfloat16
    _opm_from_host = P.Trunk._opm_from_host

    def __init__(self, refuse_upload=False):
        self.refuse_upload, self.ups = refuse_upload, 0

    def _up(self, t):
        self.ups += 1
        if self.refuse_upload:
            raise RuntimeError(REFUSAL)
        return "whole"


class _Opm:
    """Refuses a whole (device) input when `refuse_whole`, returns what it was given."""

    def __init__(self, refuse_whole=False):
        self.refuse_whole, self.calls = refuse_whole, []

    def __call__(self, x, mask, n, residual):
        self.calls.append(x)
        if x == "whole" and self.refuse_whole:
            raise RuntimeError(REFUSAL)
        return x, residual


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    P._UPLOAD_REFUSED_ROWS.clear()
    freed = []
    monkeypatch.setattr(P.ttnn, "deallocate", freed.append)
    monkeypatch.setattr(P.ttnn, "from_torch", lambda t, **kw: (t, kw))
    yield freed
    P._UPLOAD_REFUSED_ROWS.clear()


def _check_chunks(out, m, tr, z):
    parts, res = out
    assert res is z
    assert [p.shape[1] for p, _ in parts] == [512, 512, 76]
    assert torch.equal(torch.cat([p for p, _ in parts], dim=1), m)
    # Host chunks, in the dtype the whole upload would have had; OPM uploads them one at a time.
    assert all("device" not in kw and kw["dtype"] == tr.dtype for _, kw in parts)


def test_an_opm_that_fits_reads_the_whole_upload_and_frees_it(_stubs):
    tr, opm, z = _Trunk(), _Opm(), _Z()
    assert tr._opm_from_host(opm, torch.zeros(1, 1100, 64, 32), z) == ("whole", z)
    assert _stubs == ["whole"]
    assert not P._UPLOAD_REFUSED_ROWS


@pytest.mark.parametrize("where", ["upload", "opm"])
def test_a_refusal_comes_back_as_the_same_rows_in_host_chunks(_stubs, where):
    m = torch.randn(1, 1100, 64, 32).to(torch.bfloat16)
    tr, opm, z = _Trunk(refuse_upload=where == "upload"), _Opm(refuse_whole=where == "opm"), _Z()
    _check_chunks(tr._opm_from_host(opm, m, z), m, tr, z)
    # A refused OPM still gives the whole upload back before the chunks go up.
    assert _stubs == ([] if where == "upload" else ["whole"])
    # The next cycle goes straight to the chunks instead of paying the refused attempt again.
    _check_chunks(tr._opm_from_host(opm, m, z), m, tr, z)
    assert tr.ups == 1


def test_a_consumed_residual_is_not_retried():
    with pytest.raises(RuntimeError, match="consumed its residual"):
        _Trunk()._opm_from_host(_Opm(refuse_whole=True), torch.zeros(1, 64, 32, 32), _Z(False))


def test_a_non_allocator_error_is_not_retried():
    class _Broken(_Trunk):
        def _up(self, t):
            raise RuntimeError("TT_FATAL: bad shape")

    with pytest.raises(RuntimeError, match="bad shape"):
        _Broken()._opm_from_host(_Opm(), torch.zeros(1, 64, 32, 32), _Z())
