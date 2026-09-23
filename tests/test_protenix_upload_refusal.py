"""Protenix's block-0 upload of the host-resident MSA falls back to depth chunks on a refusal.

protenix-v2 at 1408 tokens against 8192 alignment rows was refused the whole upload of the
pristine m (2952790016 B, 596 MiB per bank free, 202 MiB largest block). After a refusal the
upload comes back as host-tiled depth chunks holding exactly the rows of the whole tensor, and
the shape is remembered so the next recycling cycle does not ask again.

Host-only: `_up` and `ttnn.from_torch` are stand-ins (a host tilize still queries the cluster),
so what runs is the control flow and the row partition, and no device is opened.
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


class _Trunk:
    dtype = ttnn.bfloat16
    _up_or_host_chunks = P.Trunk._up_or_host_chunks

    def __init__(self, refuse):
        self.refuse, self.ups = refuse, 0

    def _up(self, t):
        self.ups += 1
        if self.refuse:
            raise RuntimeError(REFUSAL)
        return "whole"


@pytest.fixture(autouse=True)
def _fresh_memo():
    P._UPLOAD_REFUSED_ROWS.clear()
    yield
    P._UPLOAD_REFUSED_ROWS.clear()


def test_an_upload_that_fits_is_the_whole_upload():
    tr = _Trunk(refuse=False)
    assert tr._up_or_host_chunks(torch.zeros(1, 1100, 64, 32, dtype=torch.bfloat16)) == "whole"
    assert not P._UPLOAD_REFUSED_ROWS


def test_a_refused_upload_comes_back_as_the_same_rows_in_host_chunks(monkeypatch):
    calls = []

    def from_torch(t, **kw):
        calls.append(kw)
        return t

    monkeypatch.setattr(P.ttnn, "from_torch", from_torch)
    m = torch.randn(1, 1100, 64, 32).to(torch.bfloat16)
    tr = _Trunk(refuse=True)
    parts = tr._up_or_host_chunks(m)
    assert [p.shape[1] for p in parts] == [512, 512, 76]
    assert torch.equal(torch.cat(parts, dim=1), m)
    # Host chunks, in the dtype the whole upload would have had; OPM uploads them one at a time.
    assert all("device" not in kw and kw["dtype"] == tr.dtype for kw in calls)
    # The next cycle goes straight to the chunks instead of paying the refused upload again.
    tr._up_or_host_chunks(m)
    assert tr.ups == 1


def test_a_non_allocator_error_is_not_retried():
    class _Broken(_Trunk):
        def _up(self, t):
            raise RuntimeError("TT_FATAL: bad shape")

    with pytest.raises(RuntimeError, match="bad shape"):
        _Broken(refuse=False)._up_or_host_chunks(torch.zeros(1, 64, 32, 32))
