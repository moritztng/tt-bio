"""The qkv_fused SDPA kernels are generated from the installed wheel's sources: every patched line
must still be there, and the stock line must survive in each `#else` (the switch-off path)."""
import pytest

pytest.importorskip("ttnn")
from tt_bio import sdpa_generic as SG  # noqa: E402


def test_qkv_fused_kernels_patch_every_site(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    d = SG.qkv_fused_kernel_dir() / "dataflow"
    reader = (d / "reader_interleaved.cpp").read_text()
    writer = (d / "writer_interleaved.cpp").read_text()
    common = (d / "dataflow_common.hpp").read_text()
    assert reader.count("#ifdef QKV_FUSED_W") == 3 and SG._Q_WINDOW_ORIG in reader
    assert all(line in reader for line in SG._KV_WINDOW_ORIG)
    assert writer.count("#ifdef QKV_FUSED_W") == 1 and SG._OUT_ID_ORIG in writer
    assert common.count("#ifdef QKV_FUSED_W") == len(SG._QKV_ROW_SKIPS)
