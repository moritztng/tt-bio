"""`tenstorrent.arch_name()` names the part from sysfs and never builds tt-metal's MetalContext.

`ttnn.get_arch_name()` constructs the UMD cluster, which holds the chip's driver fds until
ReleaseOwnership. Asked before any `get_device()`, it left a card open that `cleanup()` had no
handle to free (pytest_device:1/4 wh at 938d50505). Opens no device.
"""
import glob

import pytest

T = pytest.importorskip("tt_bio.tenstorrent")


def _sysfs(monkeypatch, tmp_path, ids):
    nodes = []
    for i, pci in enumerate(ids):
        d = tmp_path / f"tenstorrent!{i}" / "device"
        d.mkdir(parents=True)
        (d / "device").write_text(pci + "\n")
        nodes.append(str(d / "device"))
    monkeypatch.setattr(glob, "glob", lambda pattern: nodes)
    T.arch_name.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_cache():
    T.arch_name.cache_clear()
    yield
    T.arch_name.cache_clear()


@pytest.mark.parametrize("ids,arch", [(["0x401e"] * 32, "wormhole_b0"), (["0xb140"] * 4, "blackhole")])
def test_sysfs_names_the_part_without_ttnn(monkeypatch, tmp_path, ids, arch):
    _sysfs(monkeypatch, tmp_path, ids)
    monkeypatch.setattr(T.ttnn, "get_arch_name", lambda: pytest.fail("built MetalContext"))
    assert T.arch_name() == arch


@pytest.mark.parametrize("ids", [[], ["0x401e", "0xb140"], ["0xfaca"]])
def test_ttnn_answers_only_where_sysfs_does_not(monkeypatch, tmp_path, ids):
    _sysfs(monkeypatch, tmp_path, ids)
    monkeypatch.setattr(T.ttnn, "get_arch_name", lambda: "from-ttnn")
    assert T.arch_name() == "from-ttnn"
