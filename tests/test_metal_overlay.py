"""tt_bio.metal_overlay: the Blackhole DRAM-read split lands in a mirrored runtime root, never in the installed one."""
import os

import pytest

from tt_bio import metal_overlay as mo


def _fake_root(tmp_path):
    root = tmp_path / "ttnn"
    hdr = root / mo.HEADER
    hdr.parent.mkdir(parents=True)
    hdr.write_text("// head\n" + mo.ANCHOR + "\n    }\n")
    (hdr.parent / "sibling.h").write_text("sibling")
    desc = root / mo.ETH_DESCRIPTOR
    desc.parent.mkdir(parents=True)
    desc.write_text("dispatch_cores:\n  [[0, 0], [0, 1], [0, 11]" + mo.ETH_CORES_PAST_12 + "\n")
    (desc.parent / "wormhole_b0_80_arch.yaml").write_text("wh")
    (root / "tt_metal" / "other").mkdir()
    (root / "runtime").mkdir()
    return root


def test_patch_inserts_split_once_and_refuses_moved_anchor():
    text = "x\n" + mo.ANCHOR + "\n"
    out = mo.patched_header(text)
    assert out.count("BH_DRAM_READ_MAX_PACKET_SIZE = 2048") == 1
    assert out.index("#ifdef ARCH_BLACKHOLE") < out.index(mo.ANCHOR)
    assert mo.patched_header(out) is None          # already patched
    assert mo.patched_header("no anchor") is None  # newer tt-metal


def test_root_mirrors_installed_tree(tmp_path):
    stock = _fake_root(tmp_path)
    root = mo.build_root(stock, tmp_path / "cache")
    assert "BH_DRAM_READ_MAX_PACKET_SIZE" in (root / mo.HEADER).read_text()
    assert not (root / mo.HEADER).is_symlink()
    assert (root / mo.HEADER).parent.joinpath("sibling.h").resolve() == (stock / mo.HEADER).parent / "sibling.h"
    assert (root / "runtime").is_symlink() and (root / "tt_metal" / "other").is_symlink()
    assert "BH_DRAM" not in (stock / mo.HEADER).read_text()   # the installed header is untouched
    assert mo.build_root(stock, tmp_path / "cache") == root    # reused, not rebuilt


def test_eth_descriptor_lists_twelve_dispatch_cores(tmp_path):
    stock = _fake_root(tmp_path)
    root = mo.build_root(stock, tmp_path / "cache")
    desc = (root / mo.ETH_DESCRIPTOR).read_text()
    assert "[0, 11]]" in desc and "[0, 12]" not in desc and not (root / mo.ETH_DESCRIPTOR).is_symlink()
    assert (root / mo.ETH_DESCRIPTOR).parent.joinpath("wormhole_b0_80_arch.yaml").is_symlink()
    assert "[0, 13]" in (stock / mo.ETH_DESCRIPTOR).read_text()   # the installed descriptor is untouched
    assert mo.patched_eth_descriptor(desc) is None


@pytest.mark.parametrize("bh,off,user_root", [(False, False, False), (True, True, False), (True, False, True)])
def test_left_alone(monkeypatch, tmp_path, bh, off, user_root):
    monkeypatch.setattr(mo, "blackhole_host", lambda: bh)
    monkeypatch.setattr(mo, "stock_root", lambda: _fake_root(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("TT_METAL_RUNTIME_ROOT", raising=False)
    if off:
        monkeypatch.setenv("TT_BIO_BH_DRAM_READ_SPLIT", "0")
    if user_root:
        monkeypatch.setenv("TT_METAL_RUNTIME_ROOT", "/mine")
    assert mo.ensure_bh_dram_read_split() is None
    assert os.environ.get("TT_METAL_RUNTIME_ROOT") == ("/mine" if user_root else None)


def test_applied_on_blackhole_with_its_own_jit_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(mo, "blackhole_host", lambda: True)
    monkeypatch.setattr(mo, "stock_root", lambda: _fake_root(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("TT_METAL_RUNTIME_ROOT", raising=False)
    monkeypatch.setenv("TT_METAL_CACHE", str(tmp_path / "jc"))
    root = mo.ensure_bh_dram_read_split()
    assert os.environ["TT_METAL_RUNTIME_ROOT"] == str(root)
    assert os.environ["TT_METAL_CACHE"] == str(tmp_path / "jc" / root.name)
