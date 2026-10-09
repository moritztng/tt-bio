"""metal_overlay builds a patched view of a runtime root without writing into it."""
import os

import pytest

from tt_bio import metal_overlay as MO

SILU = """template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);
}
"""
LLK = "        ckernel::sfpu::calculate_silu<is_fp32_dest_acc_en, {it}>, dst_index, vector_mode);\n"


def _root(tmp_path):
    root = tmp_path / "ttnn"
    for arch, it in (("wormhole_b0", "8"), ("blackhole", "ITERATIONS")):
        d = root / MO._SFPU.format(arch=arch)
        d.mkdir(parents=True)
        (d / "ckernel_sfpu_silu.h").write_text(SILU)
        (d / "llk_math_eltwise_unary_sfpu_silu.h").write_text(LLK.format(it=it))
        (d / "ckernel_sfpu_exp.h").write_text("exp\n")
    (root / "ttnn" / "cpp").mkdir(parents=True)
    return root


def test_overlay_patches_and_leaves_root_alone(tmp_path):
    root = _root(tmp_path)
    before = {p: p.read_text() for p in root.rglob("*") if p.is_file()}
    out = MO.build(["silu_approx"], root=root, cache=tmp_path / "cache")
    assert {p: p.read_text() for p in root.rglob("*") if p.is_file()} == before
    for arch, it in (("wormhole_b0", "8"), ("blackhole", "ITERATIONS")):
        d = out / MO._SFPU.format(arch=arch)
        assert not (d / "ckernel_sfpu_silu.h").is_symlink()
        assert "_sfpu_sigmoid_<is_fp32_dest_acc_en && !APPROXIMATION_MODE>" in (d / "ckernel_sfpu_silu.h").read_text()
        assert f"calculate_silu<is_fp32_dest_acc_en, {it}, APPROXIMATE>" in (d / "llk_math_eltwise_unary_sfpu_silu.h").read_text()
        assert (d / "ckernel_sfpu_exp.h").is_symlink() and (d / "ckernel_sfpu_exp.h").read_text() == "exp\n"
    assert (out / "ttnn").is_symlink()
    assert MO.build(["silu_approx"], root=root, cache=tmp_path / "cache") == out


def test_missing_anchor_raises(tmp_path):
    root = _root(tmp_path)
    (root / MO._SFPU.format(arch="blackhole") / "ckernel_sfpu_silu.h").write_text("changed upstream\n")
    with pytest.raises(RuntimeError):
        MO.build(["silu_approx"], root=root, cache=tmp_path / "cache")


def _dataflow_root(tmp_path):
    root = tmp_path / "ttnn"
    hdr = root / MO.DATAFLOW_API
    hdr.parent.mkdir(parents=True)
    hdr.write_text("// head\n" + MO._READ_ANCHOR + "\n    }\n")
    (hdr.parent / "sibling.h").write_text("sibling")
    (root / "tt_metal" / "other").mkdir()
    (root / "runtime").mkdir()
    return root


def test_dram_read_split_inserts_once_and_refuses_a_moved_anchor():
    out = MO._patch_dram_read_split("x\n" + MO._READ_ANCHOR + "\n")
    assert out.count("BH_DRAM_READ_MAX_PACKET_SIZE = 2048") == 1
    assert out.index("#ifdef ARCH_BLACKHOLE") < out.index(MO._READ_ANCHOR)
    for src in (out, "no anchor"):  # already patched, newer tt-metal
        with pytest.raises(RuntimeError):
            MO._patch_dram_read_split(src)


def test_dram_read_split_mirrors_the_installed_tree(tmp_path):
    stock = _dataflow_root(tmp_path)
    out = MO.build(["bh_dram_read_split"], root=stock, cache=tmp_path / "cache")
    hdr = out / MO.DATAFLOW_API
    assert "BH_DRAM_READ_MAX_PACKET_SIZE" in hdr.read_text() and not hdr.is_symlink()
    assert hdr.parent.joinpath("sibling.h").resolve() == (stock / MO.DATAFLOW_API).parent / "sibling.h"
    assert (out / "runtime").is_symlink() and (out / "tt_metal" / "other").is_symlink()
    assert "BH_DRAM" not in (stock / MO.DATAFLOW_API).read_text()


def _clean_env(monkeypatch, tmp_path, stock):
    monkeypatch.setattr(MO, "runtime_root", lambda: stock)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    for v in ("TT_METAL_RUNTIME_ROOT", "TT_METAL_HOME", "TT_METAL_CACHE", "TT_BIO_BH_DRAM_READ_SPLIT",
              "TT_BIO_METAL_OVERLAY", "TT_BIO_METAL_OVERLAY_STOCK", "TT_BIO_METAL_OVERLAY_CACHE"):
        monkeypatch.delenv(v, raising=False)


@pytest.mark.parametrize("bh,off,user_root", [(False, False, False), (True, True, False), (True, False, True)])
def test_dram_read_split_left_alone(monkeypatch, tmp_path, bh, off, user_root):
    _clean_env(monkeypatch, tmp_path, _dataflow_root(tmp_path))
    monkeypatch.setattr(MO, "blackhole_host", lambda: bh)
    if off:
        monkeypatch.setenv("TT_BIO_BH_DRAM_READ_SPLIT", "0")
    if user_root:
        monkeypatch.setenv("TT_METAL_RUNTIME_ROOT", "/mine")
    assert MO.ensure_bh_dram_read_split() is None
    assert os.environ.get("TT_METAL_RUNTIME_ROOT") == ("/mine" if user_root else None)


def test_dram_read_split_on_blackhole_with_its_own_jit_cache(monkeypatch, tmp_path):
    _clean_env(monkeypatch, tmp_path, _dataflow_root(tmp_path))
    monkeypatch.setattr(MO, "blackhole_host", lambda: True)
    monkeypatch.setenv("TT_METAL_CACHE", str(tmp_path / "jc"))
    out = MO.ensure_bh_dram_read_split()
    assert os.environ["TT_METAL_RUNTIME_ROOT"] == str(out) and "TT_METAL_HOME" not in os.environ
    assert os.environ["TT_METAL_CACHE"] == str(tmp_path / "jc" / out.name)
    assert MO.ensure_bh_dram_read_split() == out  # a child process inherits it, does not stack it


def test_a_second_enable_adds_to_the_overlay(monkeypatch, tmp_path):
    stock = _root(tmp_path)
    hdr = stock / MO.DATAFLOW_API
    hdr.parent.mkdir(parents=True)
    hdr.write_text(MO._READ_ANCHOR + "\n")
    _clean_env(monkeypatch, tmp_path, stock)
    monkeypatch.setenv("TT_METAL_CACHE", str(tmp_path / "jc"))
    first = MO.enable(("bh_dram_read_split",))
    both = MO.enable(("silu_approx",))
    assert both != first and os.environ["TT_BIO_METAL_OVERLAY"] == "bh_dram_read_split,silu_approx"
    assert "BH_DRAM_READ_MAX_PACKET_SIZE" in (both / MO.DATAFLOW_API).read_text()
    assert "APPROXIMATION_MODE" in (both / MO._SFPU.format(arch="blackhole") / "ckernel_sfpu_silu.h").read_text()
    assert os.environ["TT_METAL_CACHE"] == str(tmp_path / "jc" / both.name)
