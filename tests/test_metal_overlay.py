"""metal_overlay builds a patched view of a runtime root without writing into it."""
import os

import pytest

from tt_bio import metal_overlay as MO

SILU = """#include "ckernel_sfpu_sigmoid.h"
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);
}
template <bool APPROXIMATION_MODE>
inline void silu_init() {
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
    bmm = root / MO.BMM
    bmm.parent.mkdir(parents=True)
    bmm.write_text(MO._BMM_INC + "main {\n" + MO._BMM_INIT + "loop {\n" + MO._BMM_LAST + "}\n}\n")
    return root


def test_overlay_patches_and_leaves_root_alone(tmp_path):
    root = _root(tmp_path)
    before = {p: p.read_text() for p in root.rglob("*") if p.is_file()}
    out = MO.build(["silu_f32"], root=root, cache=tmp_path / "cache")
    assert {p: p.read_text() for p in root.rglob("*") if p.is_file()} == before
    for arch, it in (("wormhole_b0", "8"), ("blackhole", "ITERATIONS")):
        d = out / MO._SFPU.format(arch=arch)
        assert not (d / "ckernel_sfpu_silu.h").is_symlink()
        assert "calculate_silu_f32<is_fp32_dest_acc_en, ITERATIONS>();" in (d / "ckernel_sfpu_silu.h").read_text()
        assert (d / "ckernel_sfpu_silu_f32.h").read_text() == MO._SILU_F32.read_text()
        assert f"calculate_silu<is_fp32_dest_acc_en, {it}, APPROXIMATE>" in (d / "llk_math_eltwise_unary_sfpu_silu.h").read_text()
        assert (d / "ckernel_sfpu_exp.h").is_symlink() and (d / "ckernel_sfpu_exp.h").read_text() == "exp\n"
    bmm = (out / MO.BMM).read_text()
    assert bmm.count("tb_silu_on_pack") == 4 and "if (tb_pack_silu)" in bmm
    assert MO.build(["silu_f32"], root=root, cache=tmp_path / "cache") == out


def test_missing_anchor_raises(tmp_path):
    root = _root(tmp_path)
    (root / MO._SFPU.format(arch="blackhole") / "ckernel_sfpu_silu.h").write_text("changed upstream\n")
    with pytest.raises(RuntimeError):
        MO.build(["silu_f32"], root=root, cache=tmp_path / "cache")


def test_patch_applies_to_the_installed_wheel(tmp_path):
    """The anchors are checked against the headers ttnn actually ships, not a fixture of them."""
    root = MO.runtime_root()
    if root is None:
        pytest.skip("ttnn not installed")
    out = MO.build(["silu_f32", "bh_dram_read_split"], root=root, cache=tmp_path / "cache")
    for arch in MO.ARCHES:
        d = out / MO._SFPU.format(arch=arch)
        assert "APPROXIMATE>" in (d / "llk_math_eltwise_unary_sfpu_silu.h").read_text()
        silu = (d / "ckernel_sfpu_silu.h").read_text()
        assert '#include "ckernel_sfpu_silu_f32.h"' in silu and "silu_f32_init();" in silu
        assert (d / "ckernel_sfpu_silu_f32.h").read_text() == MO._SILU_F32.read_text()
    assert "PACK((llk_math_eltwise_unary_sfpu_silu<true, DST_ACCUM_MODE>(i)));" in (out / MO.BMM).read_text()


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
        # setenv first so monkeypatch records the variable: enable() writes os.environ directly, and a
        # variable monkeypatch never touched would leak the fake overlay root into every later test.
        monkeypatch.setenv(v, "")
        monkeypatch.delenv(v)


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
    both = MO.enable(("silu_f32",))
    assert both != first and os.environ["TT_BIO_METAL_OVERLAY"] == "bh_dram_read_split,silu_f32"
    assert "BH_DRAM_READ_MAX_PACKET_SIZE" in (both / MO.DATAFLOW_API).read_text()
    assert "calculate_silu_f32" in (both / MO._SFPU.format(arch="blackhole") / "ckernel_sfpu_silu.h").read_text()
    assert os.environ["TT_METAL_CACHE"] == str(tmp_path / "jc" / both.name)


def _kernel_ld(root, kb):
    ld = root / MO.BH_KERNEL_LD
    ld.parent.mkdir(parents=True)
    ld.write_text("  .segments 0 (INFO) :\n  {\n    LONG(ADDR(.text)) LONG(ADDR(.text))\n"
                  f"    LONG(({kb} * 1024)\n         - (__fw_export_text_end - 13088)\n         )\n")


@pytest.mark.parametrize("kb,want", [
    (24, False),     # stock 0.68.0, and the first bh.eth1 build that only widened the firmware region
    (40, True),      # scripts/ttnn_bh_eth
    (None, False),   # no Blackhole toolchain at all
])
def test_bh_eth_dispatch_supported(tmp_path, kb, want):
    if kb is not None:
        _kernel_ld(tmp_path, kb)
    assert MO.bh_eth_dispatch_supported(tmp_path) is want


def test_bh_eth_dispatch_reads_the_users_runtime_root(tmp_path, monkeypatch):
    _kernel_ld(tmp_path, 40)
    monkeypatch.setenv("TT_METAL_RUNTIME_ROOT", str(tmp_path))
    assert MO.bh_eth_dispatch_supported()
