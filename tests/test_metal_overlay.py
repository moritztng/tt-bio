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


def _wheel_root():
    import importlib.util
    spec = importlib.util.find_spec("ttnn")
    if spec is None or spec.origin is None:
        pytest.skip("ttnn not installed")
    from pathlib import Path
    return Path(spec.origin).resolve().parent


@pytest.mark.parametrize("name", ["silu_approx", "silu_f32"])
def test_patches_apply_to_the_installed_wheel(tmp_path, name):
    """The anchors are checked against the headers ttnn actually ships, not a fixture of them."""
    out = MO.build([name], root=_wheel_root(), cache=tmp_path / "cache")
    for arch in MO.ARCHES:
        d = out / MO._SFPU.format(arch=arch)
        assert "APPROXIMATE>" in (d / "llk_math_eltwise_unary_sfpu_silu.h").read_text()
        if name == "silu_f32":
            silu = (d / "ckernel_sfpu_silu.h").read_text()
            assert '#include "ckernel_sfpu_silu_f32.h"' in silu and "silu_f32_init();" in silu
            assert (d / "ckernel_sfpu_silu_f32.h").read_text() == MO._SILU_F32.read_text()


def test_silu_f32_and_silu_approx_are_exclusive(tmp_path):
    with pytest.raises(RuntimeError):
        MO.build(["silu_approx", "silu_f32"], root=_wheel_root(), cache=tmp_path / "cache")


def test_enable_gives_the_overlay_its_own_jit_cache(tmp_path, monkeypatch):
    """A shared JIT cache would serve the wheel's binary to the overlay (or the reverse)."""
    root = _wheel_root()
    monkeypatch.setenv("TT_METAL_RUNTIME_ROOT", str(root))
    monkeypatch.setenv("TT_METAL_HOME", str(root))
    monkeypatch.setenv("TT_METAL_CACHE", str(tmp_path / "jit"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("TT_BIO_METAL_OVERLAY", "")  # recorded, so the enable() below is undone
    monkeypatch.delenv("TT_BIO_METAL_OVERLAY")
    out = MO.enable(("silu_f32",))
    assert os.environ["TT_METAL_CACHE"] == str(tmp_path / "jit" / f"overlay-{out.name}")
    assert os.environ["TT_METAL_RUNTIME_ROOT"] == str(out)


def test_enable_is_idempotent_and_refuses_a_second_overlay(tmp_path, monkeypatch):
    root = _wheel_root()
    monkeypatch.setenv("TT_METAL_RUNTIME_ROOT", str(root))
    monkeypatch.setenv("TT_METAL_HOME", str(root))
    monkeypatch.setenv("TT_METAL_CACHE", str(tmp_path / "jit"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("TT_BIO_METAL_OVERLAY", "")  # recorded, so the enable() below is undone
    monkeypatch.delenv("TT_BIO_METAL_OVERLAY")
    out = MO.enable(("silu_f32",))
    cache = os.environ["TT_METAL_CACHE"]
    assert MO.enable(("silu_f32",)) == out and os.environ["TT_METAL_CACHE"] == cache
    with pytest.raises(RuntimeError):
        MO.enable(("silu_approx",))
