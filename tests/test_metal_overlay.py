"""metal_overlay builds a patched view of a runtime root without writing into it."""
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
