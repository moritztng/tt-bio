"""A private, per-process view of ttnn's kernel sources with a few headers patched.

ttnn compiles its device kernels at runtime from the headers it ships, and it looks for them under
``TT_METAL_RUNTIME_ROOT`` (with ``TT_METAL_HOME`` as the data root). Pointing both at a directory
that mirrors the installed package lets tt-bio change what a kernel function does without touching
``site-packages`` or rebuilding ``_ttnn.so``. Only the directories on the way to a patched file are
real; every sibling is a symlink back into the wheel, so the overlay is a few hundred inodes.

The one patch today, ``silu_f32``: ``calculate_silu`` drops the caller's ``APPROXIMATION_MODE``
(every sibling activation honours it) and on an fp32 dest runs the accurate exp and a two-step
reciprocal, 92 SFPU instructions a row on Wormhole. The patch threads the flag through and, under
approx mode only, runs ``calculate_silu_f32`` (kernels/silu_f32): 6e-6 of float64 on the fp32
accumulator in 32 instructions. In the matmul without bias that silu also moves from the MATH thread
to the PACK thread, so it overlaps the next subblock's matmul (what tenstorrent/tt-metal#43067 does
upstream). Sites with ``math_approx_mode=False`` compute exactly as before,
and tt-bio sets that flag on a fused silu from the ``silu_f32`` lever (``tenstorrent.silu_ckc``).

``enable()`` must run before the first device open. A patch whose anchor text is missing (a ttnn
version that changed the header) raises rather than serving the unpatched kernel under a patched
name.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

ARCHES = ("wormhole_b0", "blackhole")
_SFPU = "tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu"


def _patch_silu_llk(src: str) -> str:
    src, n = re.subn(
        r"calculate_silu<is_fp32_dest_acc_en, (8|ITERATIONS)>",
        r"calculate_silu<is_fp32_dest_acc_en, \1, APPROXIMATE>",
        src,
    )
    if n != 1:
        raise RuntimeError(f"silu_f32: llk_math_eltwise_unary_sfpu_silu.h anchor matched {n} times")
    return src


_SILU_F32 = Path(__file__).resolve().parent / "kernels" / "silu_f32" / "ckernel_sfpu_silu_f32.h"


def _patch_silu_f32(src: str) -> str:
    """Under math_approx_mode, silu runs calculate_silu_f32 (a few ulp of float32, 32 instructions a row)."""
    src, n0 = re.subn(r'(#include "ckernel_sfpu_sigmoid.h"\n)', r'\1#include "ckernel_sfpu_silu_f32.h"\n', src)
    src, n1 = re.subn(
        r"template <bool is_fp32_dest_acc_en, int ITERATIONS>\s*\ninline void calculate_silu\(\) \{\n",
        "template <bool is_fp32_dest_acc_en, int ITERATIONS, bool APPROXIMATION_MODE = false>\n"
        "inline void calculate_silu() {\n"
        "    if constexpr (APPROXIMATION_MODE) {\n"
        "        calculate_silu_f32<is_fp32_dest_acc_en, ITERATIONS>();\n"
        "        return;\n"
        "    }\n",
        src,
    )
    src, n2 = re.subn(
        r"(template <bool APPROXIMATION_MODE>\s*\ninline void silu_init\(\) \{\n)",
        r"\1    if constexpr (APPROXIMATION_MODE) {\n        silu_f32_init();\n        return;\n    }\n",
        src,
    )
    if (n0, n1, n2) != (1, 1, 1):
        raise RuntimeError(f"silu_f32: ckernel_sfpu_silu.h anchors matched {n0}, {n1}, {n2} times")
    return src


def _add_silu_f32(src: str | None) -> str:
    return _SILU_F32.read_text()


BMM = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp"
_BMM_DECL = r"""
// tt-bio silu_f32: a fused silu under math_approx_mode runs on the PACK thread, so the SFPU works on one dest half
// while the FPU fills the other (tenstorrent/tt-metal#43067 does the same for every activation). The pack thread
// cannot see APPROX, so MATH sends it through the thread mailbox once. Without bias only; everything else as shipped.
#if defined(SFPU_OP_INIT_ACTIVATION) && !defined(FUSE_BIAS)
#define TB_STR2(...) #__VA_ARGS__
#define TB_STR(...) TB_STR2(__VA_ARGS__)
constexpr bool tb_streq(const char* a, const char* b) { return *a == *b && (*a == 0 || tb_streq(a + 1, b + 1)); }
constexpr bool TB_SILU = tb_streq(TB_STR(SFPU_OP_FUNC_ACTIVATION), "silu_tile(i);");
#if defined(TRISC_MATH)
inline bool tb_silu_on_pack() {
    if constexpr (TB_SILU) {
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, APPROX);
    }
    return TB_SILU && APPROX;
}
#elif defined(TRISC_PACK)
#include "llk_math_eltwise_unary_sfpu_silu.h"
inline bool tb_silu_on_pack() {
    if constexpr (TB_SILU) {
        return ckernel::mailbox_read(ckernel::ThreadId::MathThreadId) != 0;
    }
    return false;
}
#else
inline bool tb_silu_on_pack() { return false; }
#endif
#endif
"""
_BMM_INIT = """#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif
"""
_BMM_INIT_NEW = """#ifdef SFPU_OP_INIT_ACTIVATION
#ifndef FUSE_BIAS
    const bool tb_pack_silu = tb_silu_on_pack();
    if (tb_pack_silu) {
        PACK((llk_math_eltwise_unary_sfpu_silu_init<true>()));
    } else
#endif
    {
        SFPU_OP_INIT_ACTIVATION
    }
#endif
"""
_BMM_LAST = """#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                    SFPU_OP_FUNC_ACTIVATION
                                }
#endif
                                tile_regs_commit();
                                // Pack out to output buffer
                                mm_out_cb.reserve_back(out_subblock_num_tiles);
                                tile_regs_wait();
"""
_BMM_LAST_NEW = """#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                                if (!tb_pack_silu) {
                                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                        SFPU_OP_FUNC_ACTIVATION
                                    }
                                }
#endif
                                tile_regs_commit();
                                // Pack out to output buffer
                                mm_out_cb.reserve_back(out_subblock_num_tiles);
#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                                if (tb_pack_silu) {
                                    // tile_regs_wait() that also holds the SETC16 until MATH is done with this half
                                    PACK(TTI_SEMWAIT(
                                        p_stall::STALL_TDMA | p_stall::STALL_CFG,
                                        semaphore::t6_sem(semaphore::MATH_PACK),
                                        p_stall::STALL_ON_ZERO));
                                    PACK(TT_SETC16(
                                        DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                        PACK((llk_math_eltwise_unary_sfpu_silu<true, DST_ACCUM_MODE>(i)));
                                    }
                                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                                } else
#endif
                                {
                                    tile_regs_wait();
                                }
"""
_BMM_INC = '#include "api/compute/eltwise_unary/sfpu_split_includes.h"\n'


def _patch_bmm_silu_pack(src: str) -> str:
    if "tb_silu_on_pack" in src or [src.count(a) for a in (_BMM_INC, _BMM_INIT, _BMM_LAST)] != [1, 1, 1]:
        raise RuntimeError("silu_f32: bmm_large_block_zm_fused_bias_activation.cpp anchors moved or already patched")
    return (src.replace(_BMM_INC, _BMM_INC + _BMM_DECL).replace(_BMM_INIT, _BMM_INIT_NEW)
            .replace(_BMM_LAST, _BMM_LAST_NEW))


PATCHES = {
    "silu_f32": {
        f"{_SFPU.format(arch=arch)}/{name}": fn
        for arch in ARCHES
        for name, fn in (
            ("ckernel_sfpu_silu.h", _patch_silu_f32),
            ("llk_math_eltwise_unary_sfpu_silu.h", _patch_silu_llk),
            ("ckernel_sfpu_silu_f32.h", _add_silu_f32),
        )
    }
    | {BMM: _patch_bmm_silu_pack},
}


def runtime_root() -> Path:
    """The installed package's runtime root: the directory holding ``tt_metal/`` and ``ttnn/cpp``."""
    env = os.environ.get("TT_METAL_RUNTIME_ROOT")
    if env:
        return Path(env)
    import ttnn

    return Path(ttnn.__file__).resolve().parent


def build(names, root: Path | None = None, cache: Path | None = None) -> Path:
    """Build (or reuse) the overlay of ``root`` with the named patches applied; return its path."""
    root = (root or runtime_root()).resolve()
    files: dict[str, str] = {}
    for name in sorted(names):
        for rel, fn in PATCHES[name].items():
            if rel in files:
                src = files[rel]
            else:
                src = (root / rel).read_text() if (root / rel).exists() else None
            files[rel] = fn(src)
    key = hashlib.sha256(str(root).encode())
    for rel in sorted(files):
        key.update(rel.encode() + files[rel].encode())
    cache = cache or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tt_bio" / "metal_overlay"
    out = cache / key.hexdigest()[:16]
    if (out / ".complete").exists():
        return out
    tmp = cache / f".{out.name}.{os.getpid()}"
    for rel, text in files.items():
        _mirror(root, tmp, Path(rel))
        (tmp / rel).write_text(text)
    (tmp / ".complete").write_text("\n".join(sorted(files)) + "\n")
    try:
        tmp.rename(out)
    except OSError:  # another process built the same overlay first
        if not (out / ".complete").exists():
            raise
    return out


def _mirror(root: Path, dst: Path, rel: Path) -> None:
    """Make ``dst/rel``'s parents real directories whose other entries symlink into ``root``."""
    here_src, here_dst = root, dst
    for part in rel.parts[:-1]:
        here_dst.mkdir(parents=True, exist_ok=True)
        for entry in here_src.iterdir():
            link = here_dst / entry.name
            if entry.name != part and not link.exists() and not link.is_symlink():
                link.symlink_to(entry)
        here_src, here_dst = here_src / part, here_dst / part
        if here_dst.is_symlink():
            here_dst.unlink()
    here_dst.mkdir(parents=True, exist_ok=True)
    for entry in here_src.iterdir():
        link = here_dst / entry.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(entry)
    # The file itself must be a real file: writing through a link would edit the wheel.
    (here_dst / rel.name).unlink(missing_ok=True)


def enable(names=("silu_f32",)) -> Path:
    """Point this process's kernel compiler at an overlay with ``names`` applied."""
    have = os.environ.get("TT_BIO_METAL_OVERLAY")
    if have is not None:
        if have != ",".join(sorted(names)):
            raise RuntimeError(f"metal overlay {have} already enabled in this process, cannot add {names}")
        return Path(os.environ["TT_METAL_RUNTIME_ROOT"])
    out = build(names)
    os.environ["TT_METAL_RUNTIME_ROOT"] = str(out)
    os.environ["TT_METAL_HOME"] = str(out)
    # tt-metal keys a compiled kernel on its defines and compile args, not on the headers it
    # included or the root it came from, so in a shared JIT cache an overlay binary and the
    # wheel's binary for the same op and config are one entry: whichever process compiled first
    # serves both (measured on .114: wheel and overlay arms bit-identical). Give each
    # overlay its own cache under the one the process would have used.
    base = os.environ.get("TT_METAL_CACHE") or str(Path.home() / ".cache")
    os.environ["TT_METAL_CACHE"] = str(Path(base) / f"overlay-{out.name}")
    os.environ["TT_BIO_METAL_OVERLAY"] = ",".join(sorted(names))
    return out
