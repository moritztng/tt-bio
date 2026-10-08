"""A private, per-process view of ttnn's kernel sources with a few headers patched.

ttnn compiles its device kernels at runtime from the headers it ships, and it looks for them under
``TT_METAL_RUNTIME_ROOT`` (with ``TT_METAL_HOME`` as the data root). Pointing both at a directory
that mirrors the installed package lets tt-bio change what a kernel function does without touching
``site-packages`` or rebuilding ``_ttnn.so``. Only the directories on the way to a patched file are
real; every sibling is a symlink back into the wheel, so the overlay is a few hundred inodes.

The one patch today, ``silu_approx``: ``calculate_silu`` drops the caller's ``APPROXIMATION_MODE``
(every sibling activation honours it), so a matmul with ``fp32_dest_acc_en=True`` and
``math_approx_mode=True`` -- every Protenix Transition fc1 -- runs the accurate exp and a two-step
reciprocal on each element. The patch threads the flag through: under approx mode silu takes
exp_21f (about 1 ulp of bfloat16) and one Newton step, still on the fp32 accumulator, so the input
is never rounded. Sites with ``math_approx_mode=False`` compile exactly as before.

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


def _patch_silu_kernel(src: str) -> str:
    src, n1 = re.subn(
        r"template <bool is_fp32_dest_acc_en, int ITERATIONS>\s*\ninline void calculate_silu\(\)",
        "template <bool is_fp32_dest_acc_en, int ITERATIONS, bool APPROXIMATION_MODE = false>\n"
        "inline void calculate_silu()",
        src,
    )
    src, n2 = re.subn(
        r"x \* _sfpu_sigmoid_<is_fp32_dest_acc_en>\(x\)",
        "x * _sfpu_sigmoid_<is_fp32_dest_acc_en && !APPROXIMATION_MODE>(x)",
        src,
    )
    if (n1, n2) != (1, 1):
        raise RuntimeError(f"silu_approx: ckernel_sfpu_silu.h anchors matched {n1}, {n2} times")
    return src


def _patch_silu_llk(src: str) -> str:
    src, n = re.subn(
        r"calculate_silu<is_fp32_dest_acc_en, (8|ITERATIONS)>",
        r"calculate_silu<is_fp32_dest_acc_en, \1, APPROXIMATE>",
        src,
    )
    if n != 1:
        raise RuntimeError(f"silu_approx: llk_math_eltwise_unary_sfpu_silu.h anchor matched {n} times")
    return src


PATCHES = {
    "silu_approx": {
        f"{_SFPU.format(arch=arch)}/{name}": fn
        for arch in ARCHES
        for name, fn in (
            ("ckernel_sfpu_silu.h", _patch_silu_kernel),
            ("llk_math_eltwise_unary_sfpu_silu.h", _patch_silu_llk),
        )
    },
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
            src = (root / rel).read_text() if rel not in files else files[rel]
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
    (here_dst / rel.name).unlink()


def enable(names=("silu_approx",)) -> Path:
    """Point this process's kernel compiler at an overlay with ``names`` applied."""
    out = build(names)
    os.environ["TT_METAL_RUNTIME_ROOT"] = str(out)
    os.environ["TT_METAL_HOME"] = str(out)
    return out
