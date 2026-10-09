"""A private, per-process view of ttnn's kernel sources with a few headers patched.

ttnn compiles its device kernels at runtime from the headers it ships, and it looks for them under
``TT_METAL_RUNTIME_ROOT`` (with ``TT_METAL_HOME`` as the data root). Pointing both at a directory
that mirrors the installed package lets tt-bio change what a kernel function does without touching
``site-packages`` or rebuilding ``_ttnn.so``. Only the directories on the way to a patched file are
real; every sibling is a symlink back into the wheel, so the overlay is a few hundred inodes.

``bh_dram_read_split`` (Blackhole hosts, on at ``import tt_bio``): a DRAM read larger than 2 KiB is
issued as reads of at most 2 KiB. A large DRAM read alongside other cores' non-posted DRAM writes on
the same NoC can lose its read response on Blackhole and hang the chip; tt-metal works around it the
same way (tenstorrent/tt-metal#59622, issue #52270) and the ttnn wheels tt-bio runs on predate that.
Every Protenix-v2 hang diagnosed on p150a/p300c had its signature. Same bytes, so bit-identical.
``TT_BIO_BH_DRAM_READ_SPLIT=0`` turns it off; a ``TT_METAL_RUNTIME_ROOT`` the user set is respected.

``silu_f32`` (Protenix's ``silu_f32`` lever, see tenstorrent.silu_ckc): ``calculate_silu`` drops the
caller's ``APPROXIMATION_MODE`` (every sibling activation honours it) and on an fp32 dest runs the
accurate exp and a two-step reciprocal, 92 SFPU instructions a row on Wormhole. The patch threads the
flag through and, under approx mode only, runs ``calculate_silu_f32`` (kernels/silu_f32): 6e-6 of
float64 on the fp32 accumulator in 32 instructions. Sites with ``math_approx_mode=False`` compile
exactly as before.

``enable()`` must run before the first device open; a second call adds its names to the overlay
already enabled. Each overlay gets its own JIT cache: tt-metal keys a compiled kernel on its defines
and compile args, not on the headers it included, so a shared cache would serve stock binaries under
a patched root. A patch whose anchor text is missing (a ttnn version that changed the header, or
already carries the fix) raises rather than serving the unpatched kernel under a patched name.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
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


DATAFLOW_API = "tt_metal/hw/inc/api/dataflow/dataflow_api.h"
_READ_ANCHOR = """    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_read_one_packet<false>(src_noc_addr, dst_local_l1_addr, size, noc, read_req_vc);"""
_READ_SPLIT = """#ifdef ARCH_BLACKHOLE
    // tt-bio: tt-metal#59622 workaround. A read from a DRAM bank larger than 2 KiB is issued as reads of at most
    // 2 KiB; a large DRAM read alongside other cores' non-posted DRAM writes can stall the NoC on Blackhole.
    constexpr uint32_t BH_DRAM_READ_MAX_PACKET_SIZE = 2048;
    if (size > BH_DRAM_READ_MAX_PACKET_SIZE) {
        const uint32_t src_xy = static_cast<uint32_t>(src_noc_addr >> NOC_ADDR_COORD_SHIFT);
        bool src_is_dram = false;
        for (uint32_t bank = 0; bank < NUM_DRAM_BANKS; ++bank) {
            src_is_dram |= src_xy == dram_bank_to_noc_xy[noc][bank];
        }
        if (src_is_dram) {
            while (size > BH_DRAM_READ_MAX_PACKET_SIZE) {
                noc_async_read_one_packet<false>(
                    src_noc_addr, dst_local_l1_addr, BH_DRAM_READ_MAX_PACKET_SIZE, noc, read_req_vc);
                src_noc_addr += BH_DRAM_READ_MAX_PACKET_SIZE;
                dst_local_l1_addr += BH_DRAM_READ_MAX_PACKET_SIZE;
                size -= BH_DRAM_READ_MAX_PACKET_SIZE;
            }
            noc_async_read_one_packet<false>(src_noc_addr, dst_local_l1_addr, size, noc, read_req_vc);
            return;
        }
    }
#endif
"""


def _patch_dram_read_split(src: str) -> str:
    if "BH_DRAM_READ_MAX_PACKET_SIZE" in src or src.count(_READ_ANCHOR) != 1:
        raise RuntimeError("bh_dram_read_split: dataflow_api.h anchor moved or already patched")
    return src.replace(_READ_ANCHOR, _READ_SPLIT + _READ_ANCHOR)


PATCHES = {
    "bh_dram_read_split": {DATAFLOW_API: _patch_dram_read_split},
    "silu_f32": {
        f"{_SFPU.format(arch=arch)}/{name}": fn
        for arch in ARCHES
        for name, fn in (
            ("ckernel_sfpu_silu.h", _patch_silu_f32),
            ("llk_math_eltwise_unary_sfpu_silu.h", _patch_silu_llk),
            ("ckernel_sfpu_silu_f32.h", _add_silu_f32),
        )
    },
}


def runtime_root() -> Path | None:
    """The runtime root ttnn would pick itself (the directory holding ``tt_metal/``): the wheel's
    package dir or a source build's checkout. Found without importing ttnn."""
    spec = importlib.util.find_spec("ttnn")
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg = Path(next(iter(spec.submodule_search_locations))).resolve()
    for root in (pkg, pkg.parent.parent):
        if (root / "tt_metal").is_dir():
            return root
    return None


def build(names, root: Path | None = None, cache: Path | None = None) -> Path:
    """Build (or reuse) the overlay of ``root`` with the named patches applied; return its path."""
    root = (root or Path(os.environ.get("TT_BIO_METAL_OVERLAY_STOCK") or runtime_root())).resolve()
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
        shutil.rmtree(tmp, ignore_errors=True)
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


def enable(names) -> Path:
    """Point this process's kernel compiler (and its children) at an overlay with ``names`` applied,
    plus whatever an earlier call enabled."""
    have = os.environ.get("TT_BIO_METAL_OVERLAY")
    want = sorted(set(names) | set(have.split(",") if have else ()))
    if have == ",".join(want):
        return Path(os.environ["TT_METAL_RUNTIME_ROOT"])
    if have is None:
        stock = runtime_root()
        if stock is None:
            raise RuntimeError("metal overlay: no ttnn runtime root found")
        os.environ["TT_BIO_METAL_OVERLAY_STOCK"] = str(stock)
        os.environ["TT_BIO_METAL_OVERLAY_CACHE"] = os.environ.get("TT_METAL_CACHE", "")
    out = build(want)
    os.environ["TT_METAL_RUNTIME_ROOT"] = str(out)
    if "TT_METAL_HOME" in os.environ:
        os.environ["TT_METAL_HOME"] = str(out)
    base = os.environ["TT_BIO_METAL_OVERLAY_CACHE"]
    os.environ["TT_METAL_CACHE"] = str(Path(base) / out.name if base else out / "jit")
    os.environ["TT_BIO_METAL_OVERLAY"] = ",".join(want)
    return out


def blackhole_host() -> bool:
    for dev in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            if (dev / "device" / "device").read_text().strip().lower() == "0xb140":
                return True
        except OSError:
            continue
    return False


def ensure_bh_dram_read_split() -> Path | None:
    """``bh_dram_read_split`` on a Blackhole host. Returns the overlay, or None when not applied."""
    if os.environ.get("TT_BIO_BH_DRAM_READ_SPLIT", "1") == "0":
        return None
    if "TT_METAL_RUNTIME_ROOT" in os.environ and "TT_BIO_METAL_OVERLAY" not in os.environ:
        return None
    if not blackhole_host():
        return None
    try:
        return enable(("bh_dram_read_split",))
    except (OSError, RuntimeError):
        return None
