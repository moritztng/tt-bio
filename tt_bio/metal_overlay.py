"""Blackhole: split large DRAM reads into 2 KiB packets in every JIT-built kernel.

On Blackhole, a large DRAM read issued while other cores have non-posted DRAM writes in flight on the same NoC
can lose its read response, and the chip hangs until reset. tt-metal reproduced it on 11x10 p300 chips and works
around it by issuing DRAM reads of at most 2 KiB from ``noc_async_read`` (tenstorrent/tt-metal#59622, issue
#52270). The ttnn wheels tt-bio runs on predate that change. Every Protenix-v2 hang we diagnosed on p150a/p300c
had its signature: a few cores with NoC 1 DRAM reads that were sent and never answered, every other core waiting
in a write barrier.

Kernels are compiled from the headers under the runtime root, so the fix needs no new wheel: build a root that
mirrors the installed one, with ``dataflow_api.h`` carrying the split, point ``TT_METAL_RUNTIME_ROOT`` at it and
give it its own JIT cache (binaries built from the stock header must not be reused). Only the read is split, the
bytes moved are the same, so results are bit-identical. Wormhole hosts are left alone.

Runs once at ``import tt_bio``, before anything imports ttnn. ``TT_BIO_BH_DRAM_READ_SPLIT=0`` turns it off; a
``TT_METAL_RUNTIME_ROOT`` the user set is respected.
"""
import hashlib
import importlib.util
import os
from pathlib import Path

BLACKHOLE_PCI_ID = "0xb140"

ANCHOR = """    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_read_one_packet<false>(src_noc_addr, dst_local_l1_addr, size, noc, read_req_vc);"""

SPLIT = """#ifdef ARCH_BLACKHOLE
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

HEADER = Path("tt_metal/hw/inc/api/dataflow/dataflow_api.h")


def blackhole_host() -> bool:
    for dev in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            if (dev / "device" / "device").read_text().strip().lower() == BLACKHOLE_PCI_ID:
                return True
        except OSError:
            continue
    return False


def stock_root() -> Path | None:
    """The runtime root ttnn would pick itself: the wheel's package dir, or the checkout of a source build."""
    spec = importlib.util.find_spec("ttnn")
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg = Path(next(iter(spec.submodule_search_locations))).resolve()
    for root in (pkg, pkg.parent.parent):
        if (root / HEADER).is_file():
            return root
    return None


def patched_header(text: str) -> str | None:
    """``dataflow_api.h`` with the split, or None when the anchor moved (a newer tt-metal, or already patched)."""
    if "BH_DRAM_READ_MAX_PACKET_SIZE" in text or text.count(ANCHOR) != 1:
        return None
    return text.replace(ANCHOR, SPLIT + ANCHOR)


def _mirror(src: Path, dst: Path, keep: Path):
    """Symlink every entry of ``src`` into ``dst`` except the one on the path to ``keep``, which is recursed into."""
    dst.mkdir()
    head = keep.parts[0]
    for entry in src.iterdir():
        if entry.name == head and len(keep.parts) > 1:
            _mirror(entry, dst / head, Path(*keep.parts[1:]))
        elif entry.name != head:
            (dst / entry.name).symlink_to(entry)


def build_root(stock: Path, cache_home: Path) -> Path | None:
    patched = patched_header((stock / HEADER).read_text())
    if patched is None:
        return None
    tag = hashlib.sha256((str(stock) + patched).encode()).hexdigest()[:12]
    root = cache_home / f"bh-dram-split-{tag}"
    if (root / HEADER).is_file():
        return root
    tmp = cache_home / f".bh-dram-split-{tag}.{os.getpid()}"
    cache_home.mkdir(parents=True, exist_ok=True)
    _mirror(stock, tmp, HEADER)
    (tmp / HEADER).write_text(patched)
    try:
        tmp.rename(root)
    except OSError:  # another process won the race; its root is identical
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return root


def ensure_bh_dram_read_split():
    """Point the JIT at the patched root on a Blackhole host. Returns the root, or None when not applied."""
    if os.environ.get("TT_BIO_BH_DRAM_READ_SPLIT", "1") == "0" or "TT_METAL_RUNTIME_ROOT" in os.environ:
        return None
    if not blackhole_host():
        return None
    stock = stock_root()
    if stock is None:
        return None
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tt-bio"
    try:
        root = build_root(stock, cache_home)
    except OSError:
        return None
    if root is None:
        return None
    os.environ["TT_METAL_RUNTIME_ROOT"] = str(root)
    # The JIT cache key does not cover header contents, so binaries built from the stock header would be reused.
    user_cache = os.environ.get("TT_METAL_CACHE")
    os.environ["TT_METAL_CACHE"] = str(Path(user_cache) / root.name if user_cache else root / "jit")
    return root
