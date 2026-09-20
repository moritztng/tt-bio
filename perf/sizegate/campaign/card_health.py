#!/usr/bin/env python3
"""Is UMD card <n> alive, in the one sense that matters to a ladder walk: does its ARC answer?

A Blackhole that has taken a PCIe DPC containment event still enumerates, still reads its
config space and still sits in D0, so `lspci` and `/dev/tenstorrent/<n>` both look fine. What
is gone is the ARC: the driver logs "Failed to set initial power state: -5" and every
`ttnn.open_device` dies with "ARC core (8, 0) failed to start". A ladder runner without this
check marches its whole model list into that card, one 300 s model load at a time.

`tt_aiclk` is the cheap read that separates the two. A live chip reports 800 idle or 1350
under load; a chip whose ARC is dead reports 0xFFFFFFFF.

UMD numbers its chips by BDF sort order, which is NOT the /dev/tenstorrent node order: on
tt-quietbox UMD 0 is 0000:01:00.0 = node 1. So the node has to be looked up by BDF, never
assumed equal to the UMD id.
"""
import pathlib
import sys

ARC_DEAD = 0xFFFFFFFF
_TT_DRIVER = pathlib.Path("/sys/bus/pci/drivers/tenstorrent")
_TT_CLASS = pathlib.Path("/sys/class/tenstorrent")


def bdf_for_umd(umd: int) -> str:
    """UMD enumerates vendor-1e52 functions in BDF sort order, so UMD id is the index."""
    bdfs = sorted(p.name for p in _TT_DRIVER.glob("0000:*"))
    if not 0 <= umd < len(bdfs):
        raise LookupError(f"UMD {umd} out of range: host has {len(bdfs)} Tenstorrent chips")
    return bdfs[umd]


def node_for_bdf(bdf: str) -> pathlib.Path:
    for node in _TT_CLASS.glob("tenstorrent!*"):
        uevent = (node / "device" / "uevent").read_text()
        if f"PCI_SLOT_NAME={bdf}" in uevent:
            return node
    raise LookupError(f"no /dev/tenstorrent node claims {bdf}")


def aiclk(umd: int) -> int:
    return int((node_for_bdf(bdf_for_umd(umd)) / "tt_aiclk").read_text().strip())


def alive(umd: int) -> bool:
    try:
        return aiclk(umd) != ARC_DEAD
    except (LookupError, OSError, ValueError):
        return False


if __name__ == "__main__":
    umd = int(sys.argv[1])
    try:
        clk = aiclk(umd)
    except (LookupError, OSError, ValueError) as e:
        print(f"card {umd}: unreadable ({e})")
        raise SystemExit(2)
    if clk == ARC_DEAD:
        print(f"card {umd} ({bdf_for_umd(umd)}): ARC DEAD (tt_aiclk reads 0xFFFFFFFF)")
        raise SystemExit(1)
    print(f"card {umd} ({bdf_for_umd(umd)}): alive, AICLK {clk} MHz")
