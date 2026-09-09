#!/usr/bin/env python3
"""Per-core Tensix reset-state census for one Wormhole chip, read-only.

Why this exists: a chip whose ``ttnn.open_device`` dies with
``Timeout (10000 ms) waiting for physical cores to finish: (x=..,y=..)`` gives you one
core coordinate and nothing else, and it costs an 11 s device open to get it. This reads
``RISCV_DEBUG_REG_SOFT_RESET_0`` from every functional worker core over the NOC in about
a second, without opening the device, so it works on a chip that cannot be opened and it
never touches tt-metal's host-wide device-open path.

A healthy Wormhole chip reads 0x47800 (all five RISCs held in reset) on all 80 cores, and
so does a wedged one at rest. Run this straight after a failed open, before anything else
touches the chip: the core the open timed out on is left reading 0x80047800 (UMD's
staggered-start bit still set because that core never completed reset deassert), and it is
the only core on the chip that does.

Numbering: the argument is the ``/dev/tenstorrent/N`` node number, which is NOT the UMD
logical id. Resolve with ``readlink /sys/class/tenstorrent/tenstorrent!N/device`` and sort
the PCI BDFs ascending to get UMD ids.

Only reads. Run it against a chip no process holds.

Wormhole ONLY, and that is enforced rather than documented: the coordinates and
SOFT_RESET_0 below are Wormhole B0's, and reading them on a Blackhole chip wedges
its ARC. Measured on pc's p150a 2026-09-09, 3/3 cycles: detect_chips() clean before,
"ARC Status: 0 out of 1 initialized" straight after, persistent until `tt-smi -r`.
That looked for half a day like a dead card.
"""
import argparse
import json
import sys

# NOC0 physical coordinates of the functional workers on Wormhole B0. Columns 0 and 5 are
# DRAM, rows 0 and 6 are ethernet, hence the gaps.
WORKER_X = (1, 2, 3, 4, 6, 7, 8, 9)
WORKER_Y = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)
SOFT_RESET_0 = 0xFFB121B0
RESET_HELD = 0x47800


class WrongArch(Exception):
    pass


def census(node):
    from pyluwen import PciChip

    chip = PciChip(pci_interface=node)
    # Constructing the chip and asking its architecture are both safe; the NOC reads below
    # are not, on anything but Wormhole. See the module docstring.
    if chip.as_wh() is None:
        arch = "Blackhole" if chip.as_bh() is not None else "unknown"
        raise WrongArch("/dev/tenstorrent/%d (%s) is %s, not Wormhole -- refusing to read "
                        "Wormhole NOC coordinates on it, that wedges the ARC and costs a "
                        "tt-smi -r to undo" % (node, chip.get_pci_bdf(), arch))
    cores = {}
    for y in WORKER_Y:
        for x in WORKER_X:
            try:
                cores[(x, y)] = chip.noc_read32(0, x, y, SOFT_RESET_0)
            except Exception:
                cores[(x, y)] = None
    return chip.get_pci_bdf(), cores


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("node", type=int, help="/dev/tenstorrent/N node number")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    try:
        bdf, cores = census(args.node)
    except WrongArch as e:
        print(e, file=sys.stderr)
        return 2
    odd = {c: v for c, v in cores.items() if v != RESET_HELD}
    if args.json:
        print(json.dumps({
            "node": args.node,
            "bdf": bdf,
            "n_cores": len(cores),
            "n_expected": sum(1 for v in cores.values() if v == RESET_HELD),
            "anomalous": [{"core": list(c),
                           "soft_reset_0": None if v is None else "0x%08x" % v}
                          for c, v in sorted(odd.items())],
        }))
    else:
        print("node %d (%s): %d/%d cores at the expected 0x%05x"
              % (args.node, bdf, len(cores) - len(odd), len(cores), RESET_HELD))
        for (x, y), v in sorted(odd.items()):
            print("  core (x=%d,y=%d) reads %s" % (x, y, "unreadable" if v is None else "0x%08x" % v))
    return 1 if odd else 0


if __name__ == "__main__":
    sys.exit(main())
