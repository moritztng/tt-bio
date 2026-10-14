#!/usr/bin/env python3
"""cgctl.py read | set <hex> | hyst <hex>: Tensix clock-gate enable (CG_CTRL_EN, 0xFFB12244, one bit per block) on every Tensix
tile of the one chip TT_VISIBLE_DEVICES pins. Firmware 19.15 EnableTensixCG writes 0xFFFFFFFF (all 17 blocks, hysteresis
2) at boot, board reset and REINIT_TENSIX; 19.14 skipped it (inverted cg_en). `set 0` turns Tensix clock gating off
until the next reset, so the qb2 chip-0 hang can be A/B'd against CG without a reflash. Bits: 5 packers, 13 L1 return
muxes, 15 L1 banks, 16 src B (tt-system-firmware lib/tenstorrent/bh_arc/tensix_init.c). Prints the value histogram."""
import sys
from collections import Counter
from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device, write_words_to_device

CG_CTRL_EN = 0xFFB12244
CG_HYST = (0xFFB12070, 0xFFB12074, 0xFFB1207C)  # one byte per block (some share), fw writes 0x02020202
ctx = init_ttexalens()
dev = ctx.devices[0]
locs = dev.get_block_locations("functional_workers")
if sys.argv[1] == "set":
    val = int(sys.argv[2], 16)
    for loc in locs:
        write_words_to_device(loc, CG_CTRL_EN, [val], context=ctx, safe_mode=False)
if sys.argv[1] == "hyst":
    val = int(sys.argv[2], 16)
    for loc in locs:
        for a in CG_HYST:
            write_words_to_device(loc, a, [val], context=ctx, safe_mode=False)
hyst = Counter(tuple(read_word_from_device(loc, a, context=ctx, safe_mode=False) for a in CG_HYST) for loc in locs)
vals = Counter(read_word_from_device(loc, CG_CTRL_EN, context=ctx, safe_mode=False) for loc in locs)
print(f"CG_CTRL_EN tiles={len(locs)} " + " ".join(f"0x{v:08x}x{n}" for v, n in sorted(vals.items())) + " hyst " +
      " ".join("/".join(f"{w:08x}" for w in h) + f"x{n}" for h, n in sorted(hyst.items())), flush=True)
