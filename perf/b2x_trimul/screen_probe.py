"""The one edit in this branch that is NOT behind a flag: eligible_gated now screens the group
count the descriptor builds (Nrt*Nt*Ct) instead of Nt**2. That is the correct number, but it is
live on the off path, so it must not decline a shape the old screen admitted -- opendde runs E6 by
default at slice_c = 256/384."""
import sys
from pathlib import Path
sys.path.insert(0, "/home/ttuser/.coworker/wt/b2x-trimul-fusion-unlock")
import ttnn
import tt_bio.reblock_permute as RB
from tt_bio.tenstorrent import get_device

dev = get_device()
bad = 0
for N in (128, 256, 288, 298, 320, 352, 384, 448, 512, 576, 640, 768, 1024, 1536, 2113):
    nt = (N + 31) // 32
    for cz in (64, 128, 256, 384):
        ct = cz // 32
        old = RB._split_plan(dev, nt * nt) is not None
        new = RB._split_plan(dev, nt * nt * ct) is not None
        if old and not new:
            print(f"REGRESSION N={N} slice_c={cz}: old screen admits, new declines")
            bad += 1
print("grid", dev.compute_with_storage_grid_size())
print("VERDICT:", "no shape loses its plan" if bad == 0 else f"{bad} SHAPES REGRESS")
