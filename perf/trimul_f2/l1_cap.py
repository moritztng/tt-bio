#!/usr/bin/env python3
"""How much L1 can F2 actually hold? A3 of the floor assumes 160.79 MB usable.

The F2 screen hit `Not enough space to allocate 33554432 B L1 buffer ... allocated: 1220608 B,
free: 241152 B` per bank, i.e. 26.5 MB free, in the middle of a run. F2's working set is ~105 MB.
This walks the real ceiling: on a fresh device, then with tensors already resident, then after a
generic_op program has run, so the number is a steady state and not one lucky moment.
"""
import json
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.tenstorrent import get_device                                   # noqa: E402
from tt_bio import reblock_permute as RP                                    # noqa: E402

dev = get_device()
L1 = ttnn.L1_MEMORY_CONFIG
OUT = {"stages": []}


def biggest(label, held):
    """Largest single L1 tensor that allocates, MB, by walking a tile-aligned ladder."""
    best, err = 0.0, ""
    got = None
    for mb in (8, 16, 32, 48, 64, 80, 96, 112, 128, 134, 144, 152, 160):
        rows = int(mb * 1e6 / 2 / 512 // 32 * 32)
        try:
            t = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, rows, 512]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, L1)
            best = rows * 512 * 2 / 1e6
            if got is not None:
                ttnn.deallocate(got)
            got = t
        except Exception as e:                                              # noqa: BLE001
            err = str(e).split("\n")[0][:150]
            break
    if got is not None:
        ttnn.deallocate(got)
    row = {"stage": label, "held_mb": held, "max_single_l1_mb": best, "first_refusal": err}
    OUT["stages"].append(row)
    print(f"{label:44s} held={held:6.1f} MB   max single L1 tensor = {best:7.1f} MB", flush=True)
    print(f"      first refusal: {err}", flush=True)
    return best


biggest("fresh device", 0.0)

# Hold a realistic F2 working set and see what is left.
held = []
tot = 0.0
for mb in (32, 32, 32):
    rows = int(mb * 1e6 / 2 / 512 // 32 * 32)
    try:
        held.append(ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, 512]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, L1))
        tot += rows * 512 * 2 / 1e6
    except Exception as e:                                                  # noqa: BLE001
        print(f"      could not hold {mb} MB: {str(e).split(chr(10))[0][:130]}", flush=True)
        break
biggest("with an F2-sized set resident", tot)
for t in held:
    ttnn.deallocate(t)

# Now run a generic_op (the reblock kernel) and re-measure: does its program reserve stay?
x = ttnn.from_torch(torch.randn(1, 512, 512, 32).bfloat16(), layout=ttnn.TILE_LAYOUT,
                    device=dev, dtype=ttnn.bfloat16)
o = RP.reblock_permute(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=dev)
ttnn.synchronize_device(dev)
ttnn.deallocate(o)
ttnn.deallocate(x)
biggest("after one reblock generic_op", 0.0)

x = ttnn.from_torch(torch.randn(1, 512, 512, 256).bfloat16(), layout=ttnn.TILE_LAYOUT,
                    device=dev, dtype=ttnn.bfloat16)
o = RP.reblock_permute(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=dev)
ttnn.synchronize_device(dev)
ttnn.deallocate(o)
ttnn.deallocate(x)
biggest("after a C=256 reblock generic_op", 0.0)

Path(sys.argv[1] if len(sys.argv) > 1 else "l1_cap.json").write_text(json.dumps(OUT, indent=1))
print("\nwrote", sys.argv[1] if len(sys.argv) > 1 else "l1_cap.json", flush=True)
