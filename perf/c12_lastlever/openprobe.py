#!/usr/bin/env python3
"""Does a device open on this card complete? Nothing else. Prints a stage line per step."""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
t0 = time.time()
def say(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
say(f"TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES')}")
import ttnn
say("ttnn imported")
from tt_bio import tenstorrent as TT
say("tt_bio imported, calling get_device()")
dev = TT.get_device()
say(f"DEVICE OPEN OK arch={dev.arch()}")
import torch
x = ttnn.from_torch(torch.randn(1, 64, 64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                    device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
say(f"roundtrip ok, sum={float(ttnn.to_torch(x).float().sum()):.4f}")
